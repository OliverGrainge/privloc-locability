from dotenv import load_dotenv 
load_dotenv()

from utils import load_config_yaml 
from data.generate import load_dataset
from models.transforms import get_transform
from torch.utils.data import DataLoader, Subset
from models.geolocation import load_model
import pandas as pd 
import sys
import os 
import torch
import time 
import ray
import numpy as np


@ray.remote
class ResultCollector:
    """Collects and periodically saves results from workers"""
    
    def __init__(self, config, initial_df, output_path, save_interval=100):
        self.config = config
        self.df = initial_df
        self.output_path = output_path
        self.save_interval = save_interval
        self.batch_count = 0
        self.pending_results = []
        
    def add_results(self, batch_results):
        """Add batch results and save periodically"""
        self.pending_results.append(batch_results)
        self.batch_count += 1
        
        # Save every N batches
        if self.batch_count % self.save_interval == 0:
            self._merge_and_save()
            return True  # Indicate a save occurred
        return False
    
    def _merge_and_save(self):
        """Merge pending results and save to disk"""
        if not self.pending_results:
            return
        
        parameters = self.config.parameters
        model_lat = f"{parameters.model_name.lower()}_lat"
        model_lon = f"{parameters.model_name.lower()}_lon"
        
        # Flatten pending results into dataframe
        all_data = []
        for batch in self.pending_results:
            batch_df = pd.DataFrame(batch)
            all_data.append(batch_df)
        
        if not all_data:
            return
        
        new_data = pd.concat(all_data, ignore_index=True)
        
        # Add model columns if they don't exist
        if model_lat not in self.df.columns:
            self.df[model_lat] = None
        if model_lon not in self.df.columns:
            self.df[model_lon] = None
        
        # Merge with existing dataframe
        if self.df.empty:
            self.df = new_data
        else:
            # Set index for merging
            df_indexed = self.df.set_index('idx') if 'idx' in self.df.columns and self.df.index.name != 'idx' else self.df.copy()
            new_data_indexed = new_data.set_index('idx')
            
            # Update existing rows
            df_indexed.update(new_data_indexed)
            
            # Add completely new indices
            new_indices = new_data_indexed.index.difference(df_indexed.index)
            if len(new_indices) > 0:
                df_indexed = pd.concat([df_indexed, new_data_indexed.loc[new_indices]])
            
            self.df = df_indexed.reset_index()
        
        # Save to disk
        self._save_to_csv()
        
        # Clear pending results
        self.pending_results = []
    
    def _save_to_csv(self):
        """Save dataframe to CSV"""
        df_to_save = self.df.copy()
        if df_to_save.index.name == "idx":
            df_to_save.reset_index(inplace=True)
        df_to_save.to_csv(self.output_path, index=False)
        
    def finalize(self):
        """Merge any remaining results and perform final save"""
        self._merge_and_save()
        return self.df
    
    def get_stats(self):
        """Get current stats"""
        return {
            'total_rows': len(self.df),
            'batches_processed': self.batch_count
        }


@ray.remote
class ProgressTracker:
    """Shared progress tracker across workers"""
    
    def __init__(self, total_batches):
        self.total_batches = total_batches
        self.completed_batches = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.last_print_count = 0
        self.last_save_count = 0
    
    def update(self, num_batches, batch_size):
        """Update progress and return whether to print"""
        self.completed_batches += num_batches
        current_time = time.time()
        
        # Print every 2 seconds or every 10 batches
        time_since_print = current_time - self.last_print_time
        batches_since_print = self.completed_batches - self.last_print_count
        
        if time_since_print >= 2.0 or batches_since_print >= 10:
            elapsed = current_time - self.start_time
            percent = (self.completed_batches / self.total_batches) * 100
            
            # Calculate throughput since last print
            images_processed = batches_since_print * batch_size
            imgs_per_sec = images_processed / time_since_print if time_since_print > 0 else 0.0
            
            print(f"Progress: {self.completed_batches}/{self.total_batches} ({percent:.1f}%) | Imgs/s: {imgs_per_sec:.2f}")
            
            self.last_print_time = current_time
            self.last_print_count = self.completed_batches
            return True
        return False
    
    def notify_save(self, batch_num):
        """Notify that a checkpoint was saved"""
        if batch_num > self.last_save_count + 50:  # Only print if significant progress
            print(f"✓ Checkpoint saved at batch {batch_num}")
            self.last_save_count = batch_num


@ray.remote
class PredictionWorker:
    """Ray actor that runs predictions on a single GPU"""
    
    def __init__(self, config, worker_id):
        self.config = config
        self.worker_id = worker_id
        # Ray isolates each worker to see only its assigned GPU as cuda:0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Debug: Show which physical GPU Ray assigned
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        
        # Load model on this GPU
        parameters = config.parameters
        self.model = load_model(parameters.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Worker {worker_id} initialized on {self.device} (Ray assigned physical GPU: {cuda_visible})")
    
    def predict_indices(self, indices_chunk, progress_tracker, result_collector):
        """Process a chunk of dataset indices"""
        parameters = self.config.parameters
        
        # Create dataset and dataloader for this chunk
        transform = get_transform(parameters.transform_name)
        full_dataset = load_dataset(
            parameters.dataset_name, 
            transform=transform, 
            max_shards=parameters.max_shards
        )
        
        # Create subset for this worker
        subset = Subset(full_dataset, indices_chunk)
        dataloader = DataLoader(
            subset, 
            batch_size=parameters.batch_size, 
            shuffle=False, 
            num_workers=parameters.num_workers
        )
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            idxs = batch['idx'].cpu().numpy()
            true_lat = batch['lat'].cpu().numpy()
            true_lon = batch['lon'].cpu().numpy()
            
            # Get predictions
            with torch.no_grad():
                pred_lat, pred_lon = self.model(images)
                pred_lat = pred_lat.cpu().numpy()
                pred_lon = pred_lon.cpu().numpy()
            
            # Store batch results
            batch_results = {
                'idx': idxs,
                'true_lat': true_lat,
                'true_lon': true_lon,
                f"{parameters.model_name.lower()}_lat": pred_lat,
                f"{parameters.model_name.lower()}_lon": pred_lon
            }
            
            # Send to result collector and check if save occurred
            saved = ray.get(result_collector.add_results.remote(batch_results))
            if saved:
                stats = ray.get(result_collector.get_stats.remote())
                ray.get(progress_tracker.notify_save.remote(stats['batches_processed']))
            
            # Update progress tracker
            ray.get(progress_tracker.update.remote(1, parameters.batch_size))
        
        # Worker completed
        return True


def get_completed_indices(config):
    """Get indices that already have predictions"""
    parameters = config.parameters
    dataset_path = f"data/train/predictions/{parameters.dataset_name}.csv"
    
    if not os.path.exists(dataset_path):
        return set()
    
    df = pd.read_csv(dataset_path)
    model_col = f"{parameters.model_name.lower()}_lat"
    
    if model_col in df.columns and 'idx' in df.columns:
        completed = set(df[df[model_col].notna()]['idx'].values)
        return completed
    
    return set()


def load_existing_dataframe(config):
    """Load existing predictions dataframe"""
    parameters = config.parameters
    dataset_path = f"data/train/predictions/{parameters.dataset_name}.csv"
    
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    else:
        return pd.DataFrame(columns=["idx", "true_lat", "true_lon"])


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python generate_distributed.py <config.yaml>")
    
    config_path = sys.argv[1]
    config = load_config_yaml(config_path)
    parameters = config.parameters
    ray_config = config.ray
    
    print(f"Configuration:\n{config}")
    
    # Initialize Ray
    if ray_config.address:
        ray.init(address=ray_config.address)
    else:
        ray.init()
    
    print(f"Ray initialized. Available resources: {ray.available_resources()}")
    
    # Determine number of GPUs to use
    num_gpus = int(ray.available_resources().get('GPU', 0))
    if num_gpus == 0:
        print("WARNING: No GPUs detected. Falling back to CPU mode.")
        num_gpus = 1
    
    print(f"Using {num_gpus} GPU(s) for inference")
    
    # Load dataset to get total size
    print("Loading dataset metadata...")
    transform = get_transform(parameters.transform_name)
    full_dataset = load_dataset(
        parameters.dataset_name, 
        transform=transform, 
        max_shards=parameters.max_shards
    )
    total_samples = len(full_dataset)
    print(f"Total dataset size: {total_samples}")
    
    # Get completed indices
    print("Checking for existing predictions...")
    completed_indices = get_completed_indices(config)
    print(f"Found {len(completed_indices)} existing predictions")
    
    # Get remaining indices to process
    all_indices = np.arange(total_samples)
    remaining_indices = [idx for idx in all_indices if idx not in completed_indices]
    
    if len(remaining_indices) == 0:
        print("All predictions already completed!")
        ray.shutdown()
        return
    
    print(f"Remaining predictions: {len(remaining_indices)}")
    
    # Split indices across workers
    indices_chunks = np.array_split(remaining_indices, num_gpus)
    print(f"Split work into {num_gpus} chunks: {[len(chunk) for chunk in indices_chunks]}")
    
    # Calculate total batches for progress tracking
    total_batches = sum([int(np.ceil(len(chunk) / parameters.batch_size)) for chunk in indices_chunks])
    print(f"Total batches to process: {total_batches}")
    
    # Setup output path
    output_dir = f"data/train/predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{parameters.dataset_name}.csv"
    
    # Create shared actors
    print("Creating shared actors...")
    progress_tracker = ProgressTracker.remote(total_batches)
    result_collector = ResultCollector.remote(
        config, 
        load_existing_dataframe(config), 
        output_path,
        save_interval=100  # Save every 100 batches
    )
    
    # Create workers
    print("Creating prediction workers...")
    workers = []
    for i in range(num_gpus):
        worker = PredictionWorker.options(num_gpus=1).remote(config, i)
        workers.append(worker)
    
    # Distribute work and collect results
    print("Starting distributed prediction...")
    print("-" * 60)
    start_time = time.time()
    
    futures = []
    for worker, indices_chunk in zip(workers, indices_chunks):
        if len(indices_chunk) > 0:
            future = worker.predict_indices.remote(
                indices_chunk.tolist(), 
                progress_tracker,
                result_collector
            )
            futures.append(future)
    
    # Wait for all workers to complete
    ray.get(futures)
    
    print("-" * 60)
    elapsed_time = time.time() - start_time
    total_processed = len(remaining_indices)
    throughput = total_processed / elapsed_time if elapsed_time > 0 else 0
    
    print(f"Prediction completed in {elapsed_time:.2f}s ({throughput:.2f} imgs/s)")
    
    # Finalize results (merge any remaining batches and save)
    print("Finalizing results...")
    final_df = ray.get(result_collector.finalize.remote())
    
    print(f"✓ Final predictions saved to {output_path}")
    print(f"Total predictions in file: {len(final_df)}")
    
    # Shutdown Ray
    ray.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()