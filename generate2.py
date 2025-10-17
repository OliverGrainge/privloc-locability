from dotenv import load_dotenv 
load_dotenv()

from utils import load_config_yaml 
from data.datasets.generate import load_dataset
from models.transforms import get_transform
from torch.utils.data import DataLoader, Subset
from models.geolocation import load_model
from data import PredictionDatabase
import sys
import os 
import torch
import time 
import ray
import numpy as np


@ray.remote
class DatabaseWriter:
    """Handles writing predictions to the database"""
    
    def __init__(self, db_name, dataset_name, model_name, save_interval=100):
        self.db = PredictionDatabase(db_name)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.save_interval = save_interval
        self.batch_count = 0
        self.pending_results = {
            'image_names': [],
            'true_lats': [],
            'true_lons': [],
            'predicted_lats': [],
            'predicted_lons': []
        }
        
        # Ensure dataset exists
        self.db.add_dataset(dataset_name)
        
    def add_results(self, batch_results):
        """Add batch results and save periodically"""
        self.pending_results['image_names'].extend(batch_results['image_names'])
        self.pending_results['true_lats'].extend(batch_results['true_lats'])
        self.pending_results['true_lons'].extend(batch_results['true_lons'])
        self.pending_results['predicted_lats'].extend(batch_results['predicted_lats'])
        self.pending_results['predicted_lons'].extend(batch_results['predicted_lons'])
        self.batch_count += 1
        
        # Save every N batches
        if self.batch_count % self.save_interval == 0:
            self._flush_to_db()
            print(f"✓ Checkpoint saved at batch {self.batch_count}")
            return True  # Indicate a save occurred
        return False
    
    def _flush_to_db(self):
        """Write pending results to database"""
        if not self.pending_results['image_names']:
            return
        
        # First, add images with ground truth
        image_result = self.db.add_images_batch(
            self.dataset_name,
            self.pending_results['image_names'],
            self.pending_results['true_lats'],
            self.pending_results['true_lons']
        )
        
        # Then, add predictions
        pred_result = self.db.add_predictions_batch(
            self.dataset_name,
            self.model_name,
            self.pending_results['image_names'],
            self.pending_results['predicted_lats'],
            self.pending_results['predicted_lons']
        )
        
        # Clear pending results
        self.pending_results = {
            'image_names': [],
            'true_lats': [],
            'true_lons': [],
            'predicted_lats': [],
            'predicted_lons': []
        }
        
        return {'images': image_result, 'predictions': pred_result}
    
    def finalize(self):
        """Flush any remaining results"""
        return self._flush_to_db()
    
    def get_stats(self):
        """Get current stats"""
        return {
            'batches_processed': self.batch_count
        }


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
    
    def predict_indices(self, indices_chunk, db_writer):
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
            image_names = batch['image_name']  # Assuming dataset returns image names
            true_lat = batch['lat'].cpu().numpy()
            true_lon = batch['lon'].cpu().numpy()
            
            # Get predictions
            with torch.no_grad():
                pred_lat, pred_lon = self.model(images)
                pred_lat = pred_lat.cpu().numpy()
                pred_lon = pred_lon.cpu().numpy()
            
            # Store batch results (including ground truth)
            batch_results = {
                'image_names': image_names if isinstance(image_names, list) else image_names.tolist(),
                'true_lats': true_lat.tolist(),
                'true_lons': true_lon.tolist(),
                'predicted_lats': pred_lat.tolist(),
                'predicted_lons': pred_lon.tolist()
            }
            
            # Send to database writer
            ray.get(db_writer.add_results.remote(batch_results))
        
        # Worker completed
        return True


def get_completed_images(db, dataset_name, model_name):
    """Get image names that already have predictions for this model"""
    try:
        # Query all images with this model's predictions
        results = db.query_batch(dataset_name, model_name=model_name)
        completed = {result['image'] for result in results if result['predictions']}
        return completed
    except:
        return set()


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python generate_distributed.py <config.yaml>")
    
    config_path = sys.argv[1]
    config = load_config_yaml(config_path)
    parameters = config.parameters
    ray_config = config.ray
    
    print(f"Configuration:\n{config}")
    
    # Initialize database (single database for all datasets and models)
    db_name = 'prediction_database.db'
    db = PredictionDatabase(db_name)
    print(f"Using database: {db_name}")
    
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
    
    # Get completed images
    print("Checking for existing predictions...")
    completed_images = get_completed_images(db, parameters.dataset_name, parameters.model_name)
    print(f"Found {len(completed_images)} existing predictions for model '{parameters.model_name}'")
    
    # Get remaining indices to process
    all_image_names = [full_dataset[i]['image_name'] for i in range(total_samples)]
    remaining_indices = [i for i, name in enumerate(all_image_names) if name not in completed_images]
    
    if len(remaining_indices) == 0:
        print("All predictions already completed!")
        ray.shutdown()
        return
    
    print(f"Remaining predictions: {len(remaining_indices)}")
    
    # Split indices across workers
    indices_chunks = np.array_split(remaining_indices, num_gpus)
    print(f"Split work into {num_gpus} chunks: {[len(chunk) for chunk in indices_chunks]}")
    
    # Create shared actors
    print("Creating database writer...")
    db_writer = DatabaseWriter.remote(
        db_name,
        parameters.dataset_name,
        parameters.model_name,
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
                db_writer
            )
            futures.append(future)
    
    # Wait for all workers to complete
    ray.get(futures)
    
    print("-" * 60)
    elapsed_time = time.time() - start_time
    total_processed = len(remaining_indices)
    throughput = total_processed / elapsed_time if elapsed_time > 0 else 0
    
    print(f"Prediction completed in {elapsed_time:.2f}s ({throughput:.2f} imgs/s)")
    
    # Finalize results (flush any remaining batches)
    print("Finalizing results...")
    ray.get(db_writer.finalize.remote())
    
    print(f"✓ All predictions saved to database: {db_name}")
    
    # Print summary
    all_models = db.get_models(parameters.dataset_name)
    print(f"Models in database: {all_models}")
    
    # Shutdown Ray
    ray.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()