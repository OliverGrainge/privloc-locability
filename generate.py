from dotenv import load_dotenv 
load_dotenv()

from utils import load_config_yaml 
from data.datasets import load_dataset
from models.transforms import get_transform
from torch.utils.data import DataLoader, Subset
from models.geolocation import load_model
import sys
import os 
import torch
import time 
import ray
import pandas as pd
import logging
from pathlib import Path
import socket
import shutil
import numpy as np
import pandas as pd
from PIL import Image


# Configure logging for SLURM cluster (single node, multi-GPU)
def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Setup logging configuration for SLURM single-node job
    All logs go to one file + console (SLURM output)
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get job info from SLURM environment variables
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler (goes to SLURM output file: slurm-<jobid>.out)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler (single unified log file for the job)
    log_file = log_path / f"job_{job_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger


def setup_ray_actor_logging(actor_name, log_dir="logs", log_level=logging.INFO):
    """
    Setup logging for Ray actors (single node, different GPUs)
    All actors log to the same file with their name prefixed
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    logger = logging.getLogger(actor_name)
    logger.setLevel(log_level)
    logger.handlers.clear()  # Clear any existing handlers
    
    # Console handler (goes to SLURM output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        f'%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler (shared log file for all actors)
    log_file = log_path / f"job_{job_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logging()


def load_siglip_encoder(device):
    """
    Load SigLip vision encoder for generating image embeddings
    Returns the model and its transform
    """
    try:
        from transformers import AutoModel, AutoProcessor
        
        model_name = "google/siglip-so400m-patch14-384"
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Extract vision model and move to device
        vision_model = model.vision_model.to(device)
        vision_model.eval()
        
        return vision_model, processor
    except Exception as e:
        raise RuntimeError(f"Failed to load SigLip encoder: {e}")


@ray.remote
class ParquetWriter:
    """Handles writing predictions to sharded Parquet files"""
    
    def __init__(self, output_dir, dataset_name, model_name, save_interval=100, 
                 shard_size=100, log_dir="logs", include_embeddings=False):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.save_interval = save_interval
        self.shard_size = shard_size  # Number of samples per shard
        self.include_embeddings = include_embeddings
        self.batch_count = 0
        self.total_samples_written = 0
        self.current_shard = 0
        self.samples_in_current_shard = 0
        
        # Setup logger for this actor first
        self.logger = setup_ray_actor_logging("ParquetWriter", log_dir)
        
        # Create shard directory: output_dir/dataset_name_model_name/
        self.shard_dir = self.output_dir / f"{dataset_name}_{model_name}"
        
        # Clean up existing shard directory if it exists (for clean restarts)
        if self.shard_dir.exists():
            self.logger.info(f"Cleaning up existing shard directory: {self.shard_dir}")
            shutil.rmtree(self.shard_dir)
        
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        
        self.pending_results = []
        
        embeddings_status = "enabled" if include_embeddings else "disabled"
        self.logger.info(f"Initialized with shard_size={shard_size}, embeddings={embeddings_status}, shard_dir={self.shard_dir}")
        self.logger.info("Ready to start generating predictions with sharding enabled")
        
    def add_results(self, batch_results):
        """Add batch results and save periodically"""
        batch_size = len(batch_results)
        self.pending_results.extend(batch_results)
        self.batch_count += 1
        self.total_samples_written += batch_size
        
        # Check if we need to flush to current shard
        if self.samples_in_current_shard + len(self.pending_results) >= self.shard_size:
            self._flush_to_current_shard()
        
        # Save every N batches (for checkpointing)
        if self.batch_count % self.save_interval == 0:
            self.logger.info(f"Checkpoint - Batch {self.batch_count}, Total samples: {self.total_samples_written}")
            return True
        return False
    
    def _get_current_shard_path(self):
        """Get the path for the current shard file"""
        return self.shard_dir / f"part_{self.current_shard:03d}.parquet"
    
    def _flush_to_current_shard(self):
        """Write pending results to current shard file"""
        if not self.pending_results:
            return
        
        # Calculate how many samples to write to current shard
        remaining_in_shard = self.shard_size - self.samples_in_current_shard
        samples_to_write = min(len(self.pending_results), remaining_in_shard)
        
        # Write samples to current shard
        shard_data = self.pending_results[:samples_to_write]
        shard_df = pd.DataFrame(shard_data)
        
        current_shard_path = self._get_current_shard_path()
        
        if current_shard_path.exists():
            # Append to existing shard
            existing_df = pd.read_parquet(current_shard_path)
            combined_df = pd.concat([existing_df, shard_df], ignore_index=True)
            combined_df.to_parquet(current_shard_path, index=False)
        else:
            # Create new shard
            shard_df.to_parquet(current_shard_path, index=False)
        
        self.samples_in_current_shard += samples_to_write
        self.logger.debug(f"Wrote {samples_to_write} samples to shard {self.current_shard}")
        
        # Remove written samples from pending
        self.pending_results = self.pending_results[samples_to_write:]
        
        # If shard is full, move to next shard
        if self.samples_in_current_shard >= self.shard_size:
            self.current_shard += 1
            self.samples_in_current_shard = 0
            self.logger.info(f"Completed shard {self.current_shard - 1}, starting shard {self.current_shard}")
        
        return samples_to_write
    
    def finalize(self):
        """Flush any remaining results to current shard"""
        self.logger.info("Finalizing Parquet writes")
        if self.pending_results:
            self._flush_to_current_shard()
        
        self.logger.info(f"Final predictions saved to {self.current_shard + 1} shard files in: {self.shard_dir}")
        return self.total_samples_written
    
    def get_stats(self):
        """Get current stats"""
        return {
            'batches_processed': self.batch_count,
            'total_samples_written': self.total_samples_written,
            'current_shard': self.current_shard,
            'samples_in_current_shard': self.samples_in_current_shard,
            'shard_dir': str(self.shard_dir),
            'include_embeddings': self.include_embeddings
        }


@ray.remote
class PredictionWorker:
    """Ray actor that runs predictions on a single GPU"""
    
    def __init__(self, config, worker_id, log_dir="logs", include_embeddings=False):
        self.config = config
        self.worker_id = worker_id
        self.include_embeddings = include_embeddings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Setup logger for this worker
        self.logger = setup_ray_actor_logging(f"Worker-{worker_id}", log_dir)
        
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        self.logger.info(f"Starting initialization - Device: {self.device}, CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # Load geolocation model
        parameters = config.parameters
        self.logger.info(f"Loading model: {parameters.model_name}")
        self.model = load_model(parameters.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load SigLip encoder if embeddings are requested
        self.siglip_model = None
        self.siglip_processor = None
        if include_embeddings:
            self.logger.info("Loading SigLip encoder for embeddings...")
            self.siglip_model, self.siglip_processor = load_siglip_encoder(self.device)
            self.logger.info("SigLip encoder loaded successfully")
        
        embeddings_status = "enabled" if include_embeddings else "disabled"
        self.logger.info(f"Initialization complete on {self.device} (embeddings: {embeddings_status})")
    
    def _generate_siglip_embedding(self, images_pil):
        """
        Generate SigLip embeddings for a batch of PIL images
        
        Args:
            images_pil: List of PIL Image objects
            
        Returns:
            numpy array of embeddings [batch_size, embedding_dim]
        """
        # Process images with SigLip processor
        inputs = self.siglip_processor(
            images=images_pil, 
            return_tensors="pt"
        )
        
        # Move to device
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.siglip_model(pixel_values)
            # Get pooled output (CLS token representation)
            embeddings = outputs.pooler_output
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def predict_range(self, start_idx, end_idx, writer):
        """Process a range of dataset indices [start_idx, end_idx)"""
        self.logger.info(f"Processing range [{start_idx}, {end_idx})")
        parameters = self.config.parameters
        
        # Load transform and dataset
        transform = get_transform(parameters.transform_name)
        full_dataset = load_dataset(
            parameters.dataset_name, 
            transform=transform, 
            max_shards=parameters.max_shards
        )
        
        # Create index range
        indices = list(range(start_idx, min(end_idx, len(full_dataset))))
        self.logger.info(f"Processing {len(indices)} samples")
        
        # Create subset and dataloader
        subset = Subset(full_dataset, indices)
        dataloader = DataLoader(
            subset, 
            batch_size=parameters.batch_size, 
            shuffle=False, 
            num_workers=parameters.num_workers
        )
        
        # Process batches
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                progress = (batch_idx * parameters.batch_size) / len(indices) * 100
                self.logger.info(f"Batch {batch_idx} ({progress:.1f}%)")
            
            images = batch['image'].to(self.device)
            image_bytes = batch['image_bytes']
            true_lat = batch['lat'].cpu().numpy()
            true_lon = batch['lon'].cpu().numpy()
            
            # Get geolocation predictions
            with torch.no_grad():
                pred_lat, pred_lon = self.model(images)
                pred_lat = pred_lat.cpu().numpy()
                pred_lon = pred_lon.cpu().numpy()
            
            # Generate SigLip embeddings if requested
            embeddings = None
            if self.include_embeddings:
                # Convert image bytes back to PIL images for SigLip processing
                images_pil = []
                for img_bytes in image_bytes:
                    from io import BytesIO
                    pil_image = Image.open(BytesIO(img_bytes)).convert('RGB')
                    images_pil.append(pil_image)
                embeddings = self._generate_siglip_embedding(images_pil)
            
            # Prepare batch results as list of dicts
            batch_results = []
            
            for i in range(len(image_bytes)):
                result = {
                    'image_bytes': image_bytes[i],
                    'true_lat': float(true_lat[i]),
                    'true_lon': float(true_lon[i]),
                    'pred_lat': float(pred_lat[i]),
                    'pred_lon': float(pred_lon[i])
                }
                
                # Add embedding if available
                if embeddings is not None:
                    # Store embedding as bytes for efficient parquet storage
                    result['embedding'] = embeddings[i].tobytes()
                
                batch_results.append(result)
            
            # Send to writer
            ray.get(writer.add_results.remote(batch_results))
        
        self.logger.info(f"Completed range [{start_idx}, {end_idx})")
        return True


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python generate.py <config.yaml> [start_idx] [end_idx]")
    
    config_path = sys.argv[1]
    config = load_config_yaml(config_path)
    parameters = config.parameters
    ray_config = config.ray
    
    # Check if embeddings should be included
    include_embeddings = getattr(parameters, 'embeddings', False)
    
    # Get log directory and output directory from config or use defaults
    log_dir = getattr(config, 'log_dir', 'logs')
    # Use DATASET_DIR environment variable if available, otherwise fall back to config or default
    dataset_dir = os.getenv('DATASET_DIR')
    if dataset_dir:
        output_dir = os.path.join(dataset_dir, 'predictions')
    else:
        output_dir = getattr(parameters, 'output_dir', 'predictions')
    
    logger.info(f"SLURM Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"Embeddings enabled: {include_embeddings}")
    logger.info(f"Logs directory: {log_dir}")
    if dataset_dir:
        logger.info(f"Using DATASET_DIR environment variable: {dataset_dir}")
        logger.info(f"Predictions directory: {output_dir}")
    else:
        logger.info(f"Predictions directory: {output_dir}")
    logger.debug(f"Configuration:\n{config}")
    
    # Initialize Ray
    if ray_config.address:
        ray.init(address=ray_config.address)
        logger.info(f"Ray initialized with address: {ray_config.address}")
    else:
        ray.init()
        logger.info("Ray initialized in local mode")
    
    logger.info(f"Available resources: {ray.available_resources()}")
    
    # Determine number of GPUs
    num_gpus = int(ray.available_resources().get('GPU', 0))
    if num_gpus == 0:
        logger.warning("No GPUs detected. Falling back to CPU mode.")
        num_gpus = 1
    
    logger.info(f"Using {num_gpus} GPU(s) for inference")
    
    # Get dataset size (only need length, not all indices)
    logger.info("Getting dataset size...")
    transform = get_transform(parameters.transform_name)
    temp_dataset = load_dataset(
        parameters.dataset_name, 
        transform=transform, 
        max_shards=parameters.max_shards
    )
    total_samples = len(temp_dataset)
    del temp_dataset  # Free memory
    
    # Optional: Allow processing only a slice of the dataset
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end_idx = int(sys.argv[3]) if len(sys.argv) > 3 else total_samples
    
    logger.info(f"Dataset size: {total_samples}")
    logger.info(f"Processing range: [{start_idx}, {end_idx})")
    samples_to_process = end_idx - start_idx
    
    # Split work into ranges for each GPU
    indices_per_worker = samples_to_process // num_gpus
    ranges = []
    for i in range(num_gpus):
        worker_start = start_idx + (i * indices_per_worker)
        worker_end = start_idx + ((i + 1) * indices_per_worker) if i < num_gpus - 1 else end_idx
        ranges.append((worker_start, worker_end))
    
    logger.info(f"Work distribution: {[(s, e, e-s) for s, e in ranges]}")
    
    # Create sharded Parquet writer
    logger.info("Creating sharded Parquet writer...")
    writer = ParquetWriter.remote(
        output_dir,
        parameters.dataset_name,
        parameters.model_name,
        save_interval=500,
        shard_size=50000,  # 50K samples per shard (adjust as needed)
        log_dir=log_dir,
        include_embeddings=include_embeddings
    )
    
    # Create workers
    logger.info("Creating prediction workers...")
    workers = []
    for i in range(num_gpus):
        worker = PredictionWorker.options(num_gpus=1).remote(
            config, i, log_dir, include_embeddings
        )
        workers.append(worker)
    
    # Distribute work
    logger.info("Starting distributed prediction...")
    logger.info("-" * 60)
    start_time = time.time()
    
    futures = []
    for i, (worker, (range_start, range_end)) in enumerate(zip(workers, ranges)):
        if range_end > range_start:
            logger.info(f"Worker {i}: processing [{range_start}, {range_end})")
            future = worker.predict_range.remote(range_start, range_end, writer)
            futures.append(future)
    
    # Wait for completion
    ray.get(futures)
    
    logger.info("-" * 60)
    elapsed_time = time.time() - start_time
    throughput = samples_to_process / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Prediction completed in {elapsed_time:.2f}s ({throughput:.2f} imgs/s)")
    
    # Finalize
    logger.info("Finalizing results...")
    final_samples = ray.get(writer.finalize.remote())
    
    stats = ray.get(writer.get_stats.remote())
    logger.info(f"All predictions saved to {stats['current_shard'] + 1} shard files in: {stats['shard_dir']}")
    if include_embeddings:
        logger.info("Embeddings included in output parquet files")
    
    # Shutdown
    ray.shutdown()
    logger.info("Done!")


if __name__ == "__main__":
    main()