import os
from dotenv import load_dotenv
load_dotenv()
from diffusers import AutoencoderKL
import torch


def main():
    # Set cache directory from environment variable or use default
    cache_path = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if cache_path:
        print(f"Hugging Face cache directory: {cache_path}")
    else:
        print("No HF_HOME set — defaulting to ~/.cache/huggingface")
    
    vae_model = "stabilityai/sd-vae-ft-mse"
    print(f"Downloading VAE model: {vae_model} ...")
    
    # Load model - it will use HF_HOME or HUGGINGFACE_HUB_CACHE automatically
    vae = AutoencoderKL.from_pretrained(vae_model, cache_dir=cache_path)
    
    # Run a dummy forward pass to ensure weights download
    print("Running dummy forward pass to verify download...")
    with torch.no_grad():
        dummy_latent = torch.randn(1, 4, 28, 28)  # SD VAE expects [B, 4, H/8, W/8] for 224x224 input
        _ = vae.decode(dummy_latent).sample
    
    print("VAE model downloaded successfully!")
    return True


if __name__ == "__main__":
    main()

