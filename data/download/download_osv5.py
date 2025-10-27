import os
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import snapshot_download

def main():
    # Set cache directory from environment variable or use default
    cache_path = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if cache_path:
        print(f"Hugging Face cache directory: {cache_path}")
    else:
        print("No HF_HOME set — defaulting to ~/.cache/huggingface")

    dataset_id = "osv5m/osv5m"
    print(f"Downloading dataset: {dataset_id} ...")

    # Download the entire dataset using snapshot_download
    # It will use HF_HOME or HUGGINGFACE_HUB_CACHE automatically
    path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        cache_dir=cache_path  # Optional: explicitly pass cache path
    )
    print(f"Download completed. Dataset files are located at: {path}")

if __name__ == "__main__":
    main()