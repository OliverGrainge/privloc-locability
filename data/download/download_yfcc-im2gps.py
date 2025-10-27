import os
from dotenv import load_dotenv
load_dotenv()

import kagglehub

def main():
    # KaggleHub will automatically respect KAGGLEHUB_CACHE if it's set
    cache_path = os.getenv("KAGGLEHUB_CACHE")
    if cache_path:
        print(f"KaggleHub cache directory: {cache_path}")
    else:
        print("No KAGGLEHUB_CACHE set — defaulting to ~/.cache/kagglehub")

    dataset_id = "lbgan2000/img2gps-yfcc4k"
    print(f"Downloading dataset: {dataset_id} ...")

    # Just call dataset_download() — it will use KAGGLEHUB_CACHE automatically
    path = kagglehub.dataset_download(dataset_id)
    print(f"Download completed. Dataset files are located at: {path}")

if __name__ == "__main__":
    main()