# ------------------------- Standard Library Imports -------------------------

import os            # For file system operations
import requests      # For HTTP requests to download files
import tarfile       # For extracting .tar.xz archives
from tqdm import tqdm # For displaying download progress bars

# ------------------------- Data Download and Extraction Utility -------------------------

def download_and_extract(dataset: str = 'us', data_directory: str = '../data') -> None:
    """
    Download and extract the OpenCellID dataset (US or Worldwide) into the specified local directory.

    Args:
        dataset (str): Dataset to download ('us' or 'worldwide'). Defaults to 'us'.
        data_directory (str): Directory to store the extracted CSV files. Defaults to 'data'.
    """
    # Define available dataset URLs
    dataset_urls = {
        'us': 'https://data.rapids.ai/cudf/datasets/cell_towers_us.tar.xz',
        'worldwide': 'https://data.rapids.ai/cudf/datasets/cell_towers.tar.xz'
    }
    
    # Validate input dataset choice
    if dataset not in dataset_urls:
        raise ValueError("Invalid dataset selection. Choose either 'us' or 'worldwide'.")
    
    # Define download URL and local file paths
    download_url = dataset_urls[dataset]
    tarball_filename = os.path.join(data_directory, os.path.basename(download_url))
    extracted_csv_filename = tarball_filename.replace('.tar.xz', '.csv')

    # Create target data directory if it does not exist
    os.makedirs(data_directory, exist_ok=True)

    # If CSV already exists, skip download and extraction
    if os.path.exists(extracted_csv_filename):
        print(f"{extracted_csv_filename} already exists. Skipping download and extraction.")
        return

    # Download tar.xz archive if not already present
    if not os.path.exists(tarball_filename):
        print(f"Downloading {dataset} dataset from {download_url}...")
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tarball_filename, 'wb') as file_handle, tqdm(
            desc=f"Downloading {os.path.basename(tarball_filename)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file_handle.write(chunk)
                    progress_bar.update(len(chunk))
        
        print(f"Download completed: {tarball_filename}")
    else:
        print(f"{tarball_filename} already exists. Skipping download.")

    # Extract CSV file from tar.xz archive
    print(f"Extracting {tarball_filename}...")
    with tarfile.open(tarball_filename, mode='r:xz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.csv'):
                member.name = os.path.basename(member.name)  # Flatten directory structure
                tar.extract(member, path=data_directory)
    
    print(f"Extraction completed: {extracted_csv_filename}")
