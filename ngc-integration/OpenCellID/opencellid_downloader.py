import os
import requests
import tarfile
from tqdm import tqdm

def download_and_extract(dataset='us', directory='opencellid_data'):
    # Define the URLs for the datasets
    urls = {
        'us': 'https://data.rapids.ai/cudf/datasets/cell_towers_us.tar.xz',
        'worldwide': 'https://data.rapids.ai/cudf/datasets/cell_towers.tar.xz'
    }
    
    # Check if the dataset parameter is valid
    if dataset not in urls:
        raise ValueError("Invalid dataset parameter. Use 'us' or 'worldwide'.")
    
    # Get the URL for the selected dataset
    url = urls[dataset]
    
    # Define the local filename and directory
    local_filename = os.path.join(directory, url.split('/')[-1])
    csv_filename = os.path.join(directory, url.split('/')[-1].replace('.tar.xz', '.csv'))
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Check if the CSV file already exists
    if os.path.exists(csv_filename):
        print(f"{csv_filename} already exists. Skipping download and extraction.")
        return
    
    # Check if the tar.xz file already exists
    if not os.path.exists(local_filename):
        # Download the file with progress bar
        print(f"Downloading {dataset} dataset from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                size = f.write(chunk)
                bar.update(size)
        print(f"Downloaded {local_filename} successfully.")
    else:
        print(f"{local_filename} already exists. Skipping download.")
    
    # Extract the tar.xz file
    print(f"Extracting {local_filename}...")
    with tarfile.open(local_filename, 'r:xz') as tar:
        members = tar.getmembers()
        csv_members = [m for m in members if m.name.endswith('.csv')]
        for member in csv_members:
            member.name = os.path.basename(member.name)  # Remove the directory structure
            tar.extract(member, path=directory)

    print(f"Extracted {local_filename} successfully.")

