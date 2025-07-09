"""Download and prepare CelebA dataset."""

import os
import zipfile
import requests
from tqdm import tqdm
import gdown

def download_file(url, dest_path, use_gdown=False):
    """Download a file with progress bar."""
    if use_gdown:
        gdown.download(url, dest_path, quiet=False)
    else:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))

def download_celeba(data_dir='./data/celeba'):
    """Download CelebA dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    # CelebA files from Google Drive
    files = {
        'img_align_celeba.zip': '1cNIac61PSA_LqDFYFUeyaQYekYPc75NH',
        'list_eval_partition.txt': '1yCwvSVFSuiW0kNm8q7fN9F2MFEZdpZWD',
        'list_attr_celeba.txt': '1wHSvusQ1CiEMc5Nx6IjWSVDhHAeLGp3r',
        'identity_CelebA.txt': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
        'list_bbox_celeba.txt': '1MHCixPJDkx3e8MXwDbAH5vwKrxpxG5Hh',
        'list_landmarks_align_celeba.txt': '1cUBMk72sj5_D1PbBhAo9IDGuwrMuCnn-'
    }
    
    print("Downloading CelebA dataset...")
    
    # Download each file
    for filename, file_id in files.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, file_path, quiet=False)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print("Please download manually from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
                return False
    
    # Extract images if needed
    zip_path = os.path.join(data_dir, 'img_align_celeba.zip')
    extract_path = os.path.join(data_dir, 'img_align_celeba')
    
    if os.path.exists(zip_path) and not os.path.exists(extract_path):
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete!")
    
    print("CelebA dataset ready!")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/celeba',
                       help='Directory to save CelebA dataset')
    args = parser.parse_args()
    
    download_celeba(args.data_dir)