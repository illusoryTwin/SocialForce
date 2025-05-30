import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # Create directories
    os.makedirs('coco', exist_ok=True)
    os.makedirs('coco/annotations', exist_ok=True)
    
    # Download annotations
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    annotations_zip = 'coco/annotations_trainval2017.zip'
    
    print("Downloading COCO annotations...")
    download_file(annotations_url, annotations_zip)
    
    # Extract annotations
    print("Extracting annotations...")
    with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
        zip_ref.extractall('coco')
    
    # Clean up
    os.remove(annotations_zip)
    
    print("COCO annotations downloaded and extracted successfully!")
    print("Annotations are available at: coco/annotations/instances_train2017.json")

if __name__ == '__main__':
    main() 