import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def setup_depth_model():
    """
    Initialize the Depth Anything model and processor
    """
    processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")
    return processor, model

def process_image(image_path, processor, model):
    """
    Process a single image to generate depth map
    """
    # Load and preprocess image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate depth map
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Convert to numpy array and normalize
    depth_map = predicted_depth.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 65535).astype(np.uint16)  # Convert to 16-bit depth
    
    return depth_map

def main():
    # Setup paths
    main_path = "/media/ant/52F6748DF67472D9/PhD/T4/embedded system/fp/experiment1/"
    rgb_folder = os.path.join(main_path, "rgb")
    output_folder = os.path.join(main_path, "depth_synthetic")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize model
    print("Loading Depth Anything model...")
    processor, model = setup_depth_model()
    
    # Process all images in the RGB folder
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
    
    # Start from index 2000
    start_index = 2000
    rgb_files = rgb_files[start_index:]
    
    print(f"Found {len(rgb_files)} images to process (starting from index {start_index})")
    
    for rgb_file in rgb_files:
        print(f"Processing {rgb_file}...")
        
        # Generate depth map
        rgb_path = os.path.join(rgb_folder, rgb_file)
        depth_map = process_image(rgb_path, processor, model)
        
        # Save depth map
        output_path = os.path.join(output_folder, rgb_file)
        cv2.imwrite(output_path, depth_map)
        
        print(f"Saved depth map to {output_path}")

if __name__ == "__main__":
    main()
