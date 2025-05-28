import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import os
import glob

def load_sam_model():
    # Initialize SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # You'll need to download this
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    return SamPredictor(sam)

def segment_floor(image_path, predictor):
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(image)
    
    # Generate points for floor detection
    h, w = image.shape[:2]
    points = []
    
    # Add points along the bottom of the image (likely floor)
    for x in range(0, w, 20):  # Increased point density
        points.append([x, h-1])
    
    # Add points slightly above the bottom
    for x in range(0, w, 40):
        points.append([x, h-50])
    
    # Add points in the middle-bottom area
    for x in range(0, w, 60):
        points.append([x, h-100])
    
    # Add points along the edges (often part of the floor)
    for y in range(h-1, h-200, -20):
        points.append([0, y])  # Left edge
        points.append([w-1, y])  # Right edge
    
    # Convert points to numpy array
    input_points = np.array(points)
    input_labels = np.ones(len(points))  # All points are floor
    
    # Generate masks
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    
    # Get the mask with highest score
    best_mask_idx = np.argmax(scores)
    floor_mask = masks[best_mask_idx]
    
    # Post-process the mask to fill holes and remove small artifacts
    floor_mask = floor_mask.astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
    
    # Create visualization
    mask_vis = np.zeros_like(image)
    mask_vis[floor_mask == 1] = [255, 0, 0]  # Red color for floor
    
    # Blend original image with mask
    alpha = 0.5
    output = cv2.addWeighted(image, 1, mask_vis, alpha, 0)
    
    return output, floor_mask

def main():
    # Initialize SAM predictor
    predictor = load_sam_model()
    
    # Process all matching images
    image_pattern = "/media/hdd_4/PhD/T4/embedded system/fp/experiment3/rgb/rgb_0020**.png"
    image_paths = glob.glob(image_pattern)
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        
        # Segment floor
        output_image, floor_mask = segment_floor(image_path, predictor)
        
        # Save results
        output_dir = image_pattern[:-18]+"segmentation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for segmented images and masks
        segmented_dir = os.path.join(output_dir, "segmented_image")
        mask_dir = os.path.join(output_dir, "mask_image")
        os.makedirs(segmented_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path)
        output_path = os.path.join(segmented_dir, f"segmented_{base_name}")
        mask_path = os.path.join(mask_dir, f"mask_{base_name}")
        
        # Save visualization
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        # Save mask
        cv2.imwrite(mask_path, floor_mask * 255)

if __name__ == "__main__":
    main()
