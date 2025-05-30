import numpy as np
from sklearn.linear_model import RANSACRegressor
import cv2
import yaml
import torch
from segment_anything import sam_model_registry, SamPredictor
import os
import json
from pathlib import Path

def load_camera_params():
    """Load camera parameters from YAML file"""
    with open('/home/ant/projects/SocialForce/camera_params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    # Use depth camera parameters
    return params['d435_depth']

def load_sam_model():
    """Load SAM model"""
    # Use the smaller ViT-B model instead of ViT-H
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if model file exists
    if not os.path.exists(sam_checkpoint):
        print(f"Downloading SAM model checkpoint to {sam_checkpoint}...")
        os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}")
    
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        print(f"Error loading SAM model: {str(e)}")
        print("Please ensure you have downloaded the correct model checkpoint.")
        print(f"Expected file: {sam_checkpoint}")
        raise

def create_point_cloud(rgb_path, depth_path):
    """Create point cloud from RGB and depth images"""
    # Read images
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if rgb is None or depth is None:
        raise ValueError(f"Could not read input images: {rgb_path}, {depth_path}")
    
    # Ensure images are the same size
    if rgb.shape[:2] != depth.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
    
    # Load camera parameters
    camera_params = load_camera_params()
    fx = float(camera_params['fx'])
    fy = float(camera_params['fy'])
    cx = float(camera_params['cx'])
    cy = float(camera_params['cy'])
    
    # Create point cloud
    height, width = depth.shape
    
    # Create meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to 3D coordinates
    Z = depth.astype(np.float32) / 1000.0  # convert to meters
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    
    # Stack coordinates and reshape
    points = np.stack([X, Y, Z], axis=-1)
    points = points.reshape(-1, 3)
    
    # Get colors
    colors = rgb.reshape(-1, 3) / 255.0
    
    # Remove invalid points (where depth is 0)
    valid_points = Z.reshape(-1) > 0
    points = points[valid_points]
    colors = colors[valid_points]
    
    return points, colors, rgb

def segment_floor_sam(image, predictor):
    """Segment the floor using SAM"""
    # Set image in predictor
    predictor.set_image(image)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create a grid of points in the lower part of the image
    # (assuming floor is in the bottom part)
    grid_size = 10
    x_points = np.linspace(0, width-1, grid_size)
    y_points = np.linspace(height*0.7, height-1, grid_size)
    
    # Create input points for SAM
    input_points = []
    for x in x_points:
        for y in y_points:
            input_points.append([x, y])
    input_points = np.array(input_points)
    
    # Set point labels (1 for floor points)
    input_labels = np.ones(len(input_points))
    
    # Generate mask
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    
    # Select the best mask
    best_mask_idx = np.argmax(scores)
    floor_mask = masks[best_mask_idx]
    
    # Print segmentation statistics
    print(f"Total pixels in image: {floor_mask.size}")
    print(f"Floor pixels detected: {np.sum(floor_mask)}")
    
    return floor_mask

def extract_floor_points(points, colors, floor_mask):
    """Extract points corresponding to the floor from the point cloud"""
    print(f"Total points in point cloud: {len(points)}")
    
    # Convert floor mask to point cloud indices
    floor_indices = np.where(floor_mask.flatten() > 0)[0]
    print(f"Floor indices found: {len(floor_indices)}")
    
    if len(floor_indices) == 0:
        raise ValueError("No floor points detected in the segmentation mask")
    
    # Extract floor points
    floor_points = points[floor_indices]
    floor_colors = colors[floor_indices]
    
    print(f"Floor points extracted: {len(floor_points)}")
    
    return floor_points, floor_colors

def fit_plane_to_points(points):
    """Fit a plane to the floor points using RANSAC"""
    if len(points) < 3:
        raise ValueError(
            "Not enough points to fit a plane (minimum 3 points required)"
        )
        
    X = points[:, :2]  # x and y coordinates
    y = points[:, 2]   # z coordinates
    
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(X, y)
    
    # Get plane parameters
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    
    return a, b, c

def visualize_plane_on_rgb(rgb, a, b, c, camera_params, depth_path):
    """Visualize the floor plane on RGB image"""
    height, width = rgb.shape[:2]
    fx = float(camera_params['fx'])
    fy = float(camera_params['fy'])
    cx = float(camera_params['cx'])
    cy = float(camera_params['cy'])
    
    # Create meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Create visualization image
    vis_img = rgb.copy()
    
    # Create mask for points close to the plane
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(np.float32) / 1000.0  # convert to meters
    
    # Calculate distance to plane for each point
    X_3d = (x - cx) * depth / fx
    Y_3d = (y - cy) * depth / fy
    Z_3d = depth
    
    # Calculate distance to plane
    distances = np.abs(a * X_3d + b * Y_3d - Z_3d + c) / np.sqrt(a*a + b*b + 1)
    
    # Create mask for points close to the plane
    plane_mask = distances < 0.05  # 5cm threshold
    
    # Overlay plane on image
    vis_img[plane_mask] = vis_img[plane_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
    
    return vis_img

def main():
    try:
        # Base directory for the dataset
        base_dir = Path("../imgs_data/")
        
        # Load timestamps
        with open(base_dir / "frame_timestamps.json", 'r') as f:
            timestamps = json.load(f)
        
        # Initialize lists to store points from all frames
        all_floor_points = []
        all_floor_colors = []
        
        # Load SAM model once
        predictor = load_sam_model()
        
        # Process frames starting from 2000
        start_frame = 2000
        num_frames = 50  # Increased from 10 to 50 frames for better accuracy
        frame_step = 2   # Process every second frame to cover more area
        
        for frame_idx in range(start_frame, start_frame + num_frames * frame_step, frame_step):
            print(f"\nProcessing frame {frame_idx}")
            
            # Construct paths with correct filename format
            rgb_path = base_dir / "rgb" / f"rgb_{frame_idx:06d}.png"
            depth_path = base_dir / "depth" / f"depth_{frame_idx:06d}.png"
            
            if not rgb_path.exists() or not depth_path.exists():
                print(f"Skipping frame {frame_idx} - files not found")
                continue
            
            # Create point cloud for this frame
            points, colors, rgb = create_point_cloud(str(rgb_path), str(depth_path))
            
            # Segment floor using SAM
            floor_mask = segment_floor_sam(rgb, predictor)
            
            # Extract floor points
            floor_points, floor_colors = extract_floor_points(points, colors, floor_mask)
            
            # Add to our collection
            all_floor_points.append(floor_points)
            all_floor_colors.append(floor_colors)
        
        if not all_floor_points:
            raise ValueError("No valid floor points found in any frame")
        
        # Combine points from all frames
        combined_points = np.vstack(all_floor_points)
        combined_colors = np.vstack(all_floor_colors)
        
        print(f"\nTotal floor points from all frames: {len(combined_points)}")
        
        # Fit plane to all floor points
        a, b, c = fit_plane_to_points(combined_points)
        
        print(f"Final plane equation: z = {a:.3f}x + {b:.3f}y + {c:.3f}")
        
        # Visualize plane on the last processed RGB image
        camera_params = load_camera_params()
        vis_img = visualize_plane_on_rgb(rgb, a, b, c, camera_params, depth_path)
        
        # Show visualization
        cv2.imshow("Floor Plane Visualization", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
