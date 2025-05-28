import numpy as np
import cv2
import yaml
from pathlib import Path
import json
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import random

def load_camera_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    # Return depth camera parameters since we're working with depth images
    return params['d435_depth']

def depth_to_points(depth_img, mask, camera_params):
    """Convert depth image to 3D points using camera parameters"""
    height, width = depth_img.shape
    
    # Create meshgrid of pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Get camera parameters
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']
    
    # Convert to 3D points
    Z = depth_img
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    
    # Stack coordinates
    points = np.stack([X, Y, Z], axis=-1)
    
    # Apply mask
    valid_points = points[mask > 0]
    
    return valid_points

def fit_plane(points):
    """Fit a plane to 3D points using RANSAC"""
    # Normalize points
    scaler = StandardScaler()
    points_normalized = scaler.fit_transform(points)
    
    # Prepare data for RANSAC
    X = points_normalized[:, :2]  # Use X and Y coordinates
    y = points_normalized[:, 2]   # Z coordinate as target
    
    # Fit RANSAC
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(X, y)
    
    # Get plane parameters
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    
    # Denormalize parameters
    scale = scaler.scale_
    mean = scaler.mean_
    
    # Convert back to original space
    a = a * scale[2] / scale[0]
    b = b * scale[2] / scale[1]
    c = c * scale[2] + mean[2] - a * mean[0] - b * mean[1]
    
    return np.array([a, b, -1, c])  # ax + by - z + c = 0

def create_plane_mesh(plane_params, points, size=1.0):
    """Create a mesh representing the fitted plane"""
    # Get plane parameters
    a, b, c, d = plane_params
    
    # Create a grid of points on the plane
    x = np.linspace(points[:, 0].min() - size, points[:, 0].max() + size, 2)
    y = np.linspace(points[:, 1].min() - size, points[:, 1].max() + size, 2)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z using plane equation: ax + by + cz + d = 0
    Z = -(a*X + b*Y + d) / c
    
    # Create vertices
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Create triangles
    triangles = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    return mesh

def visualize_frame(points, plane_params, frame_num):
    """Visualize points and fitted plane for a single frame"""
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create plane mesh
    plane_mesh = create_plane_mesh(plane_params, points)
    
    # Generate random color for this frame
    color = [random.random(), random.random(), random.random()]
    plane_mesh.paint_uniform_color(color)
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(plane_mesh)
    
    # Set view
    vis.get_render_option().point_size = 2
    vis.get_render_option().background_color = [0, 0, 0]
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main():
    # Define paths
    base_path = Path("/media/hdd_4/PhD/T4/embedded system/fp/experiment1")
    camera_params_path = base_path / "camera_params.yaml"
    
    # Load camera parameters once
    camera_params = load_camera_params(camera_params_path)
    
    # Create results directory if it doesn't exist
    results_dir = base_path / "plane_fitting_results"
    results_dir.mkdir(exist_ok=True)
    
    # Create numpy data directory
    npy_dir = results_dir / "numpy_data"
    npy_dir.mkdir(exist_ok=True)
    
    # Initialize arrays to store all data
    all_plane_params = []
    all_frame_numbers = []
    
    # Process frames from 2000 to 2099
    for frame_num in range(2000, 2100):
        # Format frame number with leading zeros
        frame_str = f"{frame_num:06d}"
        
        # Define paths for current frame
        depth_path = base_path / f"depth/depth_{frame_str}.png"
        mask_path = base_path / f"segmentation_results/mask_image/mask_rgb_{frame_str}.png"
        
        # Check if files exist
        if not depth_path.exists() or not mask_path.exists():
            print(f"Skipping frame {frame_str} - files not found")
            continue
            
        print(f"Processing frame {frame_str}")
        
        # Load data
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if depth_img is None or mask is None:
            print(f"Error loading frame {frame_str}")
            continue
        
        # Convert depth to points
        points = depth_to_points(depth_img, mask, camera_params)
        
        if len(points) < 10:  # Skip if too few points
            print(f"Frame {frame_str} has too few points ({len(points)})")
            continue
        
        # Fit plane
        plane_params = fit_plane(points)
        
        # Visualize the frame
        # visualize_frame(points, plane_params, frame_num)
        
        # Save results
        results = {
            'frame': frame_str,
            'plane_equation': {
                'a': float(plane_params[0]),
                'b': float(plane_params[1]),
                'c': float(plane_params[2]),
                'd': float(plane_params[3])
            },
            'num_points': len(points)
        }
        
        # Save to JSON file
        result_file = results_dir / f"plane_{frame_str}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save points as numpy array
        np.save(npy_dir / f"points_{frame_str}.npy", points)
        
        # Store plane parameters and frame number
        all_plane_params.append(plane_params)
        all_frame_numbers.append(frame_num)
        
        print(f"Frame {frame_str} processed - {len(points)} points, plane equation: "
              f"{plane_params[0]:.4f}x + {plane_params[1]:.4f}y + {plane_params[2]:.4f}z + {plane_params[3]:.4f} = 0")
    
    # Convert lists to numpy arrays
    all_plane_params = np.array(all_plane_params)
    all_frame_numbers = np.array(all_frame_numbers)
    
    # Save all plane parameters and frame numbers
    np.save(npy_dir / "all_plane_params.npy", all_plane_params)
    np.save(npy_dir / "all_frame_numbers.npy", all_frame_numbers)
    
    print(f"\nSaved all data to {npy_dir}")
    print(f"Total frames processed: {len(all_frame_numbers)}")
    print(f"Shape of plane parameters array: {all_plane_params.shape}")

if __name__ == "__main__":
    main()
