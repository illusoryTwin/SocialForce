import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import sys
import os
import glob
import open3d as o3d


class PedestrianDetector:
    def __init__(self, plane_equation_path, camera_params_path):
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Load plane equation
        xyz_d0_means = np.load('xyz_d0_means.npy')
        print("Loaded plane coefficients:", xyz_d0_means)

        a, b, c, d0 = xyz_d0_means
        d0 = -d0  # Negate d0 to match 3D_visualization.py

        with open(plane_equation_path, 'r') as f:
            plane_data = yaml.safe_load(f)
            self.plane_eq = np.array([a, b, c, d0])
        
        # Load camera parameters
        with open(camera_params_path, 'r') as f:
            camera_data = yaml.safe_load(f)
            # Use color camera parameters
            color_cam = camera_data['d435_color']
            self.fx = color_cam['fx']
            self.fy = color_cam['fy']
            self.cx = color_cam['cx']
            self.cy = color_cam['cy']
        
        # Initialize pedestrian tracking
        self.next_pedestrian_id = 0
        self.pedestrian_tracks = {}  # Dictionary to store pedestrian tracks
        
        # Initialize Open3D visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # Create plane mesh
        self.plane_mesh = self.create_plane_mesh(self.plane_eq, size=200.0)
        self.plane_mesh.paint_uniform_color([0, 1, 0])  # Green color
        
        # Create coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0,
            origin=[0, 0, 0]
        )
        
        # Add geometries to visualizer
        self.vis.add_geometry(self.plane_mesh)
        self.vis.add_geometry(self.coordinate_frame)
        
        # Set up camera
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
    
    def create_plane_mesh(self, coefficients, size=200.0):
        a, b, c, d0 = coefficients
        
        # Create a grid of points
        x = np.linspace(-size, size, 20)
        y = np.linspace(-size, size, 20)
        X, Y = np.meshgrid(x, y)
        
        # Calculate Z values for the plane (ax + by + cz + d0 = 0)
        Z = -(a*X + b*Y + d0) / c
        
        # Create vertices
        vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Create triangles
        triangles = []
        for i in range(19):
            for j in range(19):
                v0 = i * 20 + j
                v1 = v0 + 1
                v2 = v0 + 20
                v3 = v2 + 1
                triangles.extend([[v0, v1, v2], [v1, v3, v2]])
        
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        
        return mesh

    def project_to_plane(self, x, y, depth):
        # Convert pixel coordinates to camera coordinates
        X = (x - self.cx) * depth / self.fx
        Y = (y - self.cy) * depth / self.fy
        Z = depth
        
        # Project point onto plane
        A, B, C, D = self.plane_eq
        t = -(A*X + B*Y + C*Z + D) / (A*A + B*B + C*C)
        
        X_plane = X + A*t
        Y_plane = Y + B*t
        Z_plane = Z + C*t
        
        return X_plane, Y_plane, Z_plane
    
    def draw_floor_plane(self, frame):
        height, width = frame.shape[:2]
        
        # Create a grid of points in camera coordinates
        grid_size = 20
        x = np.linspace(0, width, grid_size)
        y = np.linspace(0, height, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Convert grid points to camera coordinates
        Z = np.ones_like(X) * 2.0  # Assuming constant depth for visualization
        X_cam = (X - self.cx) * Z / self.fx
        Y_cam = (Y - self.cy) * Z / self.fy
        
        # Project points onto the plane
        A, B, C, D = self.plane_eq
        t = -(A*X_cam + B*Y_cam + C*Z + D) / (A*A + B*B + C*C)
        X_plane = X_cam + A*t
        Y_plane = Y_cam + B*t
        Z_plane = Z + C*t
        
        # Convert back to image coordinates
        X_img = X_plane * self.fx / Z_plane + self.cx
        Y_img = Y_plane * self.fy / Z_plane + self.cy
        
        # Create overlay for the floor plane
        overlay = frame.copy()
        
        # Draw grid lines
        for i in range(grid_size):
            # Horizontal lines
            points = np.column_stack((X_img[i, :], Y_img[i, :]))
            points = points.astype(np.int32)
            cv2.polylines(overlay, [points], False, (0, 255, 0), 1)
            
            # Vertical lines
            points = np.column_stack((X_img[:, i], Y_img[:, i]))
            points = points.astype(np.int32)
            cv2.polylines(overlay, [points], False, (0, 255, 0), 1)
        
        # Add semi-transparent overlay
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def create_point_cloud(self, rgb, depth):
        # Create meshgrid of pixel coordinates
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Convert to 3D points
        z = depth.astype(float) / 10.0  # Convert to meters
        x = (c - self.cx) * z / self.fx
        y = (r - self.cy) * z / self.fy
        
        # Stack coordinates and reshape
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)
        colors = rgb.reshape(-1, 3) / 255.0
        
        # Remove invalid points (where depth is 0)
        valid_points = z.reshape(-1) > 0
        points = points[valid_points]
        colors = colors[valid_points]
        
        return points, colors

    def create_bounding_box(self, x1, y1, x2, y2, depth):
        # Convert 2D points to 3D
        points_2d = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]  # Bottom face
        ])
        
        points_3d = []
        for x, y in points_2d:
            X = (x - self.cx) * depth / self.fx
            Y = (y - self.cy) * depth / self.fy
            Z = depth
            points_3d.append([X, Y, Z])
        
        # Create top face points
        top_points = np.array(points_3d) + np.array([0, 0, 0.5])  # Add height
        
        # Combine all points
        all_points = np.vstack([points_3d, top_points])
        
        # Create lines for the box
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
        ]
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # Red color
        
        return line_set

    def process_frame(self, frame, depth_frame):
        # Run YOLO detection
        results = self.model(frame, classes=[0])  # class 0 is person in COCO
        
        # Create point cloud from RGB and depth
        points, colors = self.create_point_cloud(frame, depth_frame)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Add RGB point cloud to visualization
        self.vis.add_geometry(pcd)
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get depth at center point
                depth = depth_frame[center_y, center_x]
                
                # Create and add bounding box
                bbox = self.create_bounding_box(x1, y1, x2, y2, depth)
                self.vis.add_geometry(bbox)
        
        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()
        
        return frame


def main():
    # Initialize detector
    detector = PedestrianDetector(
        'floor_plane_equation.yaml',
        'camera_params.yaml'
    )
    
    # Directory containing RGB frames
    rgb_dir = (
        "/media/hdd_4//PhD/T4/embedded system/"
        "fp/experiment1/rgb"
    )
    
    # Get all image files in the directory
    image_files = sorted(glob.glob(os.path.join(rgb_dir, "*.[jp][pn][g]")))
    
    if not image_files:
        print(f"Error: No image files found in {rgb_dir}")
        sys.exit(1)
    
    # Filter files to start from frame 2000
    start_frame = 2000
    image_files = [f for f in image_files if int(os.path.basename(f).split('_')[1].split('.')[0]) >= start_frame]
    
    print(f"Found {len(image_files)} images to process starting from frame {start_frame}")
    
    # Process only the first frame for debugging
    if image_files:
        img_path = image_files[0]
        print(f"Processing frame: {img_path}")
        
        # Read frame
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error: Could not read image {img_path}")
            sys.exit(1)
            
        # For demonstration, we'll use a synthetic depth frame
        # In real application, you would get this from your depth camera
        depth_frame = np.ones(
            (frame.shape[0], frame.shape[1]),
            dtype=np.float32
        ) * 2.0
        
        # Process frame
        detector.process_frame(frame, depth_frame)
        
        # Keep Open3D window open until closed
        detector.vis.run()
        detector.vis.destroy_window()


if __name__ == "__main__":
    main() 