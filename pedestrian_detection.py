import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import sys
import os
import glob
import csv

class PedestrianDetector:
    def __init__(self, plane_equation_path, camera_params_path):
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Load plane equation
        with open(plane_equation_path, 'r') as f:
            plane_data = yaml.safe_load(f)
            self.plane_eq = np.array([
                plane_data['plane_equation']['A'],
                plane_data['plane_equation']['B'],
                plane_data['plane_equation']['C'],
                plane_data['plane_equation']['D']
            ])
        
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
    
    def process_frame(self, frame, depth_frame):
        # Run YOLO detection
        results = self.model(frame, classes=[0])  # class 0 is person in COCO
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw floor plane
        vis_frame = self.draw_floor_plane(vis_frame)
        
        # List to store projected points and their IDs
        projected_points = []
        point_ids = []
        
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
                
                # Project to plane
                X, Y, Z = self.project_to_plane(center_x, center_y, depth)
                projected_points.append([X, Y, Z])
                
                # Assign or get pedestrian ID
                ped_id = self.next_pedestrian_id
                self.next_pedestrian_id += 1
                point_ids.append(ped_id)
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(vis_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Add text with ID and 3D coordinates
                text = f"ID:{ped_id} ({X:.2f}, {Y:.2f}, {Z:.2f})"
                cv2.putText(
                    vis_frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                
                # Project point to image plane
                proj_x = int(X * self.fx / Z + self.cx)
                proj_y = int(Y * self.fy / Z + self.cy)
                
                # Draw projection line
                cv2.line(
                    vis_frame,
                    (center_x, center_y),
                    (proj_x, proj_y),
                    (0, 0, 255),
                    1
                )
                
                # Draw projected point
                cv2.circle(vis_frame, (proj_x, proj_y), 3, (0, 0, 255), -1)
        
        projected_data = list(zip(point_ids, projected_points))
        return vis_frame, projected_data


def main():
    # Initialize detector
    detector = PedestrianDetector(
        'floor_plane_equation.yaml',
        'camera_params.yaml'
    )
    
    # Directory containing RGB frames
    rgb_dir = (
        "../imgs_data/rgb"
    )
    
    # Get all image files in the directory
    image_files = sorted(glob.glob(os.path.join(rgb_dir, "*.[jp][pn][g]")))
    
    if not image_files:
        print(f"Error: No image files found in {rgb_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # # Process each frame
    # for img_path in image_files:
    #     # Read frame
    #     frame = cv2.imread(img_path)
    #     if frame is None:
    #         print(f"Error: Could not read image {img_path}")
    #         continue
            
    #     # For demonstration, we'll use a synthetic depth frame
    #     # In real application, you would get this from your depth camera
    #     depth_frame = np.ones(
    #         (frame.shape[0], frame.shape[1]),
    #         dtype=np.float32
    #     ) * 2.0
        
    #     # Process frame
    #     result_frame = detector.process_frame(frame, depth_frame)
        
    #     # Display results
    #     cv2.imshow('Pedestrian Detection', result_frame)
        
    #     # Wait for key press (1ms) and break if 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

      # Open CSV file for writing
    with open("projected_points.csv", mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "pedestrian_id", "X", "Y", "Z"])

        for frame_index, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Error: Could not read image {img_path}")
                continue

            depth_frame = np.ones((frame.shape[0], frame.shape[1]), dtype=np.float32) * 2.0

            # Process frame and get result and projected data
            result_frame, projected_data = detector.process_frame(frame, depth_frame)

            # Write projected points to CSV
            for ped_id, (X, Y, Z) in projected_data:
                csv_writer.writerow([frame_index, ped_id, X, Y, Z])

            # Display result
            cv2.imshow('Pedestrian Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 