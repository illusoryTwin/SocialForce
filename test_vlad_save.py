import numpy as np
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from tqdm import tqdm

def pixel_to_world_coords(px, img_shape, fx=525.0, fy=525.0, cx=639.5, cy=359.5, H=1.65):
    """
    Transform pixel coordinates to world coordinates using camera parameters
    px: pixel coordinates (x, y)
    img_shape: (height, width) of the image
    fx, fy: focal lengths in x and y directions
    cx, cy: principal point coordinates (adjusted for 1280x720)
    H: camera height from ground
    """
    # Convert pixel coordinates to normalized image coordinates
    x = (px[0] - cx) / fx
    y = (px[1] - cy) / fy
    
    # Calculate world coordinates using perspective projection
    # Using similar triangles principle
    # X = H * x / y
    # Z = H / y
    # Y = 0 (ground plane assumption)
    
    X = H * x / y
    Z = H / y
    Y = 0  # Ground plane assumption
    
    # Scale up world coordinates by 3
    return np.array([X * 3, Z * 3, Y])  # Return in X, Z, Y order for better visualization

class TrajectorySaver:
    def __init__(self):
        self.next_id = 0
        self.tracked_pedestrians = {}  # Dictionary to store tracked pedestrians
        self.max_distance = 50  # Maximum distance to consider as same person
        self.trajectories = {}  # Dictionary to store trajectories
        self.frame_numbers = {}  # Dictionary to store frame numbers for each point
        self.timestamps = {}  # Dictionary to store timestamps for each point

    def get_id(self, box):
        # Get center point of the box
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center = np.array([center_x, center_y])
        
        # Calculate box size
        box_width = x2 - x1
        box_height = y2 - y1
        box_size = box_width * box_height

        # Check if this is a new detection or matches an existing one
        best_match = None
        best_distance = float('inf')
        
        for ped_id, last_box in self.tracked_pedestrians.items():
            last_x1, last_y1, last_x2, last_y2 = last_box
            last_center_x = (last_x1 + last_x2) / 2
            last_center_y = (last_y1 + last_y2) / 2
            last_center = np.array([last_center_x, last_center_y])

            # Calculate distance between centers
            distance = np.linalg.norm(center - last_center)
            
            # Calculate size difference
            last_width = last_x2 - last_x1
            last_height = last_y2 - last_y1
            last_size = last_width * last_height
            size_ratio = min(box_size, last_size) / max(box_size, last_size)
            
            # Only consider it a match if both distance and size are similar
            if distance < self.max_distance and size_ratio > 0.7:
                if distance < best_distance:
                    best_distance = distance
                    best_match = ped_id

        if best_match is not None:
            # Update the position of existing pedestrian
            self.tracked_pedestrians[best_match] = box
            return best_match

        # If no match found, assign new ID
        new_id = self.next_id
        self.next_id += 1
        self.tracked_pedestrians[new_id] = box
        return new_id

    def add_to_trajectory(self, ped_id, world_coords, frame_num):
        """Add a point to the trajectory of a pedestrian"""
        if ped_id not in self.trajectories:
            self.trajectories[ped_id] = []
            self.frame_numbers[ped_id] = []
            self.timestamps[ped_id] = []
        
        # Add new point
        self.trajectories[ped_id].append(world_coords)
        self.frame_numbers[ped_id].append(frame_num)
        self.timestamps[ped_id].append(datetime.now().timestamp())

    def save_trajectories(self, filename='trajectories_3.npz'):
        """Save all trajectories to numpy file"""
        # Convert trajectories to numpy arrays
        trajectory_data = {}
        for ped_id in self.trajectories:
            if len(self.trajectories[ped_id]) > 0:
                trajectory_data[f'id_{ped_id}_coords'] = np.array(self.trajectories[ped_id])
                trajectory_data[f'id_{ped_id}_frames'] = np.array(self.frame_numbers[ped_id])
                trajectory_data[f'id_{ped_id}_timestamps'] = np.array(self.timestamps[ped_id])
        
        # Save to npz file
        np.savez(filename, **trajectory_data)
        print(f"Saved trajectories for {len(trajectory_data)//3} pedestrians to {filename}")

    def process_video(self, start_frame, end_frame, base_path):
        """Process video frames and save trajectories"""
        print(f"Processing frames {start_frame} to {end_frame}")
        
        # Create progress bar
        pbar = tqdm(range(start_frame, end_frame + 1), desc="Processing frames")
        
        for frame_num in pbar:
            # Read frame
            img_path = os.path.join(base_path, f'rgb_{str(frame_num).zfill(6)}.png')
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
            results = model(img_rgb, classes=[0])
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    if conf > 0.3:
                        ped_id = self.get_id((x1, y1, x2, y2))
                        
                        # Calculate bottom center point
                        bottom_center_x = int((x1 + x2) / 2)
                        bottom_center_y = y2
                        
                        # Convert to world coordinates
                        px = np.array([bottom_center_x, bottom_center_y])
                        world_coords = pixel_to_world_coords(px, img.shape)
                        
                        # Add to trajectory
                        self.add_to_trajectory(ped_id, world_coords, frame_num)
            
            # Update progress bar description with current number of tracked pedestrians
            pbar.set_postfix({'tracked_peds': len(self.trajectories)})

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize saver
saver = TrajectorySaver()

# Base path for images
base_path = "/media/hdd_4/PhD/T4/embedded system/fp/experiment3/rgb"

# Process video and save trajectories
start_frame = 2300
end_frame = 5000
saver.process_video(start_frame, end_frame, base_path)
saver.save_trajectories() 