import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
import cv2
import numpy as np
import os
from matplotlib.animation import FuncAnimation
import pandas as pd
from datetime import datetime

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

class LivePedestrianTracker:
    def __init__(self, max_history_frames=300):  # 10 seconds at 30 fps
        self.next_id = 0
        self.tracked_pedestrians = {}  # Dictionary to store tracked pedestrians
        self.max_distance = 50  # Maximum distance to consider as same person
        self.trajectories = {}  # Dictionary to store trajectories
        self.frame_numbers = {}  # Dictionary to store frame numbers for each point
        self.current_frame = None
        self.fig = None
        self.ax1 = None  # For video frame
        self.ax2 = None  # For trajectory plot
        self.max_history_frames = max_history_frames
        # Create DataFrame to store all trajectories
        self.trajectory_df = pd.DataFrame(columns=['x', 'y', 'id', 'timestamp'])

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
        
        # Add new point
        self.trajectories[ped_id].append(world_coords)
        self.frame_numbers[ped_id].append(frame_num)
        
        # Add to DataFrame
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        new_row = pd.DataFrame({
            'x': [world_coords[0]],
            'y': [world_coords[1]],
            'id': [ped_id],
            'timestamp': [timestamp]
        })
        self.trajectory_df = pd.concat([self.trajectory_df, new_row], ignore_index=True)
        
        # Remove old points
        while len(self.frame_numbers[ped_id]) > 0 and frame_num - self.frame_numbers[ped_id][0] > self.max_history_frames:
            self.trajectories[ped_id].pop(0)
            self.frame_numbers[ped_id].pop(0)

    def save_trajectories(self, filename='trajectories.csv'):
        """Save all trajectories to CSV file"""
        self.trajectory_df.to_csv(filename, index=False)
        print(f"Saved {len(self.trajectory_df)} trajectory points to {filename}")

    def setup_animation(self):
        """Setup the figure and axes for animation"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(30, 15))
        self.ax1.set_title('Video Frame')
        self.ax2.set_title('Live Pedestrian Trajectories (Last 10s)')
        self.ax2.set_xlabel('X (meters)')
        self.ax2.set_ylabel('Y (meters)')
        self.ax2.grid(True)
        self.ax2.axis('equal')
        plt.tight_layout()

    def update_animation(self, frame_num):
        """Update function for animation"""
        # Clear previous frame
        self.ax1.clear()
        self.ax2.clear()
        
        # Read and process frame
        img_path = os.path.join(base_path, f'rgb_{str(frame_num).zfill(6)}.png')
        img = cv2.imread(img_path)
        if img is None:
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_frame = img_rgb
        
        # Draw center point on the frame
        center_x = int(img.shape[1]/2)
        center_y = int(img.shape[0]/2)
        cv2.circle(img_rgb, (center_x, center_y), 7, (0, 0, 255), -1)  # Red circle with radius 7
        
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
                    
                    # Draw on video frame for all pedestrians
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img_rgb, (bottom_center_x, bottom_center_y), 3, (255, 0, 0), -1)
                    cv2.putText(img_rgb, f"ID {ped_id}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update video frame
        self.ax1.imshow(img_rgb)
        self.ax1.set_title(f'Frame {frame_num}')
        self.ax1.axis('off')
        
        # Update trajectory plot for all pedestrians
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.trajectories)))
        for (ped_id, trajectory), color in zip(self.trajectories.items(), colors):
            if len(trajectory) > 0:
                trajectory = np.array(trajectory)
                frames = self.frame_numbers[ped_id]
                
                # Plot trajectory
                self.ax2.scatter(trajectory[:, 0], trajectory[:, 1], 
                               c=frames, cmap='viridis', alpha=0.6, s=20)
                self.ax2.plot(trajectory[:, 0], trajectory[:, 1], '-', 
                            color=color, alpha=0.8, linewidth=2, label=f'ID {ped_id}')
                
                # Add ID label at the end of trajectory
                self.ax2.text(trajectory[-1, 0], trajectory[-1, 1], f'ID {ped_id}', 
                            fontsize=10, alpha=0.8, color=color)
        
        # Set plot limits with padding
        all_x = []
        all_y = []
        for trajectory in self.trajectories.values():
            if len(trajectory) > 0:
                trajectory = np.array(trajectory)
                all_x.extend(trajectory[:, 0])
                all_y.extend(trajectory[:, 1])
        
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            
            self.ax2.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax2.set_ylim(y_min - y_padding, y_max + y_padding)
        
        self.ax2.set_title('Live Pedestrian Trajectories (Last 10s)')
        self.ax2.set_xlabel('X (meters)')
        self.ax2.set_ylabel('Y (meters)')
        self.ax2.grid(True)
        self.ax2.axis('equal')
        # Add legend
        self.ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize tracker
tracker = LivePedestrianTracker()

# Base path for images
base_path = "/media/hdd_4/PhD/T4/embedded system/fp/experiment1/rgb"

# Setup animation
tracker.setup_animation()

# Create animation
start_frame = 2300
end_frame = 5000
ani = FuncAnimation(tracker.fig, tracker.update_animation, 
                   frames=range(start_frame, end_frame + 1),
                   interval=100)  # 100ms between frames

plt.show()

# Save trajectories after animation ends
tracker.save_trajectories() 