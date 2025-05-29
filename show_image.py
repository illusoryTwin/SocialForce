import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
import cv2
import numpy as np
import os
from matplotlib.animation import FuncAnimation

def pixel_to_world_coords(px, img_shape, f=525, H=1.65):
    """
    Transform pixel coordinates to world coordinates
    px: pixel coordinates (x, y)
    img_shape: (height, width) of the image
    f: focal length
    H: camera height
    """
    # Get image dimensions
    sz = np.array(img_shape[1::-1])  # width, height
    
    # Transformation matrix and vector
    A = np.array([[1, 0], [0, -1]])
    b = 1/2 * np.array([-1, 1]) * sz
    
    # Transform pixel coordinates
    w = A @ px + b
    
    # Convert to world coordinates
    world_coords = -H * np.array([w[0]/w[1], f/w[1], 1])
    
    return world_coords

class PedestrianTracker:
    def __init__(self):
        self.next_id = 0
        self.tracked_pedestrians = {}  # Dictionary to store tracked pedestrians
        self.max_distance = 50  # Maximum distance to consider as same person
        self.trajectories = {}  # Dictionary to store trajectories
        self.frame_numbers = {}  # Dictionary to store frame numbers for each point
        self.current_frame = None
        self.fig = None
        self.ax1 = None  # For video frame
        self.ax2 = None  # For trajectory plot

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
        self.trajectories[ped_id].append(world_coords)
        self.frame_numbers[ped_id].append(frame_num)

    def setup_animation(self):
        """Setup the figure and axes for animation"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.ax1.set_title('Video Frame')
        self.ax2.set_title('Trajectory for ID 1')
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
                    
                    # Draw on video frame
                    if ped_id == 1:  # Only draw ID 1
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(img_rgb, (bottom_center_x, bottom_center_y), 3, (255, 0, 0), -1)
                        cv2.putText(img_rgb, f"ID {ped_id}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update video frame
        self.ax1.imshow(img_rgb)
        self.ax1.set_title(f'Frame {frame_num}')
        self.ax1.axis('off')
        
        # Update trajectory plot
        if 1 in self.trajectories:
            trajectory = np.array(self.trajectories[1])
            frames = self.frame_numbers[1]
            
            # Plot trajectory
            self.ax2.scatter(trajectory[:, 0], trajectory[:, 1], 
                           c=frames, cmap='viridis', alpha=0.6)
            self.ax2.plot(trajectory[:, 0], trajectory[:, 1], '--', alpha=0.3)
            
            # Set plot limits with padding
            x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
            y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            
            self.ax2.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax2.set_ylim(y_min - y_padding, y_max + y_padding)
        
        self.ax2.set_title('Trajectory for ID 1')
        self.ax2.set_xlabel('X (meters)')
        self.ax2.set_ylabel('Y (meters)')
        self.ax2.grid(True)
        self.ax2.axis('equal')

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize tracker
tracker = PedestrianTracker()

# Base path for images
base_path = "/media/hdd_4/PhD/T4/embedded system/fp/experiment1/rgb"

# Setup animation
tracker.setup_animation()

# Create animation
start_frame = 2475
end_frame = 2495
ani = FuncAnimation(tracker.fig, tracker.update_animation, 
                   frames=range(start_frame, end_frame + 1),
                   interval=100)  # 100ms between frames

plt.show() 