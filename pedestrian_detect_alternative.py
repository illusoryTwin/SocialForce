import cv2
import numpy as np
import yaml
import json
import glob
import os
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import time

class HumanTracker:
    def __init__(self, camera_params_path, model_path='yolov8n.pt'):
        # Load camera parameters
        with open(camera_params_path, 'r') as f:
            self.camera_params = yaml.safe_load(f)
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Initialize tracking variables
        self.trajectories = {}  # Dictionary to store trajectories
        self.next_id = 0
        
        # Visualization settings
        self.colors = plt.cm.rainbow(np.linspace(0, 1, 20))  # 20 different colors for trajectories
        
        # Initialize visualization windows
        cv2.namedWindow('RGB Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Trajectories', cv2.WINDOW_NORMAL)
        
    def load_frame(self, rgb_path, depth_path):
        # Load RGB image
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load depth image
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        return rgb, depth
    
    def detect_humans(self, frame):
        # Run YOLO detection
        results = self.model(frame, classes=0)  # class 0 is person in COCO dataset
        return results[0].boxes.data.cpu().numpy()  # Returns [x1, y1, x2, y2, conf, cls]
    
    def get_3d_position(self, bbox, depth):
        x1, y1, x2, y2 = map(int, bbox[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get depth value at center of bounding box
        depth_value = depth[center_y, center_x]
        
        # Convert to 3D coordinates using camera parameters
        # Using color camera parameters since we're working with RGB images
        fx = self.camera_params['d435_color']['fx']
        fy = self.camera_params['d435_color']['fy']
        cx = self.camera_params['d435_color']['cx']
        cy = self.camera_params['d435_color']['cy']
        
        Z = depth_value
        X = (center_x - cx) * Z / fx
        Y = (center_y - cy) * Z / fy
        
        return np.array([X, Y, Z])
    
    def update_trajectories(self, detections, depth, frame_id):
        current_positions = []
        
        for det in detections:
            if det[4] > 0.5:  # Confidence threshold
                pos_3d = self.get_3d_position(det, depth)
                current_positions.append((pos_3d, det))
        
        # Simple tracking: assign to nearest existing trajectory
        for pos, det in current_positions:
            if len(self.trajectories) == 0:
                self.trajectories[self.next_id] = [(frame_id, pos)]
                self.next_id += 1
                continue
            
            # Find nearest existing trajectory
            min_dist = float('inf')
            best_id = None
            
            for traj_id, traj in self.trajectories.items():
                last_pos = traj[-1][1]
                dist = np.linalg.norm(pos - last_pos)
                if dist < min_dist and dist < 1.0:  # 1 meter threshold
                    min_dist = dist
                    best_id = traj_id
            
            if best_id is not None:
                self.trajectories[best_id].append((frame_id, pos))
            else:
                self.trajectories[self.next_id] = [(frame_id, pos)]
                self.next_id += 1
    
    def visualize_frame(self, rgb, depth, detections, frame_id):
        # Create a copy of the RGB image for visualization
        vis_rgb = rgb.copy()
        
        # Draw bounding boxes and IDs
        for det in detections:
            if det[4] > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, det[:4])
                conf = det[4]
                
                # Find the trajectory ID for this detection
                pos_3d = self.get_3d_position(det, depth)
                traj_id = None
                min_dist = float('inf')
                
                for tid, traj in self.trajectories.items():
                    if len(traj) > 0:
                        last_pos = traj[-1][1]
                        dist = np.linalg.norm(pos_3d - last_pos)
                        if dist < min_dist and dist < 1.0:
                            min_dist = dist
                            traj_id = tid
                
                # Draw bounding box
                color = tuple(map(int, self.colors[traj_id % len(self.colors)][:3] * 255))
                cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and confidence
                label = f"ID: {traj_id} ({conf:.2f})"
                cv2.putText(vis_rgb, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Normalize depth for visualization
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create trajectory visualization
        traj_vis = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw trajectories
        for traj_id, traj in self.trajectories.items():
            if len(traj) > 1:
                color = tuple(map(int, self.colors[traj_id % len(self.colors)][:3] * 255))
                points = np.array([t[1] for t in traj])
                
                # Project 3D points to 2D for visualization
                fx = self.camera_params['d435_color']['fx']
                fy = self.camera_params['d435_color']['fy']
                cx = self.camera_params['d435_color']['cx']
                cy = self.camera_params['d435_color']['cy']
                
                # Project points
                x = (points[:, 0] * fx / points[:, 2] + cx).astype(int)
                y = (points[:, 1] * fy / points[:, 2] + cy).astype(int)
                
                # Draw trajectory
                for i in range(len(x)-1):
                    cv2.line(traj_vis, (x[i], y[i]), (x[i+1], y[i+1]), color, 2)
                
                # Draw current position
                cv2.circle(traj_vis, (x[-1], y[-1]), 5, color, -1)
        
        # Add frame number
        cv2.putText(vis_rgb, f"Frame: {frame_id}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show images
        cv2.imshow('RGB Detection', cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('Depth', depth_vis)
        cv2.imshow('Trajectories', traj_vis)
        
        # Wait for key press (1ms delay)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            return False
        return True
    
    def process_sequence(self, rgb_dir, depth_dir, timestamps_path):
        # Load timestamps
        with open(timestamps_path, 'r') as f:
            timestamps = json.load(f)
        
        # Get sorted list of frames
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
        
        # Filter frames between 2000 and 2100
        start_frame = 2000
        end_frame = 2500
        
        # Process only the specified range of frames
        for i, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
            # Extract frame number from filename
            frame_num = int(os.path.basename(rgb_path).split('_')[1].split('.')[0])
            
            # Skip frames outside our range
            if frame_num < start_frame or frame_num > end_frame:
                continue
                
            frame_id = frame_num  # Use actual frame number as ID
            rgb, depth = self.load_frame(rgb_path, depth_path)
            detections = self.detect_humans(rgb)
            self.update_trajectories(detections, depth, frame_id)
            
            # Visualize frame
            if not self.visualize_frame(rgb, depth, detections, frame_id):
                break
            
            print(f"Processed frame {frame_num}")
        
        cv2.destroyAllWindows()
    
    def save_trajectories(self, output_path):
        # Convert trajectories to numpy arrays
        traj_data = {}
        for traj_id, traj in self.trajectories.items():
            frames = np.array([t[0] for t in traj])
            positions = np.array([t[1] for t in traj])
            traj_data[traj_id] = {
                'frames': frames,
                'positions': positions
            }
        
        # Save to numpy file
        np.save(output_path, traj_data)
    
    def visualize_trajectories_2d(self):
        """Create a 2D top-down view of the trajectories"""
        plt.figure(figsize=(12, 12))
        for traj_id, traj in self.trajectories.items():
            positions = np.array([t[1] for t in traj])
            color = self.colors[traj_id % len(self.colors)]
            plt.plot(positions[:, 0], positions[:, 2], 
                    label=f'Person {traj_id}',
                    color=color,
                    linewidth=2)
            # Plot start and end points
            plt.scatter(positions[0, 0], positions[0, 2], 
                       color=color, marker='o', s=100)
            plt.scatter(positions[-1, 0], positions[-1, 2], 
                       color=color, marker='*', s=200)
        
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('Human Trajectories (Top View)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('trajectories_2d.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_trajectories_3d(self):
        """Create a 3D visualization of the trajectories"""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for traj_id, traj in self.trajectories.items():
            positions = np.array([t[1] for t in traj])
            color = self.colors[traj_id % len(self.colors)]
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                   label=f'Person {traj_id}',
                   color=color,
                   linewidth=2)
            # Plot start and end points
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                      color=color, marker='o', s=100)
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                      color=color, marker='*', s=200)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Human Trajectories (3D View)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('trajectories_3d.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_trajectories(self):
        """Create both 2D and 3D visualizations"""
        self.visualize_trajectories_2d()
        self.visualize_trajectories_3d()

def main():
    # Paths
    base_dir = "/media/hdd_4/PhD/T4/embedded system/fp/experiment1"
    rgb_dir = os.path.join(base_dir, "rgb")
    depth_dir = os.path.join(base_dir, "depth")
    camera_params_path = os.path.join(base_dir, "camera_params.yaml")
    timestamps_path = os.path.join(base_dir, "frame_timestamps.json")
    output_path = os.path.join(base_dir, "human_trajectories.npy")
    
    # Create tracker
    tracker = HumanTracker(camera_params_path)
    
    # Process sequence
    tracker.process_sequence(rgb_dir, depth_dir, timestamps_path)
    
    # Save trajectories
    tracker.save_trajectories(output_path)
    
    # Visualize trajectories
    tracker.visualize_trajectories()

if __name__ == "__main__":
    main()
