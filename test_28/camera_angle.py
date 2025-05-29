import cv2
import numpy as np
import json
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

class PedestrianTracker:
    def __init__(self, camera_params_path):
        # Load camera parameters
        with open(camera_params_path, 'r') as f:
            self.camera_params = yaml.safe_load(f)
        
        # Get RGB camera intrinsics
        self.rgb_fx = self.camera_params['d435_color']['fx']
        self.rgb_fy = self.camera_params['d435_color']['fy']
        self.rgb_cx = self.camera_params['d435_color']['cx']
        self.rgb_cy = self.camera_params['d435_color']['cy']
        
        # Get depth camera intrinsics
        self.depth_fx = self.camera_params['d435_depth']['fx']
        self.depth_fy = self.camera_params['d435_depth']['fy']
        self.depth_cx = self.camera_params['d435_depth']['cx']
        self.depth_cy = self.camera_params['d435_depth']['cy']
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model
        
        # Store trajectories for all three cases
        self.trajectories_rgb = {}  # RGB only trajectories (2D)
        self.trajectories_real = {}  # Real depth trajectories (3D)
        self.trajectories_synthetic = {}  # Synthetic depth trajectories (3D)

        # Camera to world transformation matrix
        # Assuming camera is mounted at height h and looking forward
        self.camera_height = 1.5  # meters (adjust based on your setup)
        self.camera_angle = 0.0   # radians (adjust based on your setup)
        
        # Create transformation matrix
        self.camera_to_world = self._create_camera_to_world_transform()

        # Initialize visualization
        self.fig = plt.figure(figsize=(15, 10))
        self.ax1 = self.fig.add_subplot(221)  # RGB
        self.ax2 = self.fig.add_subplot(222)  # Depth
        self.ax3 = self.fig.add_subplot(223)  # Real Depth Trajectories
        self.ax4 = self.fig.add_subplot(224)  # Synthetic Depth Trajectories
        plt.ion()  # Turn on interactive mode

    def _create_camera_to_world_transform(self):
        """Create transformation matrix from camera to world coordinates"""
        # Rotation matrix for camera tilt
        R = np.array([
            [1, 0, 0],
            [0, np.cos(self.camera_angle), -np.sin(self.camera_angle)],
            [0, np.sin(self.camera_angle), np.cos(self.camera_angle)]
        ])
        
        # Translation vector (camera position in world coordinates)
        t = np.array([0, -self.camera_height, 0])
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T

    def camera_to_world_coordinates(self, X, Y, Z):
        """Convert camera coordinates to world coordinates"""
        # Create homogeneous coordinates
        point_camera = np.array([X, Y, Z, 1])
        
        # Apply transformation
        point_world = np.dot(self.camera_to_world, point_camera)
        
        # Return 3D coordinates
        return point_world[:3]

    def get_depth_from_mask(self, depth_img, x1, y1, x2, y2):
        """Get depth value using bottom center point of bounding box"""
        # Get image dimensions
        height, width = depth_img.shape
        
        # Calculate bottom center point
        bottom_center_x = min((x1 + x2) // 2, width - 1)
        bottom_center_y = min(y2, height - 1)  # Bottom of the bounding box
        
        # Get depth value at bottom center point
        depth = depth_img[bottom_center_y, bottom_center_x]
        
        # Create mask for visualization
        mask = np.zeros_like(depth_img, dtype=bool)
        mask[y1:y2, x1:x2] = True
        
        # Get all depth values in the box for debugging
        box_depths = depth_img[y1:y2, x1:x2] 
        
        valid_depths = box_depths[box_depths > 0]  # Filter out zero/invalid depths
        
        if len(valid_depths) > 0:
            print(f"Bottom center point depth: {depth/1000:.2f}m at ({bottom_center_x}, {bottom_center_y})")
        else:
            print(f"No valid depth values in box at ({bottom_center_x}, {bottom_center_y})")
            return None, mask
            
        return depth, mask

    def calculate_angle_and_distance(self, x, y, depth):
        """Calculate angle and distance for a detected pedestrian"""
        # Convert depth from millimeters to meters
        depth_meters = depth / 1000.0

        # Convert pixel coordinates to depth camera coordinates (in meters)
        X = (x - self.depth_cx) * depth_meters / self.depth_fx
        Y = (y - self.depth_cy) * depth_meters / self.depth_fy
        Z = depth_meters

        # Convert to world coordinates
        X_world, Y_world, Z_world = self.camera_to_world_coordinates(X, Y, Z)

        # Calculate angle (in radians) in world coordinates
        angle = np.arctan2(X_world, Z_world)
        
        # Calculate distance in world coordinates (in meters)
        # Use Pythagorean theorem to get true distance to the point on the ground
        # where the pedestrian is standing
        ground_distance = np.sqrt(X_world**2 + Z_world**2)  # Distance in XZ plane
        true_distance = np.sqrt(ground_distance**2 + 1.65**2)  # Include camera height
        
        return angle, true_distance, (X_world, Y_world, Z_world)

    def process_frame(self, rgb_path, depth_path, depth_synthetic_path):
        """Process a single frame"""
        # Read images
        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth_synthetic_img = cv2.imread(depth_synthetic_path, cv2.IMREAD_ANYDEPTH)
        
        if rgb_img is None or depth_img is None or depth_synthetic_img is None:
            print(f"Error reading images: {rgb_path}, {depth_path}, {depth_synthetic_path}")
            return None

        # Print depth image statistics
        # print(f"\nDepth image stats:")
        # print(f"Real depth - min: {np.min(depth_img)/1000:.2f}m, max: {np.max(depth_img)/1000:.2f}m, mean: {np.mean(depth_img)/1000:.2f}m")
        # print(f"Synthetic depth - min: {np.min(depth_synthetic_img)/1000:.2f}m, max: {np.max(depth_synthetic_img)/1000:.2f}m, mean: {np.mean(depth_synthetic_img)/1000:.2f}m")

        # Get image dimensions
        height, width = rgb_img.shape[:2]
        
        # Resize synthetic depth to match real depth dimensions
        depth_synthetic_img = cv2.resize(depth_synthetic_img, (width, height))
        
        # Normalize synthetic depth to match real depth range
        depth_synthetic_normalized = cv2.normalize(depth_synthetic_img, None, 
                                                 np.min(depth_img), np.max(depth_img), 
                                                 cv2.NORM_MINMAX)
        
        # Multiply normalized synthetic depth by 100
        depth_synthetic_normalized = depth_synthetic_normalized * 100
        
        # print(f"Normalized synthetic depth - min: {np.min(depth_synthetic_normalized)/1000:.2f}m, max: {np.max(depth_synthetic_normalized)/1000:.2f}m, mean: {np.mean(depth_synthetic_normalized)/1000:.2f}m")

        # Store the last processed RGB image for dimensions
        self.last_rgb_img = rgb_img.copy()

        # Convert BGR to RGB for visualization
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Normalize depth for visualization
        depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Normalize synthetic depth for visualization
        depth_synthetic_vis = cv2.normalize(depth_synthetic_normalized, None, 0, 255, cv2.NORM_MINMAX)
        depth_synthetic_vis = cv2.applyColorMap(depth_synthetic_vis.astype(np.uint8), cv2.COLORMAP_JET)

        # Run YOLO detection only on RGB image
        results = self.model(rgb_img, classes=[0])  # class 0 is person in COCO dataset
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Get confidence
                conf = float(box.conf[0].cpu().numpy())
                
                if conf > 0.3:  # Confidence threshold
                    print(f"\nProcessing detection at ({x1}, {y1}, {x2}, {y2})")
                    
                    # Get bottom-middle point for visualization
                    bottom_x = x1 + w//2
                    bottom_y = y2  # Bottom of the bounding box
                    
                    # Get depth using bounding box as mask
                    depth_real, mask_real = self.get_depth_from_mask(depth_img * 100, x1, y1, x2, y2)
                    
                    # For synthetic depth, use coordinates relative to the top-left quarter
                    height, width = self.last_rgb_img.shape[:2]
                    # Ensure box is within top-left quarter and has non-zero size
                    synth_x1 = min(max(x1, 0), width//2 - 1)
                    synth_y1 = min(max(y1, 0), height//2 - 1)
                    synth_x2 = min(max(x2, 1), width//2)
                    synth_y2 = min(max(y2, 1), height//2)
                    
                    # print(f"\nSynthetic depth processing:")
                    # print(f"Original box: ({x1}, {y1}, {x2}, {y2})")
                    # print(f"Synthetic box: ({synth_x1}, {synth_y1}, {synth_x2}, {synth_y2})")
                    
                    # Get synthetic depth values for the box
                    synth_box_depths = depth_synthetic_normalized[synth_y1:synth_y2, synth_x1:synth_x2]
                    # print(f"Synthetic depth box shape: {synth_box_depths.shape}")
                    # print(f"Synthetic depth values in box:\n{synth_box_depths}")
                    
                    # Draw synthetic depth box on visualization
                    cv2.rectangle(depth_synthetic_vis, (synth_x1, synth_y1), (synth_x2, synth_y2), (0, 255, 0), 2)
                    cv2.putText(depth_synthetic_vis, f"Box", 
                              (synth_x1, synth_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    depth_synthetic, mask_synthetic = self.get_depth_from_mask(depth_synthetic_normalized * 100, 
                                                                             synth_x1, synth_y1, synth_x2, synth_y2)
                    
                    # Skip if both depths are invalid
                    if depth_real is None and depth_synthetic is None:
                        print("Both depths are invalid, skipping detection")
                        continue
                    
                    # Calculate angle, distance and world coordinates for both depth sources
                    angle_real, distance_real, world_coords_real = self.calculate_angle_and_distance(bottom_x, bottom_y, depth_real/100) if depth_real is not None else (None, None, None)
                    angle_synthetic, distance_synthetic, world_coords_synthetic = self.calculate_angle_and_distance(bottom_x, bottom_y, depth_synthetic/100) if depth_synthetic is not None else (None, None, None)
                    
                    # Draw detection on RGB image
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(rgb_img, (bottom_x, bottom_y), 5, (255, 0, 0), -1)
                    
                    # Find matching trajectory ID
                    traj_id = None
                    for tid, trajectory in self.trajectories_real.items():
                        if len(trajectory['positions']) > 0 and self.is_same_person(trajectory['positions'][-1], (bottom_x, bottom_y)):
                            traj_id = tid
                            break
                    
                    # Add ID label if found
                    if traj_id is not None:
                        cv2.putText(rgb_img, f"ID: {traj_id}", 
                                   (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.putText(rgb_img, f"Real: {distance_real:.2f}m" if distance_real else "Real: N/A", 
                               (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(rgb_img, f"Synth: {distance_synthetic:.2f}m" if distance_synthetic else "Synth: N/A", 
                               (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Apply masks to depth visualizations
                    if depth_real is not None:
                        # Create colored mask overlay
                        mask_overlay = np.zeros_like(depth_vis)
                        mask_overlay[mask_real] = [0, 255, 0]  # Green mask
                        # Blend mask with depth visualization
                        depth_vis = cv2.addWeighted(depth_vis, 0.7, mask_overlay, 0.3, 0)
                        cv2.circle(depth_vis, (bottom_x, bottom_y), 5, (255, 0, 0), -1)
                        # Add ID label if found
                        if traj_id is not None:
                            cv2.putText(depth_vis, f"ID: {traj_id}", 
                                      (bottom_x - 30, bottom_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(depth_vis, f"{distance_real:.2f}m", 
                                   (bottom_x - 30, bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Always show mask on synthetic depth, even if depth is None
                    # Create colored mask overlay
                    mask_overlay = np.zeros_like(depth_synthetic_vis)
                    mask_overlay[mask_synthetic] = [0, 255, 0]  # Green mask
                    # Blend mask with depth visualization
                    depth_synthetic_vis = cv2.addWeighted(depth_synthetic_vis, 0.7, mask_overlay, 0.3, 0)
                    cv2.circle(depth_synthetic_vis, (bottom_x, bottom_y), 5, (255, 0, 0), -1)
                    
                    # Add ID label if found
                    if traj_id is not None:
                        cv2.putText(depth_synthetic_vis, f"ID: {traj_id}", 
                                  (bottom_x - 30, bottom_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if depth_synthetic is not None:
                        cv2.putText(depth_synthetic_vis, f"{distance_synthetic:.2f}m", 
                                   (bottom_x - 30, bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(depth_synthetic_vis, f"N/A {distance_synthetic:.2f}m", 
                                   (bottom_x - 30, bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detections.append({
                        'position': (bottom_x, bottom_y),
                        'angle_real': angle_real,
                        'distance_real': distance_real,
                        'world_coords_real': world_coords_real,
                        'angle_synthetic': angle_synthetic,
                        'distance_synthetic': distance_synthetic,
                        'world_coords_synthetic': world_coords_synthetic,
                        'box': (x1, y1, w, h)
                    })

        return detections, rgb_img, depth_vis, depth_synthetic_vis

    def process_sequence(self, rgb_dir, depth_dir, depth_synthetic_dir, timestamps_path):
        """Process entire sequence of frames"""
        # Load timestamps
        with open(timestamps_path, 'r') as f:
            timestamps = json.load(f)

        # Get sorted list of frames
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith('rgb_0023')])
        
        for rgb_file in rgb_files:
            frame_num = rgb_file.split('_')[1].split('.')[0]
            depth_file = f"depth_{frame_num}.png"
            depth_synthetic_file = f"rgb_{frame_num}.png"
            
            rgb_path = os.path.join(rgb_dir, rgb_file)
            depth_path = os.path.join(depth_dir, depth_file)
            depth_synthetic_path = os.path.join(depth_synthetic_dir, depth_synthetic_file)
            
            result = self.process_frame(rgb_path, depth_path, depth_synthetic_path)
            if result:
                detections, rgb_vis, depth_vis, depth_synthetic_vis = result
                if detections:
                    self.update_trajectories(detections, frame_num)
                    # self.visualize_frame(rgb_vis, depth_vis, depth_synthetic_vis, frame_num)

    def save_trajectories(self, output_path):
        """Save all trajectories to numpy files"""
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save RGB trajectories (convert pixels to meters)
        rgb_data = {
            'positions': [],
            'frame_numbers': [],
            'trajectory_ids': []
        }
        for traj_id, trajectory in self.trajectories_rgb.items():
            positions = np.array(trajectory['positions'])
            # Convert pixel coordinates to meters using camera intrinsics
            X = (positions[:, 0] - self.rgb_cx) * 1.0 / self.rgb_fx  # Assuming 1 meter depth for RGB
            Y = (positions[:, 1] - self.rgb_cy) * 1.0 / self.rgb_fy
            positions_meters = np.column_stack((X, Y))
            
            frame_numbers = np.array(trajectory['frame_numbers'])
            trajectory_ids = np.full(len(positions), traj_id)
            
            rgb_data['positions'].append(positions_meters)
            rgb_data['frame_numbers'].append(frame_numbers)
            rgb_data['trajectory_ids'].append(trajectory_ids)
        
        # Convert lists to arrays
        rgb_data['positions'] = np.vstack(rgb_data['positions'])
        rgb_data['frame_numbers'] = np.concatenate(rgb_data['frame_numbers'])
        rgb_data['trajectory_ids'] = np.concatenate(rgb_data['trajectory_ids'])
        
        # Save RGB data
        np.save(os.path.join(output_path, 'rgb_positions_meters.npy'), rgb_data['positions'])
        np.save(os.path.join(output_path, 'rgb_frame_numbers.npy'), rgb_data['frame_numbers'])
        np.save(os.path.join(output_path, 'rgb_trajectory_ids.npy'), rgb_data['trajectory_ids'])
        
        # Save real depth trajectories (already in meters)
        real_data = {
            'positions': [],
            'world_coords': [],
            'distances': [],
            'frame_numbers': [],
            'trajectory_ids': []
        }
        for traj_id, trajectory in self.trajectories_real.items():
            # Convert pixel positions to meters
            positions = np.array(trajectory['positions'])
            X = (positions[:, 0] - self.rgb_cx) * 1.0 / self.rgb_fx
            Y = (positions[:, 1] - self.rgb_cy) * 1.0 / self.rgb_fy
            positions_meters = np.column_stack((X, Y))
            
            world_coords = np.array(trajectory['world_coords'])
            distances = np.array(trajectory['distances'])
            frame_numbers = np.array(trajectory['frame_numbers'])
            trajectory_ids = np.full(len(positions), traj_id)
            
            real_data['positions'].append(positions_meters)
            real_data['world_coords'].append(world_coords)
            real_data['distances'].append(distances)
            real_data['frame_numbers'].append(frame_numbers)
            real_data['trajectory_ids'].append(trajectory_ids)
        
        # Convert lists to arrays
        real_data['positions'] = np.vstack(real_data['positions'])
        real_data['world_coords'] = np.vstack(real_data['world_coords'])
        real_data['distances'] = np.concatenate(real_data['distances'])
        real_data['frame_numbers'] = np.concatenate(real_data['frame_numbers'])
        real_data['trajectory_ids'] = np.concatenate(real_data['trajectory_ids'])
        
        # Save real depth data
        np.save(os.path.join(output_path, 'real_positions_meters.npy'), real_data['positions'])
        np.save(os.path.join(output_path, 'real_world_coords.npy'), real_data['world_coords'])
        np.save(os.path.join(output_path, 'real_distances.npy'), real_data['distances'])
        np.save(os.path.join(output_path, 'real_frame_numbers.npy'), real_data['frame_numbers'])
        np.save(os.path.join(output_path, 'real_trajectory_ids.npy'), real_data['trajectory_ids'])
        
        # Save synthetic depth trajectories (already in meters)
        synth_data = {
            'positions': [],
            'world_coords': [],
            'distances': [],
            'frame_numbers': [],
            'trajectory_ids': []
        }
        for traj_id, trajectory in self.trajectories_synthetic.items():
            # Convert pixel positions to meters
            positions = np.array(trajectory['positions'])
            X = (positions[:, 0] - self.rgb_cx) * 1.0 / self.rgb_fx
            Y = (positions[:, 1] - self.rgb_cy) * 1.0 / self.rgb_fy
            positions_meters = np.column_stack((X, Y))
            
            world_coords = np.array(trajectory['world_coords'])
            distances = np.array(trajectory['distances'])
            frame_numbers = np.array(trajectory['frame_numbers'])
            trajectory_ids = np.full(len(positions), traj_id)
            
            synth_data['positions'].append(positions_meters)
            synth_data['world_coords'].append(world_coords)
            synth_data['distances'].append(distances)
            synth_data['frame_numbers'].append(frame_numbers)
            synth_data['trajectory_ids'].append(trajectory_ids)
        
        # Convert lists to arrays
        synth_data['positions'] = np.vstack(synth_data['positions'])
        synth_data['world_coords'] = np.vstack(synth_data['world_coords'])
        synth_data['distances'] = np.concatenate(synth_data['distances'])
        synth_data['frame_numbers'] = np.concatenate(synth_data['frame_numbers'])
        synth_data['trajectory_ids'] = np.concatenate(synth_data['trajectory_ids'])
        
        # Save synthetic depth data
        np.save(os.path.join(output_path, 'synthetic_positions_meters.npy'), synth_data['positions'])
        np.save(os.path.join(output_path, 'synthetic_world_coords.npy'), synth_data['world_coords'])
        np.save(os.path.join(output_path, 'synthetic_distances.npy'), synth_data['distances'])
        np.save(os.path.join(output_path, 'synthetic_frame_numbers.npy'), synth_data['frame_numbers'])
        np.save(os.path.join(output_path, 'synthetic_trajectory_ids.npy'), synth_data['trajectory_ids'])
        
        print(f"Trajectories saved to {output_path}")

    def update_trajectories(self, detections, frame_num):
        """Update trajectories with new detections"""
        for detection in detections:
            # Update RGB trajectories (2D)
            matched = False
            for traj_id, trajectory in self.trajectories_rgb.items():
                last_pos = trajectory['positions'][-1]
                if self.is_same_person(last_pos, detection['position']):
                    trajectory['positions'].append(detection['position'])
                    trajectory.setdefault('frame_numbers', []).append(frame_num)
                    matched = True
                    break
            
            if not matched:
                new_id = len(self.trajectories_rgb)
                self.trajectories_rgb[new_id] = {
                    'positions': [detection['position']],
                    'frame_numbers': [frame_num]
                }
            
            # Update real depth trajectories
            if detection['distance_real'] is not None:
                matched = False
                for traj_id, trajectory in self.trajectories_real.items():
                    last_pos = trajectory['positions'][-1]
                    if self.is_same_person(last_pos, detection['position']):
                        trajectory['positions'].append(detection['position'])
                        trajectory['world_coords'].append(detection['world_coords_real'])
                        trajectory['distances'].append(detection['distance_real'])
                        trajectory.setdefault('frame_numbers', []).append(frame_num)
                        matched = True
                        break
                
                if not matched:
                    new_id = len(self.trajectories_real)
                    self.trajectories_real[new_id] = {
                        'positions': [detection['position']],
                        'world_coords': [detection['world_coords_real']],
                        'distances': [detection['distance_real']],
                        'frame_numbers': [frame_num]
                    }
            
            # Update synthetic depth trajectories
            if detection['distance_synthetic'] is not None:
                matched = False
                for traj_id, trajectory in self.trajectories_synthetic.items():
                    last_pos = trajectory['positions'][-1]
                    if self.is_same_person(last_pos, detection['position']):
                        trajectory['positions'].append(detection['position'])
                        trajectory['world_coords'].append(detection['world_coords_synthetic'])
                        trajectory['distances'].append(detection['distance_synthetic'])
                        trajectory.setdefault('frame_numbers', []).append(frame_num)
                        matched = True
                        break
                
                if not matched:
                    new_id = len(self.trajectories_synthetic)
                    self.trajectories_synthetic[new_id] = {
                        'positions': [detection['position']],
                        'world_coords': [detection['world_coords_synthetic']],
                        'distances': [detection['distance_synthetic']],
                        'frame_numbers': [frame_num]
                    }

    def is_same_person(self, pos1, pos2, threshold=50):
        """Check if two positions likely belong to the same person"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < threshold

    def visualize_frame(self, rgb_img, depth_img, depth_synthetic_img, frame_num):
        """Visualize the current frame with detections"""
        # Get image dimensions
        height, width = rgb_img.shape[:2]
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot RGB image
        self.ax1.imshow(rgb_img)
        self.ax1.set_title(f'RGB Frame {frame_num}')
        self.ax1.axis('off')
        
        # Plot depth images
        self.ax2.imshow(depth_img)
        self.ax2.set_title(f'Real Depth Frame {frame_num}')
        self.ax2.axis('off')
        
        # Plot trajectories in image space
        for traj_id, trajectory in self.trajectories_real.items():
            positions = np.array(trajectory['positions'])
            self.ax3.plot(positions[:, 0]/100, positions[:, 1]/100, label=f'Person {traj_id}')
        self.ax3.set_title('Real Depth World Trajectories')
        self.ax3.set_xlabel('X (meters)')
        self.ax3.set_ylabel('Y (meters)')
        self.ax3.legend()
        self.ax3.grid(True)
        
        # Plot synthetic depth
        self.ax4.imshow(depth_synthetic_img)
        self.ax4.set_title(f'Synthetic Depth Frame {frame_num}')
        self.ax4.axis('off')
        
        self.fig.tight_layout()
        # plt.pause(0.2)  # 0.2 second delay between frames

    def plot_trajectories(self):
        """Plot the final trajectories"""
        # Get image dimensions from the last processed frame
        if hasattr(self, 'last_rgb_img'):
            height, width = self.last_rgb_img.shape[:2]
        else:
            width, height = 640, 480  # Default D435 resolution
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot in image coordinates
        for traj_id, trajectory in self.trajectories_real.items():
            positions = np.array(trajectory['positions'])
            self.ax3.plot(positions[:, 0]/100, positions[:, 1]/100, label=f'Person {traj_id}')
        self.ax3.set_title('Real Depth World Trajectories')
        self.ax3.set_xlabel('X (meters)')
        self.ax3.set_ylabel('Y (meters)')
        self.ax3.legend()
        self.ax3.grid(True)

        # Plot in world coordinates
        for traj_id, trajectory in self.trajectories_synthetic.items():
            world_coords = np.array(trajectory['world_coords'])
            self.ax4.plot(world_coords[:, 0]/100, world_coords[:, 1]/100, label=f'Person {traj_id}')
        self.ax4.set_title('Synthetic Depth World Trajectories')
        self.ax4.set_xlabel('X (meters)')
        self.ax4.set_ylabel('Y (meters)')
        self.ax4.legend()
        self.ax4.grid(True)
        
        self.fig.tight_layout()
        plt.show()  # Show the plot instead of closing immediately

def main():
    # Base path
    base_path = "/media/hdd_4/PhD/T4/embedded system/fp/experiment1"
    
    # Initialize tracker
    tracker = PedestrianTracker('camera_params.yaml')
    
    # Process sequence
    tracker.process_sequence(
        os.path.join(base_path, 'rgb'),
        os.path.join(base_path, 'depth'),
        os.path.join(base_path, 'depth_synthetic'),
        os.path.join(base_path, 'frame_timestamps.json')
    )
    
    # Save trajectories
    tracker.save_trajectories('trajectories')
    
    # Plot results
    # tracker.plot_trajectories()

if __name__ == "__main__":
    main()
