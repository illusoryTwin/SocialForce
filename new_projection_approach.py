# import cv2 
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from glob import glob 
# import os 
# import csv 


# def get_centroid(box):
#     """Calculate centroid of a bounding box and return (x, y)."""
#     x1, y1, x2, y2 = box
#     return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# def pixel_to_world(u, v, depth, fx, fy, cx, cy):
#     """Project pixel (u,v) with depth to 3D world coordinates."""
#     Z = depth / 1000.0  # Convert mm to meters if necessary
#     # if Z == 0:
#     #     return None  # Ignore invalid depth
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy
#     return (X, Y, Z)


# # === Camera intrinsics (D435) ===
# fx = 525.0
# fy = 525.0
# cx = 319.5
# cy = 239.5

# EXPERIMENT_NAME = "experiment1"

# # === Define paths to images ===
# base_path = "../imgs_data/"
# rgb_folder = "rgb/"
# depth_folder = "depth/"

# rgb_images = sorted(glob(os.path.join(base_path, rgb_folder, "rgb_*.png")))
# depth_images = sorted(glob(os.path.join(base_path, depth_folder, "depth_*.png")))

# # === Define models ===
# model = YOLO("yolov8n.pt")
# tracker = DeepSort(max_age=30)

# # === Prepare CSV output ===
# os.makedirs(f"../data/{EXPERIMENT_NAME}", exist_ok=True)
# csv_output_path = f"../data/{EXPERIMENT_NAME}/trajectories.csv"
# csv_file = open(csv_output_path, mode='w', newline='')
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["track_id", "X", "Y", "Z"])  # CSV header

# trajectories = {}  # {track_id: [(X, Y, Z), ...]}

# for rgb_image_path, depth_image_path in zip(rgb_images, depth_images):
#     rgb_image = cv2.imread(rgb_image_path)
#     depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
#     results = model(rgb_image)[0]
#     boxes = results.boxes.xyxy.cpu().numpy()
#     scores = results.boxes.conf.cpu().numpy()
#     class_ids = results.boxes.cls.cpu().numpy()

#     detected_people = []
#     for box, score, cls_id in zip(boxes, scores, class_ids):
#         if int(cls_id) == 0:  # Only track people
#             x1, y1, x2, y2 = box
#             w, h = x2 - x1, y2 - y1
#             detected_people.append(([x1, y1, w, h], score, "person"))

#     # Update tracker
#     tracks = tracker.update_tracks(detected_people, frame=rgb_image)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         ltrb = track.to_ltrb()
#         x1, y1, x2, y2 = map(int, ltrb)
#         centroid = get_centroid([x1, y1, x2, y2])
#         u, v = centroid

#         # Guard against out-of-bounds access
#         if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
#             continue

#         depth = depth_image[v, u]

#         world_coords = pixel_to_world(u, v, depth, fx, fy, cx, cy)
#         print("world_coords", world_coords)
#         if world_coords is None:
#             continue  # Skip if depth was invalid

#         if track_id not in trajectories:
#             trajectories[track_id] = []
#         trajectories[track_id].append(world_coords)

#         # Save to CSV
#         csv_writer.writerow([track_id, *world_coords])

#         # Draw trajectory (in image space)
#         image_points = [get_centroid([x1, y1, x2, y2]) for (X, Y, Z) in trajectories[track_id]]
#         for i in range(1, len(image_points)):
#             cv2.line(rgb_image, image_points[i - 1], image_points[i], (0, 0, 255), 2)

#         cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # cv2.circle(rgb_image, centroid, 5, (0, 0, 255), -1)

#         # Draw the centroid in red
#         print("centroid", centroid)
#         print("world_coords", world_coords)
#         cv2.circle(rgb_image, centroid, 5, (0, 0, 255), -1)
#         cv2.putText(rgb_image, f"ID {track_id}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     # Show frames
#     cv2.imshow("Tracking", rgb_image)
#     cv2.imshow("Depth Image", depth_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # === Save all trajectories to text ===
# print("trajectories", trajectories)
# with open(f"../data/{EXPERIMENT_NAME}/trajectories.txt", "w") as f:
#     for track_id, points in trajectories.items():
#         f.write(f"ID {track_id}:\n")
#         for p in points:
#             f.write(f"\t{p}\n")

# print("Trajectories saved to trajectories.txt")
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from glob import glob
import os
import csv
import yaml
import numpy as np


def get_centroid(box):
    """Calculate centroid of a bounding box and return (x, y)."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def pixel_to_world(u, v, depth, fx, fy, cx, cy):
    """Project pixel (u,v) with depth to 3D world coordinates using camera intrinsics."""
    # If depth is a numpy array or list with shape, extract scalar value
    if hasattr(depth, "shape") and depth.shape != ():
        depth = depth.item()  # extract scalar from numpy array

    if isinstance(depth, (list, tuple)):
        depth = depth[0]  # fallback if list or tuple

    if depth == 0:
        return None  # Ignore invalid depth

    Z = depth / 1000.0  # Convert depth from mm to meters
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return (X, Y, Z)


def project_point_to_plane(point_3d, plane_params):
    """Project a 3D point onto a plane defined by Ax + By + Cz + D = 0."""
    if point_3d is None:
        return None
    
    x, y, z = point_3d
    A, B, C, D = plane_params['A'], plane_params['B'], plane_params['C'], plane_params['D']
    
    # Calculate the distance from point to plane
    distance = (A * x + B * y + C * z + D) / (A**2 + B**2 + C**2)**0.5
    
    # Project point onto plane by moving along the normal vector
    projected_x = x - A * distance
    projected_y = y - B * distance
    projected_z = z - C * distance
    
    return (projected_x, projected_y, projected_z)


def load_config():
    """Load camera parameters and plane equation from YAML files."""
    with open('camera_params.yaml', 'r') as f:
        camera_params = yaml.safe_load(f)
    
    with open('floor_plane_equation.yaml', 'r') as f:
        plane_data = yaml.safe_load(f)
    
    return camera_params, plane_data['plane_equation']



# === Load configuration ===
camera_params, plane_equation = load_config()

# Use D435 depth camera parameters
fx = camera_params['d435_depth']['fx']
fy = camera_params['d435_depth']['fy']
cx = camera_params['d435_depth']['cx']
cy = camera_params['d435_depth']['cy']

EXPERIMENT_NAME = "experiment1"

# === Paths ===
base_path = "../imgs_data/"
rgb_folder = "rgb/"
depth_folder = "depth/"

rgb_images = sorted(glob(os.path.join(base_path, rgb_folder, "rgb_*.png")))
depth_images = sorted(glob(os.path.join(base_path, depth_folder, "depth_*.png")))

# === Load models ===
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# === Setup output ===
os.makedirs(f"../data/{EXPERIMENT_NAME}", exist_ok=True)
csv_output_path = f"../data/{EXPERIMENT_NAME}/trajectories.csv"

with open(csv_output_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["track_id", "X", "Y", "Z", "projected_X", "projected_Y", "projected_Z"])  # CSV header

    trajectories = {}  # {track_id: [(X, Y, Z), ...]}

    for rgb_path, depth_path in zip(rgb_images, depth_images):
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        results = model(rgb)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        # Only detect people (class 0)
        detections = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            if int(cls_id) == 0:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], score, "person"))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=rgb)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            centroid = get_centroid([x1, y1, x2, y2])
            u, v = centroid

            # Ensure centroid is inside image bounds
            if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
                continue

            depth_value = depth[v, u]
            world_coords = pixel_to_world(u, v, depth_value, fx, fy, cx, cy)
            if world_coords is None:
                continue

            # Project 3D point onto ground plane
            projected_coords = project_point_to_plane(world_coords, plane_equation)
            if projected_coords is None:
                continue

            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((world_coords, projected_coords))

            # Save both world and projected coordinates to CSV
            csv_writer.writerow([track_id, *world_coords, *projected_coords])

            # Store centroids for trajectory visualization
            if not hasattr(track, 'centroid_history'):
                track.centroid_history = []
            track.centroid_history.append(centroid)
            
            # Draw trajectory line between consecutive centroids
            if len(track.centroid_history) > 1:
                for i in range(1, len(track.centroid_history)):
                    cv2.line(rgb, track.centroid_history[i-1], track.centroid_history[i], (0, 0, 255), 2)

            # Draw bounding box and ID
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(rgb, centroid, 5, (0, 0, 255), -1)
            cv2.putText(rgb, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display frames
        cv2.imshow("Tracking", rgb)
        cv2.imshow("Depth", depth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# === Save full trajectories to text file ===
txt_output_path = f"../data/{EXPERIMENT_NAME}/trajectories.txt"
with open(txt_output_path, "w") as f:
    for track_id, trajectory_data in trajectories.items():
        f.write(f"ID {track_id}:\n")
        for world_coords, projected_coords in trajectory_data:
            f.write(f"\tWorld: {world_coords}\n")
            f.write(f"\tProjected: {projected_coords}\n")

print("Trajectories saved to CSV and TXT.")
cv2.destroyAllWindows()
