# import cv2
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# import ultralytics
# import cv2
# import os
# import numpy as np
# from glob import glob
# from ultralytics import YOLO

# def  get_centroid(box):
#     """
#     Calculate the centroid of a bounding box.
#     :param box: Bounding box in the format [x1, y1, x2, y2].
#     :return: Centroid coordinates (x, y).
#     """
#     x1, y1, x2, y2 = box
#     return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    

# base_path = "../experiment1/"
# rgb_folder = "rgb/"
# depth_folder = "depth/"


# rgb_images = sorted(glob(os.path.join(base_path, rgb_folder, "rgb_*.png")))
# depth_images = sorted(glob(os.path.join(base_path, depth_folder, "depth_*.png")))
# # if len(rgb_images) != len(depth_images):
# #     raise ValueError("Number of RGB and depth images do not match.")


# model = YOLO("yolov8n.pt")  # Load the YOLOv8 model

# for rgb_image_path, depth_image_path in zip(rgb_images, depth_images):
#     # Read RGB and depth images
#     rgb_image = cv2.imread(rgb_image_path)
#     depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

#     # Perform object detection on the RGB image
#     results = model(rgb_image)[0]

#     boxes = results.boxes.xyxy.cpu().numpy()
#     class_ids = results.boxes.cls.cpu().numpy()
    
#     people_boxes = [box for box, cls_id in zip(boxes, class_ids) if int(cls_id) == 0]  # class 0 is person
    
#     current_positions = [get_centroid(box) for box in people_boxes]
#     print("Current positions:", current_positions)

#     # Draw bounding boxes on the RGB image
#     annotated_rgb = results.plot()

#     # Display the annotated RGB image
#     cv2.imshow("Annotated RGB Image", annotated_rgb)
    
#     # Display the depth image
#     cv2.imshow("Depth Image", depth_image)

#     # Wait for a key press and break if 'q' is pressed
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break


# ==================================================

# import cv2
# import os
# import numpy as np
# from glob import glob
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort


# def get_centroid(box):
#     """Calculate the centroid of a bounding box [x1, y1, x2, y2]."""
#     x1, y1, x2, y2 = box
#     return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# # === Paths ===
# base_path = "../experiment1/"
# rgb_folder = "rgb/"
# depth_folder = "depth/"

# rgb_images = sorted(glob(os.path.join(base_path, rgb_folder, "rgb_*.png")))
# depth_images = sorted(glob(os.path.join(base_path, depth_folder, "depth_*.png")))

# # === Initialize Detector and Tracker ===
# model = YOLO("yolov8n.pt")
# tracker = DeepSort(max_age=30)

# # === Store trajectories by ID ===
# trajectories = {}  # {id: [(x, y), (x, y), ...]}

# for rgb_image_path, depth_image_path in zip(rgb_images[:20], depth_images[:20]):
#     # Read RGB and depth images
#     rgb_image = cv2.imread(rgb_image_path)
#     depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

#     # Run YOLO detection
#     results = model(rgb_image)[0]
#     boxes = results.boxes.xyxy.cpu().numpy()
#     scores = results.boxes.conf.cpu().numpy()
#     class_ids = results.boxes.cls.cpu().numpy()

#     # Prepare detections for Deep SORT
#     detections = []
#     for box, score, cls_id in zip(boxes, scores, class_ids):
#         if int(cls_id) == 0:  # Only track people
#             x1, y1, x2, y2 = box
#             w, h = x2 - x1, y2 - y1
#             detections.append(([x1, y1, w, h], score, "person"))

#     # Update tracker with current frame detections
#     tracks = tracker.update_tracks(detections, frame=rgb_image)

#     # Draw results and collect centroids
#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         track_id = track.track_id
#         ltrb = track.to_ltrb()
#         x1, y1, x2, y2 = map(int, ltrb)

#         # Get centroid
#         centroid = get_centroid([x1, y1, x2, y2])

#         # Add to trajectory
#         if track_id not in trajectories:
#             trajectories[track_id] = []
#         trajectories[track_id].append(centroid)

#         # Draw box and ID
#         cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.circle(rgb_image, centroid, 4, (255, 0, 0), -1)
#         cv2.putText(rgb_image, f"ID {track_id}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Optionally display depth
#     cv2.imshow("Annotated RGB Image", rgb_image)
#     cv2.imshow("Depth Image", depth_image)

#     key = cv2.waitKey(100)  # or 0 for manual, 100ms for auto-play
#     if key & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

# # === (Optional) Save or print trajectories ===
# for track_id, points in trajectories.items():
#     print(f"Trajectory for ID {track_id}: {points}")


# =========================================

import cv2
import os
import numpy as np
from glob import glob
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def get_centroid(box):
    """Calculate the centroid of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# === Paths ===
base_path = "../experiment1/"
rgb_folder = "rgb/"
depth_folder = "depth/"

rgb_images = sorted(glob(os.path.join(base_path, rgb_folder, "rgb_*.png")))
depth_images = sorted(glob(os.path.join(base_path, depth_folder, "depth_*.png")))

# === Initialize Detector and Tracker ===
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# === Store trajectories by ID ===
trajectories = {}  # {id: [(x, y), (x, y), ...]}

for rgb_image_path, depth_image_path in zip(rgb_images[:40], depth_images[:40]):
    # Read RGB and depth images
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # Run YOLO detection
    results = model(rgb_image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()

    # Prepare detections for Deep SORT
    detections = []
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if int(cls_id) == 0:  # Only track people
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], score, "person"))

    # Update tracker with current frame detections
    tracks = tracker.update_tracks(detections, frame=rgb_image)

    # Draw results and collect centroids
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Get centroid
        centroid = get_centroid([x1, y1, x2, y2])

        # Add to trajectory
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append(centroid)

        # Draw trajectory
        points = trajectories[track_id]
        for i in range(1, len(points)):
            cv2.line(rgb_image, points[i - 1], points[i], (0, 0, 255), 2)

        # Draw box and ID
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(rgb_image, centroid, 4, (255, 0, 0), -1)
        cv2.putText(rgb_image, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Optionally display depth
    cv2.imshow("Annotated RGB Image", rgb_image)
    cv2.imshow("Depth Image", depth_image)

    key = cv2.waitKey(100)  # or 0 for manual, 100ms for auto-play
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# # === (Optional) Save or print trajectories ===
# for track_id, points in trajectories.items():
#     print(f"Trajectory for ID {track_id}: {points}")
