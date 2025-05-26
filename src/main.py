import cv2 
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from glob import glob 
import os 
import csv 


def get_centroid(box):
    """Calculate centroid of a bound box and return 
    its coordinates in the format (x, y)."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


EXPERIMENT_NAME = "experiment1"

# === Define paths to images ===
base_path = "../../experiment1/"
rgb_folder = "rgb/"
depth_folder = "depth/"

rgb_images = sorted(glob(os.path.join(base_path, rgb_folder, "rgb_*.png")))
depth_images = sorted(glob(os.path.join(base_path, depth_folder, "depth_*.png")))



# === Define the models ===
# Define model for people detection
model = YOLO("yolov8n.pt")
# Define tracker for tracking detected people
tracker = DeepSort(max_age=30)



# Define output CSV path
csv_output_path = f"../data/{EXPERIMENT_NAME}/trajectories.csv"

# Prepare CSV file for writing
csv_file = open(csv_output_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["track_id", "x", "y"])  # CSV header


trajectories = {} # Store trajectories by ID {id: [(x, y), (x, y), ...]}

for rgb_image_path, depth_image_path in zip(rgb_images, depth_images):
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    results = model(rgb_image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()

    detected_people = []
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if int(cls_id) == 0:  # Only track people
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detected_people.append(([x1, y1, w, h], score, "person"))
    
    # Update tracker with current frame detections
    tracks = tracker.update_tracks(detected_people, frame=rgb_image)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        centroid = get_centroid([x1, y1, x2, y2])

        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append(centroid)

        # Save to CSV
        csv_writer.writerow([track_id, centroid[0], centroid[1]])

        # Draw trajectory
        points = trajectories[track_id]
        for i in range(1, len(points)):
            cv2.line(rgb_image, points[i - 1], points[i], (0, 0, 255), 2)


        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(rgb_image, centroid, 5, (0, 0, 255), -1)
        cv2.putText(rgb_image, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # Show the RGB and depth images
    cv2.imshow("Tracking", rgb_image)
    cv2.imshow("Depth Image", depth_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   


print("trajectories", trajectories)
with open(f"../data/{EXPERIMENT_NAME}/trajectories.txt", "w") as f:
    for track_id, points in trajectories.items():
        f.write(f"ID {track_id}: {points}\n")
print("Trajectories saved to trajectories.txt")

cv2.destroyAllWindows() 
