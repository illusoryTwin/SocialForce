# import pandas as pd
# import cv2
# import numpy as np
# import os
# from collections import defaultdict

# # Load data
# df_proj = pd.read_csv("projected_points.csv")
# df_impact = pd.read_csv("impact_factors.csv")

# # Camera intrinsics (must match pedestrian_detection.py)
# fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0  # Example values

# # Convert 3D to 2D image coordinates
# def project_to_image(x, y, z):
#     u = int(x * fx / z + cx)
#     v = int(y * fy / z + cy)
#     return u, v

# # Group impact data by frame
# impact_by_frame = defaultdict(list)
# for _, row in df_impact.iterrows():
#     impact_by_frame[int(row.frame)].append(row)

# # Image directory
# image_dir = "../imgs_data/rgb"
# output_dir = "impact_visualization"
# os.makedirs(output_dir, exist_ok=True)

# # Get image paths
# image_files = sorted(os.listdir(image_dir))
# image_files = [f for f in image_files if f.endswith(('.png', '.jpg', '.jpeg'))]

# for frame_index, filename in enumerate(image_files):
#     img_path = os.path.join(image_dir, filename)
#     frame = cv2.imread(img_path)
#     if frame is None:
#         continue

#     impacts = impact_by_frame.get(frame_index, [])
#     for row in impacts:
#         u1, v1 = project_to_image(row.x1, row.y1, row.z1)
#         u2, v2 = project_to_image(row.x2, row.y2, row.z2)

#         # Draw arrow from person 1 to person 2
#         cv2.arrowedLine(frame, (u1, v1), (u2, v2), (0, 0, 255), 2, tipLength=0.3)

#         # Draw impact factor value near the midpoint
#         mx, my = (u1 + u2) // 2, (v1 + v2) // 2
#         cv2.putText(
#             frame,
#             f"{row.impact_factor:.2f}",
#             (mx, my),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 255, 255),
#             1,
#             cv2.LINE_AA
#         )

#     # Save or display result
#     out_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
#     cv2.imwrite(out_path, frame)
#     cv2.imshow("Impact Visualization", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()


import pandas as pd
import cv2
import numpy as np
import os
from collections import defaultdict

# Load data
df_proj = pd.read_csv("projected_points.csv")
df_impact = pd.read_csv("impact_factors.csv")

# Camera intrinsics (adjust if needed)
fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0

# Convert 3D to 2D image coordinates
def project_to_image(x, y, z):
    if z <= 0: z = 0.001  # Prevent division by zero
    u = int(x * fx / z + cx)
    v = int(y * fy / z + cy)
    return u, v

# Group impacts by frame
impact_by_frame = defaultdict(list)
for _, row in df_impact.iterrows():
    impact_by_frame[int(row.frame)].append(row)

# Paths
image_dir = "../imgs_data/rgb"
output_dir = "impact_visualization"
os.makedirs(output_dir, exist_ok=True)

# Get sorted image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

for frame_index, filename in enumerate(image_files):
    img_path = os.path.join(image_dir, filename)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Could not load {img_path}")
        continue

    impacts = impact_by_frame.get(frame_index, [])
    for row in impacts:
        # Project both 3D points to 2D
        u1, v1 = project_to_image(row.x1, row.y1, row.z1)
        u2, v2 = project_to_image(row.x2, row.y2, row.z2)

        # Draw arrow from person 1 to person 2
        cv2.arrowedLine(frame, (u1, v1), (u2, v2), (0, 0, 255), 2, tipLength=0.3)

        # Compute mid-point for text placement
        mx, my = (u1 + u2) // 2, (v1 + v2) // 2

        # Draw impact value
        text = f"{row.impact_factor:.2f}"
        cv2.putText(
            frame,
            text,
            (mx + 5, my - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Save image
    out_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
    cv2.imwrite(out_path, frame)

    # Show preview (optional)
    cv2.imshow("Impact Visualization", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
