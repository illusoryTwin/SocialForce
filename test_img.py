import ultralytics
import cv2
import os
import numpy as np
from glob import glob
from ultralytics import YOLO

base_path = "../experiment1/"
rgb_folder = "rgb/"
depth_folder = "depth/"


rgb_images = sorted(glob(os.path.join(base_path, rgb_folder, "rgb_*.png")))
depth_images = sorted(glob(os.path.join(base_path, depth_folder, "depth_*.png")))

print("len(rgb_images)", len(rgb_images), "len(depth_images)", len(depth_images))

# if len(rgb_images) != len(depth_images):
#     raise ValueError("Number of RGB and depth images do not match.")

cv2.imshow("RGB Image", cv2.imread(rgb_images[0]))
cv2.imshow("Depth Image", cv2.imread(depth_images[0], cv2.IMREAD_UNCHANGED))
cv2.waitKey(0)
cv2.destroyAllWindows()
