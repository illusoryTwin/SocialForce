import cv2
import numpy as np
import json
import os

class TableCalibrator:
    def __init__(self):
        self.points = []
        self.image = None
        self.window_name = "Select Table Points"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                # Draw point on image
                cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(self.window_name, self.image)
                print(f"Point {len(self.points)} selected at ({x}, {y})")
                
                if len(self.points) == 2:
                    # Draw line between points
                    cv2.line(self.image, self.points[0], self.points[1], (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.image)
                    print("Both points selected. Press 's' to save or 'r' to reset.")

    def calibrate(self, image_path):
        # Read image
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Show image
        cv2.imshow(self.window_name, self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save points
                if len(self.points) == 2:
                    # Save points to JSON
                    data = {
                        'point1': self.points[0],
                        'point2': self.points[1],
                        'image_shape': self.image.shape[:2]
                    }
                    
                    # Create calibration directory if it doesn't exist
                    os.makedirs('calibration', exist_ok=True)
                    
                    # Save to file
                    with open('calibration/table_points.json', 'w') as f:
                        json.dump(data, f, indent=4)
                    print("Points saved to calibration/table_points.json")
                    break
                    
            elif key == ord('r'):  # Reset points
                self.points = []
                self.image = cv2.imread(image_path)
                cv2.imshow(self.window_name, self.image)
                print("Points reset")
                
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize calibrator
    calibrator = TableCalibrator()
    
    # Path to your image
    image_path = "/media/hdd_4/PhD/T4/embedded system/fp/experiment1/rgb/rgb_002300.png"  # Change this to your image path
    
    # Run calibration
    calibrator.calibrate(image_path) 