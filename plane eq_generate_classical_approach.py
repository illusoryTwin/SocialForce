import cv2
import numpy as np
import yaml
import random
import os

# File paths
SEGMENTED_IMAGE_PATH = '/home/ant/projects/SocialForce/segmented_image.jpg'
DEPTH_MAP_PATH = '/home/ant/projects/SocialForce/depth_002126.png'
CAMERA_PARAMS_PATH = '/home/ant/projects/SocialForce/camera_params.yaml'
PLANE_EQUATION_OUTPUT_PATH = '/home/ant/projects/SocialForce/floor_plane_equation.yaml' # Output file for plane equation

# Dataset path for visualization
# Update this path to match your actual dataset location
DATASET_RGB_FOLDER = '/media/ant/52F6748DF67472D9/PhD/T4/embedded system/fp/experiment1/rgb'
VISUALIZATION_OUTPUT_FOLDER = '/home/ant/projects/SocialForce/plane_visualizations' # Output folder for visualized images

# Assume the mask color is bright pink (R=255, G=0, B=255).
# You might need to adjust these values or add a tolerance if the pink color
# in your image is slightly different.
MASK_COLOR_HSV = (89, 241, 255)  # HSV values for pink
COLOR_TOLERANCE_HSV = (90, 128, 128)  # Tolerance for H, S, V channels

# RANSAC Parameters
RANSAC_ITERATIONS = 5000  # Increased iterations for better sampling
RANSAC_THRESHOLD = 0.1  # Increased threshold to allow more points to be considered inliers
RANSAC_SAMPLE_SIZE = 20  # Reduced sample size for initial plane fitting
RANSAC_MIN_INLIERS = 500  # Reduced minimum inliers requirement

# Visualization Parameters
PLANE_GRID_SIZE = 0.5 # Size of the grid points on the plane in meters
PLANE_VIS_COLOR = (0, 255, 0) # Color to draw the plane (BGR - Green)
PLANE_POINT_SIZE = 2 # Size of the points drawn for visualization
MASK_OVERLAY_COLOR = (0, 255, 0)  # Color for mask overlay (BGR - Green)

# HSV Color Tuning
HSV_WINDOW_NAME = "HSV Color Tuning"
HSV_DEFAULT_VALUES = {
    'H': 89,
    'S': 241,
    'V': 255,
    'H_tolerance': 90,
    'S_tolerance': 128,
    'V_tolerance': 128
}

def nothing(x):
    pass

def create_hsv_trackbars():
    """Creates trackbars for HSV color tuning."""
    cv2.namedWindow(HSV_WINDOW_NAME)
    
    # Create trackbars for HSV values
    cv2.createTrackbar('H', HSV_WINDOW_NAME, HSV_DEFAULT_VALUES['H'], 180, nothing)
    cv2.createTrackbar('S', HSV_WINDOW_NAME, HSV_DEFAULT_VALUES['S'], 255, nothing)
    cv2.createTrackbar('V', HSV_WINDOW_NAME, HSV_DEFAULT_VALUES['V'], 255, nothing)
    
    # Create trackbars for tolerances
    cv2.createTrackbar('H_tolerance', HSV_WINDOW_NAME, HSV_DEFAULT_VALUES['H_tolerance'], 90, nothing)
    cv2.createTrackbar('S_tolerance', HSV_WINDOW_NAME, HSV_DEFAULT_VALUES['S_tolerance'], 128, nothing)
    cv2.createTrackbar('V_tolerance', HSV_WINDOW_NAME, HSV_DEFAULT_VALUES['V_tolerance'], 128, nothing)
    
    # Create save button
    cv2.createButton('Save', lambda x: None, HSV_WINDOW_NAME, cv2.QT_PUSH_BUTTON)

def get_hsv_values():
    """Gets current HSV values from trackbars."""
    h = cv2.getTrackbarPos('H', HSV_WINDOW_NAME)
    s = cv2.getTrackbarPos('S', HSV_WINDOW_NAME)
    v = cv2.getTrackbarPos('V', HSV_WINDOW_NAME)
    h_tol = cv2.getTrackbarPos('H_tolerance', HSV_WINDOW_NAME)
    s_tol = cv2.getTrackbarPos('S_tolerance', HSV_WINDOW_NAME)
    v_tol = cv2.getTrackbarPos('V_tolerance', HSV_WINDOW_NAME)
    
    return (h, s, v), (h_tol, s_tol, v_tol)

def update_mask_visualization(image, hsv_values, tolerance_values):
    """Updates the mask visualization with current HSV values."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([max(0, hsv_values[0] - tolerance_values[0]),
                           max(0, hsv_values[1] - tolerance_values[1]),
                           max(0, hsv_values[2] - tolerance_values[2])])
    upper_bound = np.array([min(180, hsv_values[0] + tolerance_values[0]),
                           min(255, hsv_values[1] + tolerance_values[1]),
                           min(255, hsv_values[2] + tolerance_values[2])])
    
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Create overlay
    overlay = image.copy()
    overlay[mask > 0] = MASK_OVERLAY_COLOR
    
    # Blend original and overlay
    alpha = 0.5
    mask_visualization = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return mask_visualization, mask

def load_camera_parameters(filepath):
    """Loads camera intrinsic parameters from a YAML file."""
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
    # Assuming you want the parameters for the 'd435_depth' camera
    # Adjust this key if needed based on your YAML structure
    camera_intrinsics = params.get('d435_depth')
    if not camera_intrinsics:
        raise KeyError("Could not find 'd435_depth' parameters in the camera_params.yaml file.")
    return (camera_intrinsics['fx'], camera_intrinsics['fy'],
            camera_intrinsics['cx'], camera_intrinsics['cy'])

def save_plane_equation_to_yaml(plane_coefficients, filepath):
    """Saves the plane equation coefficients to a YAML file."""
    data = {
        'plane_equation': {
            'A': float(plane_coefficients[0]),
            'B': float(plane_coefficients[1]),
            'C': float(plane_coefficients[2]),
            'D': float(plane_coefficients[3])
        }
    }
    with open(filepath, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Plane equation saved to {filepath}")

def get_masked_pixels_with_depth(segmented_img_path, depth_map_path, mask_color_hsv, tolerance_hsv):
    """
    Identifies masked pixels and retrieves their depth values.
    Collects all points from the pink area using HSV color space.

    Args:
        segmented_img_path: Path to the segmented image.
        depth_map_path: Path to the depth map.
        mask_color_hsv: The HSV tuple of the mask color.
        tolerance_hsv: The HSV tuple of tolerances for each channel.

    Returns:
        A list of tuples (u, v, depth) for masked pixels.
    """
    segmented_image = cv2.imread(segmented_img_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)  # Load as is for potential 16-bit depth

    if segmented_image is None:
        raise FileNotFoundError(f"Segmented image not found at {segmented_img_path}")
    if depth_map is None:
        raise FileNotFoundError(f"Depth map not found at {depth_map_path}")

    # Ensure depth map is single channel, taking the first channel if it's multi-channel
    if len(depth_map.shape) > 2:
        print("Warning: Depth map has multiple channels. Using the first channel for depth.")
        depth_map = depth_map[:, :, 0]

    # Convert image to HSV
    hsv_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)

    # Get mask based on HSV color
    lower_bound = np.array([max(0, mask_color_hsv[0] - tolerance_hsv[0]),
                           max(0, mask_color_hsv[1] - tolerance_hsv[1]),
                           max(0, mask_color_hsv[2] - tolerance_hsv[2])])
    upper_bound = np.array([min(180, mask_color_hsv[0] + tolerance_hsv[0]),
                           min(255, mask_color_hsv[1] + tolerance_hsv[1]),
                           min(255, mask_color_hsv[2] + tolerance_hsv[2])])

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Get coordinates of all masked pixels
    masked_pixels = np.where(mask > 0)

    # Collect all valid points with their depths
    all_points = []
    for i in range(len(masked_pixels[0])):
        v = masked_pixels[0][i]  # row (y)
        u = masked_pixels[1][i]  # column (x)

        # Ensure pixel coordinates are within depth map bounds
        if v < depth_map.shape[0] and u < depth_map.shape[1]:
            depth = depth_map[v, u]
            if depth > 0:  # Assuming 0 or NaN represents invalid depth
                all_points.append((u, v, depth))

    print(f"Found {len(all_points)} points with valid depth")
    return [(int(u), int(v), float(d)) for u, v, d in all_points]

def convert_uvd_to_xyz(uvd_points, fx, fy, cx, cy):
    """
    Converts 2D pixel coordinates with depth to 3D points.

    Args:
        uvd_points: List of tuples (u, v, depth).
        fx, fy, cx, cy: Camera intrinsic parameters.

    Returns:
        A NumPy array of shape (N, 3) representing 3D points (x, y, z).
    """
    xyz_points = []
    for u, v, depth in uvd_points:
        # Assuming depth is in millimeters based on typical depth sensor output (like D435)
        # If your depth map is in meters, remove the /1000.0 scaling
        Z = depth / 1000.0  # Convert depth from millimeters to meters
        # Avoid division by zero for points with Z=0 (should be handled by depth>0 check, but as a safeguard)
        if Z > 1e-6:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            xyz_points.append([X, Y, Z])
        # else:
            # print(f"Warning: Skipping point ({u}, {v}) with depth Z <= 0.")


    return np.array(xyz_points)

def fit_plane_ransac(points_3d, iterations, threshold, sample_size, min_inliers):
    """
    Fits a plane (Ax + By + Cz + D = 0) to a set of 3D points using RANSAC.

    Args:
        points_3d: A NumPy array of shape (N, 3) representing 3D points.
        iterations: Number of RANSAC iterations.
        threshold: Distance threshold for a point to be considered an inlier.
        sample_size: Minimum number of points to define a plane (usually 3).
        min_inliers: Minimum number of inliers required to accept a model.

    Returns:
        A tuple (A, B, C, D) representing the plane equation coefficients of the best model.
        Returns None if no valid model is found.
    """
    best_plane = None
    best_inlier_count = 0
    num_points = points_3d.shape[0]

    if num_points < sample_size:
        print(f"Not enough points ({num_points}) for RANSAC sample.")
        return None

    # Add a check for minimum total points for RANSAC to be meaningful
    if num_points < min_inliers:
         print(f"Not enough points ({num_points}) total. Need at least {min_inliers} for meaningful RANSAC.")
         return None


    for i in range(iterations):
        # Randomly select `sample_size` points
        try:
            sample_indices = random.sample(range(num_points), sample_size)
            sample_points = points_3d[sample_indices, :]
        except ValueError:
            # Happens if num_points < sample_size, though checked above, defensive programming
            print(f"Could not sample {sample_size} points from {num_points} points. Skipping iteration.")
            continue


        # Calculate plane from sample points
        # Use SVD for more robust plane fitting even on 3 points
        # Centroid of sample points
        centroid = np.mean(sample_points, axis=0)

        # Center the sample points
        centered_sample_points = sample_points - centroid

        # Use SVD to find the normal vector
        # The normal vector is the left singular vector corresponding to the smallest singular value
        try:
            U, S, Vt = np.linalg.svd(centered_sample_points)
            normal_vector = Vt[-1, :]
        except np.linalg.LinAlgError:
            # Handle cases where SVD might not converge (e.g., collinear points)
            # print("Warning: SVD failed on sample points. Skipping iteration.")
            continue # Skip if SVD fails


        # Check if normal vector is valid (not zero)
        norm = np.linalg.norm(normal_vector)
        if norm < 1e-6:
            # print("Warning: Sample points are collinear. Skipping iteration.")
            continue # Skip if points are collinear

        A, B, C = normal_vector
        D = -np.dot(normal_vector, centroid)


        # Calculate distances of all points to this plane
        # Distance of point (x0, y0, z0) to plane Ax + By + Cz + D = 0 is |Ax0 + By0 + Cz0 + D| / sqrt(A^2 + B^2 + C^2)
        # Ensure the divisor is not zero, though handled by normal_vector norm check
        distances = np.abs(points_3d[:, 0] * A + points_3d[:, 1] * B + points_3d[:, 2] * C + D) / norm


        # Count inliers
        inlier_indices = np.where(distances < threshold)[0]
        inlier_count = len(inlier_indices)

        # Update best model if current model has more inliers and meets minimum requirement
        if inlier_count > best_inlier_count and inlier_count >= min_inliers:
            best_inlier_count = inlier_count
            best_plane = (A, B, C, D)
            # print(f"Iteration {i}: Found new best model with {inlier_count} inliers.")


    # After RANSAC iterations, refit the plane using least squares on the inliers of the best model found
    if best_plane:
        A, B, C, D = best_plane
        # Recalculate inliers based on the best found plane
        norm = np.linalg.norm([A, B, C])
        if norm > 1e-6:
            distances = np.abs(points_3d[:, 0] * A + points_3d[:, 1] * B + points_3d[:, 2] * C + D) / norm
            best_model_inlier_indices = np.where(distances < threshold)[0]
            best_model_inliers = points_3d[best_model_inlier_indices, :]

            if len(best_model_inliers) >= 3:
                 print(f"Refitting plane using {len(best_model_inliers)} inliers.")
                 # Use least squares (SVD) on the identified inliers
                 return fit_plane_least_squares(best_model_inliers)
            else:
                 print(f"Best RANSAC model had fewer than 3 inliers ({len(best_model_inliers)}) for final refitting.")
                 # Return the RANSAC result if refitting not possible with enough inliers
                 return best_plane
        else:
             print("Best RANSAC model had a near-zero normal vector. Cannot refit.")
             return None


    print("RANSAC failed to find a valid plane model meeting the minimum inlier count.")
    return None # No valid plane found

def fit_plane_least_squares(points_3d):
    """
    Fits a plane (Ax + By + Cz + D = 0) to a set of 3D points using least squares (SVD).
    This is suitable for a set of points known to be mostly inliers.

    Args:
        points_3d: A NumPy array of shape (N, 3) representing 3D points.

    Returns:
        A tuple (A, B, C, D) representing the plane equation coefficients.
        Returns None if fitting fails (e.g., not enough points).
    """
    if points_3d.shape[0] < 3:
        # print("Not enough points for least squares fitting.")
        return None

    # Centroid of the points
    centroid = np.mean(points_3d, axis=0)

    # Center the points
    centered_points = points_3d - centroid

    # Use SVD to find the normal vector
    # The normal vector is the left singular vector corresponding to the smallest singular value
    # Using full_matrices=False can be slightly more memory efficient for tall matrices
    try:
        U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
        normal_vector = Vt[-1, :]
    except np.linalg.LinAlgError:
        # print("Warning: SVD failed during least squares fitting.")
        return None


    # Check if normal vector is valid (not zero)
    norm = np.linalg.norm(normal_vector)
    if norm < 1e-6:
        # print("Warning: Points are collinear during least squares fitting.")
        return None

    A, B, C = normal_vector
    D = -np.dot(normal_vector, centroid)

    return (A, B, C, D)

def project_3d_to_2d(points_3d, fx, fy, cx, cy):
    """
    Projects 3D points to 2D pixel coordinates using camera intrinsic parameters.

    Args:
        points_3d: A NumPy array of shape (N, 3) representing 3D points (x, y, z).
        fx, fy, cx, cy: Camera intrinsic parameters.

    Returns:
        A NumPy array of shape (N, 2) representing 2D pixel coordinates (u, v).
        Points with Z <= 0 are filtered out.
    """
    if points_3d.shape[0] == 0:
        return np.array([])

    # Filter out points with Z <= 0
    valid_indices = np.where(points_3d[:, 2] > 1e-6)[0]
    valid_points_3d = points_3d[valid_indices, :]

    if valid_points_3d.shape[0] == 0:
        return np.array([])

    X = valid_points_3d[:, 0]
    Y = valid_points_3d[:, 1]
    Z = valid_points_3d[:, 2]

    # Project to 2D pixel coordinates
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    # Combine u and v into a single array
    projected_points_2d = np.stack((u, v), axis=-1)

    return projected_points_2d.astype(int) # Convert to integer pixel coordinates

def visualize_plane_on_image(image_path, plane_coefficients, fx, fy, cx, cy, image_shape):
    """
    Visualizes the fitted plane on a given image.

    Args:
        image_path: Path to the image file.
        plane_coefficients: Tuple (A, B, C, D) of the plane equation.
        fx, fy, cx, cy: Camera intrinsic parameters.
        image_shape: Tuple (height, width) of the image for clipping projected points.

    Returns:
        The image with the plane drawn, or None if reading fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image file {image_path}")
        return None

    A, B, C, D = plane_coefficients

    # Define a grid of points in X and Y to generate 3D points on the plane
    grid_size = 500  # Increased from 20 to 100 for more points
    x_range = np.linspace(-2, 2, grid_size)
    y_range = np.linspace(-1, 1, grid_size)

    grid_x, grid_y = np.meshgrid(x_range, y_range)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    # Calculate Z for each (X, Y) based on the plane equation Ax + By + Cz + D = 0
    if abs(C) > 1e-6:
        grid_z = (-A * grid_x - B * grid_y - D) / C
    else:
        print("Warning: Plane is near vertical. Visualization might be inaccurate.")
        return image

    # Combine into 3D points
    plane_points_3d = np.vstack((grid_x, grid_y, grid_z)).T

    # Project 3D points to 2D
    projected_points_2d = project_3d_to_2d(plane_points_3d, fx, fy, cx, cy)

    # Draw grid points and lines
    height, width = image_shape[:2]
    
    # Initialize grid_points with invalid coordinates (-1, -1)
    grid_points = np.full((grid_size, grid_size, 2), -1, dtype=int)
    
    # Fill in valid points
    valid_points = 0
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(projected_points_2d):
                u, v = projected_points_2d[idx]
                if 0 <= u < width and 0 <= v < height:
                    grid_points[i, j] = [u, v]
                    valid_points += 1
    
    if valid_points < 4:  # Need at least 4 points to draw a grid
        print("Warning: Not enough valid points to draw grid")
        return image
    
    # Draw grid lines
    for i in range(grid_size):
        for j in range(grid_size):
            # Draw horizontal lines
            if j < grid_size - 1:
                pt1 = grid_points[i, j]
                pt2 = grid_points[i, j+1]
                if pt1[0] >= 0 and pt2[0] >= 0:  # Check if points are valid
                    cv2.line(image, tuple(pt1), tuple(pt2), PLANE_VIS_COLOR, 1)
            
            # Draw vertical lines
            if i < grid_size - 1:
                pt1 = grid_points[i, j]
                pt2 = grid_points[i+1, j]
                if pt1[0] >= 0 and pt2[0] >= 0:  # Check if points are valid
                    cv2.line(image, tuple(pt1), tuple(pt2), PLANE_VIS_COLOR, 1)
    
    # Draw grid points
    for i in range(grid_size):
        for j in range(grid_size):
            pt = grid_points[i, j]
            if pt[0] >= 0:  # Check if point is valid
                cv2.circle(image, tuple(pt), PLANE_POINT_SIZE, PLANE_VIS_COLOR, -1)

    return image

def save_hsv_values_to_yaml(hsv_values, tolerance_values, filepath):
    """Saves HSV values and tolerances to a YAML file."""
    data = {
        'hsv_values': {
            'H': int(hsv_values[0]),
            'S': int(hsv_values[1]),
            'V': int(hsv_values[2])
        },
        'tolerance_values': {
            'H': int(tolerance_values[0]),
            'S': int(tolerance_values[1]),
            'V': int(tolerance_values[2])
        }
    }
    with open(filepath, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"HSV values saved to {filepath}")

if __name__ == "__main__":
    try:
        # 1. Load camera parameters
        fx, fy, cx, cy = load_camera_parameters(CAMERA_PARAMS_PATH)
        print(f"Loaded camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

        # Load the image for HSV tuning
        segmented_image = cv2.imread(SEGMENTED_IMAGE_PATH)
        if segmented_image is None:
            raise FileNotFoundError(f"Could not read the segmented image at {SEGMENTED_IMAGE_PATH}")

        # Create HSV tuning window
        create_hsv_trackbars()
        
        print("Adjust HSV values using trackbars. Press 's' to save and continue, 'q' to quit.")
        while True:
            # Get current HSV values
            hsv_values, tolerance_values = get_hsv_values()
            
            # Update visualization
            mask_visualization, mask = update_mask_visualization(segmented_image, hsv_values, tolerance_values)
            
            # Show the result
            cv2.imshow("Mask Visualization", mask_visualization)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Operation cancelled by user.")
                cv2.destroyAllWindows()
                exit(0)
            elif key == ord('s'):
                # Save HSV values to YAML
                save_hsv_values_to_yaml(hsv_values, tolerance_values, 'hsv_values.yaml')
                break
        
        cv2.destroyAllWindows()
        
        # Use the final HSV values for point collection
        MASK_COLOR_HSV = hsv_values
        COLOR_TOLERANCE_HSV = tolerance_values
        
        print(f"Using HSV values: {MASK_COLOR_HSV}")
        print(f"Using HSV tolerances: {COLOR_TOLERANCE_HSV}")

        # 2. & 3. Get masked pixels with depth from the segmented image and depth map
        masked_uvd_points = get_masked_pixels_with_depth(
            SEGMENTED_IMAGE_PATH, DEPTH_MAP_PATH, MASK_COLOR_HSV, COLOR_TOLERANCE_HSV
        )
        print(f"Found {len(masked_uvd_points)} masked pixels with valid depth.")

        if not masked_uvd_points:
            print("No valid masked points found. Cannot calculate plane equation.")
        else:
            # 4. Convert uvd to xyz
            xyz_points = convert_uvd_to_xyz(masked_uvd_points, fx, fy, cx, cy)
            print(f"Converted {len(xyz_points)} points to 3D.")

            # 5. Fit plane to 3D points using RANSAC with increased iterations
            print(f"Fitting plane using RANSAC with {RANSAC_ITERATIONS} iterations, threshold {RANSAC_THRESHOLD}, and minimum {RANSAC_MIN_INLIERS} inliers...")
            plane_coefficients = fit_plane_ransac(
                xyz_points, RANSAC_ITERATIONS, RANSAC_THRESHOLD,
                RANSAC_SAMPLE_SIZE, RANSAC_MIN_INLIERS
            )

            if plane_coefficients:
                A, B, C, D = plane_coefficients
                print("\nCalculated Plane Equation (Ax + By + Cz + D = 0):")
                print(f"A = {A}")
                print(f"B = {B}")
                print(f"C = {C}")
                print(f"D = {D}")

                # 6. Save plane equation to YAML
                save_plane_equation_to_yaml(plane_coefficients, PLANE_EQUATION_OUTPUT_PATH)

                # 7. Load the saved plane equation from YAML
                with open(PLANE_EQUATION_OUTPUT_PATH, 'r') as file:
                    saved_data = yaml.safe_load(file)
                    saved_plane = saved_data['plane_equation']
                    loaded_coefficients = (
                        saved_plane['A'],
                        saved_plane['B'],
                        saved_plane['C'],
                        saved_plane['D']
                    )

                # 8. Visualize the loaded plane equation on the segmented image
                print("\nVisualizing plane on the segmented image...")
                segmented_img = cv2.imread(SEGMENTED_IMAGE_PATH)
                if segmented_img is not None:
                    image_shape = segmented_img.shape[:2]  # (height, width)
                    print(f"Using image shape {image_shape} for visualization clipping.")

                    # Visualize plane on the segmented image using loaded coefficients
                    image_with_plane = visualize_plane_on_image(
                        SEGMENTED_IMAGE_PATH, loaded_coefficients, fx, fy, cx, cy, image_shape
                    )

                    if image_with_plane is not None:
                        # Display the image
                        cv2.imshow("Plane Visualization", image_with_plane)
                        cv2.waitKey(0)  # Wait for any key press
                        cv2.destroyAllWindows()
                    else:
                        print("Failed to visualize plane on the image.")
                else:
                    print(f"Could not read the segmented image at {SEGMENTED_IMAGE_PATH}")

            else:
                print("Failed to calculate plane equation.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")