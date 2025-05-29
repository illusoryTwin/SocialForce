import numpy as np

def calculate_scale_factor(p1, p2, known_distance=1.5):
    """
    Calculate scale factor from two points and known distance
    p1, p2: pixel coordinates of two points
    known_distance: actual distance between points in meters
    """
    # Calculate pixel distance
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    pixel_distance = np.sqrt(dx*dx + dy*dy)
    
    # Calculate scale factor (meters per pixel)
    scale_factor = known_distance / pixel_distance
    
    return scale_factor

def pixel_to_world_coords(px, img_shape, scale_factor, cx=639.5, cy=359.5, H=1.65):
    """
    Transform pixel coordinates to world coordinates using scale factor
    px: pixel coordinates (x, y)
    img_shape: (height, width) of the image
    scale_factor: meters per pixel
    cx, cy: principal point coordinates
    H: camera height from ground
    """
    # Calculate distance from center in pixels
    dx = px[0] - cx
    dy = px[1] - cy
    
    # Convert to world coordinates using scale factor
    X = dx * scale_factor
    Z = H  # Assuming constant height for ground plane points
    Y = 0  # Ground plane assumption
    
    return np.array([X, Z, Y])

def calculate_table_distance(scale_factor):
    # Table points
    p1 = np.array([22, 585])
    p2 = np.array([133, 575])
    
    # Convert to world coordinates
    w1 = pixel_to_world_coords(p1, (720, 1280), scale_factor)
    w2 = pixel_to_world_coords(p2, (720, 1280), scale_factor)
    
    # Calculate distance
    distance = np.sqrt((w2[0] - w1[0])**2 + (w2[2] - w1[2])**2)
    
    print(f"\nUsing scale factor: {scale_factor:.6f} meters/pixel")
    print(f"World coordinates:")
    print(f"Point 1: ({w1[0]:.2f}, {w1[1]:.2f}, {w1[2]:.2f})")
    print(f"Point 2: ({w2[0]:.2f}, {w2[1]:.2f}, {w2[2]:.2f})")
    print(f"Calculated distance: {distance:.2f} meters")
    print(f"Expected distance: 1.5 meters")
    print(f"Error: {abs(distance - 1.5):.2f} meters")
    
    return distance

if __name__ == "__main__":
    # Table points
    table_points = ((22, 585), (133, 575))
    
    # Calculate scale factor
    scale_factor = calculate_scale_factor(table_points[0], table_points[1])
    print(f"\nCalculated scale factor: {scale_factor:.6f} meters/pixel")
    
    # Calculate and verify table distance
    table_distance = calculate_table_distance(scale_factor) 