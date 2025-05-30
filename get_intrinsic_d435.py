import numpy as np
import yaml


def get_intrinsic_matrix():
    # Load camera parameters from YAML file
    with open('camera_params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Create intrinsic matrices from YAML parameters
    depth_intrinsic_matrix = np.array([
        [params['d435_depth']['fx'], 0, params['d435_depth']['cx']],
        [0, params['d435_depth']['fy'], params['d435_depth']['cy']],
        [0, 0, 1]
    ])
    
    color_intrinsic_matrix = np.array([
        [params['d435_color']['fx'], 0, params['d435_color']['cx']],
        [0, params['d435_color']['fy'], params['d435_color']['cy']],
        [0, 0, 1]
    ])
    
    print("Depth Intrinsic Matrix:")
    print(depth_intrinsic_matrix)
    print("\nColor Intrinsic Matrix:")
    print(color_intrinsic_matrix)
    
    return depth_intrinsic_matrix, color_intrinsic_matrix


if __name__ == "__main__":
    depth_matrix, color_matrix = get_intrinsic_matrix()
