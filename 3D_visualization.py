import numpy as np
import open3d as o3d
import cv2
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time

# Load YOLO model
model = YOLO('yolov8n.pt')

# Load plane coefficients
xyz_d0_means = np.load('xyz_d0_means.npy')
print("Loaded plane coefficients:", xyz_d0_means)

# Extract coefficients
a, b, c, d0 = xyz_d0_means
d0 = -d0
# c = -c
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d0:.2f} = 0")

# Load additional points
additional_points = np.load('/media/hdd_4/PhD/T4/embedded system/fp/experiment1/plane_fitting_results/numpy_data/points_002000.npy')
print("Loaded additional points shape:", additional_points.shape)

# Load camera parameters
with open('/media/hdd_4/PhD/T4/embedded system/fp/experiment1/camera_params.yaml', 'r') as f:
    camera_params = yaml.safe_load(f)

# Calculate camera extrinsics from plane equation
def calculate_camera_extrinsics(plane_coeffs, camera_params):
    a, b, c, d0 = plane_coeffs
    fx = camera_params['d435_depth']['fx']
    fy = camera_params['d435_depth']['fy']
    cx = camera_params['d435_depth']['cx']
    cy = camera_params['d435_depth']['cy']
    
    # Get image dimensions from the first frame
    rgb_path = '/media/hdd_4/PhD/T4/embedded system/fp/experiment1/rgb/rgb_002000.png'
    img = cv2.imread(rgb_path)
    height, width = img.shape[:2]
    
    # Calculate camera position using the plane equation and image center
    # We'll use the image center point and its corresponding 3D point on the plane
    center_x = width // 2
    center_y = height // 2
    
    # Get depth at image center (you might need to adjust this based on your setup)
    depth_path = '/media/hdd_4/PhD/T4/embedded system/fp/experiment1/depth/depth_002000.png'
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    z = depth[center_y, center_x] / 10.0  # Convert to meters
    
    # Calculate 3D point in camera space
    x = (center_x - cx) * z / fx
    y = (center_y - cy) * z / fy
    
    # This point should lie on the plane
    # We can use this to calculate the camera position
    # The camera position should be such that the plane normal points towards it
    plane_normal = np.array([a, b, c])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Calculate camera position (assuming camera is above the plane looking down)
    # We'll place the camera at a reasonable height above the plane
    camera_height = 2.0  # meters
    camera_pos = np.array([0, 0, camera_height])
    
    # Calculate rotation to make camera look at the plane
    # We want the camera's z-axis to align with the plane normal
    z_axis = plane_normal
    x_axis = np.array([1, 0, 0])  # Initial x-axis
    if np.abs(np.dot(z_axis, x_axis)) > 0.9:
        x_axis = np.array([0, 1, 0])
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Create rotation matrix
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # Calculate translation
    t = -R @ camera_pos
    
    return R, t

# Calculate camera extrinsics
camera_rotation, camera_translation = calculate_camera_extrinsics((a, b, c, d0), camera_params)

# Create transformation matrix from camera to world space
def create_camera_to_world_transform():
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = camera_rotation
    transform[:3, 3] = camera_translation
    
    return transform

# Transform point from camera to world space
def camera_to_world(point):
    # Add homogeneous coordinate
    point_h = np.append(point, 1)
    
    # Apply transformation
    world_point = camera_to_world_transform @ point_h
    
    # Remove homogeneous coordinate
    return world_point[:3]

# Create camera to world transformation matrix
camera_to_world_transform = create_camera_to_world_transform()

# Create camera to world transformation matrix
camera_to_world_transform = create_camera_to_world_transform()

# Print camera extrinsics for verification
print("Camera rotation matrix:")
print(camera_rotation)
print("\nCamera translation vector:")
print(camera_translation)

# Create a mesh for the plane
def create_plane_mesh(coefficients, size=100.0):
    a, b, c, d0 = coefficients
    # Create a grid of points
    x = np.linspace(-size, size, 20)
    y = np.linspace(-size, size, 20)
    X, Y = np.meshgrid(x, y)
    # Calculate Z values for the plane (ax + by + cz + d0 = 0)
    Z = -(a*X + b*Y + d0) / c
    # Create vertices
    vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    # Create triangles
    triangles = []
    for i in range(19):
        for j in range(19):
            v0 = i * 20 + j
            v1 = v0 + 1
            v2 = v0 + 20
            v3 = v2 + 1
            triangles.extend([[v0, v1, v2], [v1, v3, v2]])
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def create_point_cloud(rgb, depth, camera_params):
    # Get camera parameters
    fx = camera_params['d435_depth']['fx']
    fy = camera_params['d435_depth']['fy']
    cx = camera_params['d435_depth']['cx']
    cy = camera_params['d435_depth']['cy']
    
    # Create meshgrid of pixel coordinates
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Convert to 3D points
    z = depth.astype(float) / 1000.0  # Convert to meters
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    
    # Stack coordinates and reshape
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    
    # Remove invalid points (where depth is 0)
    valid_points = z.reshape(-1) > 0
    points = points[valid_points]
    colors = colors[valid_points]
    
    return points, colors

def get_detection_3d_position(detection, depth, camera_params):
    # Get bounding box coordinates
    x1, y1, x2, y2 = detection.boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    # Calculate center point
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Get depth at center point
    depth_value = depth[center_y, center_x]
    
    # Convert to 3D coordinates using camera parameters
    fx = camera_params['d435_depth']['fx']
    fy = camera_params['d435_depth']['fy']
    cx = camera_params['d435_depth']['cx']
    cy = camera_params['d435_depth']['cy']
    
    Z = depth_value / 1000.0  # Convert to meters
    X = (center_x - cx) * Z / fx
    Y = (center_y - cy) * Z / fy
    
    return np.array([X, Y, Z])

def project_point_to_plane(point, plane_coeffs):
    a, b, c, d0 = plane_coeffs
    x, y, z = point
    
    # Calculate the distance from point to plane
    distance = (a*x + b*y + c*z + d0) / np.sqrt(a*a + b*b + c*c)
    
    # Project point onto plane
    projected_point = point - distance * np.array([a, b, c]) / np.sqrt(a*a + b*b + c*c)
    
    return projected_point

# Create visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add plane and coordinate frame
plane_mesh = create_plane_mesh(np.array((a, b, c, d0)), size=200.0)
plane_mesh.paint_uniform_color([0, 1, 0])  # Green color for the plane
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

vis.add_geometry(plane_mesh)
vis.add_geometry(coordinate_frame)

# Process 10 frames
for frame_num in range(2000, 2010):
    print(f"Processing frame {frame_num}")
    
    # Load RGB and depth images
    rgb_path = f'/media/hdd_4/PhD/T4/embedded system/fp/experiment1/rgb/rgb_{frame_num:06d}.png'
    depth_path = f'/media/hdd_4/PhD/T4/embedded system/fp/experiment1/depth/depth_{frame_num:06d}.png'
    
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    # Create point cloud from depth image
    points, colors = create_point_cloud(rgb, depth, camera_params)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Run YOLO detection
    results = model(rgb_path)
    
    # Create lists to store original and projected pedestrian positions
    original_pedestrian_positions = []
    projected_pedestrian_positions = []
    
    # Get original 3D positions of pedestrians and project them
    for result in results:
        for detection in result:
            class_id = int(detection.boxes.cls[0].cpu().numpy())
            if class_id == 0:  # person class
                original_position = get_detection_3d_position(detection, depth, camera_params)
                if original_position is not None:
                    original_pedestrian_positions.append(original_position)
                    # Project the point onto the plane
                    projected_position = project_point_to_plane(original_position, (a, b, c, d0))
                    projected_pedestrian_positions.append(projected_position)
    
    # Create spheres for original pedestrian positions (red)
    original_spheres = []
    if original_pedestrian_positions:
        for position in original_pedestrian_positions:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(position)
            sphere.paint_uniform_color([1, 0, 0])  # Red for original positions
            original_spheres.append(sphere)
    
    # Create spheres for projected pedestrian positions (yellow)
    projected_spheres = []
    if projected_pedestrian_positions:
        for position in projected_pedestrian_positions:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(position)
            sphere.paint_uniform_color([1, 1, 0])  # Yellow for projected positions
            projected_spheres.append(sphere)
    
    # Add geometries to visualization
    vis.add_geometry(pcd)
    
    # Add original pedestrian positions (red spheres)
    for sphere in original_spheres:
        vis.add_geometry(sphere)
    
    # Add projected pedestrian positions (yellow spheres)
    for sphere in projected_spheres:
        vis.add_geometry(sphere)
    
    # Add lines connecting original to projected positions
    for orig_pos, proj_pos in zip(original_pedestrian_positions, projected_pedestrian_positions):
        line = o3d.geometry.LineSet()
        points = o3d.utility.Vector3dVector([orig_pos, proj_pos])
        lines = o3d.utility.Vector2iVector([[0, 1]])
        line.points = points
        line.lines = lines
        line.paint_uniform_color([0, 1, 0])  # Green lines
        vis.add_geometry(line)
    
    # Update visualization
    vis.poll_events()
    vis.update_renderer()
    
    # Add a small delay to make the visualization smoother
    time.sleep(0.1)

# Set up camera
ctr = vis.get_view_control()
ctr.set_zoom(0.8)
ctr.set_front([0, 0, -1])
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, -1, 0])

# Run visualization
vis.run()
vis.destroy_window()

# Create a mesh for the plane
