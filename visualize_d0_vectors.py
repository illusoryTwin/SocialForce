import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load the plane parameters
data_path = '/media/hdd_4/PhD/T4/embedded system/fp/experiment1/plane_fitting_results/numpy_data/all_plane_params.npy'
plane_params = np.load(data_path)

# Calculate d0 for each plane
d0_values = np.abs(plane_params[:, 3]) / np.sqrt(np.sum(plane_params[:, :3]**2, axis=1))

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add coordinate frame
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
vis.add_geometry(coordinate_frame)

# Create a sphere at origin
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
sphere.paint_uniform_color([1, 0, 0])  # Red color for origin
vis.add_geometry(sphere)

# Create and add planes with their d0 vectors
for i in range(len(plane_params)):
    # Get plane parameters
    a, b, c, d = plane_params[i]
    d0 = d0_values[i]
    
    # Calculate normal vector
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # Normalize
    
    # Calculate point on plane closest to origin
    point = normal * d0
    
    # Create arrow for d0 vector
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.04,
        cylinder_height=0.8,
        cone_height=0.2
    )
    
    # Position and orient the arrow
    arrow.translate([0, 0, 0])  # Start from origin
    arrow.rotate(arrow.get_rotation_matrix_from_xyz([0, 0, 0]), center=[0, 0, 0])
    
    # Scale arrow to d0 length
    arrow.scale(d0, center=[0, 0, 0])
    
    # Rotate arrow to point in normal direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, normal)
    rotation_angle = np.arccos(np.dot(z_axis, normal))
    if np.linalg.norm(rotation_axis) > 0:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        arrow.rotate(R, center=[0, 0, 0])
    
    # Color the arrow based on d0 value
    # Use a colormap to map d0 values to colors
    normalized_d0 = (d0 - np.min(d0_values)) / (np.max(d0_values) - np.min(d0_values))
    color = plt.cm.viridis(normalized_d0)[:3]  # Get RGB values
    arrow.paint_uniform_color(color)
    
    vis.add_geometry(arrow)

# Set up the view
view_control = vis.get_view_control()
view_control.set_zoom(0.8)
view_control.set_front([0, 0, -1])
view_control.set_lookat([0, 0, 0])
view_control.set_up([0, -1, 0])

# Add a colorbar to show d0 scale
plt.figure(figsize=(1, 4))
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
             label='d0 value',
             orientation='vertical')
plt.savefig('d0_colorbar.png', bbox_inches='tight', dpi=300)
plt.close()

# Run the visualizer
vis.run()
vis.destroy_window() 