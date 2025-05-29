import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_trajectory_data(data_type):
    """Load trajectory data for a specific type (real, rgb, or synthetic)."""
    base_path = Path('/home/ant/projects/es_fp/SocialForce/trajectories')
    return {
        'positions': np.load(base_path / f'{data_type}_positions_meters.npy'),
        'frame_numbers': np.load(base_path / f'{data_type}_frame_numbers.npy'),
        'trajectory_ids': np.load(base_path / f'{data_type}_trajectory_ids.npy'),
        'distances': np.load(base_path / f'{data_type}_distances.npy') if Path(f'{data_type}_distances.npy').exists() else None,
        'world_coords': np.load(base_path / f'{data_type}_world_coords.npy') if Path(f'{data_type}_world_coords.npy').exists() else None
    }

def plot_trajectories(data, title):
    """Plot trajectories in 2D space."""
    plt.figure(figsize=(10, 8))
    unique_ids = np.unique(data['trajectory_ids'])
    
    for traj_id in unique_ids:
        mask = data['trajectory_ids'] == traj_id
        positions = data['positions'][mask]
        plt.plot(positions[:, 0], positions[:, 1], '-', alpha=0.5)
    
    plt.title(title)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.axis('equal')

def analyze_trajectory_statistics(data, data_type):
    """Calculate and print basic statistics about the trajectories."""
    unique_ids = np.unique(data['trajectory_ids'])
    trajectory_lengths = [np.sum(data['trajectory_ids'] == traj_id) for traj_id in unique_ids]
    
    print(f"\nStatistics for {data_type} trajectories:")
    print(f"Number of trajectories: {len(unique_ids)}")
    print(f"Average trajectory length: {np.mean(trajectory_lengths):.2f} frames")
    print(f"Min trajectory length: {np.min(trajectory_lengths)} frames")
    print(f"Max trajectory length: {np.max(trajectory_lengths)} frames")
    
    if data['distances'] is not None:
        print(f"Average distance traveled: {np.mean(data['distances']):.2f} units")
        print(f"Total distance traveled: {np.sum(data['distances']):.2f} units")

def plot_xy_coordinates(data, data_type):
    """Create a detailed plot of XY coordinates for trajectories."""
    plt.figure(figsize=(12, 10))
    
    # Get unique trajectory IDs
    unique_ids = np.unique(data['trajectory_ids'])
    
    # Create a colormap for different trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ids)))
    
    # Plot each trajectory
    for idx, traj_id in enumerate(unique_ids):
        mask = data['trajectory_ids'] == traj_id
        positions = data['positions'][mask]
        
        # Plot the trajectory
        plt.plot(positions[:, 0], positions[:, 1], '-', 
                color=colors[idx], 
                alpha=0.7,
                linewidth=2,
                label=f'Trajectory {traj_id}')
        
        # Plot start and end points
        plt.plot(positions[0, 0], positions[0, 1], 'o', 
                color=colors[idx], 
                markersize=8,
                label=f'Start {traj_id}' if idx == 0 else "")
        plt.plot(positions[-1, 0], positions[-1, 1], 's', 
                color=colors[idx], 
                markersize=8,
                label=f'End {traj_id}' if idx == 0 else "")
    
    plt.title(f'{data_type.capitalize()} Trajectories - XY Coordinates', fontsize=14)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    # Add legend for start/end points
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
              loc='upper right', 
              bbox_to_anchor=(1.15, 1))
    
    # Save the plot
    plt.savefig(f'{data_type}_xy_coordinates.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def plot_positions(data, data_type):
    """Create a scatter plot of all positions."""
    plt.figure(figsize=(10, 8))
    
    # Plot all positions as scatter points
    plt.scatter(data['positions'][:, 0], data['positions'][:, 1], 
                alpha=0.5, s=20, label='Positions')
    
    plt.title(f'{data_type.capitalize()} Positions', fontsize=14)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.legend()
    
    plt.show()
    plt.close()

def main():
    # Load data for all types
    data_types = ['real', 'rgb', 'synthetic']
    all_data = {dtype: load_trajectory_data(dtype) for dtype in data_types}
    
    # Create position plots for each data type
    for dtype, data in all_data.items():
        plot_positions(data, dtype)
    
    # Create plots
    plt.figure(figsize=(15, 5))
    for i, (dtype, data) in enumerate(all_data.items(), 1):
        plt.subplot(1, 3, i)
        plot_trajectories(data, f'{dtype.capitalize()} Trajectories')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Analyze statistics
    for dtype, data in all_data.items():
        analyze_trajectory_statistics(data, dtype)
    
    # Create a comparison plot of trajectory lengths
    plt.figure(figsize=(10, 6))
    lengths_data = []
    for dtype, data in all_data.items():
        unique_ids = np.unique(data['trajectory_ids'])
        lengths = [np.sum(data['trajectory_ids'] == traj_id) for traj_id in unique_ids]
        lengths_data.extend([(dtype, length) for length in lengths])
    
    df = pd.DataFrame(lengths_data, columns=['Type', 'Length'])
    sns.boxplot(data=df, x='Type', y='Length')
    plt.title('Trajectory Length Distribution by Type')
    plt.show()
    plt.close()

    # Create XY coordinate plots for each data type
    for dtype, data in all_data.items():
        plot_xy_coordinates(data, dtype)
        print(f"Created XY coordinate plot for {dtype} trajectories")

if __name__ == "__main__":
    main()
