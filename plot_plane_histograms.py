import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
plt.style.use('default')
sns.set_theme()

# Load the plane parameters
data_path = '/media/hdd_4/PhD/T4/embedded system/fp/experiment1/plane_fitting_results/numpy_data/all_plane_params.npy'
plane_params = np.load(data_path)

# Calculate d0 for each plane
d0_values = np.abs(plane_params[:, 3]) / np.sqrt(np.sum(plane_params[:, :3]**2, axis=1))

# Create a figure with subplots for coefficients and d0
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

# Coefficient names
coef_names = ['a', 'b', 'c', 'd']

# Create histograms for each coefficient
for i in range(4):
    sns.histplot(data=plane_params[:, i], ax=axes[i], kde=True)
    axes[i].set_title(f'Distribution of Coefficient {coef_names[i]}')
    axes[i].set_xlabel(f'Value of {coef_names[i]}')
    axes[i].set_ylabel('Frequency')

# Create histogram for d0
sns.histplot(data=d0_values, ax=axes[4], kde=True)
axes[4].set_title('Distribution of d0 (Distance from Origin)')
axes[4].set_xlabel('d0 Value')
axes[4].set_ylabel('Frequency')

# Remove the last unused subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('plane_coefficients_and_d0_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics for coefficients and d0
print("Basic Statistics for each coefficient and d0:")
print("-" * 50)
for i in range(4):
    print(f"\nCoefficient {coef_names[i]}:")
    print(f"Mean: {np.mean(plane_params[:, i]):.4f}")
    print(f"Std: {np.std(plane_params[:, i]):.4f}")
    print(f"Min: {np.min(plane_params[:, i]):.4f}")
    print(f"Max: {np.max(plane_params[:, i]):.4f}")

print("\nd0 (Distance from Origin):")
print(f"Mean: {np.mean(d0_values):.4f}")
print(f"Std: {np.std(d0_values):.4f}")
print(f"Min: {np.min(d0_values):.4f}")
print(f"Max: {np.max(d0_values):.4f}") 