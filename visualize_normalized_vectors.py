import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the plane parameters
data_path = '/media/hdd_4/PhD/T4/embedded system/fp/experiment1/plane_fitting_results/numpy_data/all_plane_params.npy'
plane_params = np.load(data_path)

# Extract v0 = (a, b, c)
v0 = plane_params[:, :3]

# Calculate |v0| for each plane
v0_norms = np.sqrt(np.sum(v0**2, axis=1))

# Calculate normalized vectors v = v0/|v0|
v = v0 / v0_norms[:, np.newaxis]

# Calculate d0
d0_values = np.abs(plane_params[:, 3]) / v0_norms

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
component_names = ['a', 'b', 'c']

# Create histograms for each component
for i in range(3):
    # Use absolute values
    abs_values = np.abs(v[:, i])
    sns.histplot(data=abs_values, ax=axes[i], kde=True, log_scale=(False, True))
    axes[i].set_title(f'Distribution of |n_{component_names[i]}|')
    axes[i].set_xlabel(f'|n_{component_names[i]}| = |{component_names[i]}/√(a²+b²+c²)|')
    axes[i].set_ylabel('Frequency (log scale)')
    axes[i].set_xlim(0, 1)  # Normalized components should be between 0 and 1
    axes[i].grid(True, which="both", ls="-", alpha=0.2)

# Create histogram for d0
sns.histplot(data=d0_values, ax=axes[3], kde=True, log_scale=(False, True))
axes[3].set_title('Distribution of d0')
axes[3].set_xlabel('d0 = |d|/√(a²+b²+c²)')
axes[3].set_ylabel('Frequency (log scale)')
axes[3].grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('normalized_components_and_d0_histograms_linear_log.png', dpi=300, bbox_inches='tight')
plt.show()

# Print statistics for each component
print("Basic Statistics for normalized vector components:")
print("-" * 50)
for i in range(3):
    print(f"\nComponent n_{component_names[i]}:")
    print(f"Mean: {np.mean(v[:, i]):.4f}")
    print(f"Std: {np.std(v[:, i]):.4f}")
    print(f"Min: {np.min(v[:, i]):.4f}")
    print(f"Max: {np.max(v[:, i]):.4f}")

# Print statistics for absolute values
print("\nBasic Statistics for absolute values of normalized vector components:")
print("-" * 50)
for i in range(3):
    abs_values = np.abs(v[:, i])
    print(f"\n|n_{component_names[i]}|:")
    print(f"Mean: {np.mean(abs_values):.4f}")
    print(f"Std: {np.std(abs_values):.4f}")
    print(f"Min: {np.min(abs_values):.4f}")
    print(f"Max: {np.max(abs_values):.4f}")

# Print statistics for d0
print("\nBasic Statistics for d0:")
print("-" * 50)
print(f"Mean: {np.mean(d0_values):.4f}")
print(f"Std: {np.std(d0_values):.4f}")
print(f"Min: {np.min(d0_values):.4f}")
print(f"Max: {np.max(d0_values):.4f}")

# Verify that vectors are normalized
norms = np.sqrt(np.sum(v**2, axis=1))
print("\nVerification of normalization:")
print(f"Mean norm: {np.mean(norms):.6f}")
print(f"Std of norms: {np.std(norms):.6f}")
print(f"Min norm: {np.min(norms):.6f}")
print(f"Max norm: {np.max(norms):.6f}") 