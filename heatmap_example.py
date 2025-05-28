import numpy as np
import matplotlib.pyplot as plt

# Create sample data
data = np.random.rand(10, 10)  # 10x10 random matrix

# Basic heatmap
plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.title('Basic Heatmap')
plt.show()

# Customized heatmap with annotations
plt.figure(figsize=(10, 8))
heatmap = plt.imshow(data, cmap='YlOrRd')
plt.colorbar(heatmap)

# Add annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        plt.text(j, i, f'{data[i, j]:.2f}',
                ha='center', va='center',
                color='black' if data[i, j] < 0.5 else 'white')

plt.title('Customized Heatmap with Annotations')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

# Heatmap with custom colormap and labels
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(data, cmap='coolwarm')
plt.colorbar(heatmap)

# Add row and column labels
x_labels = [f'Col {i+1}' for i in range(data.shape[1])]
y_labels = [f'Row {i+1}' for i in range(data.shape[0])]
plt.xticks(range(data.shape[1]), x_labels, rotation=45)
plt.yticks(range(data.shape[0]), y_labels)

plt.title('Heatmap with Custom Labels')
plt.tight_layout()
plt.show() 