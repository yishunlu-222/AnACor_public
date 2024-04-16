import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
import open3d as o3d
import pdb 

pth ='D:/lys/studystudy/phd/0-Project_absorption_correction/Code_0_for_absorption_correction/ac/13295_.npy'
values = np.load(pth)

# Define colors for values 2 and 3
color_map = {
    2: [1.0, 1.0, 0.0],  # Yellow (normalized RGB values)
    3: [1.0, 0.0, 0.0]   # Red
}

# Extract indices where values == 2 or values == 3
indices = np.where((values == 2) | (values == 3))

# Convert these indices to points in 3D space
points = np.column_stack(indices)

# Create a point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Assign colors based on the original values
colors = np.array([color_map[values[i]] for i in zip(*indices)])
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

pdb.set_trace()

# Create the spatial reference
grid = pv.ImageData()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
grid.dimensions = np.array(values.shape) + 1

# Edit the spatial reference
grid.origin = (0, 0,0)  # The bottom left corner of the data set
grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

# Add the data values to the cell data
grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array

# plotter = pv.Plotter()

# # Define a colormap (ensure it covers the range of your scalar values)
# colors = [(0, 0, 0, 0), (0, 0, 255, 127), (255, 255, 0, 255), (255, 0, 0, 255)]
# cmap = ListedColormap(colors)
# plotter.add_volume(grid, scalars="values", cmap=cmap, n_colors=len(colors))

# # Display the plot
# plotter.show()

color_map = {
    0: [0, 0, 0, 0],       # Transparent
    1: [0, 0, 0, 0],   # Blue, 50% transparent
    2: [0, 0, 0, 0], # Yellow, fully opaque
    3: [255, 0, 0, 255]    # Red, fully opaque
}

rgba_values = np.zeros((*values.shape, 4), dtype=np.uint8)
for value, color in color_map.items():
    rgba_values[values == value] = color
# pdb.set_trace()
rgba_values_flat =rgba_values.reshape(-1, 4)
# pdb.set_trace()
# plotter = pv.Plotter(window_size=[1920, 1080])
plotter = pv.Plotter() 
plotter.add_volume(grid) #, scalars=rgba_values_flat)

plotter.show()
# rgba_values = np.array([color_map[val] for val in values.flatten()]).reshape(*values.shape, 4).astype(np.uint8)

# Set colors for values 1, 2, 3 (as RGB) and set value 0 to be fully transparent
# opacity = [0, 0.5, 1, 1]  # Opacity for values 0, 1, 2, 3
# colors = ["white", "blue", "yellow", "red"] # Colors for values 0, 1, 2, 3

# Add the volume to the plotter

