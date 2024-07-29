import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import wandb
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
WANDB_TRACKING = True

# def plot_voxels_type1(voxels):
#     """
#     Plot 3D voxels where each voxel has three specific values (0: background, 1: edge, 2: interior).
#     """
#     # Create a figure to hold the scatter plot
#     fig = go.Figure()
#
#     # Get the coordinates of the non-background voxels
#     coords = torch.nonzero(voxels > 0, as_tuple=False)
#     values = voxels[coords[:, 0], coords[:, 1], coords[:, 2]]
#
#     # Get the values of the non-background voxels
#     colors = ['#000000', '#00FF00', '#FFFFFF']  # black for background, green for edge, white for interior
#
#     values = values.cpu().tolist()  # Convert tensor to numpy array
#     color_map = [colors[val] for val in values]
#
#     # Create scatter plot
#     fig.add_trace(go.Scatter3d(
#         x=coords[:, 0].tolist(),
#         y=coords[:, 1].tolist(),
#         z=coords[:, 2].tolist(),
#         mode='markers',
#         marker=dict(size=2, color=color_map)
#     ))
#
#     # Show the plot
#     fig.show()
#
# def plot_voxels_type2(voxels):
#     """
#     Plot 3D voxels where each non-background voxel can have a different color.
#     """
#     # Create a figure to hold the scatter plot
#     fig = go.Figure()
#
#     # Get the coordinates of the non-background voxels
#     coords = torch.nonzero(voxels > 0, as_tuple=False)
#
#     # Get the values of the non-background voxels
#     values = voxels[coords[:, 0], coords[:, 1], coords[:, 2]]
#
#     # Normalize the values for color mapping
#     norm_values = (values - values.min()) / (values.max() - values.min())
#
#     # Create scatter plot
#     fig.add_trace(go.Scatter3d(
#         x=coords[:, 0].tolist(),
#         y=coords[:, 1].tolist(),
#         z=coords[:, 2].tolist(),
#         mode='markers',
#         marker=dict(size=2, color=norm_values.tolist(), colorscale='Viridis')
#     ))
#
#     # Show the plot
#     fig.show()
#
#
def detect_edges(mask, threshold=0.25):
    # Compute the gradients along rows and columns
    gradient_x = torch.gradient(mask, dim=2)[0]
    gradient_y = torch.gradient(mask, dim=1)[0]
    gradient_z = torch.gradient(mask, dim=0)[0]

    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)
    masked_gradient_magnitude = gradient_magnitude * mask
    edge_mask = (masked_gradient_magnitude > threshold).to(torch.int)

    return edge_mask


def split_mask(mask):
    # mask: torch.Size([batch, D, H, W])
    three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
    for batch_idx in range(mask.size()[0]):
        unique_elements = torch.unique(mask[batch_idx].flatten())
        for element in unique_elements:
            if element != 0:
                element_mask = (mask[batch_idx] == element).to(torch.int)
                edges = detect_edges(element_mask)
                element_mask -= edges
                three_classes_mask[batch_idx][edges == 1] = 1         # edge
                three_classes_mask[batch_idx][element_mask == 1] = 2  # interior

    return three_classes_mask





def visualize_3d_image_2(tensor, save_path, wand_log=False):
    # Ensure tensor is on the CPU and convert to numpy array
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.numpy()

    # Create a figure to hold all scatter plots
    fig = go.Figure()

    # Get the dimensions of the input tensor
    print(data.shape)
    depth, height, width = data.shape[-3:]
    # Iterate through the tensor to add scatter points based on the two types of images
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                value = data[0, z, y, x]
                if value > 0:  # Only plot non-background voxels
                    if value == 1:
                        # Edge voxel (green color)
                        color = 'green'
                    elif value == 2:
                        # Interior voxel (white color)
                        color = 'white'
                    else:
                        # For the second type, use the voxel value for color
                        color = f'rgb({int(value % 256)}, {int(value * 2 % 256)}, {int(value * 3 % 256)})'

                    print(f"point cords:{[y,x,z]}, color:{color}")
                    fig.add_trace(go.Scatter3d(
                        x=[x],
                        y=[y],
                        z=[z],
                        mode='markers',
                        marker=dict(size=2, color=color),
                        name=f'Voxel ({x},{y},{z})'
                    ))

    # Save the plot
    fig.write_html(save_path, auto_play=False)

    if wand_log:
        # Create a table
        table = wandb.Table(columns=["plotly_figure"])

        # Add Plotly figure as HTML file into Table
        table.add_data(wandb.Html(save_path))

        # Log Table
        wandb.log({save_path: table})





def visualize_3d_image(input_tensor, save_path):
    input_tensor = input_tensor.long()
    depth, height, width = input_tensor.shape
    print(f"input_tensor.shape: {input_tensor.shape}")
    # # Create a color map tensor based on input values
    # color_map = torch.tensor([
    #     [0, 0, 0],  # Black for value 0
    #     [0, 255, 0],  # Green for value 1
    #     [255, 255, 255]  # White for value 2
    # ], dtype=torch.float, device=input_tensor.device)  # Use input_tensor's device
    #
    # # Index the color_map tensor with input_tensor to assign colors
    # output_tensor = color_map[input_tensor]
    # print(f"output_tensor.shape: {output_tensor.shape}")
    # output_tensor = output_tensor.squeeze(0)  # .permute(3, 0, 1, 2)
    # output_array = output_tensor.cpu().numpy().astype(np.uint8)
    print("before fig")
    # Create the grid of coordinates
    x, y, z = np.meshgrid(
        np.arange(width),
        np.arange(height),
        np.arange(depth)
    )

    # Flatten the coordinate arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    values_flat = input_tensor.cpu().numpy().flatten()
    print("before fig")
    fig = go.Figure(data=go.Volume(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=values_flat,
        isomin=0,
        isomax=2,
        opacity=0.1,  # Needs to be small to see through all surfaces
        surface_count=3,  # Number of isosurfaces
        colorscale=[[0, 'black'], [0.5, 'green'], [1, 'white']]  # Custom colorscale for black, green, and white
    ))
    print("after fig")

    fig.update_layout(scene=dict(
        xaxis_title='Width',
        yaxis_title='Height',
        zaxis_title='Depth'
    ))
    print("after fig.update_layout")
    if WANDB_TRACKING:
        # Create a table
        # table = wandb.Table(columns=["plotly_figure"])
        # table.add_data(wandb.Html(save_path))

        # Log Table
        wandb.log({save_path: fig})

if WANDB_TRACKING:
    wandb.login(key="12b9b358323faf2af56dc288334e6247c1e8bc63")
    wandb.init(project="seg_unet_3D")
# Save path for the plot
save_path1 = '3d_image1.html'
save_path2 = '3d_image2.html'

# Visualize and save the 3D image


mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG/man_seg070.tif"
voxels_type2 = torch.from_numpy(tiff.imread(mask_path).astype(np.float32)).to(DEVICE)
voxels_type1 = split_mask(voxels_type2.unsqueeze(0)).to(DEVICE)  # 0/1/2
print("going to visualize_3d_image")
visualize_3d_image(voxels_type1.squeeze(0), save_path1)
# visualize_3d_image(voxels_type1, save_path2, wand_log=True)

wandb.finish()
# plot_voxels_type1(voxels_type1)

# plot_voxels_type1(voxels_type2)


















#
# import plotly.graph_objects as go
# import numpy as np
#
# def plot_voxels_type1(voxels):
#     # Create a figure to hold the scatter plot
#     fig1 = go.Figure()
#
#     # Get the coordinates of the non-background voxels
#     x, y, z = np.where(voxels > 0)
#
#     # Get the values of the non-background voxels
#     values = voxels[x, y, z]
#
#     # Map values to colors
#     colors = ['#000000', '#00FF00', '#FFFFFF']  # black for background, green for edge, white for interior
#     color_map = [colors[val] for val in values]
#
#     # Create scatter plot
#     fig1.add_trace(go.Scatter3d(
#         x=x, y=y, z=z,
#         mode='markers',
#         marker=dict(size=2, color=color_map)
#     ))
#
#     return fig1
#
# def plot_voxels_type2(voxels):
#     # Create a figure to hold the scatter plot
#     fig2 = go.Figure()
#
#     # Get the coordinates of the non-background voxels
#     x, y, z = np.where(voxels > 0)
#
#     # Get the values of the non-background voxels
#     values = voxels[x, y, z]
#
#     # Normalize the values for color mapping
#     norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
#
#     # Create scatter plot
#     fig2.add_trace(go.Scatter3d(
#         x=x, y=y, z=z,
#         mode='markers',
#         marker=dict(size=2, color=norm_values, colorscale='Viridis')
#     ))
#
#     return fig2
#
# def combine_figures_with_plane(fig1, fig2, save_path=None):
#     # Extract data from both figures
#     data1 = fig1.data
#     data2 = fig2.data
#
#     # Create a new figure to combine the data
#     combined_fig = go.Figure()
#
#     # Add data from the first figure
#     for trace in data1:
#         combined_fig.add_trace(trace)
#
#     # Add a red plane between the two plots
#     red_plane = go.Surface(
#         z=[[0, 0], [0, 0]],
#         x=[0, 0, 0, 0],
#         y=[[0, 64], [64, 64]],
#         showscale=False,
#         opacity=0.8,
#         colorscale=[[0, 'red'], [1, 'red']]
#     )
#     combined_fig.add_trace(red_plane)
#
#     # Add data from the second figure, shifting the x-coordinates
#     for trace in data2:
#         shifted_trace = trace
#         shifted_trace.update(x=[x+128 for x in shifted_trace.x])  # Shift x-coordinates to separate the plots
#         combined_fig.add_trace(shifted_trace)
#
#     # Update layout
#     combined_fig.update_layout(
#         scene=dict(
#             xaxis=dict(range=[-10, 150]),
#             yaxis=dict(range=[-10, 70]),
#             zaxis=dict(range=[-10, 70])
#         )
#     )
#
#     # Show the plot
#     combined_fig.show()
#
#     # Save the plot if a path is provided
#     if save_path:
#         combined_fig.write_html(save_path, auto_open=True)
#         print(f"Plot saved to {save_path}")
#
# # Example usage
# voxels_type1 = np.random.randint(0, 3, size=(64, 64, 64))  # Replace with your voxel data
# voxels_type2 = np.random.randint(0, 10, size=(64, 64, 64))  # Replace with your voxel data
#
# fig1 = plot_voxels_type1(voxels_type1)
# fig2 = plot_voxels_type2(voxels_type2)
#
# save_path = "combined_voxel_plot.html"
# combine_figures_with_plane(fig1, fig2, save_path)
