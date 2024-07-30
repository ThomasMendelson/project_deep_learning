import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import wandb
import plotly.graph_objects as go
import numpy as np

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
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

def visualize_3d_image(input_tensor, save_path):
    image_np = input_tensor.cpu().numpy()

    x, y, z = np.indices(image_np.shape)
    x, y, z = x[image_np > 0], y[image_np > 0], z[image_np > 0]  # Get indices where the voxel value is not zero
    values = image_np[image_np > 0]  # Get voxel values that are not zero

    colors = np.where(values == 1, 'green', 'red')

    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            mode='markers',
            marker=dict(
                size=5,
                color=colors,  # Color by voxel value
                opacity=0.2
            )
        )
    ])

    # Update layout for better visualization
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=[0, image_np.shape[0]]),
        yaxis=dict(nticks=4, range=[0, image_np.shape[1]]),
        zaxis=dict(nticks=4, range=[0, image_np.shape[2]]),
        aspectratio=dict(x=1, y=1, z=1)
    ))
    # Save the figure to a temporary file
    fig.write_html(save_path)

    table = wandb.Table(columns=["plotly_figure"])
    table.add_data(wandb.Html(save_path))
    # Log the image to wandb
    wandb.log({os.path.basename(save_path): table})
    # wandb.log({"3D Plot": wandb.Image(html_file)})

def random_crop(image, crop_size=(32,128,128) , num_crops=10, threshold=500):
    depth, height, width = image.shape[-3:]
    crop_depth, crop_height, crop_width = crop_size

    if crop_depth > depth or crop_height > height or crop_width > width:
        raise ValueError("Crop shape is larger than the image dimensions")

    while True:
        # Random starting points
        start_d = torch.randint(0, depth - crop_depth + 1, (1,)).item()
        start_h = torch.randint(0, height - crop_height + 1, (1,)).item()
        start_w = torch.randint(0, width - crop_width + 1, (1,)).item()
        # Crop the image
        cropped_image = image[:, start_d:start_d + crop_depth, start_h:start_h + crop_height,
                        start_w:start_w + crop_width]
        if torch.sum(cropped_image > 0) > threshold * crop_depth:
            # print(f"depth: {depth}, height: {height}, width: {width}")
            # print(f"crop_depth: {crop_depth}, crop_height: {crop_height}, crop_width: {crop_width}")
            # print(f"start_d: {start_d}, start_h: {start_h}, start_w: {start_w}")
            break

    return cropped_image

# if WANDB_TRACKING:
#     wandb.login(key="12b9b358323faf2af56dc288334e6247c1e8bc63")
#     wandb.init(project="seg_unet_3D")
# # Save path for the plot
# save_path1 = '3d_image1.html'
# save_path2 = '3d_image2.html'
#
# # Visualize and save the 3D image
#
#
# mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG/man_seg070.tif"
# voxels_type2 = torch.from_numpy(tiff.imread(mask_path).astype(np.float32)).to(DEVICE)  # every cell have different value
# voxels_type1 = split_mask(voxels_type2.unsqueeze(0)).to(DEVICE)  # 0/1/2
# print("going to visualize_3d_image")
# visualize_3d_image(random_crop(voxels_type1)[0], save_path1)
#
#
# wandb.finish()
import torch.nn as nn
from monai.losses import DiceCELoss
CLASS_WEIGHTS = [0.1, 0.7, 0.2]
class_weights = torch.FloatTensor(CLASS_WEIGHTS).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

if isinstance(criterion, DiceCELoss):
    print("in then")
else:
    print("in else")
print(criterion)

criterion = DiceCELoss(to_onehot_y=False, softmax=True, squared_pred=True, weight=class_weights)

if isinstance(criterion, DiceCELoss):
    print("in then")
else:
    print("in else")
print(criterion)
















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
