import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import torch
import wandb
import plotly.graph_objects as go
import numpy as np
import skfmm
from scipy.spatial import KDTree
import time
from scipy import ndimage
import os
from PIL import Image
import numpy as np
from scipy.ndimage import label, binary_erosion
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
WANDB_TRACKING = True

def detect_edges2d(mask, threshold=0.25):
    # Compute the gradients along rows and columns
    gradient_x = torch.gradient(mask, dim=1)
    gradient_y = torch.gradient(mask, dim=0)

    # Extract gradient components from the tuple
    gradient_x = gradient_x[0]
    gradient_y = gradient_y[0]

    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
    masked_gradient_magnitude = gradient_magnitude * mask
    edge_mask = (masked_gradient_magnitude > threshold).to(torch.int)

    return edge_mask
def split_mask2d(mask):
    three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
    for batch_idx in range(mask.size()[0]):
        unique_elements = torch.unique(mask[batch_idx].flatten())
        for element in unique_elements:
            if element != 0:
                element_mask = (mask[batch_idx] == element).to(torch.int)
                edges = detect_edges2d(element_mask)
                element_mask -= edges
                three_classes_mask[batch_idx][edges == 1] = 1
                three_classes_mask[batch_idx][element_mask == 1] = 2

    return three_classes_mask


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

def random_crop2D(image, crop_size=(128,128), threshold=500):
    height, width = image.shape[-2:]
    crop_height, crop_width = crop_size

    if crop_height > height or crop_width > width:
        raise ValueError("Crop shape is larger than the image dimensions")

    while True:
        # Random starting points
        start_h = torch.randint(0, height - crop_height + 1, (1,)).item()
        start_w = torch.randint(0, width - crop_width + 1, (1,)).item()
        # Crop the image
        cropped_image = image[:, start_h:start_h + crop_height,
                        start_w:start_w + crop_width]
        if torch.sum(cropped_image > 0) > threshold:
            # print(f"depth: {depth}, height: {height}, width: {width}")
            # print(f"crop_depth: {crop_depth}, crop_height: {crop_height}, crop_width: {crop_width}")
            # print(f"start_d: {start_d}, start_h: {start_h}, start_w: {start_w}")
            break

    return cropped_image

def get_cell_instances(input_np, marker=False,  three_d=False):
    if three_d:
        strel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    else:
        strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    if marker:
        foreground_mask = input_np
    else:
        foreground_mask = (input_np == 2).astype(np.uint8)
    labeled, max_num = ndimage.label(foreground_mask, structure=strel)
    return labeled, max_num


def inference(class_predictions, marker_predictions, three_d=False):
    def find_closest_marker_fmm(foreground, labeled_markers):
        """
        Finds the closest marker for each pixel in the foreground using Fast Marching Method.

        Parameters:
        - foreground: numpy array, binary map of the foreground region (1 for foreground, 0 for background)
        - labeled_markers: numpy array, labeled markers for different regions (0 for no marker, >0 for markers)

        Returns:
        - closest_marker_map: numpy array, same shape as foreground, where each pixel is labeled with the nearest marker
        """
        foreground = np.array(foreground, dtype=bool)
        labeled_markers = np.array(labeled_markers, dtype=np.int32)

        closest_marker_map = np.zeros_like(foreground, dtype=np.int32)
        min_distances = np.full(foreground.shape, np.inf)

        unique_markers = np.unique(labeled_markers)
        unique_markers = unique_markers[unique_markers > 0]

        for marker in unique_markers:
            marker_mask = (labeled_markers == marker)

            if np.any(marker_mask):
                phi = np.ma.MaskedArray(np.ones_like(foreground), mask=~foreground)
                phi[marker_mask] = 0

                distances = skfmm.distance(phi)
                distances[distances.mask] = np.inf

                update_mask = (distances < min_distances) & foreground

                # Ensure update only happens where update_mask is True
                min_distances[update_mask] = distances[update_mask]
                closest_marker_map[update_mask] = marker

        return closest_marker_map

    def find_nearest_markers_KDTree(foreground, markers):
        """
        Finds the nearest marker for each pixel in the foreground with a value of 1 and
        returns an array where each pixel in the foreground is labeled with the value of the nearest marker.

        Parameters:
        - foreground (np.array): 2D array where pixels of interest are marked with 1.
        - markers (np.array): 2D array where non-zero values indicate different marker types.
        - num_features (int): The number of unique non-zero elements in the markers array.

        Returns:
        - np.array: Array where each pixel in the foreground has the value of the nearest marker.
        """
        print(f"foreground.shape {foreground.shape}, markers.shape {markers.shape}")
        # Find coordinates of pixels with value 1 in the foreground
        foreground_coords = np.argwhere(foreground == 1)

        # Find coordinates of markers and their values
        marker_coords = np.argwhere(markers > 0)
        marker_values = markers[marker_coords[:, 0], marker_coords[:, 1]]

        # Create KDTree for marker coordinates
        tree = KDTree(marker_coords)

        # Find the nearest marker for each foreground pixel
        distances, indices = tree.query(foreground_coords)
        nearest_marker_values = marker_values[indices]

        # Create an output array with the same shape as foreground, initialized to 0
        result_array = np.zeros(foreground.shape, dtype=markers.dtype)

        # Use advanced indexing to set the nearest marker values
        result_array[foreground == 1] = nearest_marker_values

        return result_array

    # predicted_classes = predict_classes(class_predictions).cpu().numpy()
    predicted_classes = class_predictions.cpu().numpy()
    predicted_foregound = (predicted_classes == 2).astype(np.uint8)
    labeled_markers, _ = get_cell_instances(marker_predictions.cpu().numpy()[0], marker=True, three_d=three_d)

    fmm_start_time = time.time()
    labeled_preds_fmm = find_closest_marker_fmm(predicted_foregound[0], labeled_markers)
    fmm_end_time = kd_tree_start_time = time.time()
    # labeled_preds = find_nearest_markers_KDTree(predicted_foregound[0], labeled_markers)
    kd_tree_end_time = time.time()
    print(f"fmm time:{fmm_end_time - fmm_start_time}\nkd_tree time:{kd_tree_end_time - kd_tree_start_time}")
    return labeled_preds_fmm  # , labeled_preds


def t_inference():
    markers = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0]])

    pred = np.array([[0, 1, 1, 0, 0],
                     [2, 2, 1, 2, 0],
                     [1, 2, 1, 1, 0],
                     [1, 2, 1, 1, 1],
                     [2, 2, 2, 2, 1]], )

    mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG/man_seg045.tif"

    tra_image = torch.from_numpy(tiff.imread(mask_path).astype(np.float32)).to(DEVICE).unsqueeze(
        0)  # every cell have different value
    # tra_image = random_crop(tra_image)
    tra_image = random_crop(tra_image)

    seg_image = split_mask(tra_image).to(DEVICE)  # 0/1/2
    tra_image[tra_image > 0] = 1

    print(f"seg_image.shape: {seg_image.shape} ")
    print(f"tra_image.shape: {tra_image.shape} ")
    labeled_preds_fmm = inference(seg_image, tra_image, True)
    # labeled_preds_fmm, labeled_preds_tree = inference(seg_image, tra_image, False)


    print("result", labeled_preds_fmm)

    # print("result2", labeled_preds_tree)




def save_shrink_images_to_new_dir():
    # Define the source and target directories
    source_dir = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_GT/TRA"
    target_dir = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_GT/TRA2"

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)


    #
    def process_image(image, num_layers_to_shrink, three_d):
        """
           Shrinks each cell in a 3D image by num_layers_to_shrink voxel layers without changing the overall image size.

           Parameters:
           - tensor_3d (torch.Tensor): A 3D tensor containing cell labels.

           Returns:
           - torch.Tensor: A 3D tensor with each cell shrunk by num_layers_to_shrink voxel layers.
           """
        if three_d:
            strel = np.ones((3, 3, 3))
        else:
            strel = np.ones((3, 3))

        # Create an empty array to store the shrunk cells
        shrunk_array = np.zeros_like(image)

        # Get unique cell values (excluding background, assumed to be 0)
        cell_values = np.unique(image)
        cell_values = cell_values[cell_values != 0]

        # Perform erosion on each cell separately
        for value in cell_values:
            cell_mask = (image == value)
            eroded_mask = binary_erosion(cell_mask, structure=strel, iterations=num_layers_to_shrink)
            shrunk_array[eroded_mask] = value

        print(f"shrunk_array.shape: {shrunk_array.shape}")
        return shrunk_array

    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        # Construct the full file path
        file_path = os.path.join(source_dir, filename)

        # Check if the file is an image (optional: based on file extension)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            # Open the image
            image = tiff.imread(file_path)

            # Process the image
            processed_image = process_image(image, 5, False)

            # Save the processed image to the target directory with the same name
            target_path = os.path.join(target_dir, filename)
            tiff.imwrite(target_path, processed_image)

    print("Processing complete.")


def t_save_shrink_images_to_new_dir(three_d):
    source_dir = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_GT/TRA/man_track050.tif"
    target_dir = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_GT/TRA2/man_track050.tif"

    # Load images
    print("Loading source image...")
    ory_image = tiff.imread(source_dir)
    print("Loading target image...")
    trg_image = tiff.imread(target_dir)
    print(f"trg_image.shape: {trg_image.shape}\nory_image.shape: {ory_image.shape}")
    # Compute middle slice and difference
    if three_d:
        mid_slice = ory_image.shape[0] // 2
        dif = ory_image[mid_slice] - trg_image[mid_slice]
    else:
        dif = ory_image - trg_image
    print("Computed difference of middle slice.")

    # Plot and save the difference image
    plt.imshow(dif, cmap='gray')  # Use 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off axis labels
    output_path = 'dif.png'
    plt.savefig(output_path)  # Save the plot as an image file
    plt.close()  # Close the figure to release memory
    print(f"Image saved to {output_path}.")

def three_d_to_two_d_represantation(images):
    """
    Args:
    - image: 5D tensor of shape (batch_size, channel, depth, height, width)

    Returns:
    - A 4D tensor of shape (batch_size * depth, 3, height, width) containing the slices
    """
    batch_size, channel, depth, height, width = images.shape

    expanded_images = torch.cat([images[:, :, 0:1, :, :], images, images[:, :, -1:, :, :]], dim=2)
    slices = torch.zeros((batch_size, depth, 3, height, width), dtype=images.dtype)

    slices[:, :, 0] = expanded_images[:, :, 0:depth]
    slices[:, :, 1] = expanded_images[:, :, 1:depth + 1]
    slices[:, :, 2] = expanded_images[:, :, 2:depth + 2]

    slices = slices.view(batch_size * depth, 3, height, width)
    return slices


def two_d_to_three_d_represantation(images, batch_size, depth):
    """
        Args:
        - images: 4D tensor of shape (batch_size * depth, channel, height, width) containing the slices
        - batch_size: The original batch size
        - depth: The original depth size

        Returns:
        - 5D tensor of shape (batch_size, channel, depth, height, width)
        """
    batch_size_depth, channel, height, width = images.shape
    assert batch_size * depth == batch_size_depth, "The input images shape does not match the provided batch_size and depth."
    reshaped_images = images.view(batch_size, depth, channel, height, width)

    return reshaped_images.permute(0, 2, 1, 3, 4)

def t_2_to_3_and_back():
    target_dir = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02_GT/TRA2/man_track050.tif"
    original_image = torch.from_numpy(tiff.imread(target_dir).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    print(f"original image.shape: {original_image.shape}")
    image = three_d_to_two_d_represantation(original_image)
    print(f"after 3D->2D image.shape: {image.shape}")
    image = image[:, 1:2, :, :]
    print(f"after extract the midlle slice image.shape: {image.shape}")
    image = two_d_to_three_d_represantation(image, 1, original_image.shape[2])
    print(f"after 2D->3D image.shape: {image.shape}")
    if torch.allclose(image, original_image):
        print("The reconstructed image is approximately the same as the original image.")
    else:
        print("The reconstructed image does not match the original image.")


def detect_edges(mask, threshold=0.25):
    inverted_mask = 1 - mask
    # Compute the gradients along rows and columns
    gradient_x = torch.gradient(mask, dim=2)[0]
    gradient_y = torch.gradient(mask, dim=1)[0]
    gradient_z = torch.gradient(mask, dim=0)[0]

    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)
    # masked_gradient_magnitude = gradient_magnitude * mask
    masked_gradient_magnitude = gradient_magnitude * inverted_mask

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

def detect_edges2(mask, dilation_layers=0, threshold=0.25):
    inverted_mask = 1 - mask
    # Compute the gradients along rows and columns
    gradient_x = torch.gradient(mask, dim=2)[0]
    gradient_y = torch.gradient(mask, dim=1)[0]
    gradient_z = torch.gradient(mask, dim=0)[0]

    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)
    # masked_gradient_magnitude = gradient_magnitude * mask
    masked_gradient_magnitude = gradient_magnitude * inverted_mask
    edge_mask = (masked_gradient_magnitude > threshold).to(torch.int)
    edge_mask = edge_mask.unsqueeze(0)
    if dilation_layers > 0:
        kernel_size = dilation_layers * 2 + 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
        edge_mask = torch.nn.functional.conv3d(edge_mask.unsqueeze(1).float(), kernel, padding=dilation_layers)
        edge_mask = (edge_mask > 0).to(torch.int)
    edge_mask = edge_mask.squeeze(0).squeeze(0)
    return edge_mask

def split_mask2(mask, dilation_layers):
    # mask: torch.Size([batch, D, H, W])
    three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
    for batch_idx in range(mask.size()[0]):
        unique_elements = torch.unique(mask[batch_idx].flatten())
        for element in unique_elements:
            if element != 0:
                element_mask = (mask[batch_idx] == element).to(torch.int)
                edges = detect_edges2(element_mask, dilation_layers)
                element_mask -= edges
                three_classes_mask[batch_idx][edges == 1] = 1         # edge
                three_classes_mask[batch_idx][element_mask == 1] = 2  # interior

    return three_classes_mask

def t_watermalon():
    save_path = "watermalon.html"
    if WANDB_TRACKING:
        wandb.login(key="12b9b358323faf2af56dc288334e6247c1e8bc63")
        wandb.init(project="seg_unet_3D")
    # Define the size of the 3D grid
    grid_size = (32, 128, 128)

    center1 = np.array([32, 100, 75])  # x=0 is on the edge
    radius1 = 4

    center2 = np.array([15, 95, 35])  # x=0 is on the edge
    radius2 = 5

    x, y, z = np.ogrid[:grid_size[0], :grid_size[1], :grid_size[2]]

    squared_distances1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 + (z - center1[2]) ** 2
    squared_distances2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 + (z - center2[2]) ** 2

    ball1 = squared_distances1 <= radius1 ** 2
    ball2 = squared_distances2 <= radius2 ** 2
    ball_array1 = ball1.astype(int)
    ball_array2 = ball2.astype(int)
    image = ball_array1 | ball_array2
    print(image.shape)

    x, y, z = np.indices(image.shape)
    x, y, z = x[image > 0], y[image > 0], z[image > 0]
    values = image[image > 0]

    # Get unique class labels
    unique_classes = np.unique(values)

    # Generate a color for each unique class
    class_colors = {
        cls: f'rgb({0}, {0}, {0})'
        for cls in unique_classes
    }

    # Assign colors based on class labels
    colors = np.array([class_colors[val] for val in values])

    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            mode='markers',
            marker=dict(
                size=5,
                color=colors,  # Color by class value
                opacity=0.2
            )
        )
    ])

    # Update layout for better visualization
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=[0, image.shape[0]]),
        yaxis=dict(nticks=4, range=[0, image.shape[1]]),
        zaxis=dict(nticks=4, range=[0, image.shape[2]]),
        aspectratio=dict(x=1, y=1, z=1)
    ))

    # Save the figure to a temporary file
    fig.write_html(save_path)

    # Optionally log to Weights & Biases
    if WANDB_TRACKING:
        table = wandb.Table(columns=["plotly_figure"])
        table.add_data(wandb.Html(save_path))
        wandb.log({os.path.basename(save_path): table})


if __name__ == "__main__":
    seg_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG/man_seg070.tif"
    seg_path2 = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG/man_seg071.tif"
    # tra_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/TRA/man_track070.tif"
    # voxels_type2 = torch.from_numpy(tiff.imread(mask_path).astype(np.float32)).to(DEVICE).unsqueeze(0)  # every cell have different value
    seg_img = torch.from_numpy(tiff.imread(seg_path).astype(np.float32)).to(DEVICE).unsqueeze(0)
    seg_img2 = torch.from_numpy(tiff.imread(seg_path2).astype(np.float32)).to(DEVICE).unsqueeze(0)
    # seg_img = torch.cat((seg_img, seg_img2), 0)
    print(seg_img.shape)
    d = seg_img.shape[-3]
    edge_image1 = split_mask(seg_img).cpu().numpy()

    edge_image2 = split_mask2(seg_img, 1).cpu().numpy()

    ffig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(seg_img[0, d//2].cpu(), cmap='gray')
    axs[0].set_title('Middle Slice of Original image')
    axs[0].axis('off')  # Hide the axes
    axs[2].imshow(edge_image2[0, d//2], cmap='gray')
    axs[2].set_title('Middle Slice  3 classes')
    axs[2].axis('off')  # Hide the axes
    axs[1].imshow((edge_image2[0, d//2] == 1), cmap='gray')
    axs[1].set_title('Middle Slice  Edges')
    axs[1].axis('off')  # Hide the axes

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"middle_slices.png")
    # image = (edge_image1[0, d//2] == 0) & ~(edge_image2[0, d//2] == 0)
    # plt.imshow(image, cmap='gray')
    # plt.show()

    # t_watermalon()

    # save_shrink_images_to_new_dir()
    # t_save_shrink_images_to_new_dir(False)
#     t_inference()
#     t_2_to_3_and_back()











