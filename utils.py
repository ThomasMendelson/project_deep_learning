import os
import torch
import wandb
import skfmm
from scipy.spatial import KDTree
import tifffile as tiff
import time
from PIL import Image
import numpy as np
from scipy.ndimage import label, binary_erosion
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from dataset import Dataset
from dataset3D import Dataset3D
from torch.utils.data import DataLoader
import plotly.graph_objects as go


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    try:
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        print("=> Checkpoint loaded successfully")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{checkpoint}'")
    except Exception as e:
        print(f"Error: Failed to load checkpoint - {str(e)}")


def custom_collate_fn(batch):
    # batch is a list of tuples (images, seg_masks, tra_masks)
    # images and masks are of shape (in_batch, C, H, W)

    images = torch.cat([item[0] for item in batch], dim=0)  # Concatenate along the first dimension
    seg_masks = torch.cat([item[1] for item in batch], dim=0)  # Concatenate along the first dimension
    tra_masks = torch.cat([item[2] for item in batch], dim=0)  # Concatenate along the first dimension

    return images, seg_masks, tra_masks


def get_loader(dir, seg_dir, tra_dir, train_aug, shuffle, batch_size, crop_size,
               device, num_workers=0, pin_memory=True, three_d=False, ):
    if not three_d:
        ds = Dataset(
            image_dir=dir,
            seg_dir=seg_dir,
            tra_dir=tra_dir,
            crop_size=crop_size,
            train_aug=train_aug
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )
    else:
        ds = Dataset3D(
            image_dir=dir,
            seg_dir=seg_dir,
            tra_dir=tra_dir,
            crop_size=crop_size,
            train_aug=train_aug,
            device=device,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=True,
        )

    return loader


def get_cell_instances(input_np, marker=False,  three_d=False):
    if three_d:
        strel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    else:
        strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    if marker:
        foreground_mask = input_np.astype(np.uint8)
    else:
        foreground_mask = (input_np == 2).astype(np.uint8)
    labeled, max_num = label(foreground_mask, structure=strel)
    return labeled, max_num


def check_accuracy(loader, model, device="cuda", num_image=None, three_d=False, three_d_by_two_d=False,
                   save_path=None, name=None):
    print("=> Checking accuracy")
    seg = 0
    num_iters = 0
    model.eval()
    with torch.no_grad():
        for data, class_targets, marker_targets in loader:
            if three_d and three_d_by_two_d:
                depth = data.shape[-3]
                batch_size = data.shape[0]
                data = three_d_to_two_d_represantation(data)
            class_predictions, marker_predictions = model(data.to(device))
            if three_d and three_d_by_two_d:
                class_predictions = two_d_to_three_d_represantation(images=class_predictions,
                                                                              batch_size=batch_size, depth=depth)
                marker_predictions = two_d_to_three_d_represantation(images=marker_predictions,
                                                                               batch_size=batch_size,
                                                                               depth=depth)
            predicted_classes = predict_classes(class_predictions)
            marker_predictions = (torch.sigmoid(marker_predictions) > 0.5).int().squeeze(1)

            gt = class_targets.numpy()
            for i in range(predicted_classes.shape[0]):
                if torch.any(class_predictions > 0):
                    pred_labels_mask = inference(predicted_classes[i], marker_predictions[i], three_d=three_d)
                    accuracy, _ = calc_SEG_measure(pred_labels_mask, gt[i])
                    if accuracy != -1:
                        seg += accuracy
                        num_iters += 1
                    if num_image is not None and num_iters == num_image:
                        seg_score = seg / num_iters
                        print(f"seg score for {num_image} image: {seg_score}")
                        model.train()
                        if save_path is not None and (seg_score >= 0.58):
                            checkpoint = {
                                "state_dict": model.state_dict(),
                            }
                            save_checkpoint(checkpoint, filename=f"{save_path}{name}_{seg_score:.4f}.pth.tar")
                        return

    print(f"seg score: {seg / num_iters}")
    model.train()


def check_accuracy_without_markers(loader, model, device="cuda", num_image=None, three_d=False, three_d_by_two_d=False,
                                   save_path=None, name=None):
    print("=> Checking accuracy")
    seg = 0
    num_iters = 0
    model.eval()
    with torch.no_grad():
        for data, class_targets, marker_targets in loader:
            if three_d and three_d_by_two_d:
                depth = data.shape[-3]
                batch_size = data.shape[0]
                data = three_d_to_two_d_represantation(data)
            class_predictions, marker_predictions = model(data.to(device))
            if three_d and three_d_by_two_d:
                class_predictions = two_d_to_three_d_represantation(images=class_predictions,
                                                                              batch_size=batch_size, depth=depth)

            predicted_classes = predict_classes(class_predictions).cpu().numpy()
            gt = class_targets.numpy()
            for i in range(predicted_classes.shape[0]):
                if torch.any(class_predictions > 0):
                    pred_labels_mask, _ = get_cell_instances(predicted_classes[i], three_d=three_d)
                    accuracy, _ = calc_SEG_measure(pred_labels_mask, gt[i])
                    if accuracy != -1:
                        seg += accuracy
                        num_iters += 1
                    if num_image is not None and num_iters == num_image:
                        seg_score = seg / num_iters
                        print(f"seg score for {num_image} image: {seg_score}")
                        model.train()
                        if save_path is not None and (seg_score >= 0.58):
                            checkpoint = {
                                "state_dict": model.state_dict(),
                            }
                            save_checkpoint(checkpoint, filename=f"{save_path}{name}_{seg_score:.4f}.pth.tar")
                        return

    print(f"seg score: {seg / num_iters}")
    model.train()


def predict_classes(preds):
    preds_softmax = F.softmax(preds, dim=1)  # Apply softmax along the class dimension
    _, predicted_classes = torch.max(preds_softmax, dim=1)  # Get the index of the maximum probability
    return predicted_classes


def get_instance_color(image):
    unique_labels = np.unique(image)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background label if present

    # Generate a unique color for each label
    color_map = plt.get_cmap('hsv', len(unique_labels))

    # Create an empty color image
    colored_image = np.zeros((*image.shape, 3), dtype=np.uint8)

    for i, label in enumerate(unique_labels):
        color = (np.array(color_map(i)[:3]) * 255).astype(np.uint8)
        colored_image[image == label] = color

    return colored_image



def save_instance_by_colors(loader, model, folder, wandb_step, device="cuda", three_d=False, three_d_by_two_d=False, wandb_tracking=False):
    print("=> saving instance images")
    model.eval()
    for idx, (data, class_targets, marker_targets) in enumerate(loader):
        if three_d and three_d_by_two_d:
            depth = data.shape[-3]
            batch_size = data.shape[0]
            data = three_d_to_two_d_represantation(data)
        data = data.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            class_predictions, marker_predictions = model(data.to(device=device))
            if three_d and three_d_by_two_d:
                class_predictions = two_d_to_three_d_represantation(images=class_predictions,
                                                                              batch_size=batch_size, depth=depth)
                marker_predictions = two_d_to_three_d_represantation(images=marker_predictions,
                                                                               batch_size=batch_size,depth=depth)
            predicted_classes = predict_classes(class_predictions)
            # marker_predictions = predict_classes(marker_predictions)
            marker_predictions = (torch.sigmoid(marker_predictions) > 0.5).int().squeeze(1)

        predicted = inference(predicted_classes[0], marker_predictions[0], three_d=three_d)
        gt = class_targets[0].cpu().numpy().astype(np.uint8)
        if three_d:
            visualize_3d_image_instances(image=predicted, save_path=f"{folder}/pred_instances.html",
                                         wandb_tracking=wandb_tracking, wandb_step=wandb_step)
            visualize_3d_image_instances(image=gt, save_path=f"{folder}/gt_instances.html",
                                         wandb_tracking=wandb_tracking, wandb_step=wandb_step)
            visualize_3d_image_instances(image=marker_predictions[0].cpu().numpy().astype(np.uint8), save_path=f"{folder}/markers_pred.html",
                                         wandb_tracking=wandb_tracking, wandb_step=wandb_step)

        else:
            colored_instance_preds = Image.fromarray(get_instance_color(predicted))
            colored_instance_gt = Image.fromarray(get_instance_color(gt))
            colored_instance_preds.save(f"{folder}/pred_instances.png")
            colored_instance_gt.save(f"{folder}/gt_instances.png")
            if wandb_tracking:
                wandb.log({"pred_instances": wandb.Image(colored_instance_preds)}, step=wandb_step)
                wandb.log({"gt_instances": wandb.Image(colored_instance_gt)}, step=wandb_step)
        break
F

def save_instance_by_colors_without_markers(loader, model, folder, wandb_step, device="cuda", three_d=False, three_d_by_two_d=False, wandb_tracking=False):
    print("=> saving instance images")
    model.eval()
    for idx, (data, class_targets, marker_targets) in enumerate(loader):
        if three_d and three_d_by_two_d:
            depth = data.shape[-3]
            batch_size = data.shape[0]
            data = three_d_to_two_d_represantation(data)
        data = data.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            class_predictions, marker_predictions = model(data.to(device=device))
            if three_d and three_d_by_two_d:
                class_predictions = two_d_to_three_d_represantation(images=class_predictions,
                                                                              batch_size=batch_size, depth=depth)

            predicted_classes = predict_classes(class_predictions).cpu().numpy()

        predicted, _ = get_cell_instances(predicted_classes[0], three_d=three_d)
        gt = class_targets[0].cpu().numpy().astype(np.uint8)
        if three_d:
            visualize_3d_image_instances(image=predicted, save_path=f"{folder}/pred_instances.html",
                                         wandb_tracking=wandb_tracking, wandb_step=wandb_step)
            visualize_3d_image_instances(image=gt, save_path=f"{folder}/gt_instances.html",
                                         wandb_tracking=wandb_tracking, wandb_step=wandb_step)

        else:
            colored_instance_preds = Image.fromarray(get_instance_color(predicted))
            colored_instance_gt = Image.fromarray(get_instance_color(gt))
            colored_instance_preds.save(f"{folder}/pred_instances.png")
            colored_instance_gt.save(f"{folder}/gt_instances.png")
            if wandb_tracking:
                wandb.log({"pred_instances": wandb.Image(colored_instance_preds)}, step=wandb_step)
                wandb.log({"gt_instances": wandb.Image(colored_instance_gt)}, step=wandb_step)
        break


def visualize_3d_image_instances(image, save_path, wandb_step, wandb_tracking=False):
    # Get indices and values where the voxel value is not zero
    x, y, z = np.indices(image.shape)
    x, y, z = x[image > 0], y[image > 0], z[image > 0]
    values = image[image > 0]

    # Get unique class labels
    unique_classes = np.unique(values)

    # Generate a color for each unique class
    class_colors = {
        cls: f'rgb({int(cls * 80) % 256}, {int(cls * 50) % 256}, {int(cls * 100) % 256})'
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
    if wandb_tracking:
        table = wandb.Table(columns=["plotly_figure"])
        table.add_data(wandb.Html(save_path))
        wandb.log({os.path.basename(save_path): table}, step=wandb_step)


def separate_masks(instance_mask):
    unique_labels = np.unique(instance_mask)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background label if present

    binary_masks = []

    for label in unique_labels:
        binary_mask = np.zeros_like(instance_mask)
        binary_mask[instance_mask == label] = 1
        binary_masks.append(binary_mask)

    return binary_masks


def calc_SEG_measure(pred_labels_mask, gt_labels_mask):
    # separete labels masks to binary masks
    binary_masks_predicted = separate_masks(pred_labels_mask)
    binary_masks_gt = separate_masks(gt_labels_mask)

    len_binary_masks_gt = len(binary_masks_gt)
    if len_binary_masks_gt == 0:
        return -1, np.zeros(len_binary_masks_gt)
    SEG_measure_array = np.zeros(len_binary_masks_gt)
    for i, r in enumerate(binary_masks_gt):
        r_or_s = None
        r_and_s = None
        # find match |R and S| > 0.5|R|
        for s in binary_masks_predicted:
            r_and_s = np.logical_and(r, s)
            if np.sum(r_and_s) > 0.5 * np.sum(r):
                # match !
                r_or_s = np.logical_or(r, s)
                break

        # calc Jaccard similarity index
        if r_or_s is not None:
            j_similarity = np.sum(r_and_s) / np.sum(r_or_s)
            SEG_measure_array[i] = j_similarity
            if np.isnan(j_similarity):
                print(f"np.sum(r_and_s): {np.sum(r_and_s)}, np.sum(r_or_s): {np.sum(r_or_s)}")

    if np.any(np.isnan(SEG_measure_array)):
        print("SEG_measure_array contains NaN values")
    SEG_measure_avg = np.mean(SEG_measure_array)
    return SEG_measure_avg, SEG_measure_array

def plot_middle_slices_3d_with_gt(pred_labels_mask, gt, num_slices=5):
    depth = pred_labels_mask.shape[0]
    start_idx = (depth - num_slices) // 2
    end_idx = start_idx + num_slices
    middle_slices_pred = pred_labels_mask[start_idx:end_idx]
    middle_slices_gt = gt[start_idx:end_idx]
    fig, axes = plt.subplots(2, num_slices, figsize=(15, 5))
    for j in range(num_slices):
        axes[0, j].imshow(middle_slices_pred[j], cmap='gray')
        axes[0, j].set_title(f'Slice {start_idx + j}')
        axes[0, j].axis('off')

        axes[1, j].imshow(middle_slices_gt[j], cmap='gray')
        axes[1, j].set_title(f'GT Slice {start_idx + j}')
        axes[1, j].axis('off')
    print("ploting")
    plt.savefig(f"{time.time()}.png")
    plt.close(fig)
    plt.show()

def save_slices(loader, model, device, num_slices=5, three_d=False):
    print("=> Saving slices")
    loader = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for data, class_targets, marker_targets in loader:
            class_predictions, marker_predictions = model(data.to(device))
            predicted_classes = predict_classes(class_predictions)
            marker_predictions = (torch.sigmoid(marker_predictions) > 0.5).int().squeeze(1)

            gt = class_targets.numpy()
            for i in range(predicted_classes.shape[0]):

                # pred_labels_mask = inference(predicted_classes[i], marker_predictions[i], three_d=three_d)
                # plot_middle_slices_3d_with_gt(pred_labels_mask, gt[i], num_slices=5)

                pred_labels_mask = predicted_classes[i].cpu().numpy()
                pred_labels_mask = (pred_labels_mask == 2).astype(np.uint8)
                plot_middle_slices_3d_with_gt(pred_labels_mask, gt[i], num_slices=5)
                plot_middle_slices_3d_with_gt(marker_predictions[i].cpu().numpy(), gt[i], num_slices=5)

            break



def inference(class_predictions, marker_predictions, three_d=False):
    def find_closest_marker_fmm(foreground, labeled_markers, three_d):
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
        if np.any((min_distances == np.inf) & foreground):
            mask_inf_and_foreground = (min_distances == np.inf) & foreground
            labeled_preds, _ = get_cell_instances(mask_inf_and_foreground, marker=True, three_d=three_d)
            labeled_preds = np.where(labeled_preds != 0, labeled_preds + len(unique_markers), labeled_preds)
            closest_marker_map = np.where(labeled_preds != 0, labeled_preds, closest_marker_map)

        return closest_marker_map

    # predicted_classes = predict_classes(class_predictions).cpu().numpy()
    predicted_classes = class_predictions.cpu().numpy()
    predicted_foregound = (predicted_classes == 2).astype(np.uint8)
    labeled_markers, _ = get_cell_instances(marker_predictions.cpu().numpy(), marker=True, three_d=three_d)

    labeled_preds = find_closest_marker_fmm(predicted_foregound, labeled_markers, three_d)
    return labeled_preds

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

def shrink_cells(input, num_layers_to_shrink, three_d):
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
    # Convert PyTorch tensor to NumPy array
    images = input.numpy()

    # Create an empty array to store the shrunk cells
    shrunk_array = np.zeros_like(images)
    for i in range(images.shape[0]):
        # Get unique cell values (excluding background, assumed to be 0)
        cell_values = np.unique(images[i])
        cell_values = cell_values[cell_values != 0]

        # Perform erosion on each cell separately
        for value in cell_values:
            cell_mask = (images[i] == value)
            eroded_mask = binary_erosion(cell_mask, structure=strel, iterations=num_layers_to_shrink)
            shrunk_array[i][eroded_mask] = value

    # Convert back to PyTorch tensor
    shrunk_tensor = torch.from_numpy(shrunk_array)

    return shrunk_tensor


def save_images_to_check_accuracy(loader, model, save_path, device, three_d, three_d_by_two_d):
    print("=> Checking accuracy")
    counter = 0
    loader = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for data, class_targets, marker_targets in loader:
            if three_d and three_d_by_two_d:
                depth = data.shape[-3]
                batch_size = data.shape[0]
                data = three_d_to_two_d_represantation(data)
            class_predictions, marker_predictions = model(data.to(device))
            if three_d and three_d_by_two_d:
                class_predictions = two_d_to_three_d_represantation(images=class_predictions,
                                                                    batch_size=batch_size, depth=depth)
                marker_predictions = two_d_to_three_d_represantation(images=marker_predictions,
                                                                     batch_size=batch_size,
                                                                     depth=depth)
            predicted_classes = predict_classes(class_predictions)
            # marker_predictions = predict_classes(marker_predictions)
            marker_predictions = (torch.sigmoid(marker_predictions) > 0.5).int().squeeze(1)

            gt = class_targets.numpy().astype(np.uint16)
            for i in range(predicted_classes.shape[0]):
                if torch.any(class_predictions > 0):
                    pred_labels_mask = inference(predicted_classes[i], marker_predictions[i], three_d=three_d).astype(np.uint16)
                    tiff.imwrite(f"{save_path}01_RES/mask{counter:03}.tif", pred_labels_mask, dtype=np.uint16)
                    tiff.imwrite(f"{save_path}01_GT/SEG/man_seg{counter:03}.tif", gt[i], dtype=np.uint16)
                    counter +=1


################################### for tests ####################################################

def t_inference():
    markers = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])

    pred = np.array([[0, 1, 1, 0, 0],
                     [2, 2, 1, 2, 0],
                     [1, 2, 1, 1, 0],
                     [1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1]],)

    pred, markers = torch.from_numpy(pred), torch.from_numpy(markers)
    image = inference(pred, markers, False)
    print("result", image)
    # plt.imshow(image, cmap='gray')  # Use 'gray' colormap for grayscale images
    # plt.axis('off')  # Turn off axis labels
    # plt.savefig('output.png')  # Save the plot as an image file
    # plt.close()  # Close the figure to release memory

def t_shrink_cells():

    image = shrink_cells(torch.ones((2, 3, 12, 12)), 1, True)
    print(image)

if __name__ == "__main__":
    t_inference()
    # t_shrink_cells()
