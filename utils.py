import os
import torch
import wandb
import skfmm
from scipy.spatial import KDTree
import time
import torchvision
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
from dataset import Dataset
from dataset3D import Dataset3D
from torch.utils.data import DataLoader
import plotly.graph_objects as go


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# def load_checkpoint(checkpoint, model):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])
def load_checkpoint(checkpoint, model):
    try:
        print("=> Loading checkpoint")
        # checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint["state_dict"])
        print("=> Checkpoint loaded successfully")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{checkpoint}'")
    except Exception as e:
        print(f"Error: Failed to load checkpoint - {str(e)}")


def custom_collate_fn(batch):
    # batch is a list of tuples (images, masks)
    # images and masks are of shape (in_batch, C, H, W)

    images = torch.cat([item[0] for item in batch], dim=0)  # Concatenate along the first dimension
    masks = torch.cat([item[1] for item in batch], dim=0)  # Concatenate along the first dimension

    return images, masks


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
        foreground_mask = input_np
    else:
        foreground_mask = (input_np == 2).astype(np.uint8)
    labeled, max_num = ndimage.label(foreground_mask, structure=strel)
    return labeled, max_num


def check_accuracy(loader, model, device="cuda", one_image=False, three_d=False):
    print("=> Checking accuracy")
    loader = tqdm(loader)
    seg = 0
    num_iters = 0
    model.eval()
    with torch.no_grad():
        for data, class_targets, marker_targets in loader:
            class_predictions, marker_predictions = model(data.to(device))
            predicted_classes = predict_classes(class_predictions)
            marker_predictions = predict_classes(marker_predictions)

            gt = class_targets.numpy()
            for i in range(predicted_classes.shape[0]):
                pred_labels_mask = inference(predicted_classes[i], marker_predictions[i], three_d=three_d)
                accuracy, _ = calc_SEG_measure(pred_labels_mask, gt[i])
                if accuracy != -1:
                    seg += accuracy
                    num_iters += 1
                if one_image and num_iters == 1:
                    print(f"seg score for one image: {seg / num_iters}")
                    model.train()
                    return

    print(f"seg score: {seg / num_iters}")
    model.train()


def check_accuracy_multy_models(loader, models, device="cuda", one_image=False, three_d=False):
    print("=> Checking accuracy")
    loader = tqdm(loader)
    seg = 0
    num_iters = 0
    for model in models:
        model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            model_preds = [model(x) for model in models]
            preds = torch.mean(torch.stack(model_preds), dim=0)
            predicted_classes = predict_classes(preds).cpu().numpy()  # predict the class 0/1/2
            gt = y.numpy()
            for i in range(predicted_classes.shape[0]):
                pred_labels_mask, _ = get_cell_instances(predicted_classes[i], three_d=three_d)
                accuracy, _ = calc_SEG_measure(pred_labels_mask, gt[i])
                if accuracy != -1:
                    seg += accuracy
                    num_iters += 1

                if one_image and num_iters == 1:
                    print(f"seg score for avg models and one image: {seg / num_iters}")
                    for model in models:
                        model.train()
                    return

    print(f"seg score for avg models : {seg / num_iters}")
    for model in models:
        model.train()


def apply_color_map(input_tensor, three_d=False):
    if three_d:
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if single image
    else:
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if single image

    # Create a color map tensor based on input values
    color_map = torch.tensor([
        [0, 0, 0],  # Black for value 0
        [0, 255, 0],  # Green for value 1
        [255, 255, 255]  # White for value 2
    ], dtype=torch.float, device=input_tensor.device)  # Use input_tensor's device

    # Index the color_map tensor with input_tensor to assign colors
    output_tensor = color_map[input_tensor]
    if three_d:
        return output_tensor.permute(0, 4, 1, 2, 3)
    return output_tensor.permute(0, 3, 1, 2)


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


def save_predictions_as_imgs(loader, model, folder, device="cuda", three_d=False, wandb_tracking=False):
    print("=> saving images")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device)
        with torch.no_grad():
            preds = model(x)
            predicted_classes = predict_classes(preds)

        for i in range(predicted_classes.shape[0]):  # Loop through the batch
            if three_d and idx == 0:
                visualize_3d_image_from_classes(image=predicted_classes[i], save_path=f"{folder}/pred_{idx}_{i}.html",
                                                wandb_tracking=wandb_tracking)
                visualize_3d_image_from_classes(image=y[i], save_path=f"{folder}/gt_{idx}_{i}.html",
                                                wandb_tracking=wandb_tracking)
                # pred_img = colored_preds[i].permute(1, 2, 3, 0).cpu().numpy()
                # gt_img = colored_gt[i].permute(1, 2, 3, 0).cpu().numpy()
                # middle_slice = pred_img.shape[0]//2
                # pred_img = pred_img[middle_slice]
                # gt_img = gt_img[middle_slice]

            else:
                colored_preds = apply_color_map(predicted_classes, three_d=three_d).type(torch.uint8)
                colored_gt = apply_color_map(Dataset.split_mask(y).long(), three_d=three_d).type(torch.uint8)

                pred_img = colored_preds[i].permute(1, 2, 0).cpu().numpy()
                gt_img = colored_gt[i].permute(1, 2, 0).cpu().numpy()
                separator_line = np.ones((pred_img.shape[0], 5, 3), dtype=np.uint8) * 255  # Red line
                concatenated_img = np.concatenate([pred_img, separator_line, gt_img], axis=1)

                # Convert to PIL Image
                concatenated_img_pil = Image.fromarray(concatenated_img)

                # Save the concatenated image
                concatenated_img_pil.save(f"{folder}/pred_gt_{idx}_{i}.png")

        if three_d:
            break

    model.train()


def visualize_3d_image_from_classes(image, save_path, wandb_tracking=False):
    # if on GPU
    image_np = image.cpu().numpy()

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
    if wandb_tracking:
        table = wandb.Table(columns=["plotly_figure"])
        table.add_data(wandb.Html(save_path))
        # Log the image to wandb
        wandb.log({os.path.basename(save_path): table})


def save_instance_by_colors(loader, model, folder, device="cuda", three_d=False, wandb_tracking=False):
    print("=> saving instance images")
    model.eval()
    for idx, (data, class_targets, marker_targets) in enumerate(loader):
        data = data.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            class_predictions, marker_predictions = model(data)
            predicted_classes = predict_classes(class_predictions)
            marker_predictions = predict_classes(marker_predictions)

        predicted = inference(predicted_classes[0], marker_predictions[0], three_d=three_d)
        gt = class_targets[0].cpu().numpy().astype(np.uint8)
        if three_d:
            visualize_3d_image_instances(image=predicted, save_path=f"{folder}/pred_instances.html",
                                         wandb_tracking=wandb_tracking)
            visualize_3d_image_instances(image=gt, save_path=f"{folder}/gt_instances.html",
                                         wandb_tracking=wandb_tracking)
        else:
            colored_instance_preds = Image.fromarray(get_instance_color(predicted))
            colored_instance_gt = Image.fromarray(get_instance_color(gt))
            colored_instance_preds.save(f"{folder}/pred_instances.png")
            colored_instance_gt.save(f"{folder}/gt_instances.png")
            wandb.log({"pred_instances": wandb.Image(colored_instance_preds)})
            wandb.log({"gt_instances": wandb.Image(colored_instance_gt)})
        break


def visualize_3d_image_instances(image, save_path, wandb_tracking=False):
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
        wandb.log({os.path.basename(save_path): table})


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
    labeled_markers, _ = get_cell_instances(marker_predictions.cpu().numpy(), marker=True, three_d=three_d)

    labeled_preds = find_closest_marker_fmm(predicted_foregound, labeled_markers)
    return labeled_preds


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
                     [2, 2, 2, 2, 1]],)



    pred, markers = torch.from_numpy(pred), torch.from_numpy(markers)
    image = inference(pred, markers, False)
    print("result", image)
    # plt.imshow(image, cmap='gray')  # Use 'gray' colormap for grayscale images
    # plt.axis('off')  # Turn off axis labels
    # plt.savefig('output.png')  # Save the plot as an image file
    # plt.close()  # Close the figure to release memory


if __name__ == "__main__":
    t_inference()
