import os

import torch
import torchvision
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
from dataset import Fluo_N2DH_SIM_PLUS
from torch.utils.data import DataLoader


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

def get_loader(
        dir,
        maskdir,
        train_aug,
        shuffle,
        batch_size,
        crop_size,
        num_workers=0,
        pin_memory=True
):
    ds = Fluo_N2DH_SIM_PLUS(
        image_dir=dir,
        mask_dir=maskdir,
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

    return loader


def get_cell_instances(input_np):
    strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    foreground_mask = (input_np == 2).astype(np.uint8)
    labeled, max_num = ndimage.label(foreground_mask, structure=strel)
    return labeled  # , np.array(max_num).astype(np.float32)


def check_accuracy(loader, model, device="cuda", one_image=False):
    print("=> Checking accuracy")
    loader = tqdm(loader)
    seg = 0
    num_iters = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device))
            predicted_classes = predict_classes(preds).cpu().numpy()  # predict the class 0/1/2
            gt = y.numpy()
            for i in range(predicted_classes.shape[0]):
                pred_labels_mask = get_cell_instances(predicted_classes[i])
                accuracy, _ = calc_SEG_measure(pred_labels_mask, gt[i])
                seg += accuracy
                num_iters += 1
                if one_image:
                    print(f"seg score for one image: {seg / num_iters}")
                    model.train()
                    return

    print(f"seg score: {seg / num_iters}")
    model.train()

def check_accuracy_multy_models(loader, models, device="cuda", one_image=False):
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
                pred_labels_mask = get_cell_instances(predicted_classes[i])
                accuracy, _ = calc_SEG_measure(pred_labels_mask, gt[i])
                seg += accuracy
                num_iters += 1
                if one_image:
                    print(f"seg score for avg models and one image: {seg / num_iters}")
                    for model in models:
                        model.train()
                    return

    print(f"seg score for avg models : {seg / num_iters}")
    for model in models:
        model.train()
def apply_color_map(input_tensor):
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

    return output_tensor.permute(0, 3, 1, 2)


def predict_classes(preds):
    preds_softmax = F.softmax(preds, dim=1)  # Apply softmax along the class dimension
    _, predicted_classes = torch.max(preds_softmax, dim=1)  # Get the index of the maximum probability
    return predicted_classes


def get_instance_color(image):
    unique_labels = np.unique(image)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background label if present

    # Generate a unique color for each label
    color_map = plt.cm.get_cmap('hsv', len(unique_labels))

    # Create an empty color image
    colored_image = np.zeros((*image.shape, 3), dtype=np.uint8)

    for i, label in enumerate(unique_labels):
        color = (np.array(color_map(i)[:3]) * 255).astype(np.uint8)
        colored_image[image == label] = color

    return colored_image



def save_instance_by_colors(loader, model, folder, device="cuda"):
    print("=> saving instance images")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            preds = model(x)
            predicted_classes = predict_classes(preds).cpu().numpy()
        labeled_preds =get_cell_instances(predicted_classes[0])
        gt = y[0].cpu().numpy().astype(np.uint8)
        colored_instance_preds = Image.fromarray(get_instance_color(labeled_preds))
        colored_instance_gt = Image.fromarray(get_instance_color(gt))

        colored_instance_preds.save(f"{folder}/pred_instances.png")
        colored_instance_gt.save(f"{folder}/gt_instances.png")
        break

def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    print("=> saving images")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device)
        with torch.no_grad():
            preds = model(x)
            predicted_classes = predict_classes(preds)

        colored_preds = apply_color_map(predicted_classes).type(torch.uint8)

        colored_gt = apply_color_map(Fluo_N2DH_SIM_PLUS.split_mask(y).long()).type(torch.uint8)

        for i in range(colored_preds.shape[0]):  # Loop through the batch
            # Permute and move to CPU
            pred_img = colored_preds[i].permute(1, 2, 0).cpu().numpy()
            gt_img = colored_gt[i].permute(1, 2, 0).cpu().numpy()
            separator_line = np.ones((pred_img.shape[0], 5, 3), dtype=np.uint8) * 255  # Red line
            concatenated_img = np.concatenate([pred_img, separator_line, gt_img], axis=1)

            # Convert to PIL Image
            concatenated_img_pil = Image.fromarray(concatenated_img)

            # Save the concatenated image
            concatenated_img_pil.save(f"{folder}/pred_gt_{idx}_{i}.png")

    model.train()


def save_test_predictions_as_imgs(
        loader, model, folder=r"C:\BGU\seg_unet\Fluo-N2DH-SIM+_training-datasets\saved", device="cuda", type="train"
):
    model.eval()
    if type == "train":

        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
            torchvision.utils.save_image(Fluo_N2DH_SIM_PLUS.split_mask(y), f"{folder}/{idx}.png")

    else:
        for idx, x in enumerate(loader):
            x = x.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
    model.train()

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

    SEG_measure_array = np.zeros(len(binary_masks_gt))
    if not binary_masks_predicted:
        return 0, SEG_measure_array
    for i, r in enumerate(binary_masks_gt):
        # find match |R and S| > 0.5|R|
        for s in binary_masks_predicted:
            r_and_s = np.logical_and(r, s)
            if np.sum(r_and_s) > 0.5 * np.sum(r):
                # match !
                break

        # calc Jaccard similarity index
        j_similarity = np.sum(r_and_s) / np.sum(np.logical_or(r, s))
        SEG_measure_array[i] = j_similarity

    SEG_measure_avg = np.average(SEG_measure_array)
    return SEG_measure_avg, SEG_measure_array

# shakeds:
# def check_accuracy(loader, model, device="cuda"):
#     three_d = False
#     channel_axis = 4 if not three_d else 5
#     calc_seg_meas = seg_measure(channel_axis=channel_axis, three_d=three_d,
#                                 foreground_class_index=2)
#     accuracy = 0
#     num_iters = 0
#     model.eval()
#     with torch.no_grad():
#         for x, y in loader:
#             # preds = model(x.to(device))
#             # preds = preds.unsqueeze(1).permute(0,1,3,4,2).to(device)

#             preds = apply_color_map(Fluo_N2DH_SIM_PLUS.split_mask(y).long())
#             preds = preds.unsqueeze(1).permute(0,1,3,4,2).to(device)

#             y = y.unsqueeze(1).unsqueeze(4).to(device)
#             print(f"preds.size: {preds.size()}, gt.size: {y.size()}")
#             accuracy += calc_seg_meas(y,preds)
#             num_iters += 1
#             print(accuracy)
#     print(f"seg score: {accuracy/num_iters}")
#     model.train()

# def seg_measure(channel_axis, three_d=False, foreground_class_index=2):
#     if not three_d:
#         strel = np.zeros([3, 3])
#         strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
#     else:
#         strel = np.zeros([3, 3, 3, 3, 3])
#         strel[1][1] = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
#                                 [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

#     def connected_components(input_np):
#         labeled = np.zeros_like(input_np, dtype=np.uint16)
#         max_num = 0
#         for d1, images in enumerate(input_np):
#             for d2, image in enumerate(images):
#                 labeled_image, max_num_temp = ndimage.label(image, structure=strel)
#                 labeled[d1, d2] = labeled_image
#                 max_num = np.maximum(max_num, max_num_temp)

#         return labeled, np.array(max_num).astype(np.float32)

#     def seg_numpy(gt, seg):
#         gt_labeled, _ = connected_components(gt)
#         seg_labeled, _ = connected_components(seg)
#         all_iou = []
#         for gt1, seg1 in zip(gt_labeled, seg_labeled):
#             for gt, seg in zip(gt1, seg1):
#                 for this_label in np.unique(gt):
#                     if this_label == 0:
#                         continue
#                     all_iou.append(0.)
#                     bw = gt == this_label
#                     l_area = np.sum(bw).astype(np.float32)
#                     overlaping_inds = seg[bw]
#                     for s in np.unique(overlaping_inds):
#                         if s == 0:
#                             continue
#                         intersection = np.sum(overlaping_inds == s).astype(np.float32)
#                         overlap = intersection / l_area
#                         if overlap > 0.5:
#                             s_area = np.sum(seg == s).astype(np.float32)
#                             iou = intersection / (l_area + s_area - intersection)
#                             all_iou[-1] = iou
#         if not len(all_iou):
#             return np.nan
#         return np.mean(all_iou)


#     def calc_seg(gt_sequence, output_sequence):
#         gt_sequence = torch.squeeze(gt_sequence, dim=channel_axis)
#         gt_valid = gt_sequence > -1
#         gt_sequence = gt_sequence.float() * gt_valid.float()
#         gt_fg = (gt_sequence == foreground_class_index).float()
#         output_classes = torch.argmax(output_sequence, dim=channel_axis)
#         output_foreground = (output_classes == foreground_class_index).float()

#         gt_fg_np = gt_fg.cpu().numpy() if gt_fg.is_cuda else gt_fg.numpy()
#         output_foreground_np = output_foreground.cpu().numpy() if output_foreground.is_cuda else output_foreground.numpy()

#         seg_measure_value = seg_numpy(gt_fg_np, output_foreground_np)

#         return seg_measure_value

#     return calc_seg


