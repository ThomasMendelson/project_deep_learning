import os
import cv2
import random
import torch
import torchio as tio
from PIL import Image
import tifffile as tiff
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np


# import albumentations as A
# from albumentations.pytorch import ToTensorV2


class Dataset3D(Dataset):
    def __init__(self, image_dir, crop_size, mask_dir=None, train_aug=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train_aug = train_aug
        self.images = os.listdir(image_dir)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.mask_dir is not None:
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index].replace("t", "man_seg", 1))
            image = tiff.imread(img_path).astype(np.float32)
            image = (image - image.mean()) / (image.std())

            mask = tiff.imread(mask_path).astype(np.float32)

            transform = self.get_transform()
            image, mask = transform(image=image, mask=mask)

            return image, mask

    @staticmethod
    def detect_edges(mask, threshold=0.25):
        # Compute the gradients along rows and columns
        gradient_x = torch.gradient(mask, dim=2)
        gradient_y = torch.gradient(mask, dim=1)
        gradient_z = torch.gradient(mask, dim=0)

        # Extract gradient components from the tuple
        gradient_x = gradient_x[0]
        gradient_y = gradient_y[0]
        gradient_z = gradient_z[0]

        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)
        masked_gradient_magnitude = gradient_magnitude * mask
        edge_mask = (masked_gradient_magnitude > threshold).to(torch.int)

        return edge_mask

    @staticmethod
    def split_mask(mask):
        # mask: torch.Size([batch, 5, 256, 256])
        three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
        for batch_idx in range(mask.size()[0]):
            unique_elements = torch.unique(mask[batch_idx].flatten())
            for element in unique_elements:
                if element != 0:
                    element_mask = (mask[batch_idx] == element).to(torch.int)
                    edges = Dataset3D.detect_edges(element_mask)
                    element_mask -= edges
                    three_classes_mask[batch_idx][edges == 1] = 1         # edge
                    three_classes_mask[batch_idx][element_mask == 1] = 2  # interior

        return three_classes_mask

    def get_transform(self):
        def affine(image, mask, p=0.5, max_degrees=15, max_scale=0.2):
            if random.random() < p:
                scales = []
                degrees = []
                for i in range(3):
                    degree = random.uniform(-max_degrees, max_degrees)
                    scale = random.uniform(-max_scale, max_scale) + 1
                    scales.append(scale)
                    scales.append(scale)
                    degrees.append(degree)
                    degrees.append(degree)
                transforms = tio.Compose([
                    tio.RandomAffine(
                        scales=tuple(scales),
                        degrees=tuple(degrees),
                        default_pad_value=0,
                    ),
                ])
                image = transforms(image)
                mask = transforms(mask)
            return image, mask

        def horizontal_flip(image, mask, p=0.5):
            if random.random() < p:
                transforms = tio.Compose([
                    tio.RandomFlip(axes=2, flip_probability=1),  # Horizontal
                ])
                image = transforms(image)
                mask = transforms(mask)
            return image, mask

        def vertical_flip(image, mask, p=0.5):
            if random.random() < p:
                transforms = tio.Compose([
                    tio.RandomFlip(axes=1, flip_probability=1),  # Horizontal
                ])
                image = transforms(image)
                mask = transforms(mask)
            return image, mask

        def depth_flip(image, mask, p=0.5):
            if random.random() < p:
                transforms = tio.Compose([
                    tio.RandomFlip(axes=0, flip_probability=1),  # Horizontal
                ])
                image = transforms(image)
                mask = transforms(mask)
            return image, mask

        def random_crop(image, mask, crop_size, threshold=500):
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
                cropped_mask = mask[:, start_d:start_d + crop_depth, start_h:start_h + crop_height,
                               start_w:start_w + crop_width]
                if torch.sum(cropped_image > 0) > threshold * crop_depth:
                    break

            return cropped_image, cropped_mask

        def to_tensor(image, mask):
            image, mask = torch.from_numpy(image), torch.from_numpy(mask)
            return image.unsqueeze(0), mask.unsqueeze(0)

        def transform(image, mask):
            image, mask = to_tensor(image, mask)
            if self.train_aug:
                image, mask = affine(image, mask, p=0.5, max_degrees=15, max_scale=0.2)
                image, mask = horizontal_flip(image, mask, p=0.5)
                image, mask = vertical_flip(image, mask, p=0.5)
                image, mask = depth_flip(image, mask, p=0.5)
                image, mask = random_crop(image, mask, crop_size=self.crop_size)
            return image, mask.squeeze(0)

        return transform


def get_transform(train_aug):
    def affine(image, mask, p=0.5, max_degrees=15, max_scale=0.2):
        if random.random() < p:
            scales = []
            degrees = []
            for i in range(3):
                degree = random.uniform(-max_degrees, max_degrees)
                scale = random.uniform(-max_scale, max_scale) + 1
                scales.append(scale)
                scales.append(scale)
                degrees.append(degree)
                degrees.append(degree)
            transforms = tio.Compose([
                tio.RandomAffine(
                    scales=tuple(scales),
                    degrees=tuple(degrees),
                    default_pad_value=0,
                ),
            ])
            image = transforms(image)
            mask = transforms(mask)
            # print(f"degrees: {degrees[1::2]}")
            # print(f"scales: {scales[1::2]}")
        return image, mask

    def horizontal_flip(image, mask, p=0.5):
        if random.random() < p:
            transforms = tio.Compose([
                tio.RandomFlip(axes=2, flip_probability=1),  # Horizontal
            ])
            image = transforms(image)
            mask = transforms(mask)
        return image, mask

    def vertical_flip(image, mask, p=0.5):
        if random.random() < p:
            transforms = tio.Compose([
                tio.RandomFlip(axes=1, flip_probability=1),  # Horizontal
            ])
            image = transforms(image)
            mask = transforms(mask)
        return image, mask

    def depth_flip(image, mask, p=0.5):
        if random.random() < p:
            transforms = tio.Compose([
                tio.RandomFlip(axes=0, flip_probability=1),  # Horizontal
            ])
            image = transforms(image)
            mask = transforms(mask)
        return image, mask

    def random_crop(image, mask, crop_size, num_crops=10, threshold=500):
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
            cropped_mask = mask[:, start_d:start_d + crop_depth, start_h:start_h + crop_height,
                           start_w:start_w + crop_width]
            if torch.sum(cropped_image > 0) > threshold * crop_depth:
                # print(f"depth: {depth}, height: {height}, width: {width}")
                # print(f"crop_depth: {crop_depth}, crop_height: {crop_height}, crop_width: {crop_width}")
                # print(f"start_d: {start_d}, start_h: {start_h}, start_w: {start_w}")
                break

        return cropped_image, cropped_mask

    def to_tensor(image, mask):
        image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        return image.unsqueeze(0), mask.unsqueeze(0)

    def transform(image, mask):
        image, mask = to_tensor(image, mask)
        if train_aug:
            image, mask = affine(image, mask, p=0.5, max_degrees=15, max_scale=0.2)
            image, mask = horizontal_flip(image, mask, p=0.5)
            image, mask = vertical_flip(image, mask, p=0.5)
            image, mask = depth_flip(image, mask, p=0.5)
            image, mask = random_crop(image, mask, crop_size=(5, 256, 256))
        return image, mask

    return transform


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


def t_transform():
    train_aug = True
    # get mask
    # img_path = "/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N3DH-SIM+/01/t070.tif"
    # mask_path ="/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N3DH-SIM+/01_GT/SEG/man_seg070.tif"
    # mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG/man_seg070.tif"
    # img_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01/t070.tif"

    # Aviv
    mask_path = r"C:\Users\beaviv\PycharmProjects\ImageProcessing\Datasets\Fluo-N3DH-SIM+\01_GT\SEG\man_seg070.tif"
    img_path = r"C:\Users\beaviv\PycharmProjects\ImageProcessing\Datasets\Fluo-N3DH-SIM+\01\t070.tif"

    image = tiff.imread(img_path).astype(np.float32)
    mask = tiff.imread(mask_path).astype(np.float32)

    image = (image - image.mean()) / (image.std())

    transform = get_transform(train_aug)
    tras_image, tras_mask = transform(image=image, mask=mask)

    image_middle_index = tras_image.shape[1] // 2

    ffig, axs = plt.subplots(2, 5, figsize=(15, 5))
    for i in range(5):
        axs[0, i].imshow(tras_image[0, image_middle_index - 2 + i], cmap='gray')
        axs[0, i].set_title(f"{i - 2}")
        axs[0, i].axis('off')  # Hide the axes
        axs[1, i].imshow(tras_mask[0, image_middle_index - 2 + i], cmap='gray')
        axs[1, i].set_title(f"{i - 2}")
        axs[1, i].axis('off')  # Hide the axes

    # Adjust spacing between plots
    ffig.suptitle("crop aug", fontsize=16)
    plt.tight_layout()
    plt.show()

    ffig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image[image_middle_index], cmap='gray')
    axs[0].set_title(f"original image middle slice")
    axs[0].axis('off')  # Hide the axes
    axs[1].imshow(tras_image[0, image_middle_index], cmap='gray')
    axs[1].set_title(f"affine image middle slice")
    axs[1].axis('off')  # Hide the axes
    axs[2].imshow(tras_mask[0, image_middle_index], cmap='gray')
    axs[2].set_title(f"affine mask middle slice")
    axs[2].axis('off')  # Hide the axes

    # Adjust spacing between plots
    # ffig.suptitle("all aug", fontsize=16)
    plt.tight_layout()
    plt.show()






    # image_middle_index = image.shape[0] // 2
    #
    # gt = mask.astype(np.uint8)
    # colored_instance_gt = get_instance_color(gt)
    # print(f"image size: {image.shape}")
    # plt.figure()
    # plt.imshow(image[image_middle_index])
    # plt.axis('off')  # Hide axes
    # plt.title('original image middle slice')
    # plt.show()
    #
    # transform = get_transform(train_aug)
    # image, mask = transform(image=image, mask=mask)
    # image = image.cpu().numpy()
    # if train_aug:
    #     # gt = mask.cpu().numpy().astype(np.uint8)
    #     for i in range(5):
    #         # colored_instance_gt = get_instance_color(gt[i, 0])
    #         plt.figure()
    #         # plt.imshow(colored_instance_gt)
    #         plt.imshow(image[i, 0, image_middle_index])
    #         plt.axis('off')  # Hide axes
    #         plt.title('all aug image')
    #         plt.show()
    # else:
    #     plt.figure()
    #     plt.imshow(image[0, 0, image_middle_index])
    #     plt.axis('off')  # Hide axes
    #     plt.title('h_flip image')
    #     plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t_transform()












