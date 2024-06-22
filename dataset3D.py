import os
import cv2
import random
from PIL import Image
import tifffile as tiff
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch


# import albumentations as A
# from albumentations.pytorch import ToTensorV2


class Fluo_N3DH_SIM_PLUS(Dataset):
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
            image = tiff.imread(img_path)
            image = image.astype(np.float32)
            image = (image - image.mean()) / (image.std())

            mask = tiff.imread(mask_path)
            mask = mask.astype(np.float32)

            transform = self.get_transform()

            return transform(image=image, mask=mask)

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
        three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
        for batch_idx in range(mask.size()[0]):
            unique_elements = torch.unique(mask[batch_idx].flatten())
            for element in unique_elements:
                if element != 0:
                    element_mask = (mask[batch_idx] == element).to(torch.int)
                    edges = Fluo_N3DH_SIM_PLUS.detect_edges(element_mask)
                    element_mask -= edges
                    three_classes_mask[batch_idx][edges == 1] = 1
                    three_classes_mask[batch_idx][element_mask == 1] = 2

        return three_classes_mask

    def get_transform(self):
        # def affine(image, mask, p=0.5, max_degrees=45, max_scale=0.2,
        #            max_shear=10):  # todo need to change to handle 3d
        #     if random.random() < p:
        #         degrees = random.uniform(-max_degrees, max_degrees)
        #         scale = random.uniform(-max_scale, max_scale)
        #         scale += 1
        #         shear_x = random.uniform(-max_shear, max_shear)
        #         shear_y = random.uniform(-max_shear, max_shear)
        #         aff_image = TF.affine(image, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
        #         aff_mask = TF.affine(mask, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
        #         return aff_image, aff_mask
        #     return image, mask

        def horizontal_flip(image, mask, p=0.5):
            if random.random() < p:
                # Flip width dimension
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            return image, mask

        def vertical_flip(image, mask, p=0.5):
            if random.random() < p:
                # Flip height dimension
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])
                return TF.vflip(image), TF.vflip(mask)
            return image, mask

        def depth_flip(image, mask, p=0.5):
            if random.random() < p:
                # Flip depth dimension
                image = torch.flip(image, dims=[0])
                mask = torch.flip(mask, dims=[0])
            return image, mask

        def random_crop(image, mask, crop_size, num_crops=10, threshold=500):
            # image, mask are PIL
            count = 0
            depth, height, width = crop_size
            images = np.zeros((num_crops, depth, height, width), dtype=np.float32)
            masks = np.zeros((num_crops, depth, height, width), dtype=np.float32)
            while count < num_crops:
                d, h, w = [random.randint(0, s - cs) for s, cs in zip(image.shape, crop_size)]
                cropped_image = image[d:d + depth, h:h + height, w:w + width]
                cropped_mask = mask[d:d + depth, h:h + height, w:w + width]

                if np.count_nonzero(cropped_mask) > threshold:
                    images[count] = cropped_image
                    masks[count] = cropped_mask
                    count += 1

            return images, masks

        def to_tensor(images, masks):
            # Convert batch of images(batch_size, D, H, W) and masks to PyTorch tensors (batch_size, C, D, H, W)
            batch_size = images.shape[0]
            tensor_images = torch.zeros((batch_size, 1, images.shape[-3], images.shape[-2], images.shape[-1]), dtype=torch.float32)
            tensor_masks = torch.zeros((batch_size, images.shape[-3], images.shape[-2], images.shape[-1]), dtype=torch.float32)

            for i in range(batch_size):
                tensor_images[i] = transforms.ToTensor()(images[i])
                tensor_masks[i] = transforms.ToTensor()(masks[i])

            return tensor_images, tensor_masks

        def val_to_tensor(image, mask):
            # Convert  images and masks to PyTorch tensors (1, 1, D, H, W)
            # input (D, H, W)
            if len(image.shape) == 3:  # Grayscale image
                image = np.expand_dims(image, axis=0)

            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            tensor_image = torch.from_numpy(image).float()
            tensor_mask = torch.from_numpy(mask).float()
            return tensor_image, tensor_mask

        def transform(image, mask):
            if self.train_aug:
                # image, mask = rotate(image, mask, p=0.5)
                image, mask = Image.fromarray(image), Image.fromarray(mask)
                # image, mask = affine(image, mask, p=0.5, max_degrees=45, max_scale=0.2, max_shear=10)
                image, mask = horizontal_flip(image, mask, p=0.5)
                image, mask = vertical_flip(image, mask, p=0.5)
                image, mask = depth_flip(image, mask, p=0.5)
                image, mask = random_crop(image, mask, crop_size=self.crop_size)
                return to_tensor(np.array(image), np.array(mask))

            return val_to_tensor(image, mask)

        return transform


def get_transform(train_aug):
    # def affine(image, mask, p=0.5, max_degrees=45, max_scale=0.2,
    #            max_shear=10):  # todo need to change to handle 3d
    #     if random.random() < p:
    #         degrees = random.uniform(-max_degrees, max_degrees)
    #         scale = random.uniform(-max_scale, max_scale)
    #         scale += 1
    #         shear_x = random.uniform(-max_shear, max_shear)
    #         shear_y = random.uniform(-max_shear, max_shear)
    #         aff_image = TF.affine(image, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
    #         aff_mask = TF.affine(mask, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
    #         return aff_image, aff_mask
    #     return image, mask

    def horizontal_flip(image, mask, p=0.5):
        if random.random() < p:
            # Flip width dimension
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        return image, mask

    def vertical_flip(image, mask, p=0.5):
        if random.random() < p:
            # Flip height dimension
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            return TF.vflip(image), TF.vflip(mask)
        return image, mask

    def depth_flip(image, mask, p=0.5):
        if random.random() < p:
            # Flip depth dimension
            image = torch.flip(image, dims=[0])
            mask = torch.flip(mask, dims=[0])
        return image, mask

    def random_crop(image, mask, crop_size, num_crops=10, threshold=500):
        # image, mask are PIL
        count = 0
        depth, height, width = crop_size
        images = np.zeros((num_crops, depth, height, width), dtype=np.float32)
        masks = np.zeros((num_crops, depth, height, width), dtype=np.float32)
        while count < num_crops:
            d, h, w = [random.randint(0, s - cs) for s, cs in zip(image.shape, crop_size)]
            cropped_image = image[d:d + depth, h:h + height, w:w + width]
            cropped_mask = mask[d:d + depth, h:h + height, w:w + width]

            if np.count_nonzero(cropped_mask) > threshold:
                images[count] = cropped_image
                masks[count] = cropped_mask
                count += 1

        return images, masks

    def to_tensor(images, masks):
        # Convert batch of images(batch_size, D, H, W) and masks to PyTorch tensors (batch_size, C, D, H, W)
        batch_size = images.shape[0]
        tensor_images = torch.zeros((batch_size, 1, images.shape[-3], images.shape[-2], images.shape[-1]),
                                    dtype=torch.float32)
        tensor_masks = torch.zeros((batch_size, images.shape[-3], images.shape[-2], images.shape[-1]),
                                   dtype=torch.float32)

        for i in range(batch_size):
            tensor_images[i] = transforms.ToTensor()(images[i])
            tensor_masks[i] = transforms.ToTensor()(masks[i])

        return tensor_images, tensor_masks

    def val_to_tensor(image, mask):
        # Convert  images and masks to PyTorch tensors (1, 1, D, H, W)
        # input (D, H, W)
        if len(image.shape) == 3:  # Grayscale image
            image = np.expand_dims(image, axis=0)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        tensor_image = torch.from_numpy(image).float()
        tensor_mask = torch.from_numpy(mask).float()
        return tensor_image, tensor_mask

    def transform(image, mask):
        if train_aug:
            # image, mask = rotate(image, mask, p=0.5)
            image, mask = Image.fromarray(image), Image.fromarray(mask)
            # image, mask = affine(image, mask, p=0.5, max_degrees=45, max_scale=0.2, max_shear=10)
            image, mask = horizontal_flip(image, mask, p=0.5)
            image, mask = vertical_flip(image, mask, p=0.5)
            image, mask = depth_flip(image, mask, p=0.5)
            image, mask = random_crop(image, mask, crop_size=(5, 256, 256))
            return to_tensor(np.array(image), np.array(mask))

        return val_to_tensor(image, mask)

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
    mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02_GT/SEG/man_seg140.tif"
    img_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02/t140.tif"
    image = tiff.imread(img_path)
    mask = tiff.imread(mask_path)
    mask = mask.astype(np.float32)
    image = image.astype(np.float32)
    image = (image - image.mean()) / (image.std())

    # test the augmantations one by one
    # gt = mask.astype(np.uint8)
    # colored_instance_gt = get_instance_color(gt)
    # plt.figure()
    # plt.imshow(image)
    # plt.axis('off')  # Hide axes
    # plt.title('original image')
    # plt.show()

    transform = get_transform(train_aug)
    image, mask = transform(image=image, mask=mask)
    image = image.cpu().numpy()
    if train_aug:
        # gt = mask.cpu().numpy().astype(np.uint8)
        for i in range(5):
            # colored_instance_gt = get_instance_color(gt[i, 0])
            plt.figure()
            # plt.imshow(colored_instance_gt)
            plt.imshow(image[i, 0])
            plt.axis('off')  # Hide axes
            plt.title('all aug image')
            plt.show()
    else:
        plt.figure()
        plt.imshow(image[0, 0])
        plt.axis('off')  # Hide axes
        plt.title('v_flip image')
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t_transform()











#todo look if it is working?
def affine_3d(image, mask, max_degrees=45, max_scale=0.2, max_shear=10):
    """
    Apply a 3D affine transformation to the image and mask.

    Parameters:
    image (torch.Tensor): The input 3D image tensor of shape (C, D, H, W).
    mask (torch.Tensor): The input 3D mask tensor of shape (C, D, H, W).
    max_degrees (float): Maximum rotation degrees.
    max_scale (float): Maximum scaling factor.
    max_shear (float): Maximum shearing factor.

    Returns:
    torch.Tensor: Transformed image.
    torch.Tensor: Transformed mask.
    """

    # Get the dimensions
    _, D, H, W = image.shape

    # Rotation
    angle_x = random.uniform(-max_degrees, max_degrees)
    angle_y = random.uniform(-max_degrees, max_degrees)
    angle_z = random.uniform(-max_degrees, max_degrees)

    # Scale
    scale = random.uniform(1 - max_scale, 1 + max_scale)

    # Shear
    shear_xy = random.uniform(-max_shear, max_shear)
    shear_xz = random.uniform(-max_shear, max_shear)
    shear_yx = random.uniform(-max_shear, max_shear)
    shear_yz = random.uniform(-max_shear, max_shear)
    shear_zx = random.uniform(-max_shear, max_shear)
    shear_zy = random.uniform(-max_shear, max_shear)

    # Create the affine transformation matrix
    transform_matrix = get_affine_matrix_3d(angle_x, angle_y, angle_z, scale, shear_xy, shear_xz, shear_yx, shear_yz,
                                            shear_zx, shear_zy, D, H, W)

    # Apply the transformation
    image = apply_affine_transform_3d(image, transform_matrix)
    mask = apply_affine_transform_3d(mask, transform_matrix)

    return image, mask


def get_affine_matrix_3d(angle_x, angle_y, angle_z, scale, shear_xy, shear_xz, shear_yx, shear_yz, shear_zx, shear_zy,
                         D, H, W):
    """
    Generate a 3D affine transformation matrix.

    Parameters:
    (same as above)

    Returns:
    torch.Tensor: The affine transformation matrix.
    """

    # Convert angles from degrees to radians
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)

    # Rotation matrices around x, y, and z axes
    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x), 0],
                    [0, np.sin(angle_x), np.cos(angle_x), 0],
                    [0, 0, 0, 1]])

    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0, 0],
                    [np.sin(angle_z), np.cos(angle_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Scaling matrix
    S = np.array([[scale, 0, 0, 0],
                  [0, scale, 0, 0],
                  [0, 0, scale, 0],
                  [0, 0, 0, 1]])

    # Shearing matrix
    Sh = np.array([[1, shear_xy, shear_xz, 0],
                   [shear_yx, 1, shear_yz, 0],
                   [shear_zx, shear_zy, 1, 0],
                   [0, 0, 0, 1]])

    # Combined affine transformation matrix
    affine_matrix = R_x @ R_y @ R_z @ S @ Sh

    # Adjust for the center of the image
    center = np.array([D / 2, H / 2, W / 2, 1])
    T_center = np.eye(4)
    T_center[:3, 3] = center[:3]

    T_center_inv = np.eye(4)
    T_center_inv[:3, 3] = -center[:3]

    affine_matrix = T_center @ affine_matrix @ T_center_inv

    return torch.tensor(affine_matrix, dtype=torch.float32)


def apply_affine_transform_3d(tensor, matrix):
    """
    Apply a 3D affine transformation to a tensor.

    Parameters:
    tensor (torch.Tensor): The input 3D tensor.
    matrix (torch.Tensor): The 3D affine transformation matrix.

    Returns:
    torch.Tensor: The transformed tensor.
    """

    # Get the shape of the tensor
    C, D, H, W = tensor.shape

    # Generate grid of coordinates
    grid = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W))
    grid = torch.stack(grid, dim=-1).float()

    # Reshape to (N, 4) where N is the number of points
    grid = grid.view(-1, 3)
    grid = torch.cat([grid, torch.ones(grid.shape[0], 1)], dim=1)  # Add a column of ones for affine transformation

    # Apply the affine transformation
    new_grid = grid @ matrix.T

    # Remove the homogeneous coordinate
    new_grid = new_grid[:, :3]

    # Reshape back to (D, H, W, 3)
    new_grid = new_grid.view(D, H, W, 3)

    # Interpolate the tensor values at the new coordinates
    new_grid = new_grid.unsqueeze(0).repeat(C, 1, 1, 1, 1)  # Repeat for each channel
    new_tensor = F.grid_sample(tensor.unsqueeze(0), new_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return new_tensor.squeeze(0)