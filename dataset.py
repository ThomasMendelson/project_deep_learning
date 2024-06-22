import os
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch


# import albumentations as A
# from albumentations.pytorch import ToTensorV2


class Fluo_N2DH_SIM_PLUS(Dataset):
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
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            image = image.astype(np.float32)
            image = (image - image.mean()) / (image.std())
            # if len(image.shape) == 2:  # (height, width)
            #     image = np.expand_dims(image, axis=-1)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = mask.astype(np.float32)
            # print("\nin dataset before augmentations = self.transform(image=image, mask=mask)")
            # print(f"image shape: {image.shape}, type: {image.dtype}, min value: {np.min(image)}, max value: {np.max(image)}")
            # print(f"mask shape: {mask.shape}, type: {mask.dtype}, min value: {np.min(mask)}, max value: {np.max(mask)}")

            # if self.train_aug:
            #     crop_size = int(min(image.shape[0], image.shape[1]) * random.uniform(0.8, 1.0))
            #     transform = get_train_transform(crop_size, self.resize)
            # else:
            #     transform = get_val_transform(self.resize)
            transform = self.get_transform()
            augmentations = transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            return image, mask
        # else:
        #     img_path = os.path.join(self.image_dir, self.images[index])
        #     image = np.array(Image.open(img_path).convert("RGB"))
        #
        #     if self.transform is not None:
        #         augmentations = self.transform(image=image)
        #         image = augmentations["image"]
        #
        #     return image

    @staticmethod
    def detect_edges(mask, threshold=0.25):
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

    @staticmethod
    def split_mask(mask):
        three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
        for batch_idx in range(mask.size()[0]):
            unique_elements = torch.unique(mask[batch_idx].flatten())
            for element in unique_elements:
                if element != 0:
                    element_mask = (mask[batch_idx] == element).to(torch.int)
                    edges = Fluo_N2DH_SIM_PLUS.detect_edges(element_mask)
                    element_mask -= edges
                    three_classes_mask[batch_idx][edges == 1] = 1
                    three_classes_mask[batch_idx][element_mask == 1] = 2

        return three_classes_mask

    def get_transform(self):
        def affine(image, mask, p=0.5, max_degrees=45, max_scale=0.2, max_shear=10):
            if random.random() < p:
                degrees = random.uniform(-max_degrees, max_degrees)
                scale = random.uniform(-max_scale, max_scale)
                scale += 1
                shear_x = random.uniform(-max_shear, max_shear)
                shear_y = random.uniform(-max_shear, max_shear)
                aff_image = TF.affine(image, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
                aff_mask = TF.affine(mask, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
                return aff_image, aff_mask
            return image, mask

        def horizontal_flip(image, mask, p=0.5):
            if random.random() < p:
                return TF.hflip(image), TF.hflip(mask)
            return image, mask

        def vertical_flip(image, mask, p=0.5):
            if random.random() < p:
                return TF.vflip(image), TF.vflip(mask)
            return image, mask

        def random_crop(image, mask, crop_size, num_crops=10, threshold=500):
            # image, mask are PIL
            count = 0
            images = np.zeros((num_crops, crop_size, crop_size), dtype=np.float32)
            masks = np.zeros((num_crops, crop_size, crop_size), dtype=np.float32)
            while count < num_crops:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(crop_size, crop_size))
                cropped_image, cropped_mask = TF.crop(image, i, j, h, w), TF.crop(mask, i, j, h, w)
                cropped_image, cropped_mask = np.array(cropped_image), np.array(cropped_mask)
                if np.count_nonzero(cropped_mask) > threshold:
                    images[count] = cropped_image
                    masks[count] = cropped_mask
                    count += 1

            return images, masks

        def to_tensor(images, masks):
            # Convert batch of images and masks to PyTorch tensors (batch_size, C, H, W)
            batch_size = images.shape[0]
            tensor_images = torch.zeros((batch_size, 1, images.shape[1], images.shape[2]), dtype=torch.float32)
            tensor_masks = torch.zeros((batch_size, masks.shape[1], masks.shape[2]), dtype=torch.float32)

            for i in range(batch_size):
                tensor_images[i] = transforms.ToTensor()(images[i])
                tensor_masks[i] = transforms.ToTensor()(masks[i])

            return tensor_images, tensor_masks

        def val_to_tensor(image, mask):
            # Convert  images and masks to PyTorch tensors (1, 1, H, W)
            # input (H, W)
            if len(image.shape) == 2:  # Grayscale image
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
                image, mask = affine(image, mask, p=0.5, max_degrees=45, max_scale=0.2, max_shear=10)
                image, mask = horizontal_flip(image, mask, p=0.5)
                image, mask = vertical_flip(image, mask, p=0.5)
                image, mask = random_crop(image, mask, crop_size=self.crop_size)
                image, mask = to_tensor(np.array(image), np.array(mask))
                return {'image': image, 'mask': mask}
            image, mask = val_to_tensor(image, mask)
            return {'image': image, 'mask': mask}

        return transform


# def get_train_transform(crop_size, resize):
#     train_transform = A.Compose(
#         [
#             A.ToFloat(max_value=65535.0),
#             # A.Rotate(limit=35, p=1.0),
#             A.RandomCrop(height=crop_size, width=crop_size),
#             A.Resize(height=resize, width=resize),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Normalize(mean=[0.0], std=[1.0], ),
#             # A.FromFloat(max_value=65535.0),
#             # A.Lambda(image=lambda x, **kwargs: x.astype(np.float32)),
#             ToTensorV2(),
#         ],
#     )
#
#     return train_transform
#
#
# def get_val_transform(resize):
#     val_transform = A.Compose(
#         [
#             A.ToFloat(max_value=65535.0),
#             A.Resize(height=resize, width=resize),
#             A.Normalize(mean=[0.0], std=[1.0], ),
#             # A.FromFloat(max_value=65535.0),
#             # A.Lambda(image=lambda x, **kwargs: x.astype(np.float32)),
#             ToTensorV2(),
#         ],
#     )
#     return val_transform


# import matplotlib.pyplot as plt
# def plot_img_befor_transform(image):
#     image = image.astype(np.float32) / np.iinfo(np.uint16).max
#     plt.imshow(image, cmap='gray')
#     plt.show()
#
# def plot_img_after_transform(image, mask):
#     fig, axs = plt.subplots(1, len(image), figsize=(15, 5))
#     fig2, axs2 = plt.subplots(1, len(mask), figsize=(15, 5))
#     for i, image in enumerate(image):
#         axs[i].imshow(image.squeeze()/torch.max(image), cmap='gray')
#         axs[i].axis("off")
#         print(f"min{i}: {torch.min(image)}, max{i}: {torch.max(image)}")
#     for i, mask in enumerate(mask):
#         axs2[i].imshow(mask.squeeze(), cmap='gray')
#         axs2[i].axis("off")
#         print(f"mask-min{i}: {torch.min(image)}, max{i}: {torch.max(image)}")
#
#     plt.show()


def detect_edges(mask, threshold=0.25):
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


def split_mask(mask):
    three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
    for batch_idx in range(mask.size()[0]):
        unique_elements = torch.unique(mask[batch_idx].flatten())
        for element in unique_elements:
            if element != 0:
                element_mask = (mask[batch_idx] == element).to(torch.int)
                print(element_mask.sum().item())  # todo delete
                edges = detect_edges(element_mask)
                element_mask -= edges
                three_classes_mask[batch_idx][edges == 1] = 1
                three_classes_mask[batch_idx][element_mask == 1] = 2

    return three_classes_mask


def get_transform(train_aug):
    def affine(image, mask, p=0.5, max_degrees=45, max_scale=0.2, max_shear=10):
        if random.random() < p:
            degrees = random.uniform(-max_degrees, max_degrees)
            scale = random.uniform(-max_scale, max_scale)
            scale += 1
            shear_x = random.uniform(-max_shear, max_shear)
            shear_y = random.uniform(-max_shear, max_shear)
            aff_image = TF.affine(image, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
            aff_mask = TF.affine(mask, angle=degrees, translate=(0, 0), scale=scale, shear=(shear_x, shear_y))
            print(f"angle= {degrees:.2f}, scale= {scale:.2f}, shear=({shear_x:.2f}, {shear_y:.2f})")
            return aff_image, aff_mask
        return image, mask

    def horizontal_flip(image, mask, p=0.5):
        if random.random() < p:
            return TF.hflip(image), TF.hflip(mask)
        return image, mask

    def vertical_flip(image, mask, p=0.5):
        if random.random() < p:
            return TF.vflip(image), TF.vflip(mask)
        return image, mask

    def random_crop(image, mask, crop_size, num_crops=10, threshold=500):
        # image, mask are PIL
        count = 0
        images = np.zeros((num_crops, crop_size, crop_size), dtype=np.float32)
        masks = np.zeros((num_crops, crop_size, crop_size), dtype=np.float32)
        while count < num_crops:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(crop_size, crop_size))
            cropped_image, cropped_mask = TF.crop(image, i, j, h, w), TF.crop(mask, i, j, h, w)
            cropped_image, cropped_mask = np.array(cropped_image), np.array(cropped_mask)
            if np.count_nonzero(cropped_mask) > threshold:
                images[count] = cropped_image
                masks[count] = cropped_mask
                count += 1

        return images, masks

    def to_tensor(images, masks):
        # Convert batch of images and masks to PyTorch tensors (batch_size, C, H, W)
        batch_size = images.shape[0]
        tensor_images = torch.zeros((batch_size, 1, images.shape[1], images.shape[2]), dtype=torch.float32)
        tensor_masks = torch.zeros((batch_size, masks.shape[1], masks.shape[2]), dtype=torch.float32)

        for i in range(batch_size):
            tensor_images[i] = transforms.ToTensor()(images[i])
            tensor_masks[i] = transforms.ToTensor()(masks[i])

        return tensor_images, tensor_masks

    def val_to_tensor(image, mask):
        # Convert  images and masks to PyTorch tensors (1, 1, H, W)
        # input (H, W)
        if len(image.shape) == 2:  # Grayscale image
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
            image, mask = affine(image, mask, p=1, max_degrees=45, max_scale=0.2, max_shear=10)
            image, mask = horizontal_flip(image, mask, p=1)
            image, mask = vertical_flip(image, mask, p=1)
            image, mask = random_crop(image, mask, crop_size=256)
            image, mask = to_tensor(np.array(image), np.array(mask))
            return {'image': image, 'mask': mask}
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        # image, mask = affine(image, mask, p=1, max_degrees=45, max_scale=0.2, max_shear=10)
        # image, mask = horizontal_flip(image, mask, p=1)
        image, mask = vertical_flip(image, mask, p=1)
        image, mask = val_to_tensor(np.array(image), np.array(mask))
        return {'image': image, 'mask': mask}

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


def t_transform_and_cell_size():
    train_aug = True
    # get mask
    mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_GT/SEG/man_seg140.tif"
    img_path = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02/t140.tif"
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
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
    augmentations = transform(image=image, mask=mask)
    image = augmentations["image"].cpu().numpy()
    mask = augmentations["mask"]
    if train_aug:
        gt = mask.cpu().numpy().astype(np.uint8)
        for i in range(5):
            colored_instance_gt = get_instance_color(gt[i,0])
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



    # # find cell size
    # for i in range(5):
    #     split_mask(mask[i])


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    t_transform_and_cell_size()
