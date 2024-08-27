import os
import torch
import random
import numpy as np
import torchio as tio
import tifffile as tiff
from torch.utils.data import Dataset



class Dataset3D(Dataset):
    def __init__(self, image_dir, crop_size, device, seg_dir=None, tra_dir=None, train_aug=False):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.tra_dir = tra_dir
        self.train_aug = train_aug
        self.images = os.listdir(image_dir)
        self.crop_size = crop_size
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.seg_dir is not None:
            img_path = os.path.join(self.image_dir, self.images[index])
            seg_path = os.path.join(self.seg_dir, self.images[index].replace("t", "man_seg", 1))
            tra_path = os.path.join(self.tra_dir, self.images[index].replace("t", "man_track", 1))
            image = tiff.imread(img_path).astype(np.float32)
            image = (image - image.mean()) / (image.std())

            seg_mask = tiff.imread(seg_path).astype(np.float32)
            tra_mask = tiff.imread(tra_path).astype(np.float32)

            transform = self.get_transform()
            image, seg_mask, tra_mask = transform(image=image, seg_mask=seg_mask, tra_mask=tra_mask)
            return image, seg_mask, tra_mask

    @staticmethod
    def detect_edges(mask, dilation_layers=0, threshold=0.25):
        inverted_mask = 1 - mask
        # Compute the gradients along rows and columns
        gradient_x = torch.gradient(mask, dim=2)[0]
        gradient_y = torch.gradient(mask, dim=1)[0]
        gradient_z = torch.gradient(mask, dim=0)[0]

        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)
        # masked_gradient_magnitude = gradient_magnitude * mask  # the edge is inside the cell
        masked_gradient_magnitude = gradient_magnitude * inverted_mask  # the edge outside the cell
        edge_mask = (masked_gradient_magnitude > threshold).to(torch.int)

        edge_mask = edge_mask.unsqueeze(0)
        if dilation_layers > 0:
            kernel_size = dilation_layers * 2 + 1
            kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
            edge_mask = torch.nn.functional.conv3d(edge_mask.unsqueeze(1).float(), kernel, padding=dilation_layers)
            edge_mask = (edge_mask > 0).to(torch.int)
        edge_mask = edge_mask.squeeze(0).squeeze(0)

        return edge_mask

    @staticmethod
    def split_mask(mask):
        # mask: torch.Size([batch, D, H, W])
        three_classes_mask = torch.zeros_like(mask, dtype=torch.int32)
        for batch_idx in range(mask.size()[0]):
            unique_elements = torch.unique(mask[batch_idx].flatten())
            for element in unique_elements:
                if element != 0:
                    element_mask = (mask[batch_idx] == element).to(torch.int)
                    edges = Dataset3D.detect_edges(mask=element_mask, dilation_layers=0)
                    element_mask -= edges
                    three_classes_mask[batch_idx][edges == 1] = 1         # edge
                    three_classes_mask[batch_idx][element_mask == 1] = 2  # interior

        return three_classes_mask

    def get_transform(self):
        def affine(subj, p=0.5, max_degrees=15, max_scale=0.2, translation=5):
            if random.random() < p:
                random_affine = tio.RandomAffine(scales=max_scale,
                                                degrees=max_degrees,
                                                translation=translation,
                                                default_pad_value=0,
                                                )
                subj = random_affine(subj)
            return subj

        def elastic_deformation(subj, num_control_points, locked_borders, p=0.5):
            if random.random() < p:
                transforms = tio.RandomElasticDeformation(num_control_points=num_control_points,
                                                          locked_borders=locked_borders)
                subj = transforms(subj)
            return subj
        def horizontal_flip(subj, p=0.5):
            if random.random() < p:
                transforms = tio.Compose([
                    tio.RandomFlip(axes=2, flip_probability=1),  # Horizontal
                ])
                subj = transforms(subj)
            return subj

        def vertical_flip(subj, p=0.5):
            if random.random() < p:
                transforms = tio.Compose([
                    tio.RandomFlip(axes=1, flip_probability=1),  # Horizontal
                ])
                subj = transforms(subj)
            return subj

        def depth_flip(subj, p=0.5):
            if random.random() < p:
                transforms = tio.Compose([
                    tio.RandomFlip(axes=0, flip_probability=1),  # Horizontal
                ])
                subj = transforms(subj)
            return subj

        def random_crop(subj, crop_size, threshold=500):
            image = subj.image.tensor
            seg_mask = subj.seg_mask.tensor
            tra_mask = subj.tra_mask.tensor
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
                cropped_seg_mask = seg_mask[:, start_d:start_d + crop_depth, start_h:start_h + crop_height,
                               start_w:start_w + crop_width]
                cropped_tra_mask = tra_mask[:, start_d:start_d + crop_depth, start_h:start_h + crop_height,
                                   start_w:start_w + crop_width]
                if torch.sum(cropped_image > 0) > threshold * crop_depth:
                    break

            return cropped_image, cropped_seg_mask, cropped_tra_mask

        def random_gamma(image, max_gamma=0.3, p=0.5):
            if random.random() < p:
                randomgamma = tio.RandomGamma(log_gamma=max_gamma)
                image = randomgamma(image)
            return image

        def random_noise(image, std,  p=0.5):
            if random.random() < p:
                randomnoise = tio.RandomNoise(std=std)
                image = randomnoise(image)
            return image


        def to_tensor(image, seg_mask, tra_mask):
            image, seg_mask, tra_mask = torch.from_numpy(image), torch.from_numpy(seg_mask), torch.from_numpy(tra_mask)
            subj = tio.Subject(image=tio.ScalarImage(tensor=image.unsqueeze(0)),
                               seg_mask=tio.LabelMap(tensor=seg_mask.unsqueeze(0)),
                               tra_mask=tio.LabelMap(tensor=tra_mask.unsqueeze(0)))
            return subj

        def transform(image, seg_mask, tra_mask):
            subj = to_tensor(image, seg_mask, tra_mask)
            if self.train_aug:
                subj = affine(subj, p=0.5, max_degrees=10, max_scale=0.2, translation=5)
                # subj = elastic_deformation(subj, num_control_points=7, locked_borders=2, p=0.5)
                subj = horizontal_flip(subj, p=0.5)
                subj = vertical_flip(subj, p=0.5)
                subj = depth_flip(subj, p=0.5)
                image, seg_mask, tra_mask = random_crop(subj, crop_size=self.crop_size)
                image = random_gamma(image, max_gamma=0.3, p=0.5)
                # image = random_noise(image, std=0.015, p=0.5)
            else:
                image, seg_mask, tra_mask = random_crop(subj, crop_size=self.crop_size)
            return image, seg_mask.squeeze(0), tra_mask.squeeze(0)

        return transform



def get_transform(train_aug=True):
    def affine(subj, p=0.5, max_degrees=15, max_scale=0.2, translation=5):
        if random.random() < p:
            random_affine = tio.RandomAffine(scales=max_scale,
                                             degrees=max_degrees,
                                             translation=translation,
                                             default_pad_value=0,
                                             )
            subj = random_affine(subj)
        return subj

    def elastic_deformation(subj, num_control_points, locked_borders, p=0.5):
        if random.random() < p:
            transforms = tio.RandomElasticDeformation(num_control_points=num_control_points,
                                                      locked_borders=locked_borders)
            subj = transforms(subj)
        return subj
    def horizontal_flip(subj, p=0.5):
        if random.random() < p:
            transforms = tio.Compose([
                tio.RandomFlip(axes=2, flip_probability=1),  # Horizontal
            ])
            subj = transforms(subj)
        return subj

    def vertical_flip(subj, p=0.5):
        if random.random() < p:
            transforms = tio.Compose([
                tio.RandomFlip(axes=1, flip_probability=1),  # Horizontal
            ])
            subj = transforms(subj)
        return subj

    def depth_flip(subj, p=0.5):
        if random.random() < p:
            transforms = tio.Compose([
                tio.RandomFlip(axes=0, flip_probability=1),  # Horizontal
            ])
            subj = transforms(subj)
        return subj

    def random_crop(subj, crop_size, threshold=500):
        image = subj.image.tensor
        seg_mask = subj.seg_mask.tensor
        tra_mask = subj.tra_mask.tensor
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
            cropped_seg_mask = seg_mask[:, start_d:start_d + crop_depth, start_h:start_h + crop_height,
                           start_w:start_w + crop_width]
            cropped_tra_mask = tra_mask[:, start_d:start_d + crop_depth, start_h:start_h + crop_height,
                               start_w:start_w + crop_width]
            if torch.sum(cropped_image > 0) > threshold * crop_depth:
                break

        return cropped_image, cropped_seg_mask, cropped_tra_mask

    def random_gamma(image, max_gamma=0.3, p=0.5):
        if random.random() < p:
            randomgamma = tio.RandomGamma(log_gamma=max_gamma)
            image = randomgamma(image)
        return image

    def random_noise(image, std,  p=0.5):
        if random.random() < p:
            randomnoise = tio.RandomNoise(std=std)
            image = randomnoise(image)
        return image


    def to_tensor(image, seg_mask, tra_mask):
        image, seg_mask, tra_mask = torch.from_numpy(image), torch.from_numpy(seg_mask), torch.from_numpy(tra_mask)
        subj = tio.Subject(image=tio.ScalarImage(tensor=image.unsqueeze(0)),
                           seg_mask=tio.LabelMap(tensor=seg_mask.unsqueeze(0)),
                           tra_mask=tio.LabelMap(tensor=tra_mask.unsqueeze(0)))
        return subj

    def transform(image, seg_mask, tra_mask):
        image_d = image.shape[0]
        subj = to_tensor(image, seg_mask, tra_mask)
        if train_aug:
            subj = affine(subj, p=1, max_degrees=10, max_scale=0.2, translation=5)
            # subj = elastic_deformation(subj, num_control_points=6, locked_borders=2, p=1)
            subj = horizontal_flip(subj, p=1)
            subj = vertical_flip(subj, p=1)
            subj = depth_flip(subj, p=1)
            image, seg_mask, tra_mask = random_crop(subj, crop_size=(image_d, 256, 256))
            # image = subj.image.tensor
            # seg_mask = subj.seg_mask.tensor
            # tra_mask = subj.tra_mask.tensor
            image = random_gamma(image, max_gamma=0.3, p=1)
            # image = random_noise(image, std=0.5, p=1)

        else:
            image, seg_mask, tra_mask = random_crop(subj, crop_size=(32, 256, 256))
        return image, seg_mask.squeeze(0), tra_mask.squeeze(0)

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
    mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG/man_seg070.tif"
    tra_mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/TRA/man_track070.tif"
    img_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01/t070.tif"

    # Aviv
    # mask_path = r"C:\Users\beaviv\PycharmProjects\ImageProcessing\Datasets\Fluo-N3DH-SIM+\01_GT\SEG\man_seg070.tif"
    # img_path = r"C:\Users\beaviv\PycharmProjects\ImageProcessing\Datasets\Fluo-N3DH-SIM+\01\t070.tif"

    image = tiff.imread(img_path).astype(np.float32)
    mask = tiff.imread(mask_path).astype(np.float32)
    tra_mask = tiff.imread(tra_mask_path).astype(np.float32)

    image = (image - image.mean()) / (image.std())

    transform = get_transform(train_aug)
    tras_image, tras_mask, _ = transform(image=image, seg_mask=mask, tra_mask=tra_mask)

    image_middle_index = tras_image.shape[1] // 2
    #
    # ffig, axs = plt.subplots(2, 5, figsize=(15, 5))
    # for i in range(5):
    #     axs[0, i].imshow(tras_image[0, image_middle_index - 2 + i], cmap='gray')
    #     axs[0, i].set_title(f"{i - 2}")
    #     axs[0, i].axis('off')  # Hide the axes
    #     axs[1, i].imshow(tras_mask[0, image_middle_index - 2 + i], cmap='gray')
    #     axs[1, i].set_title(f"{i - 2}")
    #     axs[1, i].axis('off')  # Hide the axes
    #
    # # Adjust spacing between plots
    # ffig.suptitle("crop aug", fontsize=16)
    # plt.tight_layout()
    # plt.show()
    #
    ffig, axs = plt.subplots(2, 2, figsize=(15, 5))
    axs[0, 0].imshow(image[image_middle_index], cmap='gray')
    axs[0, 0].set_title(f"original image middle slice")
    axs[0, 0].axis('off')  # Hide the axes
    axs[0, 1].imshow(mask[image_middle_index], cmap='gray')
    axs[0, 1].set_title(f"original mask middle slice")
    axs[0, 1].axis('off')  # Hide the axes
    axs[1, 0].imshow(tras_image[0, image_middle_index], cmap='gray')
    axs[1, 0].set_title(f"all augmentations image middle slice")
    axs[1, 0].axis('off')  # Hide the axes
    axs[1, 1].imshow(tras_mask[image_middle_index], cmap='gray')
    axs[1, 1].set_title(f"all augmentations mask middle slice")
    axs[1, 1].axis('off')  # Hide the axes

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












