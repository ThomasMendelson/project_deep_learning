import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
# import os
# import torch
# checkpoint_dir = "checkpoint"
# TRAIN_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02"             # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02"
# TRAIN_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_ERR_SEG"    # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02_ERR_SEG"
# VAL_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01"               # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01"
# VAL_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01_ERR_SEG"      # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01_ERR_SEG"
#
# print(f"{os.listdir(checkpoint_dir)} ,os.listdir(checkpoint_dir)")
# print(f"{len(os.listdir(TRAIN_IMG_DIR))} ,os.listdir(TRAIN_IMG_DIR)")
# print("cuda" if torch.cuda.is_available() else "cpu")
#
#
# if torch.cuda.is_available():
#     # torch.cuda.set_device(2)
#     current_device = torch.cuda.current_device()
#     print(f"Current CUDA device index: {current_device}")
#     print(f"Current CUDA device name: {torch.cuda.get_device_name(current_device)}")
# else:
#     print("CUDA is not available.")




# TRAIN_IMG_DIR = "/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/DIC-C2DH-HeLa/02/t067.tif"
# TRAIN_SEG_DIR = "/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/DIC-C2DH-HeLa/02_GT/SEG/man_seg067.tif"
# TRAIN_TRA_DIR = "/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/DIC-C2DH-HeLa/02_GT/TRA/man_track067.tif"
# image = cv2.imread(TRAIN_IMG_DIR, cv2.IMREAD_UNCHANGED)
# seg_mask = cv2.imread(TRAIN_SEG_DIR, cv2.IMREAD_UNCHANGED)
# track_mask = cv2.imread(TRAIN_TRA_DIR, cv2.IMREAD_UNCHANGED)

# image = image.astype(np.float32)
# image = (image - image.mean()) / (image.std())
#
# plt.figure()
# plt.imshow(image)
# plt.axis('off')  # Hide axes
# plt.title('original image')
# plt.show()
#
# plt.figure()
# plt.imshow(seg_mask)
# plt.axis('off')  # Hide axes
# plt.title('original seg mask')
# plt.show()

# plt.figure()
# plt.imshow(track_mask)
# plt.axis('off')  # Hide axes
# plt.title('original track mask')
# plt.show()

# TRAIN_TRA_DIR = "/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N2DH-SIM+/02_GT/TRA/man_track140.tif"
# track_mask = cv2.imread(TRAIN_TRA_DIR, cv2.IMREAD_UNCHANGED)
# plt.figure()
# plt.imshow(track_mask)
# plt.axis('off')  # Hide axes
# plt.title('original track mask')
# plt.show()

# image_3d_dir = "/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N3DH-SIM+/02_GT/SEG/man_seg079.tif"
# image_3d = tiff.imread(image_3d_dir)
# print(image_3d.shape)
#
# print(f"unique elements: \n{np.unique(image_3d)}")
#
#
# middle_index = image_3d.shape[0] // 2
# middle_slice = image_3d[middle_index]
# print(f"middle_slice unique elements: \n{np.unique(middle_slice)}")

# ffig, axs = plt.subplots(1, 5, figsize=(15, 5))
# for i in range(5):
#     axs[i].imshow(image_3d[middle_index-2+i], cmap='gray')
#     axs[i].set_title(f"{i-2}")
#     axs[i].axis('off')  # Hide the axes
#
# # Adjust spacing between plots
# plt.tight_layout()
# plt.show()

# plt.imshow(middle_slice, cmap='gray')
# plt.colorbar()
# plt.title(f'Middle Slice (index: {middle_index})')
# plt.show()

# plt.figure()
# plt.imshow(image_3d)
# plt.axis('off')  # Hide axes
# plt.title('original image 3d')
# plt.show()

import torch
import torchio as tio
#
# class RandomCrop3D:
#     def __init__(self, crop_shape, threshold=500):
#         """
#         Initialize the transform with the desired crop shape.
#
#         :param crop_shape: tuple of 3 integers (crop_depth, crop_height, crop_width)
#         """
#         self.crop_shape = crop_shape
#         self.threshold = threshold
#
#     def __call__(self, image, mask):
#         """
#         Apply the random crop transform to the 3D tensor image.
#
#         :param image: torch tensor of shape (batch, depth, height, width)
#         :param mask: torch tensor of shape (batch, depth, height, width)
#         :return: cropped 3D tensor of shape crop_shape
#         """
#         depth, height, width = image.shape[-3:]
#         crop_depth, crop_height, crop_width = self.crop_shape
#
#         if crop_depth > depth or crop_height > height or crop_width > width:
#             raise ValueError("Crop shape is larger than the image dimensions")
#
#         while True:
#             # Random starting points
#             start_d = torch.randint(0, depth - crop_depth + 1, (1,)).item()
#             start_h = torch.randint(0, height - crop_height + 1, (1,)).item()
#             start_w = torch.randint(0, width - crop_width + 1, (1,)).item()
#             # Crop the image
#             cropped_image = image[:, start_d:start_d + crop_depth, start_h:start_h + crop_height, start_w:start_w + crop_width]
#             cropped_mask = mask[:, start_d:start_d + crop_depth, start_h:start_h + crop_height, start_w:start_w + crop_width]
#             if torch.sum(cropped_image > 0) > self.threshold * crop_depth:
#                 print(f"depth: {depth}, height: {height}, width: {width}")
#                 print(f"crop_depth: {crop_depth}, crop_height: {crop_height}, crop_width: {crop_width}")
#                 print(f"start_d: {start_d}, start_h: {start_h}, start_w: {start_w}")
#                 break
#
#         return cropped_image, cropped_mask
#
# image_dir = "/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N3DH-SIM+/02_GT/SEG/man_seg079.tif"
# img_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01/t070.tif"
# mask_path = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG/man_seg070.tif"
#
# image = tiff.imread(img_path).astype(np.float32)
# image = (image - image.mean()) / (image.std())
# mask = tiff.imread(mask_path).astype(np.float32)
#
# # image_middle_index = image.shape[0] // 2
#
# transforms = tio.Compose([
#     tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),  # Horizontal=2, vertical=1, depth=0 flip
#     # tio.RandomFlip(axes=2, flip_probability=1),  # Horizontal
#     # tio.RandomFlip(axes=1, flip_probability=1),  # vertical
#     # tio.RandomFlip(axes=0, flip_probability=1),  # depth
#     tio.RandomAffine(
#         scales=0.2,
#         degrees=15,
#     default_pad_value=0,
#     ),
#     RandomCrop3D(crop_shape=(5, 256, 256)),
# ])
# image_tensor, mask_tensor = torch.from_numpy(image), torch.from_numpy(mask)
# tras_image = transforms(image_tensor.unsqueeze(0))
# tras_mask = transforms(mask_tensor.unsqueeze(0))
# image_middle_index = tras_image.shape[1] // 2
# print(f"tras_image shape: {tras_image.shape}, middle = {image_middle_index}")
# ffig, axs = plt.subplots(2, 2, figsize=(15, 5))
#
# axs[0, 0].imshow(image[image_middle_index], cmap='gray')
# axs[0, 0].set_title(f"original image middle  slice")
# axs[0, 0].axis('off')  # Hide the axes
#
# axs[0, 1].imshow(image[image_middle_index + 5], cmap='gray')
# axs[0, 1].set_title(f"original image middle + 5 slice")
# axs[0, 1].axis('off')  # Hide the axes
#
# axs[1, 0].imshow(tras_image[0, -1-image_middle_index], cmap='gray')
# axs[1, 0].set_title(f"h_v_d flip image middle slice")
# axs[1, 0].axis('off')  # Hide the axes
#
# axs[1, 1].imshow(tras_image[0, image_middle_index-5], cmap='gray')
# axs[1, 1].set_title(f"h_v_d flip image middle - 5 slice")
# axs[1, 1].axis('off')  # Hide the axes
#
# # Adjust spacing between plots
# plt.tight_layout()
# plt.show()

# ffig, axs = plt.subplots(1, 2, figsize=(15, 5))
#
# axs[0].imshow(image[image_middle_index], cmap='gray')
# axs[0].set_title(f"original image middle  slice")
# axs[0].axis('off')  # Hide the axes
#
# axs[1].imshow(tras_image[0, image_middle_index], cmap='gray')
# axs[1].set_title(f"Affine image middle slice")
# axs[1].axis('off')  # Hide the axes
#
# # Adjust spacing between plots
# plt.tight_layout()
# plt.show()


# ffig, axs = plt.subplots(2, 5, figsize=(15, 5))
# for i in range(5):
#     axs[0, i].imshow(tras_image[0, image_middle_index-2+i], cmap='gray')
#     axs[0, i].set_title(f"{i-2}")
#     axs[0, i].axis('off')  # Hide the axes
#     axs[1, i].imshow(tras_mask[0, image_middle_index - 2 + i], cmap='gray')
#     axs[1, i].set_title(f"{i - 2}")
#     axs[1, i].axis('off')  # Hide the axes
#
# # Adjust spacing between plots
# ffig.suptitle("all aug", fontsize=16)
# plt.tight_layout()
# plt.show()


DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
print(DEVICE)