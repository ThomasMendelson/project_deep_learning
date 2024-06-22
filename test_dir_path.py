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

image_3d_dir = "/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N3DH-SIM+/02_GT/SEG/man_seg079.tif"
image_3d = tiff.imread(image_3d_dir)
print(image_3d.shape)

print(f"unique elements: \n{np.unique(image_3d)}")


middle_index = image_3d.shape[0] // 2
middle_slice = image_3d[middle_index]
print(f"middle_slice unique elements: \n{np.unique(middle_slice)}")

ffig, axs = plt.subplots(1, 5, figsize=(15, 5))

axs[0].imshow(image_3d[middle_index-2], cmap='gray')
axs[0].set_title("-2")
axs[0].axis('off')  # Hide the axes

axs[1].imshow(image_3d[middle_index-1], cmap='gray')
axs[1].set_title("-1")
axs[1].axis('off')  # Hide the axes


axs[2].imshow(image_3d[middle_index], cmap='gray')
axs[2].set_title("0")
axs[2].axis('off')  # Hide the axes


axs[3].imshow(image_3d[middle_index+1], cmap='gray')
axs[3].set_title("1")
axs[3].axis('off')  # Hide the axes

axs[4].imshow(image_3d[middle_index+2], cmap='gray')
axs[4].set_title("2")
axs[4].axis('off')  # Hide the axes

# Adjust spacing between plots
plt.tight_layout()
plt.show()

# plt.imshow(middle_slice, cmap='gray')
# plt.colorbar()
# plt.title(f'Middle Slice (index: {middle_index})')
# plt.show()

# plt.figure()
# plt.imshow(image_3d)
# plt.axis('off')  # Hide axes
# plt.title('original image 3d')
# plt.show()