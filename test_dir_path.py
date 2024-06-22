import os
import torch
checkpoint_dir = "checkpoint"
TRAIN_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02"             # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02"
TRAIN_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_ERR_SEG"    # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02_ERR_SEG"
VAL_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01"               # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01"
VAL_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01_ERR_SEG"      # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01_ERR_SEG"

print(f"{os.listdir(checkpoint_dir)} ,os.listdir(checkpoint_dir)")
print(f"{len(os.listdir(TRAIN_IMG_DIR))} ,os.listdir(TRAIN_IMG_DIR)")
print("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    # torch.cuda.set_device(2)
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device index: {current_device}")
    print(f"Current CUDA device name: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available.")