import argparse
from train_vit2 import main as vit_unet
from train_swin import main as swin_unetr
from train3D import main as unet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="model type to train")
    args = parser.parse_args()

    if args.model == "ViT_UNet":
        vit_unet()
    elif args.model == "SwinUNETR":
        swin_unetr()
    elif args.model == "unet":
        unet()



