import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class CustomSwinUNETR(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, feature_size=48):
        super(CustomSwinUNETR, self).__init__()
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=True
        )
        self.class_head = nn.Conv3d(feature_size, out_channels, kernel_size=1)
        self.marker_head = nn.Conv3d(feature_size, 1, kernel_size=1)

        self.beforelast_classes = DoubleConv3D(feature_size, 32, 32)
        self.beforelast_marker = DoubleConv3D(feature_size, 32, 32)

        self.out_classes = nn.Conv3d(32, out_channels, kernel_size=1)
        self.out_marker = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.swin_unetr(x)
        x_class = self.beforelast_classes(x)
        x_class = self.out_classes(x_class)

        x_marker = self.beforelast_marker(x)
        x_marker = self.out_marker(x_marker)
        # Crop the outputs back to the original dimensions (16, 128, 128)
        # class_output = class_output[:, :, 8:24, :, :]
        # marker_output = marker_output[:, :, 8:24, :, :]
        return x_class, x_marker


def get_SwinUNETR_model(crop_size, device, pretrained_path):
    model = CustomSwinUNETR(
        img_size=crop_size,
        in_channels=1,
        out_channels=3,
        feature_size=48,
    ).to(device)

    # Load the pre-trained weights
    # pretrained_path = r'C:\Users\beaviv\PycharmProjects\ImageProcessing\pre_trained_models\swin_unetr_btcv_segmentation\models\model.pt'

    pretrained_dict = torch.load(pretrained_path, map_location=device)

    # Get the state dict of the current model
    model_dict = model.state_dict()

    # Filter out the keys with mismatched shapes
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and v.shape == model_dict[k].shape}

    # Update the current model's state dict
    model_dict.update(pretrained_dict)

    # Load the updated state dict into the model
    model.load_state_dict(model_dict)
    return model

