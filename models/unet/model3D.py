import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET3D(nn.Module):
    def __init__(
            # self, in_channels=1, out_channels=3, features=[64, 128, 256, 512, 1024],
            self, in_channels=1, out_channels=3, features=[64, 128, 256],
            # self, in_channels=1, out_channels=3, features=[64, 128, 256],
    ):

        super(UNET3D, self).__init__()
        self.ups = nn.ModuleList()
        self.up_marker = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv3D(feature*2, feature))
        self.up_marker.append(nn.ConvTranspose3d(features[0]*2, features[0], kernel_size=2, stride=2,))
        self.up_marker.append(DoubleConv3D(features[0] * 2, features[0]))

        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)
        self.out_classes = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.out_marker = nn.Conv3d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
             if idx == len(self.ups)-2:
                 x_classes, x_markers = self.ups[idx](x), self.up_marker[0](x)
                 skip_connection = skip_connections[idx // 2]
                 if x_classes.shape != skip_connection.shape:
                     x_classes = F.interpolate(x_classes, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
                 if x_markers.shape != skip_connection.shape:
                     x_markers = F.interpolate(x_markers, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)

                 x_classes, x_markers = torch.cat((skip_connection, x_classes), dim=1), torch.cat((skip_connection, x_markers), dim=1)
                 x_classes, x_markers = self.ups[idx + 1](x_classes), self.up_marker[1](x_markers)

             else:
                 x = self.ups[idx](x)
                 skip_connection = skip_connections[idx//2]

                 if x.shape != skip_connection.shape:
                     x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)

                 concat_skip = torch.cat((skip_connection, x), dim=1)
                 x = self.ups[idx+1](concat_skip)

        return self.out_classes(x_classes), self.out_marker(x_markers)

        #     x = self.ups[idx](x)
        #     skip_connection = skip_connections[idx//2]
        #
        #     if x.shape != skip_connection.shape:
        #         x = F.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
        #
        #     concat_skip = torch.cat((skip_connection, x), dim=1)
        #     x = self.ups[idx+1](concat_skip)
        # return self.out_classes(x), None



def t():
    DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
    x = torch.randn((16, 1, 16,  256, 256))#.to(DEVICE)
    model = UNET3D(in_channels=1, out_channels=3)#.to(DEVICE)
    class_predictions, marker_predictions = model(x)
    print(f"class_predictions.shape: {class_predictions.shape}")
    print(f"marker_predictions.shape: {marker_predictions.shape}")
    print(f"x.shape: {x.shape}")
    # summary(model, input_size=(16, 1, 16,  256, 256))

if __name__ == "__main__":
    from torchinfo import summary
    # t()

    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    CROP_SIZE = (32, 128, 128)
    model = UNET3D(in_channels=1, out_channels=3).to(DEVICE)

    print("Summary for model3D:")
    print(summary(model, depth=3, input_size=(1, 1, 32, 128, 128),
                  col_names=["input_size", "output_size", "num_params"], device=DEVICE))
