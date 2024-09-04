import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=3, features=[64, 128, 256, 512, 1024],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.up_marker = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        self.up_marker.append(nn.ConvTranspose2d(features[0] * 2, features[0], kernel_size=2, stride=2, ))
        self.up_marker.append(DoubleConv(features[0] * 2, features[0]))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.out_classes = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.out_marker = nn.Conv2d(features[0], 1, kernel_size=1)

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
                    # x_classes = F.interpolate(x_classes, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
                    x_classes = TF.resize(x_classes, size=skip_connection.shape[2:])
                if x_markers.shape != skip_connection.shape:
                    # x_markers = F.interpolate(x_markers, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
                    x_markers = TF.resize(x_markers, size=skip_connection.shape[2:])

                x_classes, x_markers = torch.cat((skip_connection, x_classes), dim=1), torch.cat((skip_connection, x_markers), dim=1)
                x_classes, x_markers = self.ups[idx + 1](x_classes), self.up_marker[1](x_markers)

            else:
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]

                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])

                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)

        return self.out_classes(x_classes), self.out_marker(x_markers)

def t():
    x = torch.randn((3, 1, 256, 256))
    model = UNET(in_channels=1, out_channels=3)
    class_predictions, marker_predictions = model(x)
    print(f"class_predictions.shape: {class_predictions.shape}")
    print(f"marker_predictions.shape: {marker_predictions.shape}")
    print(f"x.shape: {x.shape}")

if __name__ == "__main__":
    t()