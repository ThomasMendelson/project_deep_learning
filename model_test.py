import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange

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

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.embedding = nn.Conv2d(dim, dim, kernel_size=1)
        self.transformer = nn.Transformer(
            d_model=dim, nhead=num_heads, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dropout=dropout, dim_feedforward=dim * 4
        )
        self.project_back = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.embedding(x)
        x = rearrange(x, 'b c h w -> (h w) b c')  # Prepare for transformer [(h * w), batch, channel]

        # Since nn.Transformer expects [sequence, batch, dim], we need both src and tgt
        src = x
        tgt = x

        x = self.transformer(src, tgt)
        x = rearrange(x, '(h w) b c -> b c h w', h=h, w=w)  # Back to image shape
        x = self.project_back(x)
        return x


class MultiUNetWithTransformer(nn.Module):
    def __init__(self, unet_params, transformer_params):
        super(MultiUNetWithTransformer, self).__init__()
        self.unet1 = UNET(**unet_params)
        self.unet2 = UNET(**unet_params)
        self.unet3 = UNET(**unet_params)
        self.transformer = TransformerBlock(**transformer_params)

    def forward(self, x):
        out1 = self.unet1(x)
        out2 = self.unet2(x)
        out3 = self.unet3(x)

        # Compute the mean of the outputs
        mean_output = (out1 + out2 + out3) / 3.0

        # Pass the mean output through the transformer
        transformed_output = self.transformer(mean_output)

        return transformed_output

def t():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    t()

    unet_params = {
        'in_channels': 1,
        'out_channels': 3,
        'features': [64, 128, 256, 512, 1024]
    }

    transformer_params = {
        'dim': 3,  # Should match the output channels of the UNET
        'num_heads': 1,
        'num_layers': 4,
        'dropout': 0.1
    }

    model = MultiUNetWithTransformer(unet_params, transformer_params)

    # Example input
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(output.shape)
