import torch
import torch.nn as nn
from monai.networks.nets import ViT


class Dconv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dconv3DBlock, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.relu(self.bn1(self.deconv(x)))
        res = self.relu(self.bn2(self.conv(res)))
        return res


class DconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.relu(self.bn1(self.deconv(x)))
        res = self.relu(self.bn2(self.conv(res)))
        return res




class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, final_layer=False):
        super(UpBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.final_layer = final_layer
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        if x2 is not None:
            diffH = x2.size()[2] - x1.size()[2]
            diffW = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
            x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.final_layer:
            return x
        return self.deconv(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, final_layer=False):
        super(UpBlock3D, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.final_layer = final_layer
        self.deconv = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        if x2 is not None:
            diffD = x2.size()[2] - x1.size()[2]
            diffH = x2.size()[3] - x1.size()[3]
            diffW = x2.size()[4] - x1.size()[4]
            x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffD // 2,
                                        diffD - diffD // 2])
            x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.final_layer:
            return x
        return self.deconv(x)


class ViT_UNet(nn.Module):
    def __init__(self, device, in_channels=1, out_channels=3, img_size=(256, 256), patch_size=16, hidden_size=512,
                 mlp_dim=3072, num_layers=12, num_heads=12, proj_type="conv", dropout_rate=0., spatial_dims=2,
                 qkv_bias=True, classification=False, three_d=False, with_markers=True):
        super(ViT_UNet, self).__init__()
        self.device = device
        self.three_d = three_d
        self.with_markers = with_markers
        if self.three_d:
            self.vit = ViT(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                proj_type=proj_type,
                dropout_rate=dropout_rate,
                classification=classification,
                spatial_dims=spatial_dims,
                qkv_bias=qkv_bias,
            )
            self.doubleconv = UpBlock3D(in_channels, 32, 16, True)

            self.up1 = UpBlock3D(512, 256, 512)
            self.up2 = UpBlock3D(2 * 256, 128, 256)
            self.up3 = UpBlock3D(2 * 128, 64, 128)
            self.up4 = UpBlock3D(2 * 64, 32, 64)

            self.beforelast_classes = UpBlock3D(2 * 32, 32, 32, True)
            self.beforelast_marker = UpBlock3D(2 * 32, 32, 32, True)

            self.out_classes = nn.Conv3d(32, out_channels, kernel_size=1)
            self.out_marker = nn.Conv3d(32, 1, kernel_size=1)
        else:
            self.vit = ViT(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                proj_type=proj_type,
                dropout_rate=dropout_rate,
                classification=classification,
                spatial_dims=spatial_dims,
                qkv_bias=qkv_bias,
            )
            self.doubleconv = UpBlock(in_channels, 32, 16, True)

            self.up1 = UpBlock(512, 256, 512)
            self.up2 = UpBlock(2 * 256, 128, 256)
            self.up3 = UpBlock(2 * 128, 64, 128)
            self.up4 = UpBlock(2 * 64, 32, 64)

            self.beforelast_classes = UpBlock(2 * 32, 32, 32, True)
            self.beforelast_marker = UpBlock(2 * 32, 32, 32, True)

            self.out_classes = nn.Conv2d(32, out_channels, kernel_size=1)
            self.out_marker = nn.Conv2d(32, 1, kernel_size=1)

    def reconstruct_image_from_patches(self, patches, img_size, patch_size):
        """
           Reconstructs an image from its patches.

           Args:
               patches (torch.Tensor): Tensor of shape (num_states, batch_size, num_patches, hidden_size) containing the image patches (ViT output).
               img_size (tuple): Tuple indicating the size of the original image. For 3D images, this would be (D, H, W). For 2D images, this would be (H, W).
               patch_size (int or tuple): Size of each patch. If int, it is assumed that the patch size is the same in all dimensions. If tuple, it specifies the patch size in each dimension.
           Returns:
               torch.Tensor: Reconstructed image tensor. For 3D images, the shape will be (num_states, batch_size, hidden_size, D, H, W). For 2D images, the shape will be (num_states, batch_size, hidden_size, H, W).
           """
        if self.three_d:
            if isinstance(patch_size, int):
                num_patches_d = img_size[0] // patch_size
                num_patches_h = img_size[1] // patch_size
                num_patches_w = img_size[2] // patch_size
            elif len(patch_size) == len(img_size):
                num_patches_d = img_size[0] // patch_size[0]
                num_patches_h = img_size[1] // patch_size[1]
                num_patches_w = img_size[2] // patch_size[2]
            else:
                raise ValueError(
                    f"Image size dimensions {len(img_size)} and patch size dimensions {len(patch_size)} do not match.")
            # Reshape patches to (num_states, batch_size, num_patches_d, num_patches_h, num_patches_w, hidden_size)
            image = patches.view(patches.shape[0], patches.shape[1], num_patches_d, num_patches_h, num_patches_w,
                                 patches.shape[-1])
            image = image.permute(0, 1, 5, 2, 3, 4)
        else:
            if isinstance(patch_size, int):
                num_patches_h = img_size[0] // patch_size
                num_patches_w = img_size[1] // patch_size
            elif len(patch_size) == len(img_size):
                num_patches_h = img_size[0] // patch_size[-2]
                num_patches_w = img_size[1] // patch_size[-1]
            else:
                raise ValueError(
                    f"Image size dimensions {len(img_size)} and patch size dimensions {len(patch_size)} do not match.")

            image = patches.view(patches.shape[0], patches.shape[1], num_patches_h, num_patches_w, patches.shape[-1])
            image = image.permute(0, 1, 4, 2, 3)
        return image

    def create_skip_connection(self, hidden_state, img_size, patch_size):
        """
               Reconstructs an image from its patches.

               Args:
                   hidden_state (torch.Tensor): Tensor of shape (num_hidden_state, batch_size, num_patches, hidden_size) containing the image patches (ViT output).
                   img_size (tuple): Tuple indicating the size of the original image. For 3D images, this would be (D, H, W). For 2D images, this would be (H, W).
                   patch_size (int or tuple): Size of each patch. If int, it is assumed that the patch size is the same in all dimensions. If tuple, it specifies the patch size in each dimension.
               Returns:
                    list: A list of torch.Tensors, each representing a reconstructed image with applied deconvolutional blocks.
                    The length of the list corresponds to the number of skip connections, and each tensor in the list has the shape:
                    - For 3D images: (batch_size, out_channels, D, H, W)
                    - For 2D images: (batch_size, out_channels, H, W)
               """

        images_to_skip_connections = self.reconstruct_image_from_patches(patches=hidden_state[-3:], img_size=img_size,
                                                                         patch_size=patch_size)
        skip_connections = []

        for i, image in enumerate(images_to_skip_connections):
            in_channels = hidden_state.shape[-1]
            for _ in range(i + 1):
                if self.three_d:
                    block = Dconv3DBlock(in_channels=in_channels, out_channels=(in_channels // 2)).to(self.device)
                else:
                    block = DconvBlock(in_channels=in_channels, out_channels=(in_channels // 2)).to(self.device)
                in_channels = in_channels // 2
                image = block(image)
            skip_connections.append(image)
        return skip_connections

    def forward(self, x):
        # Forward pass through ViT with hidden states
        if self.three_d:
            img_size = x.shape[-3:]
            x, skip_connections = self.vit(x), [self.doubleconv(x, None)]
        else:
            img_size = x.shape[-2:]
            x, skip_connections = self.vit(x), [self.doubleconv(x, None)]

        if isinstance(x, tuple):
            x, hidden_states_out = x
        x = self.reconstruct_image_from_patches(patches=x.unsqueeze(0), img_size=img_size, patch_size=16).squeeze(0)
        hidden_states_out = torch.stack(hidden_states_out, dim=0)
        skip_connections = self.create_skip_connection(hidden_state=hidden_states_out, img_size=img_size,
                                                       patch_size=16) + skip_connections
        # skip_connections = self.create_skip_connection(hidden_state=hidden_states_out, img_size=img_size,
        #                                                patch_size=16)
        x = self.up1(x, None)
        x = self.up2(x, skip_connections[0])
        x = self.up3(x, skip_connections[1])
        x = self.up4(x, skip_connections[2])
        x_classes = self.beforelast_classes(x, skip_connections[3])
        if self.with_markers:
            x_marker = self.beforelast_marker(x, skip_connections[3])
            return self.out_classes(x_classes), self.out_marker(x_marker)
        return self.out_classes(x_classes), None



# #-----------------------------------------------------------------------------------------


if __name__ == "__main__":
    from torchinfo import summary

    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    #
    # input_tensor2d = torch.rand((64, 1, 256, 256)).to(DEVICE)
    # # model2D = ViT_UNet(in_channels=1, out_channels=3, img_size=(256, 256), patch_size=16, hidden_size=512,
    # #                    mlp_dim=2048, num_layers=12, num_heads=8, proj_type="conv", dropout_rate=0.,
    # #                    classification=False, three_d=False, device=DEVICE).to(DEVICE)
    #
    # input_tensor2d = torch.rand((64, 1, 256, 256)).to(DEVICE)
    # model2D = ViT(
    #     in_channels=1,
    #     img_size=(256, 256),
    #     patch_size=16,
    #     hidden_size=512,
    #     mlp_dim=2048,
    #     num_layers=12,
    #     num_heads=8,
    #     proj_type="conv",
    #     dropout_rate=0.,
    #     classification=False,
    #     spatial_dims=2,
    # ).to(DEVICE)
    # # Input tensor (batch size of 2, 1 channel, 16x128x128 image)
    #
    # # Forward pass
    # output2D = model2D(input_tensor2d)
    #
    # # Check if output is a tuple and handle accordingly
    # if isinstance(output2D, tuple):
    #     main_output2D, hidden_states_out = output2D
    # else:
    #     main_output2D = output2D
    #     hidden_states_out = None
    #
    # # Print the shape of the main output
    # print(main_output2D.shape)  # Output shape will be (batch_size, num_patches, hidden_size)
    #
    # print("Summary for model2D:")
    # print(summary(model2D, depth=3, input_size=(1, 1, 256, 256), col_names=["input_size", "output_size", "num_params"],
    #               device=DEVICE))

    model3D = ViT_UNet(in_channels=1, out_channels=3, img_size=(32, 128, 128), patch_size=16, hidden_size=512,
                       mlp_dim=2048, num_layers=4, num_heads=8, proj_type="conv", dropout_rate=0.1,qkv_bias=True,
                       classification=False, three_d=True, device=DEVICE, spatial_dims=3, with_markers=True).to(DEVICE)

    # input_tensor3d = torch.rand((1, 1, 32, 128, 128)).to(DEVICE)
    # # Forward pass
    # output3D = model3D(input_tensor3d)
    #
    # # Check if output is a tuple and handle accordingly
    # if isinstance(output3D, tuple):
    #     main_output3D, hidden_states_out = output3D
    # else:
    #     main_output3D = output3D
    #     hidden_states_out = None
    #
    # # Print the shape of the main output
    # print(main_output3D.shape)  # Output shape will be (batch_size, num_patches, hidden_size)

    print("Summary for model3D:")
    print(summary(model3D, depth=3, input_size=(1, 1, 32, 128, 128),
                  col_names=["input_size", "output_size", "num_params"], device=DEVICE))
