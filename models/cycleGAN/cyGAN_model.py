import torch
import torch.nn.functional as F
from torch import nn

from utils.arguments import CycleGAN_Option


class ResidualBlock(nn.Module):
    def __init__(self, in_features, net_G_drop_out):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(net_G_drop_out),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, net_G_depth=9, net_G_downsampling=2, net_G_channel=64, net_G_drop_out=0.5, **args):
        super().__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, net_G_channel, 7),
            nn.InstanceNorm2d(net_G_channel),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = net_G_channel
        out_features = in_features * 2
        for _ in range(net_G_downsampling):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(net_G_depth):
            model += [ResidualBlock(in_features, net_G_drop_out)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(net_G_downsampling):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(in_features, output_nc, 7), nn.Tanh()]
        # model += [nn.Conv2d(in_features, output_nc, 1), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc, depth=4, channels=64):
        super(Discriminator, self).__init__()
        # nn.Tanh(),
        model = [nn.Conv2d(input_nc, channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        # A bunch of convolutions one after another
        for _ in range(depth - 1):
            model += [
                nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels *= 2

        """
        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]
        """
        # FCN classification layer
        model += [nn.Conv2d(channels, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor):
        # if seg is not None:
        #    x = torch.cat((x, seg), 1)
        x = self.model(x)
        # if seg is not None:
        #    with torch.no_grad():
        #        seg -= seg.min()
        #        seg[seg != 0] = 1
        #        seg = torch.nn.functional.interpolate(seg, x.shape[2:], mode="nearest")
        #        seg = seg.detach()
        #    x = x * seg
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Discriminator3D(nn.Module):
    def __init__(self, input_nc, depth=4, channels=64):
        super(Discriminator3D, self).__init__()
        # nn.Tanh(),
        model = [nn.Conv3d(input_nc, channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        # A bunch of convolutions one after another
        for _ in range(depth - 1):
            model += [
                nn.Conv3d(channels, channels * 2, 4, stride=2, padding=1),
                nn.InstanceNorm3d(128),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels *= 2
        # FCN classification layer
        model += [nn.Conv3d(channels, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)


class ChannelNormPatchDiscriminator(nn.Module):
    def __init__(self, input_nc, opt: CycleGAN_Option):  # input_nc, depth=4, channels=64,
        super().__init__()
        from models.nn import conv_nd, normalization

        channels = opt.net_D_channel
        depth = opt.net_D_depth
        activation = nn.SiLU()
        # nn.Tanh(),
        model = [conv_nd(opt.dims, input_nc, channels, 4, stride=2, padding=1), activation]
        # A bunch of convolutions one after another
        for _ in range(depth - 1):
            model += [
                conv_nd(opt.dims, channels, channels * 2, 4, stride=2, padding=1),
                normalization(channels * 2, target=16),
                activation,
            ]
            channels *= 2
        # FCN classification layer
        model += [conv_nd(opt.dims, channels, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        from torch.nn.utils import spectral_norm

        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
