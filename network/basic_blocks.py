import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0), groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        # add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv2(x)
        return x


class single_conv(nn.Module):
    '''(conv => BN => ReLU) * 1'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(single_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class single_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 1'''

    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(single_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        # add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch, group_conv=False, dilation=dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
            else:
                self.conv = double_conv(in_ch, out_ch, group_conv=False, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=False, group_conv=False, use_dropblock=False,
                 drop_p=0.5):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2, groups=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch, group_conv=group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch, group_conv=group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
