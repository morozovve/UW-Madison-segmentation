import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            # first conv
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # second conv
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.double_conv(x)

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(self.mp(x))

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1up = self.up(x1)
        return self.conv(torch.cat([x1up, x2], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=3) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.c1 = DoubleConv(1, 64)
        self.d1 = DownConv(64, 128)
        self.d2 = DownConv(128, 256)
        self.d3 = DownConv(256, 512)
        self.d4 = DownConv(512, 1024)

        self.u4 = UpConv(1024, 512)
        self.u3 = UpConv(512, 256)
        self.u2 = UpConv(256, 128)
        self.u1 = UpConv(128, 64)
        self.out = OutConv(64, self.n_classes)

    def forward(self, x):
        f1 = self.c1(x)
        f2 = self.d1(f1)
        f3 = self.d2(f2)
        f4 = self.d3(f3)

        feat = self.d4(f4)

        feat = self.u4(feat, f4)
        feat = self.u3(feat, f3)
        feat = self.u2(feat, f2)
        feat = self.u1(feat, f1)

        logits = self.out(feat)
        return logits

