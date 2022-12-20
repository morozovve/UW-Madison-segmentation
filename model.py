import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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
    def __init__(self, n_classes=3, model_depth=64) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.c1 = DoubleConv(1, model_depth)
        self.d1 = DownConv(model_depth, model_depth * 2)
        model_depth *= 2
        self.d2 = DownConv(model_depth, model_depth * 2)
        model_depth *= 2
        self.d3 = DownConv(model_depth, model_depth * 2)
        model_depth *= 2
        self.d4 = DownConv(model_depth, model_depth * 2)

        self.u4 = UpConv(model_depth * 2, model_depth)
        self.u3 = UpConv(model_depth, model_depth // 2)
        model_depth //= 2
        self.u2 = UpConv(model_depth, model_depth // 2)
        model_depth //= 2
        self.u1 = UpConv(model_depth, model_depth // 2)
        model_depth //= 2
        self.out = OutConv(model_depth, self.n_classes)

        self.model_weight_initializer()

    def model_weight_initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight'):
                    init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    init.normal_(m.weight.data, mean=1, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

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
