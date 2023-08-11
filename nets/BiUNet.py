import torch.nn as nn
import torch
from .blocks import PixelMerging, PixelExpanding
from .bra_legacy import BiLevelRoutingAttention

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, use_att=False, n_heads=2, n_win=7, topk=1, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.down = PixelMerging(in_channels, out_channels)
        self.nConvs = _make_nConv(out_channels, out_channels, nb_Conv, activation)
        self.use_att = use_att
        if use_att:
            self.att = BiLevelRoutingAttention(dim=out_channels, num_heads=n_heads, n_win=n_win, topk=topk)

    def forward(self, x):
        out = self.down(x)
        out = self.nConvs(out)
        if self.use_att:
            return self.att(out)
        else:
            return out

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, use_att=False, n_heads=2, n_win=7, topk=1, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up = PixelExpanding(in_channels//2,in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.use_att = use_att
        if use_att:
            self.att = BiLevelRoutingAttention(dim=out_channels, num_heads=n_heads, n_win=n_win, topk=topk)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        x = self.nConvs(x)
        if self.use_att:
            return self.att(x)
        else:
            return x

class BiUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2, use_att=True, n_heads=8, n_win=7, topk=16)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2, use_att=True, n_heads=4, n_win=7, topk=4)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2, use_att=True, n_heads=2, n_win=7, topk=1)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))

        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up4(x5, x4)
        x3 = self.up3(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up1(x2, x1)

        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x1))
        else:
            logits = self.outc(x1)
        return logits


