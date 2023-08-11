import torch.nn as nn
import torch
from einops import rearrange

class PixelMerging(nn.Module):

    def __init__(self,in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.merge = nn.Conv2d(4 * in_dim, out_dim, kernel_size=1, bias=False)
        self.norm = norm_layer(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        x = self.merge(x)
        x = rearrange(x, 'b c h w-> b h w c') # B H/2 W/2 C
        x = self.norm(x)
        x = rearrange(x, 'b h w c-> b c h w') # B C H/2 W/2
        x = self.relu(x)

        return x


class PixelExpanding(nn.Module):

    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Conv2d(in_dim, 4*out_dim, kernel_size=1, bias=False)
        self.norm = norm_layer(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x = self.expand(x)
        C = x.shape[1]

        x = rearrange(x, 'b (p1 p2 c) h w-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = self.norm(x)
        x = rearrange(x, 'b h w c-> b c h w')
        x = self.relu(x)

        return x