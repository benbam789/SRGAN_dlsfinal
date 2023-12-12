import torch
import torch.nn as nn
from dwsc import*

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = DWSC(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=True)
        self.key = DWSC(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=True)
        self.value = DWSC(in_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        in_channels = num_channels  # Get the actual number of channels from the input tensor
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, height * width)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, num_channels, height, width)
        out = self.gamma * out + x

        return out


class Generator(nn.Module):
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, num_block=16, act=nn.PReLU(), scale=4):
        super(Generator, self).__init__()

        self.conv01 = conv(in_channel=img_feat, out_channel=n_feats, kernel_size=9, BN=False, act=act)

        resblocks = [ResBlock(channels=n_feats, kernel_size=3, act=act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)

        # Add self-attention after certain blocks or layers
        self.attention = SelfAttention(n_feats)

        self.conv02 = conv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=True, act=None)

        if scale == 4:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=scale, act=act)]

        self.tail = nn.Sequential(*upsample_blocks)

        self.last_conv = conv(in_channel=n_feats, out_channel=img_feat, kernel_size=3, BN=False, act=nn.Tanh())

    def forward(self, x):
        x = self.conv01(x)
        _skip_connection = x

        x = self.body(x)

        # Apply self-attention
        x = self.attention(x)

        x = self.conv02(x)
        feat = x + _skip_connection

        x = self.tail(feat)
        x = self.last_conv(x)

        return x, feat

class Discriminator(nn.Module):
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, act=nn.LeakyReLU(inplace=True), num_of_block=3, patch_size=96):
        super(Discriminator, self).__init__()
        self.act = act

        self.conv01 = DWSC(in_channels=img_feat, out_channels=n_feats, kernel_size=3, stride=1, bias=False)
        self.conv02 = DWSC(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, bias=False)

        # Add self-attention after certain blocks or layers
        self.attention = SelfAttention(n_feats * 2 ** (num_of_block - 1))

        body = [discrim_block(in_feats=n_feats * (2 ** i), out_feats=n_feats * (2 ** (i + 1)), kernel_size=3, act=self.act) for i in range(num_of_block)]
        self.body = nn.Sequential(*body)

        self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))

        tail = []

        tail.append(nn.Linear(self.linear_size, 1024))
        tail.append(self.act)
        tail.append(nn.Linear(1024, 1))
        tail.append(nn.Sigmoid())

        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.conv01(x)
        x = self.conv02(x)

        # Apply self-attention
        x = self.attention(x)

        x = self.body(x)
        x = x.view(-1, self.linear_size)
        x = self.tail(x)

        return x

