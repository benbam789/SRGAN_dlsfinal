import torch
import torch.nn as nn
import torch.nn.functional as F

class DWSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(DWSC, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, (kernel_size) // 2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, BN=False, act=None, stride=1, bias=True):
        super(conv, self).__init__()
        m = []
        m.append(DWSC(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, bias=True))

        if BN:
            m.append(nn.BatchNorm2d(num_features=out_channel))

        if act is not None:
            m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        out = self.body(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, act = nn.ReLU(inplace = True), bias = True):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(channels, channels, kernel_size, BN = True, act = act))
        m.append(conv(channels, channels, kernel_size, BN = True, act = None))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_res_block, act = nn.ReLU(inplace = True)):
        super(BasicBlock, self).__init__()
        m = []
        
        self.conv = conv(in_channels, out_channels, kernel_size, BN = False, act = act)
        for i in range(num_res_block):
            m.append(ResBlock(out_channels, kernel_size, act))
        
        m.append(conv(out_channels, out_channels, kernel_size, BN = True, act = None))
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.conv(x)
        out = self.body(res)
        out += res
        
        return out
        
class Upsampler(nn.Module):
    def __init__(self, channel, kernel_size, scale, act = nn.ReLU(inplace = True)):
        super(Upsampler, self).__init__()
        m = []
        m.append(conv(channel, channel * scale * scale, kernel_size))
        m.append(nn.PixelShuffle(scale))
    
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        out = self.body(x)
        return out

class discrim_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, act=nn.LeakyReLU(inplace=True)):
        super(discrim_block, self).__init__()
        m = []
        m.append(DWSC(in_channels=in_feats, out_channels=out_feats, kernel_size=kernel_size, stride=1, bias=True))
        m.append(DWSC(in_channels=out_feats, out_channels=out_feats, kernel_size=kernel_size, stride=2, bias=True))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        out = self.body(x)
        return out
