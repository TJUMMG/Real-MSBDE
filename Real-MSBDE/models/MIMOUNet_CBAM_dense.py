import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel):
        super(EBlock, self).__init__()
        self.Res1 = ResBlock(out_channel, out_channel)
        self.Res2 = ResBlock(out_channel, out_channel)
        self.Res3 = ResBlock(out_channel, out_channel)
        self.Res4 = ResBlock(out_channel, out_channel)
        self.conv1 = BasicConv(out_channel * 2, out_channel, kernel_size=1, stride=1, relu=False)
        self.conv2 = BasicConv(out_channel * 3, out_channel, kernel_size=1, stride=1, relu=False)
        self.conv3 = BasicConv(out_channel * 4, out_channel, kernel_size=1, stride=1, relu=False)


    def forward(self, x):
        res1 = self.Res1(x)
        z = torch.cat([x, res1], dim=1)
        z = self.conv1(z)
        res2 = self.Res2(z)
        z = torch.cat([x, res1, res2], dim=1)
        z = self.conv2(z)
        res3 = self.Res3(z)
        z = torch.cat([x, res1, res2, res3], dim=1)
        z = self.conv3(z)
        res4 = self.Res4(z)
        return res4


class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()

        self.Res1 = ResBlock(channel, channel)
        self.Res2 = ResBlock(channel, channel)
        self.Res3 = ResBlock(channel, channel)
        self.Res4 = ResBlock(channel, channel)
        self.conv1 = BasicConv(channel * 2, channel, kernel_size=1, stride=1, relu=False)
        self.conv2 = BasicConv(channel * 3, channel, kernel_size=1, stride=1, relu=False)
        self.conv3 = BasicConv(channel * 4, channel, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        res1 = self.Res1(x)
        z = torch.cat([x, res1], dim=1)
        z = self.conv1(z)
        res2 = self.Res2(z)
        z = torch.cat([x, res1, res2], dim=1)
        z = self.conv2(z)
        res3 = self.Res3(z)
        z = torch.cat([x, res1, res2, res3], dim=1)
        z = self.conv3(z)
        res4 = self.Res4(z)
        return res4


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = BasicConv(channel, channel // ratio, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(channel // ratio, channel, kernel_size=1, stride=1, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.conv1(self.avg_pool(x)))
        max_out = self.conv2(self.conv1(self.max_pool(x)))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = self.conv1 = BasicConv(2, 1, kernel_size=7, stride=1, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x



class MIMOUNet(nn.Module):
    def __init__(self):
        super(MIMOUNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel),
            EBlock(base_channel*2),
            EBlock(base_channel*4),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4),
            DBlock(base_channel * 2),
            DBlock(base_channel)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])
        self.CBAM = nn.ModuleList([
            CBAM(base_channel, base_channel),
            CBAM(base_channel*2, base_channel*2),
            CBAM(base_channel*4, base_channel*4),
            CBAM(base_channel, base_channel),
            CBAM(base_channel * 2, base_channel * 2),
            CBAM(base_channel * 4, base_channel * 4)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):

        x = torch.tensor(x, dtype=torch.float)
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        c11 = self.CBAM[0](res1)
        c12 = self.CBAM[1](z21)
        c13 = self.CBAM[2](z41)
        c21 = self.CBAM[3](z12)
        c22 = self.CBAM[4](res2)
        c23 = self.CBAM[5](z42)

        res1 = self.AFFs[0](c11, c12, c13)
        res2 = self.AFFs[1](c21, c22, c23)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs




def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNet":
        return MIMOUNet()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')