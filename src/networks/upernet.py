"""
MIT License

Copyright (c) 2020 Yassine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Link to the original code: https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPModule(nn.Module):
    
    # In the original implementation they use precise RoI pooling
    # Instead of using adaptive average pooling
    def __init__(self, in_channels, bin_sizes):
        super().__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList(
            [
                self._make_stages(in_channels, out_channels, b_s)
                for b_s in bin_sizes
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + (out_channels * len(bin_sizes)), in_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend(
            [F.interpolate(
                stage(features), size=(h, w), mode='bilinear',
                align_corners=True
            ) for stage in self.stages]
        )
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    
    def __init__(self, feature_channels):
        super().__init__()
        # assert feature_channels[0] == fpn_out
        fpn_out = feature_channels[0]
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(ft_size, fpn_out, kernel_size=1)
             for ft_size in feature_channels[1:]]
        )
        self.smooth_conv = nn.ModuleList(
            [nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
            * (len(feature_channels) - 1)
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]
        
        x = self.conv_fusion(torch.cat((P), dim=1))
        return x
