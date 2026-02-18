import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d


class MultiLevelNeck(nn.Module):
    
    def __init__(self, in_channels: list[int], out_channels: int, scales: list[float]):
        super().__init__()
        assert isinstance(in_channels, list)
        assert len(in_channels) == len(scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        kernel_size=1,
                        padding="same"
                    ),
                    nn.ReLU()
                )
                for in_channel in in_channels
            ]
        )
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding="same"
                    ),
                    nn.ReLU()
                )
                for _ in range(self.num_outs)
            ]
        )
    
    def forward(self, inputs):
        return [
            self.convs[i](
                F.interpolate(
                    self.lateral_convs(inputs[i]), scale_factor=self.scales[i], mode='bilinear',
                    align_corners=True
                )
            )
            for i in range(self.num_outs)
        ]
