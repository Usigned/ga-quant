"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize_utils import QConv2d


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, conv_layer=nn.Conv2d):
        super().__init__()

        self.residual = nn.Sequential(
            conv_layer(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            conv_layer(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            conv_layer(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100, conv_layer=nn.Conv2d):
        super().__init__()

        self.pre = nn.Sequential(
            conv_layer(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1, conv_layer=conv_layer)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6, conv_layer=conv_layer)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6, conv_layer=conv_layer)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6, conv_layer=conv_layer)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6, conv_layer=conv_layer)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6, conv_layer=conv_layer)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6, conv_layer=conv_layer)

        self.conv1 = nn.Sequential(
            conv_layer(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = conv_layer(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t, **kwargs):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t, **kwargs))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, **kwargs))
            repeat -= 1

        return nn.Sequential(*layers)

def mobilenetv2():
    return MobileNetV2()

def qmobilenetv2(path=None):
    '''
    no quant-policy, all QModule are, by default, not quantized
    more details on how to use this model to quantization, see env.linear_quantize_env._build_index
    '''
    qnet = MobileNetV2(conv_layer=QConv2d)
    if path is not None:
        ch = {n.replace('module.', ''): v for n, v in torch.load(path).items()}
        qnet.load_state_dict(ch, strict=False)
    return qnet