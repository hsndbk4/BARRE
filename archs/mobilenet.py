import math
import functools
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import torch.nn.init as init



class MobileNetV1(nn.Module):
    def __init__(self, width_mult = 1, res_mul =1, num_classes=10):
        super(MobileNetV1, self).__init__()
        #assert(width_mult <= 1 and width_mult >0)

        def conv_bn(inp, oup, stride):
            inp = round(inp*1.0)
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            inp = int(round(inp*width_mult))
            oup = int(round(oup*width_mult))
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 1),
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 1),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.d_in = int(round(1024*width_mult))
        self.classifier = nn.Linear(int(round(1024*width_mult)), num_classes)
    def forward(self, x):
            x = self.features(x)
            x = x.mean([2, 3])
            x = self.classifier(x)
            return x
