#BSD 3-Clause License
#=======
#Based on Xilinx BNN training codes
#Copyright (c) 2019, Xilinx Inc.
#All rights reserved.
#
#Based on Matthieu Courbariaux's BinaryNet example
#Copyright (c) 2015-2016, Matthieu Courbariaux
#All rights reserved
#

import torch.nn as nn
from torch.nn import Parameter,Module
import torch.nn.functional as F
from quantization import *

class HingeLoss(Module):
    def __init__(self):
        super(HingeLoss,self).__init__()

    def forward(self, output, target):
        target_=2*torch.eye(5)[target]-1
        hinge_loss = 1 - torch.mul(output, target_)
        hinge_loss[hinge_loss < 0] = 0
        hinge_loss=torch.sqrt(hinge_loss)
        hinge_loss=torch.mean(hinge_loss)
        return hinge_loss


class FixedHardTanH(Module):
    def __init__(self, quantization):
        super(FixedHardTanH,self).__init__()
        self.quantization = quantization

    def forward(self,input):
        y = torch.clamp(input, torch.tensor(-1), torch.tensor(1))
        out = self.quantization.quantize(y)
        return out


class BinarizedFCLayer(Module):
    def __init__(self,in_features, out_features, quantization, bias=False):
        super(BinarizedFCLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization=quantization
        if bias is not False:
            self.bias=Parameter(torch.zeros(out_features).float())
        else:
            self.bias = None
        self.weight=Parameter(torch.randn(out_features,in_features).float())

    def quantize(self, X):
        return self.quantization.quantizeWeights(X)

    def clip(self):
        self.weight.data=self.quantization.clipWeights(self.weight.data)

    def forward(self, X):
        self.weight.org = self.weight.data.clone()
        self.weight.data = self.quantize(self.weight.org)
        if self.bias is not None:
            out = F.linear(input=X, weight=self.weight, bias=self.bias)
        else:
            out = F.linear(input=X, weight=self.weight)
        self.weight.data=self.weight.org
        return out

    def eval(self, X):
        binarized_weights = self.quantize(self.weight.data)
        if self.bias is not None:
            out = F.linear(input=X, weight=binarized_weights, bias=self.bias)
        else:
            out = F.linear(input=X, weight=binarized_weights)
        return out

    def reset_parameters(self):
        # Glorot initialization
        nn.init.xavier_uniform_(self.weight, gain=15)
        if self.bias is not None:
            self.bias.data.zero_()


class BinarizedConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, quantization, bias=False):
        super(BinarizedConv2d, self).__init__()
        self.stride=stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.quantization=quantization
        if bias is not False:
            self.bias = Parameter(torch.zeros(out_channels).float())
        else:
            self.bias = None
        self.weight = Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size).float())

    def quantize(self, X):
        return self.quantization.quantizeWeights(X)

    def clip(self):
        self.weight.data=self.quantization.clipWeights(self.weight.data)

    def forward(self, X):
        self.weight.org = self.weight.data.clone()

        self.weight.data = self.quantize(self.weight.org)

        if self.bias is not None:
            out = F.conv2d(input=X, weight=self.weight,
                           stride=self.stride,bias=self.bias)
        else:
            out = F.conv2d(input=X, weight=self.weight,stride=self.stride)

        self.weight.data= self.weight.org

        return out


    def eval(self, X):
        binarized_weights = self.quantize(self.weight.data)
        if self.bias is not None:
            out = F.conv2d(input=X, weight=binarized_weights,stride=self.stride,bias=self.bias)
        else:
            out = F.conv2d(input=X, weight=binarized_weights, stride=self.stride)
        return out

    def reset_parameters(self):
        # Glorot initialization
        nn.init.xavier_uniform_(self.weight, gain=15)
        if self.bias is not None:
            self.bias.data.zero_()
