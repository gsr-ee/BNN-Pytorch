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

import torch
from torch.autograd import Function


#Rounding function that not set gradients to zero
class Round_function(Function):
    @staticmethod
    def forward(ctx, input):
        rvalue=torch.round(input)
        return rvalue
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input[grad_input.ge(1)] = 0
        grad_input[grad_input.le(-1)] = 0
        return grad_input

roundf=Round_function.apply

# A fixed point quantization scheme

class QuantizationFixed():
    def __init__(self, wordlength, fraclength, narrow_range=True):
        self.wordlength = wordlength
        self.fraclength = fraclength
        self.narrow_range = narrow_range
        self.set_quantization_params()

    # Set up the parameters for run-time quantization
    def set_quantization_params(self):
        self.set_min_max()
        self.set_scale_shift()

    # Work out how much to shift and scale prior to rounding in the current scheme
    def set_scale_shift(self):
        if self.narrow_range:
            sub = 2
        else:
            sub = 1
        self.scale = (2.**self.wordlength - sub) / (self.max - self.min)
        self.shift = -self.min

    # Find the minimum and maximum representable parameters in this scheme
    def set_min_max(self):
        min_val = - (2.**(self.wordlength - self.fraclength - 1))
        max_val = - min_val - 2.**-self.fraclength
        if self.narrow_range:
            min_val = - max_val
        self.min = min_val
        self.max = max_val

    # Quantize activations
    def quantize(self, X):
        return roundf((torch.clamp(X, self.min, self.max) + self.shift)*self.scale) / self.scale - self.shift

    # Quantize weights (can use Theano's built-in round function - should be faster)
    def quantizeWeights(self, X):
        return roundf((torch.clamp(X, self.min, self.max) + self.shift)*self.scale) / self.scale - self.shift

    def clipWeights(self, X):
        return torch.clamp(X, self.min, self.max)


# From https://github.com/MatthieuCourbariaux/BinaryNet
def hard_sigmoid(x):
    return torch.clamp((x + 1.) / 2., torch.tensor(0).float(), torch.tensor(1).float())


# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
# From https://github.com/MatthieuCourbariaux/BinaryNet
def binary_tanh_unit(x):
    return 2. * roundf(hard_sigmoid(x)).float() - 1.


# From https://github.com/MatthieuCourbariaux/BinaryNet
def binary_sigmoid_unit(x):
    return roundf(hard_sigmoid(x)).float()


class QuantizationBinary():
    def __init__(self, scale=1.0):
        super(QuantizationBinary,self).__init__()
        self.scale = scale
        self.min = -scale
        self.max = scale

    # Quantize activations
    def quantize(self, X):
        return binary_tanh_unit(X / self.scale)*self.scale

    # Quantize weights (can use Theano's built-in round function - should be faster?)
    def quantizeWeights(self, X):
        # [-1,1] -> [0,1]
        Xa = hard_sigmoid(X / self.scale)
        Xb = roundf(Xa).byte()
        # 0 or 1 -> -1 or 1
        return torch.where(Xb==0,torch.tensor(-1),torch.tensor(1)).float()

    def clipWeights(self, X):
        return torch.clamp(X, torch.tensor(-self.scale).float(), torch.tensor(self.scale).float())
