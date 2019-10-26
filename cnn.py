#BSD 3-Clause License
#=======
#Based on Xilinx BNN training codes
#Copyright (c) 2019, Xilinx Inc.
#All rights reserved.
#
#Based on Matthieu Courbariaux's BinaryNet example
#Copyright (c) 2015-2016, Matthieu Courbariaux, Itay Hubara et al.
#All rights reserved
#

from quantized_net import *
from quantization import *


class BNN_Pynq(nn.Module):
    def __init__(self, num_classes=5, **kwargs):
        super(BNN_Pynq, self).__init__()
        if 'learning_parameters' in kwargs.keys():
            learning_parameters=kwargs['learning_parameters']
        else:
            raise Exception("no learning parameters")

        if num_classes < 1 or num_classes > 64:
            assert ("num_outputs should be in the range of 1 to 64.")
        input_quantization = QuantizationFixed(8, 7, False)
        if learning_parameters.weight_bits==1:
            weigth_quantization=QuantizationBinary()
        else:
            weigth_quantization=QuantizationFixed(learning_parameters.weight_bits,
                                                  learning_parameters.weight_bits-2)
        if learning_parameters.activation_bits==1:
            activation_quantization=QuantizationBinary()
        else:
            activation_quantization = QuantizationFixed(learning_parameters.activation_bits,
                                                        learning_parameters.activation_bits-2,True)

        affine=learning_parameters.affine
        bias=learning_parameters.bias
        lr_decay=learning_parameters.lr_decay
        lr_start=learning_parameters.lr_start
        momentum=learning_parameters.momentum
        epsilon=learning_parameters.epsilon

        self.features=nn.ModuleDict([
            ['input_quantization', FixedHardTanH(quantization=input_quantization)],
            #Fist Convolucional Layer
            ['conv_0', BinarizedConv2d(in_channels=3,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            quantization=weigth_quantization)],
            ['batchnorm_0', nn.BatchNorm2d(num_features=64,eps=epsilon, momentum=momentum, affine=affine)],
            ['activation_0', FixedHardTanH(quantization=activation_quantization)],
            # Second Convolucional Layer
            ['conv_1', BinarizedConv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            quantization=weigth_quantization)],
            ['max-pool_0', nn.MaxPool2d(kernel_size=2,stride=2)],
            ['batchnorm_1', nn.BatchNorm2d(num_features=64,eps=epsilon, momentum=momentum, affine=affine)],
            ['activation_1', FixedHardTanH(quantization=activation_quantization)],
            # Third Convolucional Layer
            ['conv_2', BinarizedConv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            quantization=weigth_quantization)],
            ['batchnorm_2', nn.BatchNorm2d(num_features=128,eps=epsilon, momentum=momentum, affine=affine)],
            ['activation_2', FixedHardTanH(quantization=activation_quantization)],
            # Fourth Convolucional Layer
            ['conv_3', BinarizedConv2d(in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            quantization=weigth_quantization)],
            ['max-pool_1', nn.MaxPool2d(kernel_size=2, stride=2)],
            ['batchnorm_3', nn.BatchNorm2d(num_features=128,eps=epsilon, momentum=momentum, affine=affine)],
            ['activation_3', FixedHardTanH(quantization=activation_quantization)],
            # Fifth Convolucional Layer
            ['conv_4', BinarizedConv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            quantization=weigth_quantization)],
            ['batchnorm_4', nn.BatchNorm2d(num_features=256,eps=epsilon, momentum=momentum, affine=affine)],
            ['activation_4', FixedHardTanH(quantization=activation_quantization)],
            # Sixth Convolucional Layer
            ['conv_5', BinarizedConv2d(in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            quantization=weigth_quantization)],
            ['batchnorm_5', nn.BatchNorm2d(num_features=256,eps=epsilon, momentum=momentum, affine=affine)],
            ['activation_5', FixedHardTanH(quantization=activation_quantization)]
        ])

        self.classifier = nn.ModuleDict([
            #Fist Fully-connected Layer
            ['fc_0', BinarizedFCLayer(in_features=256,
                                out_features=512,
                                bias=bias,
                                quantization=weigth_quantization)],
            ['batchnorm_0', nn.BatchNorm1d(num_features=512,eps=epsilon, momentum=momentum, affine=affine)],
            ['activation_0', FixedHardTanH(quantization=activation_quantization)],
            # Second Fully-connected Layer
            ['fc_1', BinarizedFCLayer(in_features=512,
                                out_features=512,
                                bias=bias,
                                quantization=weigth_quantization)],
            ['batchnorm_1', nn.BatchNorm1d(num_features=512,eps=epsilon, momentum=momentum, affine=affine)],
            ['activation_1', FixedHardTanH(quantization=activation_quantization)],
            # Third Fully-connected Layer
            ['fc_2', BinarizedFCLayer(in_features=512,
                                out_features=num_classes,
                                bias=False,
                                quantization=weigth_quantization)],
            #['softmax', nn.Softmax()]
            ['batchnorm_2', nn.BatchNorm1d(num_features=num_classes,eps=epsilon, momentum=momentum, affine=False)]
        ])
        # Hyper parameters
        self.hyper_parameters = {
            'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': lr_start,
                'lr_decay': lr_decay}

        #Softmax for evaluation
        self.softmax=nn.Softmax(dim=1)

    def forward(self,cnn):
        # Conv Layer forward
        output = self.features['input_quantization'](cnn)
        # First Layer
        output = self.features['conv_0'](output)
        output = self.features['batchnorm_0'](output)
        output = self.features['activation_0'](output)
        # Second Layer
        output = self.features['conv_1'](output)
        output = self.features['batchnorm_1'](output)
        output = self.features['activation_1'](output)
        output = self.features['max-pool_0'](output)
        # Third Layer
        output = self.features['conv_2'](output)
        output = self.features['batchnorm_2'](output)
        output = self.features['activation_2'](output)
        # Fourth Layer
        output = self.features['conv_3'](output)
        output = self.features['batchnorm_3'](output)
        output = self.features['activation_3'](output)
        output = self.features['max-pool_1'](output)
        # Fifth Layer
        output = self.features['conv_4'](output)
        output = self.features['batchnorm_4'](output)
        output = self.features['activation_4'](output)
        # Sixth Layer
        output = self.features['conv_5'](output)
        output = self.features['batchnorm_5'](output)
        output = self.features['activation_5'](output)

        # Fully-connected Layer forward
        output = output.flatten(1)
        # First Layer
        output = self.classifier['fc_0'](output)
        output = self.classifier['batchnorm_0'](output)
        output = self.classifier['activation_0'](output)
        # Second Layer
        output = self.classifier['fc_1'](output)
        output = self.classifier['batchnorm_1'](output)
        output = self.classifier['activation_1'](output)
        # Third Layer
        output = self.classifier['fc_2'](output)
        output = self.classifier['batchnorm_2'](output)

        return output

    def evaluation(self, cnn):
        # Conv Layer evaluation
        output = self.features['input_quantization'](cnn)
        #First Layer
        output = self.features['conv_0'].eval(output)
        output = self.features['batchnorm_0'](output)
        output = self.features['activation_0'](output)
        #Second Layer
        output = self.features['conv_1'].eval(output)
        output = self.features['batchnorm_1'](output)
        output = self.features['activation_1'](output)
        output = self.features['max-pool_0'](output)
        #print(output)
        # Third Layer
        output = self.features['conv_2'].eval(output)
        output = self.features['batchnorm_2'](output)
        output = self.features['activation_2'](output)
        # Fourth Layer
        output = self.features['conv_3'].eval(output)
        output = self.features['batchnorm_3'](output)
        output = self.features['activation_3'](output)
        output = self.features['max-pool_1'](output)
        # Fifth Layer
        output = self.features['conv_4'].eval(output)
        output = self.features['batchnorm_4'](output)
        output = self.features['activation_4'](output)
        # Sixth Layer
        output = self.features['conv_5'].eval(output)
        output = self.features['batchnorm_5'](output)
        output = self.features['activation_5'](output)

        # Fully-connected Layer forward
        output = output.flatten(1)
        #output = output.view(output.size(0), -1)
        # First Layer
        output = self.classifier['fc_0'].eval(output)
        output = self.classifier['batchnorm_0'](output)
        output = self.classifier['activation_0'](output)
        # Second Layer
        output = self.classifier['fc_1'].eval(output)
        output = self.classifier['batchnorm_1'](output)
        output = self.classifier['activation_1'](output)
        # Third Layer
        output = self.classifier['fc_2'].eval(output)
        # No activation for the last layer, use in evaluation softmax instead
        bnn = self.softmax(output)
        output = self.classifier['batchnorm_2'](output)
        #print(output)
        return output, bnn

    def clipWeights(self):
        for i in range(6):
            self.features['conv_%i' %i].clip()

        for i in range(3):
            self.classifier['fc_%i' % i].clip()


