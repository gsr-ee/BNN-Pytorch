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

from aedes import *
from cnn import *
from utils import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import OrderedDict
from torchvision.transforms import *


def main(args):
    learning_parameters = OrderedDict()
    # Quantization parameters
    print('Quantization parameters:')
    learning_parameters.activation_bits = args.activation_bits
    print('activation_bits = ' + str(learning_parameters.activation_bits))
    learning_parameters.weight_bits = args.weight_bits
    print('weight_bits = ' + str(learning_parameters.weight_bits))
    # BN parameters
    # alpha is the exponential moving average factor
    print('BN parameters:')
    learning_parameters.momentum = .1
    print('momentum = ' + str(learning_parameters.momentum))
    learning_parameters.epsilon = 1e-4
    print('epsilon = ' + str(learning_parameters.epsilon))
    learning_parameters.bias=False
    learning_parameters.affine=True
    print('affine = ' + str(learning_parameters.affine))
    # Training parameters
    print('Training parameters:')
    num_epochs = 200
    print('num_epochs = ' + str(num_epochs))
    # Decaying LR
    lr_start = 0.005
    learning_parameters.lr_start = lr_start
    print('LR_start = ' + str(lr_start))
    lr_fin = 0.00003
    print('LR_fin = ' + str(lr_fin))
    lr_decay = (lr_fin / lr_start) ** (1. / num_epochs)
    learning_parameters.lr_decay = lr_decay
    print('LR_decay = ' + str(lr_decay))
    # BTW, LR decay might good for the BN moving average...
    if not os.path.exists('models'):
        os.mkdir('models')
    save_path = 'models/aedes-{}w-{}a' .format(learning_parameters.weight_bits,
                                              learning_parameters.activation_bits)
    print('save_path = ' + str(save_path))
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Hyper parameters
    num_classes = 5
    print('num_classes = ' + str(num_classes))

    print('Loading the dataset...')

    transform=RandomChoice([RandomResizedCrop(size=32,scale=(0.64,0.64), ratio=(0.75,0.75)),
                            RandomResizedCrop(size=32, scale=(1, 1), ratio=(1, 1)),
                            RandomRotation(degrees=(-90,+90)),
                            ColorJitter(brightness=(0,2), contrast=(0,3)),
                            RandomHorizontalFlip(1),
                            RandomVerticalFlip(1),
                            RandomResizedCrop(size=32, scale=(1, 1), ratio=(1, 1))])


    train_dataset = Aedes_Dataset(root_dir='data', prefix='train',
                                  transform=transform)
    train_set_size = len(train_dataset)
    print('train_set_size = ' + str(train_set_size))
    test_dataset = Aedes_Dataset(root_dir='data', prefix='test')
    batch_size_train = int(len(train_dataset) / 10)
    print('batch_size_train = ' + str(batch_size_train))

    test_set_size = len(test_dataset)
    print('test_set_size = ' + str(test_set_size))
    batch_size_test = int(len(test_dataset) / 10)
    print('batch_size_test = ' + str(batch_size_test))

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size_train,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size_test,
                                              shuffle=False)

    print('Building the CNN model...')
    model = BNN_Pynq(num_classes=num_classes, learning_parameters=learning_parameters)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    print('Model set!')
    print('Training...')
    train_loss, accuracy = quantized_training(model=model,optimizer=optimizer,
                                              criterion=criterion, train_loader=train_loader,
                                              test_loader=test_loader, num_epochs=num_epochs,
                                              device=device,save_path=save_path)

    print('Plotting figures...')
    if not os.path.exists('notebooks'):
        os.mkdir('notebooks')

    plt.plot((torch.arange(num_epochs)+1).numpy(),100*(1-torch.tensor(accuracy).numpy()),color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.title('Error in inference cnv{}w{}a per Epoch' .format(learning_parameters.weight_bits,
                                                      learning_parameters.activation_bits),
              fontsize=14,fontweight="bold")
    plt.grid()
    plt.savefig('notebooks/inference_error_cnv{}w{}a.png'.format(learning_parameters.weight_bits,
                                                                 learning_parameters.activation_bits))
    plt.show()

    plt.plot((torch.arange(num_epochs)+1).numpy(),train_loss,color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss cnv{}w{}a per Epoch'.format(learning_parameters.weight_bits,
                                                         learning_parameters.activation_bits),
              fontsize=14,fontweight="bold")
    plt.grid()
    plt.savefig('notebooks/train_loss_cnv{}w{}a.png' .format(learning_parameters.weight_bits,
                                                             learning_parameters.activation_bits))
    plt.show()
    print('Done!')



if __name__ == "__main__":
    # Parse some command line options
    parser = ArgumentParser(
        description='Train the CNV network on the own dataset')
    parser.add_argument('-ab', '--activation-bits', type=int, default=2,
        help='Quantized the activations to the specified number of bits, default: %(default)s')
    parser.add_argument('-wb', '--weight-bits', type=int, default=1,
        help='Quantized the weights to the specified number of bits, default: %(default)s')
    args = parser.parse_args()
    main(args)

