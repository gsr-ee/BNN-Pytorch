import torch
import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class loadDataset():
    def __init__(self,path):
        """
        Args:
            :param path: Directory to the binaries files
        """
        self.dir=path

    def load_images(self,filename):
        with open(os.path.join(self.dir,filename),'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=0)

        data=data.reshape(-1,3,32,32)
        data=np.transpose(data, [0, 3, 2, 1])
        return data

    def load_labels(self,filename):
        with open(os.path.join(self.dir,filename),'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=0)
        return labels

class Aedes_Dataset(Dataset,loadDataset):

    def __init__(self, root_dir, prefix, transform=None):
        """
        Args:
            :param root_dir: Directory to the binary files
            :param transform: To.Tensor() is the unique transformation available
            :param prefix: train or test data to load
        """
        super(Aedes_Dataset,self).__init__(path=root_dir)
        self.labels = Aedes_Dataset.load_labels(self, filename='{}-labels.bin'.format(prefix))
        self.length = len(self.labels)
        self.images = Aedes_Dataset.load_images(self, filename='{}-images.bin'.format(prefix))
        self.transform=transform
        self.tensor=ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Args:
            :param idx: index
            :return: tuple {'image','label'}
        """
        label=np.array(self.labels[idx],dtype=np.float32)
        image=self.images[idx]
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        sample=self.tensor(sample)


        #sample = {'image': image, 'label': label}

        #image = np.subtract(np.multiply(2. / 255., self.images[idx]), 1., dtype=np.float32)
        return sample

class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image=np.array(image)
        image = np.subtract(np.multiply(2. / 255., image), 1., dtype=np.float32)
        image=image.transpose([2,0,1])

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label).long()}

