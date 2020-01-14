import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset


class DistortedDataSet(Dataset):
    def __init__(self, images_folder, labels_folder, transform, is_train=True):
        assert os.path.exists(images_folder), 'Images folder not exists'
        assert os.path.exists(labels_folder), 'Labels folder not exists'
        
        self.images_folder = os.path.abspath(images_folder)
        self.labels_folder = os.path.abspath(labels_folder)

        self.images = os.listdir(self.images_folder)  
        self.labels = os.listdir(self.labels_folder)

        self.transform = transform
        self.is_train = is_train

        assert len(self.images) == len(self.labels), 'The number of samples and labels is inconsistent'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        filename = img_filename[:img_filename.index('.')]
        label_filename = filename + '.npz'

        img = mpimg.imread(self.images_folder + '/' + img_filename)
        label = np.load(self.labels_folder + '/' + label_filename)

        img = self.transform(img)

        label_x, label_y = label['x'], label['y']

        return img, label_x, label_y

if __name__ == '__main__':
    data_path = '/home/cc/wulei/data_gen'
    images_folder = os.path.join(data_path, 'image')
    labels_folder = os.path.join(data_path, 'label')

    ds = DistortedDataSet(images_folder, labels_folder, None, is_train=False)

    # batch_size = 10
    # dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    # for image, label in dl:
        
    #     print(image.shape, label['x'].size(), label['y'].size())

    #     break
    # for i, data in enumerate(dl):
    #     print(len(data))
    #     break

    img, label_x, label_y = ds.__getitem__(99)
