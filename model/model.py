import torch
import torch.nn as nn
from data import DistortedDataSet
from torch.utils.data import DataLoader


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_block, self).__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)

        self.down_conv1 = Conv_block(3, 64)
        self.down_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(64, 128),
        )
        self.down_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(128, 256),
        )
        self.down_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(256, 512),
        )
        self.down_conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(512, 1024),
        )

        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1_later = Conv_block(1024, 512)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2_later = Conv_block(512, 256)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3_later = Conv_block(256, 128)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4_later = Conv_block(128, 64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1)

    def forward(self, x):
        features_stack = []
        x0 = self.down_conv1(x)
        features_stack.append(x0)
        x1 = self.down_conv2(x0)
        features_stack.append(x1)
        x2 = self.down_conv3(x1)
        features_stack.append(x2)
        x3 = self.down_conv4(x2)
        features_stack.append(x3)
        x4 = self.down_conv5(x3)
        print(x3.size())

        x = self.up_conv1(x4)
        print(x.size())
        x = torch.cat((features_stack.pop(), x), dim=1)
        x = self.up_conv1_later(x)

        x = self.up_conv2(x)
        x = torch.cat((features_stack.pop(), x), dim=1)
        x = self.up_conv2_later(x)

        x = self.up_conv3(x)
        x = torch.cat((features_stack.pop(), x), dim=1)
        x = self.up_conv3_later(x)

        x = self.up_conv4(x)
        x = torch.cat((features_stack.pop(), x), dim=1)
        x = self.up_conv4_later(x)

        x = self.out_conv(x)

        return x

if __name__ == '__main__':
    images_folder = '/home/wulei/DocUNet/data_gen/gen_test/image/'
    labels_folder = '/home/wulei/DocUNet/data_gen/gen_test/label/'
    batch_size = 1

    data_set = DistortedDataSet(images_folder, labels_folder)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    net = Net()

    for i, data in enumerate(data_loader):
        images = data[0]
        labels = data[1]

        out = net(images)
        print(out.size())
        

