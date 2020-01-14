import torch
import torch.nn as nn
from dataset import DistortedDataSet
from torch.utils.data import DataLoader


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_block, self).__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
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

        # second unet
        self.down2_conv1 = Conv_block(66, 64)
        self.down2_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(64, 128),
        )
        self.down2_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(128, 256),
        )
        self.down2_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(256, 512),
        )
        self.down2_conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(512, 1024),
        )
        self.up2_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up2_conv1_later = Conv_block(1024, 512)
        self.up2_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up2_conv2_later = Conv_block(512, 256)
        self.up2_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up2_conv3_later = Conv_block(256, 128)
        self.up2_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up2_conv4_later = Conv_block(128, 64)

        self.out2_conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1)

    def forward(self, x):
        features_stack = []
        features_stack_2 = []
        x = self.down_conv1(x)
        features_stack.append(x)
        x = self.down_conv2(x)
        features_stack.append(x)
        x = self.down_conv3(x)
        features_stack.append(x)
        x = self.down_conv4(x)
        features_stack.append(x)
        x = self.down_conv5(x)
        # print(x3.size())

        x = self.up_conv1(x)
        # print(x.size())
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

        out1 = self.out_conv(x)

        x = torch.cat((x, out1), dim=1)
        x = self.down2_conv1(x)
        features_stack_2.append(x)
        x = self.down2_conv2(x)
        features_stack_2.append(x)
        x = self.down2_conv3(x)
        features_stack_2.append(x)
        x = self.down2_conv4(x)
        features_stack_2.append(x)
        x = self.down2_conv5(x)
        
        x = self.up2_conv1(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv1_later(x)

        x = self.up2_conv2(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv2_later(x)
        
        x = self.up2_conv3(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv3_later(x)
        
        x = self.up2_conv4(x)
        x = torch.cat((features_stack_2.pop(), x), dim=1)
        x = self.up2_conv4_later(x)

        out2 = self.out2_conv(x)
        
        return out2

if __name__ == '__main__':
    # images_folder = '/home/wulei/DocUNet/data_gen/gen_test/image/'
    # labels_folder = '/home/wulei/DocUNet/data_gen/gen_test/label/'
    # batch_size = 1

    # data_set = DistortedDataSet(images_folder, labels_folder)

    # data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    # net = Net()
    # # print(net)

    # for i, data in enumerate(data_loader):
    #     images = data[0]
    #     labels = data[1]

    #     out = net(images)
    #     print(out.size())
        


    x = torch.randn((10, 3, 352, 272))
    net = Net()
    out = net(x)
    print(out.size())
