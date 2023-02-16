import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import get_config, print_usage
import os
import torch.utils.data as Data
from dataset import Data_Loader

"""
class Net(nn.Module):

    def __init__(self):
        # 继承原有模型
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # 定义了两个卷积层
        # 第一层是输入1维的（说明是单通道，灰色的图片）图片，输出6维的的卷积层（说明用到了6个卷积核，而每个卷积核是5*5的）。

        假设图像大小为N * N矩阵
        卷积核的尺寸为K * K矩阵
        卷积的方式（边缘像素填充方式）:P
        卷积的步伐为S * S
        那么经过一层这样的卷积后出来的图像为：
        大小：(N - K + 2P) / S + 1

        .conv_bn(3, 32, 1, name='conv0')
        .conv_bn(3, 32, 1, name='conv1')
        .conv_bn(3, 64, 2, name='conv2')
        .conv_bn(3, 64, 1, name='conv3')
        .conv_bn(3, 128, 2, name='conv4')
        .conv_bn(3, 128, 1, name='conv5')

        kernel_size,
        filters,
        strides,


        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=3, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=3, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=3, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=3, bias=True)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=3, bias=True)

        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv4 = nn.Conv2d(in_channels=101, out_channels=101, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv5 = nn.Conv2d(in_channels=101, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, y, x_initial):

        :param
        x: 左图像
        :param
        y: 右图像
        :param
        z: 左坐标
        :return:


        print(x.shape)
        # x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2, padding=1)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        # x = F.relu(self.conv3(x))

        # x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2, padding=1)
        y = F.relu(self.conv1(y))
        print(y.shape)
        y = F.relu(self.conv2(y))
        print(y.shape)
        # y = F.relu(self.conv3(y))
        # x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2, padding=1)
        # x = F.max_pool2d((F.relu(self.conv5(F.relu(self.conv4((self.conv3(x))))))), kernel_size=3, stride=2, padding=1)

        x_initial = x_initial.squeeze(1)
        x_left_initial = x_initial[:, :, 0:2]
        x_right_initial = x_initial[:, :, 2:]
        print(x_left_initial.shape, x_left_initial[:, :, 0].shape)
        print(x_left_initial[0, :, :].max(dim=0))
        print(x_right_initial[0, :, :])

        D = torch.Tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]).type(
            torch.LongTensor)
        # D = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).type(torch.LongTensor)

        # print(Left_Image.shape)
        z = x[D, :, x_left_initial[:, :, 0], x_left_initial[:, :, 1]]
        k = y[D, :, x_right_initial[:, :, 0], x_right_initial[:, :, 1]]
        print(z.shape)

        return z
"""


class Net(nn.Module):

    def __init__(self):
        # 继承原有模型
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # 定义了两个卷积层
        # 第一层是输入1维的（说明是单通道，灰色的图片）图片，输出6维的的卷积层（说明用到了6个卷积核，而每个卷积核是5*5的）。
        """
        假设图像大小为N * N矩阵
        卷积核的尺寸为K * K矩阵
        卷积的方式（边缘像素填充方式）:P
        卷积的步伐为S * S
        那么经过一层这样的卷积后出来的图像为：
        大小：(N - K + 2P) / S + 1

        .conv_bn(3, 32, 1, name='conv0')
        .conv_bn(3, 32, 1, name='conv1')
        .conv_bn(3, 64, 2, name='conv2')
        .conv_bn(3, 64, 1, name='conv3')
        .conv_bn(3, 128, 2, name='conv4')
        .conv_bn(3, 128, 1, name='conv5')

        kernel_size,
        filters,
        strides,
        """

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(1, 1), bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(1, 1), bias=True)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=(0, 0), bias=True)

        self.bn1 = nn.BatchNorm2d(32, eps=0.0001)
        self.bn2 = nn.BatchNorm2d(32, eps=0.0001)
        self.bn3 = nn.BatchNorm2d(64, eps=0.0001)
        self.bn4 = nn.BatchNorm2d(64, eps=0.0001)
        self.bn5 = nn.BatchNorm2d(128, eps=0.0001)
        self.bn6 = nn.BatchNorm2d(128, eps=0.0001)
        self.bn7 = nn.BatchNorm2d(128, eps=0.0001)

        self.fc1 = nn.Linear(128, 128)  # 8192
        # self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        # self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        # self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)

    def ContextNorm1(self, input):
        """

                下面为原版github上的代码 个人感觉写错了  重新写一份

        """

        mid = input
        mean = mid.mean(dim=3).unsqueeze(-1).expand_as(input)
        std = mid.std(dim=3).unsqueeze(-1).expand_as(input)

        return (input - mean) / (std + 0.0001)

    def forward(self, x, y):
        """
        :param
        x: 左图像
        :param
        y: 右图像
        :param
        z: 左坐标
        :return:
        """
        # print("Start", x.shape, y.shape)
        """
                第一幅图像
        """
        x = F.relu(self.bn1(self.conv1(x)))
        # print(111111, x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(222222, x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(333333, x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        # print(444444, x.shape)
        x = F.relu(self.bn5(self.conv5(x)))
        # print(555555, x.shape)
        x = F.relu(self.bn6(self.conv6(x)))
        x1 = x.reshape(2000, 128, -1).unsqueeze(1)
        # print(666666, x1.shape)

        """
            第二幅图像
        """
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))
        y = F.relu(self.bn6(self.conv6(y)))
        # print(666666, y.shape)
        y1 = y.reshape(2000, 128, -1).unsqueeze(1)      # 2000 1 128 64
        # print(666666, y1.shape)

        c1 = torch.cat((x1, y1), 3)
        # print("合并", c1.shape)

        c2 = F.relu(self.fc1(c1))
        # c3 = F.relu(self.fc2(c2))
        c4 = F.relu(self.fc3(c2))
        # c5 = F.relu(self.fc4(c4))
        c6 = F.relu(self.fc5(c4))
        # c7 = F.relu(self.fc6(c6))
        c8 = F.relu(self.fc7(c6))


        # c2_temp = torch.div(c2, torch.norm(c2))
        # c4_temp = torch.div(c4, torch.norm(c4))
        # c6_temp = torch.div(c6, torch.norm(c6))
        # c8_temp = torch.div(c8, torch.norm(c8))
        c2_temp = self.ContextNorm1(c2)
        c4_temp = self.ContextNorm1(c4)
        c6_temp = self.ContextNorm1(c6)
        c8_temp = self.ContextNorm1(c8)


        """
        data = {}
        data[3] = c2_temp
        # data[4] = c2
        data[5] = c2_temp
        # data[6] = c4
        data[7] = c4_temp
        # data[8] = c6
        data[9] = c4_temp
        # c1, c2, c3, c4, c5, c6, c7, c8
        """
        data = {}
        data[3] = c2_temp
        # data[4] = c2
        data[5] = c4_temp
        # data[6] = c4
        data[7] = c6_temp
        # data[8] = c6
        data[9] = c8_temp
        # data[10] = c8

        return data


class ResidualBlock(nn.Module):
    def __init__(self, pre=False, use_weight=False, weight_index=None):
        super(ResidualBlock, self).__init__()
        self.pre = pre
        self.use_weight = use_weight
        self.weight_index = weight_index

        self.right = nn.Sequential(
            nn.Conv2d(1, 128, (1, 4)),
        )
        self.conv = nn.Conv2d(128, 128, (1, 1))
        self.conv1 = nn.Conv2d(128, 128, (1, 1))
        self.block = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.BN = nn.BatchNorm2d(128)
        self.BN1 = nn.BatchNorm2d(128)

        # self.softmax = nn.Softmax(dim=2)
        # self.conv1 = nn.Conv2d(128, 1, (1, 1))
        # self.conv2 = nn.Conv2d(128, 128, (1, 1))

    def ContextNorm(self, input, No_Attention=True):
        """

                下面为原版github上的代码 个人感觉写错了  重新写一份

        """
        mid = input.view(input.size(0), 128, 2000)
        mean = mid.mean(dim=2).unsqueeze(-1).unsqueeze(-1).expand_as(input)
        std = mid.std(dim=2).unsqueeze(-1).unsqueeze(-1).expand_as(input)
        return (input - mean) / (std + 0.0001)

    def forward(self, x_data):
        # print(55555, len(x_data))
        x = x_data[0]
        data = x_data[1]
        x = self.right(x) if self.pre is True else x
        out = self.conv(x)
        out = self.ContextNorm(out)
        out = self.block(out)
        if self.use_weight:
            # out = out * self.weight
            out1 = out.permute(2, 0, 3, 1).squeeze(1)
            # print("55555", self.weight_index, data[self.weight_index], data[self.weight_index + 1])
            data1 = data[self.weight_index].squeeze(1)
            # print(8888888, data[self.weight_index])
            # print(999999, out1.shape, data[self.weight_index].shape)
            out = torch.matmul(out1, data1)
            out = out.permute(1, 2, 0).unsqueeze(3)
            # out = torch.bmm(out, data[self.weight_index])
        else:
            out = self.conv1(out)

        out = self.ContextNorm(out)
        out = self.BN(out)
        out = out + x
        return (F.relu(out), data)


# LGCNet
class NM_Net(nn.Module):
    def __init__(self):
        super(NM_Net, self).__init__()
        self.cnn_network = Net()
        self.layer1 = self.make_layers(ResidualBlock)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 1, (1, 1)),
        )
        self.initialize_weights()

    def make_layers(self, ResidualBlock):
        layers = []
        layers.append(ResidualBlock(pre=True, use_weight=False, weight_index=None))
        for i in range(1, 12):
            if i == 7 or i == 3 or i == 5 or i == 9:
            # if i == 7:
                layers.append(ResidualBlock(pre=False, use_weight=True, weight_index=i))
            else:
                layers.append(ResidualBlock(pre=False, use_weight=False, weight_index=None))

        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)

    def forward(self, img1, img2, x, epoch=None):
        data = self.cnn_network(img1, img2)
        # print(data)

        out, _ = self.layer1((x, data))
        out1 = self.layer2(out)
        out1 = out1.view(out1.size(0), -1)
        w = F.tanh(out1)
        w = F.relu(w)

        return out1, w

