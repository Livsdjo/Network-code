import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.fc1 = nn.Linear(256, 256)   # 8192
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.PointNet = NM_Net()

        self.start = 0

    def ContextNorm(self, input):
        """

                下面为原版github上的代码 个人感觉写错了  重新写一份

        """

        mid = input.view(2000, 128)
        mean = mid.mean(dim=1).unsqueeze(-1).expand_as(input)
        std = mid.std(dim=1).unsqueeze(-1).expand_as(input)

        return (input - mean) / (std + 0.0001)


    def forward(self, x, y, xs_4, epoch):
        """
        :param
        x: 左图像
        :param
        y: 右图像
        :param
        z: 左坐标
        :return:
        """

       #         第一幅图像

       # print(000000, x.shape)
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
      #  print(666666, x.shape)
        x = F.relu(self.bn7(self.conv7(x)))
       # print(777777, x.shape)
        x = x.squeeze(3)
        x = x.squeeze(2)

        a0 = x.float()
        a2 = torch.norm(a0, p=2, dim=1)
        a2 = a2.unsqueeze(1)
        x = torch.div(x, a2)

        #    第二幅图像

        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))
        y = F.relu(self.bn6(self.conv6(y)))
        y = F.relu(self.bn7(self.conv7(y)))
        y = y.squeeze(3)
        y = y.squeeze(2)

        a0 = y.float()
        a2 = torch.norm(a0, p=2, dim=1)
        a2 = a2.unsqueeze(1)
        y = torch.div(y, a2)

        # print("输出维度", x.shape, y.shape)
        point_out, point_w = self.PointNet(xs_4)
        # print("输出维度3", point_out.shape, point_w.shape)
        point_out = point_out.permute(0, 2, 1, 3).squeeze(3).reshape(-1, 128)
        # print("输出维度5", point_out.shape)

        z = torch.cat([x, y], 1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))

        if epoch <= 13:     # 3  10    15
            pass
        else:
            pass
            if 1:
                if self.start == 3:
                    fh = open('heheba_test16.txt', 'a', encoding='utf-8')
                    str1 = "分支开启"
                    fh.write(str1)
                    fh.close()
                    self.start = 1
                z = z.detach()
        # print(66666, point_out.shape, z.shape)
        # print("11111", point_out, z)


        z = self.ContextNorm(z) + self.ContextNorm(point_out)
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        w = F.tanh(z)
        w = F.relu(w)

        # print("输出维度2", z.shape)
        z = z.reshape(-1, 2000)
        w = w.reshape(-1, 2000)
        # print("输出维度最后", z.shape, w.shape)
        return z, w


class ResidualBlock(nn.Module):
    def __init__(self, pre=False):
        super(ResidualBlock, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(1, 128, (1, 4)),
        )
        self.conv = nn.Conv2d(128, 128, (1, 1))
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


    def forward(self, x):
        x = self.right(x) if self.pre is True else x
        out = self.conv(x)
        out = self.ContextNorm(out)
        out = self.block(out)
        out = self.conv(out)
        out = self.ContextNorm(out)
        out = self.BN(out)
        out = out + x
        return F.relu(out)

# LGCNet
class NM_Net(nn.Module):
    def __init__(self):
        super(NM_Net, self).__init__()
        self.layer1 = self.make_layers()
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 1, (1, 1)),
        )
        self.initialize_weights()

    def make_layers(self):
        layers = []
        layers.append(ResidualBlock(pre=True))
        for i in range(1, 12):
            layers.append(ResidualBlock(pre=False))
        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        out = self.layer1(x)
        out1 = self.layer2(out)
        out1 = out1.view(out1.size(0), -1)
        w = F.tanh(out1)
        w = F.relu(w)
        # print("shfjksh", out1.shape)

        return out, w

