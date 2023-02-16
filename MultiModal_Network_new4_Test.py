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

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(1, 1), bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(1, 1), bias=True)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=(0, 0), bias=True)
        self.pool1 = nn.AvgPool2d(kernel_size=8, stride=0, padding=0, ceil_mode=False, count_include_pad=True)

        self.bn1 = nn.BatchNorm2d(32, eps=0.0001)
        self.bn2 = nn.BatchNorm2d(32, eps=0.0001)
        self.bn3 = nn.BatchNorm2d(64, eps=0.0001)
        self.bn4 = nn.BatchNorm2d(64, eps=0.0001)
        self.bn5 = nn.BatchNorm2d(128, eps=0.0001)
        self.bn6 = nn.BatchNorm2d(128, eps=0.0001)
        self.bn7 = nn.BatchNorm2d(128, eps=0.0001)

        self.fc1 = nn.Linear(128, 128)  # 8192
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.PointNet = NM_Net()
        self.start = 0

    def ContextNorm(self, input):
        """

                下面为原版github上的代码 个人感觉写错了  重新写一份

        """

        # print(1111111, input.shape)
        if input.dim() == 3:
            mid = input.view(2000, 128, 5)
            # print(mid.mean(dim=1).shape)
            mean = mid.mean(dim=1).unsqueeze(-1).permute(0, 2, 1).expand_as(input)
            std = mid.std(dim=1).unsqueeze(-1).permute(0, 2, 1).expand_as(input)
        else:
            # print("yyyyyyyyyyyyyyyyyy", input.shape)
            mid = input.view(2000, 128)
            # print(mid.mean(dim=1).shape)
            mean = mid.mean(dim=1).unsqueeze(-1).expand_as(input)
            std = mid.std(dim=1).unsqueeze(-1).expand_as(input)

        return (input - mean) / (std + 0.0001)

    """
    def forward(self, x, y, xs_4, epoch):

        :param
        x: 左图像
        :param
        y: 右图像
        :param
        z: 左坐标
        :return:

                第一幅图像

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
        # center_feature =
        center_feature1 = (x[:, :, 3, 3] + x[:, :, 3, 4] + x[:, :, 4, 3] + x[:, :, 4, 4]) / 4
        center_feature1 = center_feature1.reshape(2000, 128, 1)
        # print(center_feature1.shape)
        # print(5555555, x.shape)

        x = self.pool1(x)
        x = x.reshape(2000, 128, -1)
        # print(7777777, x.shape)
        x = torch.cat([x, center_feature1], dim=2)
        # print(666666, x.shape)

        # x = F.relu(self.bn7(self.conv7(x)))
        # print(777777, x.shape)

        a0 = x.float()
        # print("jjjjjjjjjjj", x.shape, a0.shape)
        a2 = torch.norm(a0, p=2, dim=1)
        # print(8888888, a2.shape)
        a2 = a2.unsqueeze(1)
        x = torch.div(x, a2)

            第二幅图像

        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))
        y = F.relu(self.bn6(self.conv6(y)))
        center_feature2 = (y[:, :, 3, 3] + y[:, :, 3, 4] + y[:, :, 4, 3] + y[:, :, 4, 4]) / 4
        center_feature2 = center_feature2.reshape(2000, 128, 1)
        # print(center_feature2.shape)

        y = self.pool2(y)
        y = y.reshape(2000, 128, -1)
        print(111111111, y.shape)
        y = torch.cat([y, center_feature2], dim=2)

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
        # print(379423497, z.shape)
        z = z.permute(0, 2, 1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = z.permute(0, 2, 1)
        # print("egwigriuw", z.shape)

        # 剪枝操作

        if epoch <= 15:  # 3  10
            pass
        else:
            pass
            if 1:
                if self.start == 0:
                    fh = open('heheba_test16.txt', 'a', encoding='utf-8')
                    str1 = "分支开启"
                    fh.write(str1)
                    fh.close()
                    self.start = 1
                z = z.detach()


        # OK了
        z = self.fusion(z,  self.ContextNorm(point_out))

        z = z.permute(0, 2, 1)
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
    """

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
        """
                第一幅图像
        """
        # print(000000, x.shape, y.shape)
        x = x.squeeze(1)
        y = y.squeeze(1)
        x = torch.cat([x, y], dim=-1)
        x = x.permute(0, 3, 1, 2)
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
        # print(7777777777, x.shape)
        x = self.pool1(x)
        x = x.reshape(2000, 128, -1)


        a0 = x.float()
        # print("jjjjjjjjjjj", x.shape, a0.shape)
        a2 = torch.norm(a0, p=2, dim=1)
        # print(8888888, a2.shape)
        a2 = a2.unsqueeze(1)
        x = torch.div(x, a2)
        """
            分界线-----------------
        """
        point_out, point_w = self.PointNet(xs_4)
        # print("输出维度3", point_out.shape, point_w.shape)
        point_out = point_out.permute(0, 2, 1, 3).squeeze(3).reshape(-1, 128)
        # print("输出维度5", point_out.shape)

        # OK了
        z = x.squeeze(2)
        z = self.ContextNorm(z) + self.ContextNorm(point_out)
        z = z.unsqueeze(2)

        z = z.permute(0, 2, 1)
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


"""
          # 整个attention 代码
          # 做了稍微的修改

"""

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 2, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 2, 1, bias=False)
        # self.q_conv.conv.weight = self.k_conv.conv.weight
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, 2 * channels, 1)
        self.trans_conv1 = nn.Conv1d(2 * channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        """
        self.mlp_forward = PositionwiseFeedForward(128, 256)
        self.layer_norm = LayerNorm(128)
        """

    def ContextNorm(self, input):
        """

                下面为原版github上的代码 个人感觉写错了  重新写一份

        """
        # print(1111111, input.shape)
        mid = input.view(2000, 128)
        # print(mid.mean(dim=1).shape)
        mean = mid.mean(dim=1).unsqueeze(-1).unsqueeze(-1).expand_as(input)
        std = mid.std(dim=1).unsqueeze(-1).unsqueeze(-1).expand_as(input)

        return (input - mean) / (std + 0.0001)

    def forward(self, image_f, point_f):
        point_f = point_f.reshape(2000, 128, 1)
        # print("qqqqqqqq", image_f.shape, point_f.shape)
        x_q = self.q_conv(point_f).permute(0, 2, 1)  # b, n, c
        # print(11111111, image_f.shape)
        x_k = self.k_conv(image_f)  # b, c, n    , point_f
        x_v = self.v_conv(image_f)  # , point_f
        energy = torch.bmm(x_q, x_k)  # b, n, n
        # print(energy.shape)
        # print("&&&&&&&&&&", energy)
        attention = self.softmax(energy)
        # print("kkkkkkkkkk", attention.shape)
        attention = attention.permute(0, 2, 1)
        # print("attention ! ", attention.max(1)[0])
        # print("attention !", attention.shape, x_v.shape, point_f)
        # print("point_f", point_f.shape)

        # print(attention[0], attention.sum(dim=1))
        # print("llllllllll", attention.shape)

        # attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        # print("ooooooooo", attention.shape, x_v.shape, attention, attention[0], attention[0].sum(dim=0))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        # print("pppppppp", x_r.shape)
        # print("dads", self.trans_conv1(self.trans_conv(x_r)).shape)
        x_r = self.ContextNorm((self.trans_conv1(self.trans_conv(x_r))))  # point_f - x_r
        # print("fffffffffffff", x_r.shape, x_r, x_r.sum(dim=1))
        # print("eeeeeeeeeeeee", point_f.shape, point_f, point_f.sum(dim=1))
        """
        if 0:
            result = point_f * 0.5 + x_r * 0.5
        else:
            result = point_f + x_r
        """

        return x_r


