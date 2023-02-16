#!/usr/bin/env python3
from config import get_config, print_usage
from pytorch_test import test_process
from tqdm import trange
import numpy as np
import torch
from model import NM_Net
import loss
import math
from torch import optim
import os
import sys
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as utils
# import matplotlib.pyplot as plt
from dataset import Data_Loader

import pickle
import numpy as np
from tqdm import tqdm, trange
import joblib


class Generate_K_LinYu:
    def __init__(self, xs_initial_data_path, others_data_path, save_data_path):
        self.xs_initial_data_path = xs_initial_data_path
        self.others_data_path = others_data_path
        self.save_data_path = save_data_path

    def read_pkl(self, data_path):
        pickle_file = open(data_path, 'rb')
        if 0:
            my_list2 = pickle.load(pickle_file)
        else:
            my_list2 = joblib.load(pickle_file)
        # print(my_list2.shape, type(my_list2))
        # print(my_list2)
        pickle_file.close()
        return my_list2

    def save_pkl(self, data_path, data):
        pickle_file = open(data_path, 'wb')  # 创建一个pickle文件，文件后缀名随意,但是打开方式必须是wb（以二进制形式写入）
        joblib.dump(data, pickle_file)  # 将列表倒入文件
        pickle_file.close()

    # 计算一个矩阵内两两向量的距离
    def EuclideanDistances(self, A, B):
        BT = B.transpose()
        # vecProd = A * BT
        vecProd = np.dot(A, BT)
        # print(vecProd)
        SqA = A ** 2
        # print(SqA)
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
        # print(sumSqAEx)

        SqB = B ** 2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
        SqED = sumSqBEx + sumSqAEx - 2 * vecProd
        SqED[SqED < 0] = 0.0
        ED = np.sqrt(SqED)
        # print(ED)
        return ED

    def Get_4_dimension_k_neighbor_index(self):
        xs_initial = self.read_pkl(self.xs_initial_data_path)
        xs_initial = xs_initial.squeeze(1)

        Idx_Mat = []
        for i in trange(xs_initial.shape[0]):
            # print(i)
            # print(xs_initial[i])
            # print(xs_initial[i].shape)
            EuclideanD_Mat = self.EuclideanDistances(xs_initial[i], xs_initial[i])
            EuclideanD_Mat = np.transpose(EuclideanD_Mat)
            # print(EuclideanD_Mat)
            # print(EuclideanD_Mat[0])
            idx = EuclideanD_Mat.argsort(axis=1)
            # print(idx)
            idx = np.array(idx)
            idx = idx[:, 0:32]
            # print("变换后", idx.shape)

            if i == 0:
                # print(type(idx))
                # print(idx.shape)
                Idx_Mat = np.reshape(idx, (1, 2000, 32))
                # print(Idx_Mat.shape)
                # print(idx.shape)
            else:
                idx = idx.reshape(1, 2000, 32)
                Idx_Mat = np.append(Idx_Mat, idx, axis=0)

        self.save_pkl(self.save_data_path, Idx_Mat)

    """
        第二个网络代码
    """
    def Get_2_dimension_64_neighbor_index(self):
        xs_initial = self.read_pkl(self.xs_initial_data_path)
        print(len(xs_initial))
        print(xs_initial[0][0])

        Idx_Mat = []
        # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
        for i in trange(len(xs_initial)):
            xs_initial_temp = xs_initial[i][0]

            kpts0 = np.stack([xs_initial_temp[:, 0], xs_initial_temp[:, 1]], axis=-1)
            kpts1 = np.stack([xs_initial_temp[:, 2], xs_initial_temp[:, 3]], axis=-1)

            img_size0 = np.array((250, 250))               # 1000 1000
            img_size1 = np.array((250, 250))               # 1000 1000
            kpts0 = kpts0 * img_size0 / 2 + img_size0 / 2
            kpts1 = kpts1 * img_size1 / 2 + img_size1 / 2

            xs_initial_temp = np.concatenate((kpts0, kpts1), axis=1)

            # return
            # print(i)
            # print(xs_initial[i])
            # print(xs_initial[i][:, 0:2])
            # print(xs_initial[i][:, 0:2].shape)

            EuclideanD_Mat = self.EuclideanDistances(xs_initial_temp, xs_initial_temp)
            EuclideanD_Mat = np.transpose(EuclideanD_Mat)
            # print(EuclideanD_Mat)
            # print(EuclideanD_Mat[0])
            idx = EuclideanD_Mat.argsort(axis=1)
            # print(idx)
            idx = np.array(idx)
            idx = idx[:, 0:64]                            # 8

            if i == 0:
                # print(type(idx))
                # print(idx.shape)
                Idx_Mat = np.reshape(idx, (1, 2000, 64))     # 8
                # print(Idx_Mat.shape)
                # print(idx.shape)
            else:
                idx = idx.reshape(1, 2000, 64)              # 8
                Idx_Mat = np.append(Idx_Mat, idx, axis=0)

        self.save_pkl(self.save_data_path, Idx_Mat)

    """
         第一个网络NM-net
    """
    def Get_2_dimension_NM_Net_8_neighbor_index(self, datasets):
        xs_initial = self.read_pkl(self.xs_initial_data_path)
        others = self.read_pkl(self.others_data_path)
        print(len(xs_initial))
        print(xs_initial[0][0])

        Idx_Mat = []
        # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
        for i in trange(len(xs_initial)):
            xs_initial_temp = xs_initial[i][0]

            kpts0 = np.stack([xs_initial_temp[:, 0], xs_initial_temp[:, 1]], axis=-1)
            kpts1 = np.stack([xs_initial_temp[:, 2], xs_initial_temp[:, 3]], axis=-1)

            if datasets == "gl3d":
                img_size0 = np.array((1000, 1000))               # 1000 1000
                img_size1 = np.array((1000, 1000))               # 1000 1000
                kpts0 = kpts0 * img_size0 / 2 + img_size0 / 2
                kpts1 = kpts1 * img_size1 / 2 + img_size1 / 2
            elif datasets == "Hpatch":
                W = others[i][3][1]
                H = others[i][3][0]
                kpts0[:, 0] = kpts0[:, 0] * W / 2 + W / 2
                kpts0[:, 1] = kpts0[:, 1] * H / 2 + H / 2

                kpts1[:, 0] = kpts1[:, 0] * W / 2 + W / 2
                kpts1[:, 1] = kpts1[:, 1] * H / 2 + H / 2
            else:
                pass

            xs_initial_temp = np.concatenate((kpts0, kpts1), axis=1)

            # return
            # print(i)
            # print(xs_initial[i])
            # print(xs_initial[i][:, 0:2])
            # print(xs_initial[i][:, 0:2].shape)

            EuclideanD_Mat = self.EuclideanDistances(xs_initial_temp, xs_initial_temp)
            EuclideanD_Mat = np.transpose(EuclideanD_Mat)
            # print(EuclideanD_Mat)
            # print(EuclideanD_Mat[0])
            idx = EuclideanD_Mat.argsort(axis=1)
            # print(idx)
            idx = np.array(idx)
            idx = idx[:, 0: 8]                            # 8

            if i == 0:
                # print(type(idx))
                # print(idx.shape)
                Idx_Mat = np.reshape(idx, (1, 2000, 8))     # 8
                # print(Idx_Mat.shape)
                # print(idx.shape)
            else:
                idx = idx.reshape(1, 2000, 8)              # 8
                Idx_Mat = np.append(Idx_Mat, idx, axis=0)

        self.save_pkl(self.save_data_path, Idx_Mat)


"""
def CaoGao_Function():
    wide_test_xs_initial_data_path = 'E:/NM-Net-Initial/datasets/WIDE/train/xs_initial.pkl'
    xs_initial = read_pkl(wide_test_xs_initial_data_path)

    wide_test_index_data_path = 'E:/NM-Net-Initial/datasets/WIDE/train/index.pkl'
    index = read_pkl(wide_test_index_data_path)
    print(".........................")


    # wide_test_my_list_data_path = 'E:/NM-Net-Initial/code/my_list.pkl'
    # my_list = read_pkl(wide_test_my_list_data_path)
    # print("OK", type(my_list), my_list)
    # return


    A = np.array([[1, 0, 1],
                  [1, 0, 0],
                  [1, 1, 1]])

    EuclideanDistances(A, A)

    xs_initial = xs_initial.squeeze(1)
    # print(xs_initial.shape)
    # print(xs_initial.shape[0])

    Idx_Mat = []
    for i in trange(xs_initial.shape[0]):
        # print(i)
        # print(xs_initial[i])
        # print(xs_initial[i].shape)
        EuclideanD_Mat = EuclideanDistances(xs_initial[i], xs_initial[i])
        EuclideanD_Mat = np.transpose(EuclideanD_Mat)
        # print(EuclideanD_Mat)
        # print(EuclideanD_Mat[0])
        idx = EuclideanD_Mat.argsort(axis=1)
        # print(idx)
        idx = np.array(idx)
        idx = idx[:, 0:8]
        # print("变换后", idx.shape)

        if i == 0:
            # print(type(idx))
            # print(idx.shape)
            Idx_Mat = np.reshape(idx, (1, 2000, 8))
            # print(Idx_Mat.shape)
            # print(idx.shape)
        else:
            idx = idx.reshape(1, 2000, 8)
            Idx_Mat = np.append(Idx_Mat, idx, axis=0)

        # if i == 50:
        #     break

    print(Idx_Mat.shape)

    # data = []
    # data.append(Idx_Mat)

    pickle_file = open('my_list.pkl', 'wb')  # 创建一个pickle文件，文件后缀名随意,但是打开方式必须是wb（以二进制形式写入）
    pickle.dump(Idx_Mat, pickle_file)  # 将列表倒入文件
    pickle_file.close()
"""

if __name__ == '__main__':
    # wide_test_xs_initial_data_path = 'E:/NM-Net-Initial/datasets/WIDE/train/xs_initial.pkl'

    # wide_test_xs_initial_data_path = "E:/NM-net-xiexie/datasets/COLMAP/train/xs_4.pkl"
    # test_data_save_path = "E:/NM-net-xiexie/datasets/COLMAP/train/index.pkl"
    """
    wide_test_xs_initial_data_path = "E:/NM-net-xiexie/datasets/COLMAP_hehe_j/test/xs_4.pkl"
    test_data_save_path = "E:/NM-net-xiexie/datasets/COLMAP_hehe_j/test/index.pkl"
    """

    wide_test_xs_initial_data_path = "E:/NM-net-xiexie/datasets/Hpatch_Data_new/train/xs_4.pkl"
    wide_test_others_data_path = "E:/NM-net-xiexie/datasets/Hpatch_Data_new/train/others.pkl"
    test_data_save_path = "E:/NM-net-xiexie/datasets/Hpatch_Data_new/train/index.pkl"

    Generate_K_LinYu_Instance = Generate_K_LinYu(wide_test_xs_initial_data_path, wide_test_others_data_path, test_data_save_path)
    # Generate_K_LinYu_Instance.Get_2_dimension_k_neighbor_index()
    Generate_K_LinYu_Instance.Get_2_dimension_NM_Net_8_neighbor_index("Hpatch")
