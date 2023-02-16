#!/usr/bin/env python3
import datetime
import os
import time
import torch
import math
from dataset import Data_Loader
import pickle
import numpy as np
from torch.autograd import Variable
import tensorflow as tf
import torch.utils.data as Data
import cv2
from six.moves import xrange
from transformations import quaternion_from_matrix
import loss
import preprocess
import torch as th
from tqdm import trange
import matplotlib.pyplot as plt
from tqdm import tqdm


def draw_matches(img0, img1, kpts0, kpts1, match_idx, label, mask,
                 downscale_ratio=1, color=(255, 0, 0), radius=4, thickness=2):
    # Args:
    #     img: color image.
    #     kpts: Nx2 numpy array.
    #     match_idx: Mx2 numpy array indicating the matching index.
    # Returns:
    #     display: image with drawn matches.

    print("标签", label, label.sum())

    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))
    resize1 = cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]
    rows1, cols1 = resize1.shape[:2]

    kpts0 *= downscale_ratio
    kpts1 *= downscale_ratio

    # 之前是行
    display = np.zeros((rows0 + rows1, max(cols0, cols1), 3))
    display[:rows0, :cols0, :] = resize0
    display[rows0:(rows0 + rows1), :cols0, :] = resize1

    """
    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1
    """

    if 1:
        count = 0
        for idx in range(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            val = match_idx[idx]
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            # 原来是行
            # pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))
            pt1 = (int(kpts1[val[1]][0]), int(kpts1[val[1]][1]) + rows0)

            # 只显示正确匹配
            label_temp = label.reshape(-1)
            # print(5555555555, label_temp, val[0])
            if int(label_temp[val[0]]) == 1 and int(mask[val[0]]) == 1:
                # count += 1
                # print("正确", label_temp[val[0]], mask[val[0]])
                cv2.circle(display, pt0, radius, (0, 255, 0), thickness)
                cv2.circle(display, pt1, radius, (0, 255, 0), thickness)
                cv2.line(display, pt0, pt1, (0, 255, 0), thickness)  # color

                # cv2.arrowedLine(display, pt0, pt1, (255, 0, 0), 9, 20, 0, 0.3)  # 画箭头
                # print(pt0, pt1)
                """
                if count >= 80:
                    break
                """
            elif int(label_temp[val[0]]) == 0: # and int(mask[val[0]]) == 0:
                count += 1

                if 0:
                    cv2.circle(display, pt0, radius, (255, 0, 0), thickness)
                    cv2.circle(display, pt1, radius, (255, 0, 0), thickness)
                    cv2.line(display, pt0, pt1, (255, 0, 0), thickness)  # color
                else:
                    pass
            
                if count >= 200:
                    break

            else:
                pass

            name = "E:/NM-net-xiexie/contrat_experiment/plot_diaplay/" + "display3_" + str(count) + ".png"

    cv2.imwrite(name, display)
    display /= 255
    return display


def draw_matches_hengxiang(img0, img1, kpts0, kpts1, match_idx, label, mask,
                 downscale_ratio=1, color=(255, 0, 0), radius=4, thickness=2):
    # Args:
    #     img: color image.
    #     kpts: Nx2 numpy array.
    #     match_idx: Mx2 numpy array indicating the matching index.
    # Returns:
    #     display: image with drawn matches.

    print("标签", label, label.sum())

    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))
    resize1 = cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]
    rows1, cols1 = resize1.shape[:2]

    kpts0 *= downscale_ratio
    kpts1 *= downscale_ratio

    # 之前是行
    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1

    if 1:
        count = 0
        for idx in range(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            val = match_idx[idx]
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            # 原来是行
            pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))

            # 只显示正确匹配
            label_temp = label.reshape(-1)
            # print(5555555555, label_temp, val[0])
            if int(label_temp[val[0]]) == 1 and int(mask[val[0]]) == 1:
                # count += 1
                # print("正确", label_temp[val[0]], mask[val[0]])
                cv2.circle(display, pt0, radius, (0, 255, 0), thickness)
                cv2.circle(display, pt1, radius, (0, 255, 0), thickness)
                cv2.line(display, pt0, pt1, (0, 255, 0), thickness)  # color
                # print(pt0, pt1)
                """
                if count >= 80:
                    break
                """
            elif int(label_temp[val[0]]) == 1 and int(mask[val[0]]) == 0:
                # if idx % 10 == 0:
                """
                count += 1

                if count > 200:
                    # print("count进入")
                    continue
                """

                if 1:
                    cv2.circle(display, pt0, radius, (255, 0, 0), thickness)
                    cv2.circle(display, pt1, radius, (255, 0, 0), thickness)
                    cv2.line(display, pt0, pt1, (255, 0, 0), thickness)  # color
                else:
                    pass

            else:
                pass

            name = "E:/NM-net-xiexie/contrat_experiment/plot_diaplay/" + "display3_" + str(count) + ".png"

    # cv2.imwrite(name, display)
    display /= 255
    return display


def draw_vector_field(img0, kpts0, kpts1, match_idx, label, mask,
                 downscale_ratio=1, color=(255, 0, 0), radius=4, thickness=2):
    # Args:
    #     img: color image.
    #     kpts: Nx2 numpy array.
    #     match_idx: Mx2 numpy array indicating the matching index.
    # Returns:
    #     display: image with drawn matches.

    print("标签", label, label.sum())

    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]

    kpts0 *= downscale_ratio
    kpts1 *= downscale_ratio

    # 之前是行
    display = np.zeros((rows0, cols0, 3))
    # display = np.full((rows0, cols0, 3), 300)
    # print("display2", display)
    # display[:rows0, :cols0, :] = resize0
    # display[:rows1, cols0:(cols0 + cols1), :] = resize1


    for x in range(rows0):
        # print("x", x)
        # for y in range(cols0):
        cv2.line(display, (x, 0), (x, 999), (255, 255, 255), thickness)  # color

    if 1:
        count = 0
        for idx in range(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            val = match_idx[idx]
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            # 原来是行
            pt1 = (int(kpts1[val[1]][0]), int(kpts1[val[1]][1]))

            # 只显示正确匹配
            label_temp = label.reshape(-1)
            if int(label_temp[val[0]]) == 1:
                count += 1
                cv2.circle(display, pt0, radius, (0, 0, 255), thickness)
                cv2.circle(display, pt1, radius, (0, 0, 255), thickness)
                # cv2.line(display, pt0, pt1, (0, 0, 255), thickness)  # color
                cv2.arrowedLine(display, pt0, pt1, (0, 0, 255), 2, 2, 0, 0.1)  # 画箭头
                # print(pt0, pt1)
                """
                if count >= 80:
                    break
                """
            elif int(label_temp[val[0]]) == 0 and int(mask[val[0]]) == 0:
                count += 1
                cv2.arrowedLine(display, pt0, pt1, (255, 0, 0), 2, 2, 0, 0.1)  # 画箭头
                """
                if 1:
                    cv2.circle(display, pt0, radius, (0, 255, 0), thickness)
                    cv2.circle(display, pt1, radius, (0, 255, 0), thickness)
                    cv2.line(display, pt0, pt1, (0, 255, 0), thickness)  # color
                else:
                    pass

                if count >= 80:
                    break
                """
            else:
                pass

            name = "E:/NM-net-xiexie/contrat_experiment/plot_diaplay/" + "display3_" + str(count) + ".png"

    # cv2.imwrite(name, display)
    display /= 255
    # print("display", display.shape, display)
    return display


def draw_location(img0, img1, location0, location1, match_idx,
                 downscale_ratio=1, color=(255, 0, 0), radius=4, thickness=2):
    """
    Args:
        img: color image.
        kpts: Nx2 numpy array.
        match_idx: Mx2 numpy array indicating the matching index.
    Returns:
        display: image with drawn matches.
    """
    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))
    resize1 = cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]
    rows1, cols1 = resize1.shape[:2]

    location0 *= downscale_ratio
    location1 *= downscale_ratio

    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1

    # print(kpts0)
    print("画边缘")
    print(location0.shape, location1.shape)
    # print(location0)

    if 1:
        """
        for idx in range(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            print(678, "hhehe")
            val = match_idx[idx]
            print(val, len(kpts0))
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))
            print(pt0, pt1)

            cv2.circle(display, pt0, radius, color, thickness)
            cv2.circle(display, pt1, radius, color, thickness)
            # cv2.line(display, pt0, pt1, color, thickness)
        """

        # 画方框
        for id_x in range(32):
            if (id_x == 0 or id_x == 31):  # or id_y!= 0 or id_y != 31 or id_x != 31
                for id_y in range(32):
                    # print("进入1111111")
                    pt0 = (int(location0[id_x][id_y][0]), int(location0[id_x][id_y][1]))
                    pt1 = (int(location1[id_x][id_y][0]) + cols0, int(location1[id_x][id_y][1]))

                    cv2.circle(display, pt0, radius, color, thickness)
                    cv2.circle(display, pt1, radius, color, thickness)
            else:
                pt0 = (int(location0[id_x][0][0]), int(location0[id_x][0][1]))
                pt1 = (int(location1[id_x][0][0]) + cols0, int(location1[id_x][0][1]))

                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)

                pt0 = (int(location0[id_x][31][0]), int(location0[id_x][31][1]))
                pt1 = (int(location1[id_x][31][0]) + cols0, int(location1[id_x][31][1]))

                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)

        # 画中心
        for pianyi_x in range(-1, 2):
            for pianyi_y in range(-1, 2):
                pt0 = (int(location0[15 + pianyi_x][15 + pianyi_y][0]), int(location0[15 + pianyi_x][15 + pianyi_y][1]))
                pt1 = (int(location1[15 + pianyi_x][15 + pianyi_y][0]) + cols0, int(location1[15 + pianyi_x][15 + pianyi_y][1]))
                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)

        # 画局部一致性
        if 1:
            for pianyi_x in range(-12, 13, 8):
                for pianyi_y in range(-12, 13, 8):
                    if(pianyi_x == pianyi_y):
                        pass
                    else:
                        continue
                    pt0 = (int(location0[15 + pianyi_x][15 + pianyi_y][0]), int(location0[15 + pianyi_x][15 + pianyi_y][1]))
                    pt1 = (int(location1[15 + pianyi_x][15 + pianyi_y][0]) + cols0, int(location1[15 + pianyi_x][15 + pianyi_y][1]))
                    cv2.line(display, pt0, pt1, (0, 255, 0), thickness)

        # 画全局结构一致性
        if 0:
            point0 = [[505, 877], [500, 761], [497, 676], [588, 669], [592, 792], [655, 774], [688, 846]]
            point1 = [[1624, 732], [1633, 648], [1645, 581], [1738, 581], [1719, 680], [1779, 682], [1799, 761]]
            for i in range(len(point0)):
                cv2.circle(display, tuple(point0[i]), radius, color, thickness)
                cv2.circle(display, tuple(point1[i]), radius, color, thickness)
                cv2.line(display, tuple(point0[i]), tuple(point1[i]), (0, 255, 0), thickness)

        # cv2.arrowedLine(display, (300, 300), (500, 500), (255, 0, 0), 20, 20, 0, 0.3)  # 画箭头
        # cv2.imshow('q', img)
        # cv2.waitKey()
    display /= 255
    return display

def draw_location2(img0, img1, location0, location1, index,
                 downscale_ratio=1, color=(255, 0, 0), radius=4, thickness=2):
    """
    Args:
        img: color image.
        kpts: Nx2 numpy array.
        match_idx: Mx2 numpy array indicating the matching index.
    Returns:
        display: image with drawn matches.
    """
    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))
    resize1 = cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]
    rows1, cols1 = resize1.shape[:2]

    location0 *= downscale_ratio
    location1 *= downscale_ratio

    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1

    # print(kpts0)
    print("画边缘")
    print(location0.shape, location1.shape)
    # print(location0)
    if 0:
        image1 = np.zeros((64, 64, 3))
        image2 = np.zeros((64, 64, 3))
        for id_x in range(64):
            for id_y in range(64):
                # print("进入1111111")
                pt0 = (int(location0[id_x][id_y][0]), int(location0[id_x][id_y][1]))
                pt1 = (int(location1[id_x][id_y][0]), int(location1[id_x][id_y][1]))
                if pt0[0] >= 1000 or pt0[0] < 0:
                    continue
                if pt0[1] >= 1000 or pt0[1] < 0:
                    continue
                if pt1[0] >= 1000 or pt1[0] < 0:
                    continue
                if pt1[1] >= 1000 or pt1[1] < 0:
                    continue
                image1[id_x, id_y, :] = resize0[pt0[0], pt0[1], :]
                image2[id_x, id_y, :] = resize1[pt1[0], pt1[1], :]

        image1 = cv2.resize(image1, (64, 64))
        image2 = cv2.resize(image2, (64, 64))
        image = np.concatenate([image1, image2], 1)

        path = "C:/Users/Administrator.DESKTOP-3TQ2JAH/Desktop/author_huitu/test15/"
        path = path + "image_" + str(index) + ".png"
        cv2.imwrite(path, image)

    if 1:
        """
        for idx in range(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            print(678, "hhehe")
            val = match_idx[idx]
            print(val, len(kpts0))
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))
            print(pt0, pt1)

            cv2.circle(display, pt0, radius, color, thickness)
            cv2.circle(display, pt1, radius, color, thickness)
            # cv2.line(display, pt0, pt1, color, thickness)
        """

        # 画方框
        for id_x in range(16):
            if (id_x == 0 or id_x == 15):  # or id_y!= 0 or id_y != 31 or id_x != 31
                for id_y in range(16):
                    # print("进入1111111")
                    pt0 = (int(location0[id_x][id_y][0]), int(location0[id_x][id_y][1]))
                    pt1 = (int(location1[id_x][id_y][0]) + cols0, int(location1[id_x][id_y][1]))

                    cv2.circle(display, pt0, radius, color, thickness)
                    cv2.circle(display, pt1, radius, color, thickness)
            else:
                pt0 = (int(location0[id_x][0][0]), int(location0[id_x][0][1]))
                pt1 = (int(location1[id_x][0][0]) + cols0, int(location1[id_x][0][1]))

                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)

                pt0 = (int(location0[id_x][15][0]), int(location0[id_x][15][1]))
                pt1 = (int(location1[id_x][15][0]) + cols0, int(location1[id_x][15][1]))

                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)

    display /= 255
    return display


def draw_patch(img0, img1, kpts0, kpts1, match_idx, label, tuxiang_count, downscale_ratio=1, color=(255, 0, 0), radius=4, thickness=2):
    """
    Args:
        img: color image.
        kpts: Nx2 numpy array.
        match_idx: Mx2 numpy array indicating the matching index.
    Returns:
        display: image with drawn matches.
    """
    resize0 = cv2.resize(
        img0, (int(img0.shape[1] * downscale_ratio), int(img0.shape[0] * downscale_ratio)))
    resize1 = cv2.resize(
        img1, (int(img1.shape[1] * downscale_ratio), int(img1.shape[0] * downscale_ratio)))

    rows0, cols0 = resize0.shape[:2]
    rows1, cols1 = resize1.shape[:2]

    # location0 *= downscale_ratio
    # location1 *= downscale_ratio

    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    """
    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1
    """
    """
    print("loc", location0.shape)
    location0 = location0.astype(np.uint8)
    location1 = location1.astype(np.uint8)
    remap0 = img0[location0[:, :, 0], location0[:, :, 1], :]
    remap1 = img1[location1[:, :, 0], location1[:, :, 1], :]
    display = np.zeros((32, 64, 3))
    display[:32, :32, :] = remap0
    display[:32, 32: 64, :] = remap1
    """

    count = 0
    for idx in range(match_idx.shape[0]):
        # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
        val = match_idx[idx]
        pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
        # 原来是行
        pt1 = (int(kpts1[val[1]][0]), int(kpts1[val[1]][1]))

        # 只显示正确匹配
        label_temp = label.reshape(-1)
        if int(label_temp[val[0]]) == 1:  # and int(mask[val[0]]) == 1:
            img0_patch = img0[max(pt0[0] - 32, 0):min(pt0[0] + 32, 1000), max(pt0[1] - 32, 0):min(pt0[1] + 32, 1000), :]
            img1_patch = img1[max(pt1[0] - 32, 0):min(pt1[0] + 32, 1000), max(pt1[1] - 32, 0):min(pt1[1] + 32, 1000), :]
            img0_patch = cv2.resize(img0_patch, (64, 64))
            img1_patch = cv2.resize(img1_patch, (64, 64))
            img_hebing = np.concatenate((img0_patch, img1_patch), 1)

            # cv2.namedWindow("123", cv2.WINDOW_NORMAL)
            # cv2.imshow("123", img0_patch)
            if 0:
                print("进入图像imwrite")
                path = "C:/Users/Administrator.DESKTOP-3TQ2JAH/Desktop/author_huitu/test_" + str(tuxiang_count) + "/"
                if not os.path.exists(path):
                    os.mkdir(path)
                name = "img" + str(idx) + ".png"
                cv2.imwrite(path + name, img_hebing)
            # cv2.waitKey(-1)

    if 1:
        """
        for idx in range(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            print(678, "hhehe")
            val = match_idx[idx]
            print(val, len(kpts0))
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))
            print(pt0, pt1)

            cv2.circle(display, pt0, radius, color, thickness)
            cv2.circle(display, pt1, radius, color, thickness)
            # cv2.line(display, pt0, pt1, color, thickness)
        """
        # 画方框
        for id_x in range(32):
            if (id_x == 0 or id_x == 31):  # or id_y!= 0 or id_y != 31 or id_x != 31
                for id_y in range(32):
                    # print("进入1111111")
                    pt0 = (int(location0[id_x][id_y][0]), int(location0[id_x][id_y][1]))
                    pt1 = (int(location1[id_x][id_y][0]) + cols0, int(location1[id_x][id_y][1]))

                    cv2.circle(display, pt0, radius, color, thickness)
                    cv2.circle(display, pt1, radius, color, thickness)
            else:
                pt0 = (int(location0[id_x][0][0]), int(location0[id_x][0][1]))
                pt1 = (int(location1[id_x][0][0]) + cols0, int(location1[id_x][0][1]))

                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)

                pt0 = (int(location0[id_x][31][0]), int(location0[id_x][31][1]))
                pt1 = (int(location1[id_x][31][0]) + cols0, int(location1[id_x][31][1]))

                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)

        # 画中心
        for pianyi_x in range(-1, 2):
            for pianyi_y in range(-1, 2):
                pt0 = (int(location0[15 + pianyi_x][15 + pianyi_y][0]), int(location0[15 + pianyi_x][15 + pianyi_y][1]))
                pt1 = (int(location1[15 + pianyi_x][15 + pianyi_y][0]) + cols0, int(location1[15 + pianyi_x][15 + pianyi_y][1]))
                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)

        # 画局部一致性
        for pianyi_x in range(-12, 13, 8):
            for pianyi_y in range(-12, 13, 8):
                if(pianyi_x == pianyi_y):
                    pass
                else:
                    continue
                pt0 = (int(location0[15 + pianyi_x][15 + pianyi_y][0]), int(location0[15 + pianyi_x][15 + pianyi_y][1]))
                pt1 = (int(location1[15 + pianyi_x][15 + pianyi_y][0]) + cols0, int(location1[15 + pianyi_x][15 + pianyi_y][1]))
                cv2.line(display, pt0, pt1, (0, 255, 0), thickness)

        # cv2.arrowedLine(display, (300, 300), (500, 500), (255, 0, 0), 20, 20, 0, 0.3)  # 画箭头
        # cv2.imshow('q', img)
        # cv2.waitKey()
    display /= 255
    return display


min_kp_num = 500
margin = 0.05


def estimate_E(input, weight):
    E = []
    data = []
    data.append(torch.unsqueeze(input[:, 0, :, 0], -1))  # u1
    data.append(torch.unsqueeze(input[:, 0, :, 1], -1))  # v1
    data.append(torch.unsqueeze(input[:, 0, :, 2], -1))  # u2
    data.append(torch.unsqueeze(input[:, 0, :, 3], -1))
    data.append(torch.unsqueeze(torch.ones(input.size(0), input.size(2)).cuda(), -1))
    X = torch.cat((data[2] * data[0], data[2] * data[1], data[2], data[3] * data[0], data[3] * data[1],
                   data[3], data[0], data[1], data[4]), -1)

    W = torch.stack([torch.diag(weight[i]) for i in range(1)])
    M = torch.bmm(X.transpose(1, 2), W)
    M = torch.bmm(M, X)
    svd = torch.stack([torch.svd(M[i])[2][:, 8] for i in range(1)])
    E = (svd / (torch.norm(svd, 2, dim=-1, keepdim=True) + 1e-6))
    return E


def CV_estimate_E(input, weight):
    try:
        inlier = input.squeeze()[weight.squeeze() > 0].cpu().data.numpy()
        x = inlier[:, :2]
        y = inlier[:, 2:]
        e, mask = cv2.findFundamentalMat(x, y, cv2.FM_8POINT)
        e = np.reshape(e, (1, 9))
        e = (e / np.linalg.norm(e, ord=2, axis=1, keepdims=True))
        return e, 1
    except (IndexError, ValueError):
        return 0, 0


def local_consistency(xs_initial, affine):  # 16*1*2000*4 16*2000*18
    x = xs_initial[:, 0, :, 0:2]
    y = xs_initial[:, 0, :, 2:4]
    affine = affine.view(-1, 18)
    affine_x = torch.stack([torch.inverse(affine[i][:9].view(3, 3)) for i in range(affine.shape[0])])
    affine_y = affine[:, 9:].view(-1, 3, 3)

    H = torch.bmm(affine_y, affine_x).view(xs_initial.size(0), xs_initial.size(2), 3, 3)

    ones = torch.ones(x.size(0), x.size(1), 1)
    x = torch.cat((x, ones), dim=-1)
    y = torch.cat((y, ones), dim=-1)
    index = []
    for i in range(x.size(0)):
        x_repeat = x[i].t().repeat(x[i].size(0), 1, 1)
        y_prj = torch.bmm(H[i], x_repeat)
        y_prj = (y_prj / y_prj[:, 2, :].unsqueeze(1))[:, 0:2, :]

        prj_dis = torch.stack(
            [torch.sum(torch.pow((y_prj[idx] - y_prj[idx, :, idx].unsqueeze(-1)), 2), dim=0) for idx in
             range(x[i].size(0))])
        prj_dis = prj_dis + prj_dis.t()
        index.append(torch.sort(prj_dis, dim=1, descending=False)[1][:, :8])
    return torch.stack(index)


def local_score(xs_initial, affine):  # 16*1*2000*4 16*2000*18
    x = xs_initial[:, 0, :, 0:2]
    y = xs_initial[:, 0, :, 2:4]
    affine = affine.view(-1, 18)
    affine_x = torch.stack([torch.inverse(affine[i][:9].view(3, 3)) for i in range(affine.shape[0])])
    affine_y = affine[:, 9:].view(-1, 3, 3)

    H = torch.bmm(affine_y, affine_x).view(xs_initial.size(0), xs_initial.size(2), 3, 3)

    ones = torch.ones(x.size(0), x.size(1), 1)
    x = torch.cat((x, ones), dim=-1)
    y = torch.cat((y, ones), dim=-1)
    score = []
    for i in range(x.size(0)):
        x_repeat = x[i].t().repeat(x[i].size(0), 1, 1)
        y_prj = torch.bmm(H[i], x_repeat)
        y_prj = (y_prj / y_prj[:, 2, :].unsqueeze(1))[:, 0:2, :]

        prj_dis = torch.stack(
            [torch.sum(torch.pow((y_prj[idx] - y_prj[idx, :, idx].unsqueeze(-1)), 2), dim=0) for idx in
             range(x[i].size(0))])
        prj_dis = prj_dis + prj_dis.t()
        score.append(torch.sort(prj_dis, dim=1, descending=False)[0][:, :8])
    score = torch.stack(score)
    score = (-margin * score).exp()

    return score


def image_join(image1, image2):
    """
    水平合并两个opencv图像矩阵为一个图像矩阵
    :param image1:
    :param image2:
    :return:
    """
    h1, w1 = image1.shape[0:2]
    h2, w2 = image2.shape[0:2]

    if h1 > h2:
        margin_height = h1 - h2
        if margin_height % 2 == 1:
            # margin_top = int(margin_height / 2)
            # margin_top = margin_height
            # margin_bottom = margin_top + 1
            margin_top = 0
            margin_bottom = margin_height
        else:
            # margin_top = margin_bottom = int((h1 - h2) / 2)
            # margin_top = margin_height
            margin_top = 0
            margin_bottom = margin_height
        image2 = cv2.copyMakeBorder(image2, margin_top, margin_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        print("hehhe", image2.shape)
    elif h2 > h1:
        margin_height = h2 - h1
        if margin_height % 2 == 1:
            # margin_top = int(margin_height / 2)
            # margin_top = margin_height
            # margin_bottom = margin_top + 1
            margin_top = 0
            margin_bottom = margin_height
        else:
            # margin_top = margin_bottom = int(margin_height / 2)
            # margin_top = margin_height
            margin_top = 0
            margin_bottom = margin_height
        image1 = cv2.copyMakeBorder(image1, margin_top, margin_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return np.concatenate((image1, image2), axis=1)


def test_process(mode, save_file_cur, model_name, data_name, config, f1, f2, epoch, datasets, adjacency_num=8,
                 Image_Preprocess_XiaoRong=False, display=False):
    d = Data_Loader(config, data_name, None, mode, initialize=False)

    data = Data.DataLoader(d, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    Network = torch.load(save_file_cur + model_name).cuda()
    Network.eval()
    # print(Network)

    loss_func = loss.Loss_classi().cuda()
    loss_list_hehe = []

    P = []
    R = []
    F = []
    MSE = []
    MAE = []

    for i, (img1s_w, img2s_w, img1s_z_g, img2s_z_g, xs_4, label, xs_12, index, others, xs_6) in enumerate(tqdm(data, 0)):
        if config.use_which_network == 0:
            if datasets == "gl3d":
                # 原来的  gl3d
                xs1s = xs_12[:, :, 0: 6]
                xs2s = xs_12[:, :, 6: 12]
            elif datasets == "Hpatch":
                # print(22222222222222, xs_12.shape)
                # 现在的   HpData
                xs1s = xs_12[:, :, :, 0: 6]
                xs2s = xs_12[:, :, :, 6: 12]
                xs1s = xs1s.squeeze(0)
                xs2s = xs2s.squeeze(0)
            else:
                pass

            if Image_Preprocess_XiaoRong == True:
                # print("图像预处理消融实验开启！！！！")
                xs1s[:, :, 0] = 1
                xs1s[:, :, 1] = 0
                xs1s[:, :, 3] = 0
                xs1s[:, :, 4] = 1

                xs2s[:, :, 0] = 1
                xs2s[:, :, 1] = 0
                xs2s[:, :, 3] = 0
                xs2s[:, :, 4] = 1
            else:
                pass

            # print(22222, type(img1s_z.numpy().squeeze(0)), img2s_z.numpy().squeeze(0).shape)
            # img1s_z_g = cv2.cvtColor(img1s_z.numpy().squeeze(0), cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
            # img2s_z_g = cv2.cvtColor(img2s_z.numpy().squeeze(0), cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B

            if datasets == "gl3d":
                img1s_z_g = img1s_z_g.reshape(1, 1000, 1000, 1)
                img2s_z_g = img2s_z_g.reshape(1, 1000, 1000, 1)
            elif datasets == "Hpatch":
                img1s_z_g = img1s_z_g.reshape(1, others[3][0], others[3][1], 1)
                img2s_z_g = img2s_z_g.reshape(1, others[4][0], others[4][1], 1)
            else:
                pass


            if 0:
                """
                    现在不确定这代码的作用到底啥 反正跑步通
                """
                # 第一张图片
                img1s_temp = th.tensor(img1s_z_g, dtype=th.float32, requires_grad=False)
                xs1s_temp = th.tensor(xs1s, dtype=th.float32, requires_grad=False)
                # 第二张图片
                img2s_temp = th.tensor(img2s_z_g, dtype=th.float32, requires_grad=False)
                xs2s_temp = th.tensor(xs2s, dtype=th.float32, requires_grad=False)

                img1s = f1(img1s_temp, xs1s_temp)
                img2s = f2(img2s_temp, xs2s_temp)

                img1s = torch.Tensor(img1s)
                img1s = img1s.permute(0, 3, 1, 2)
                img2s = torch.Tensor(img2s)
                img2s = img2s.permute(0, 3, 1, 2)
            else:
                # img1s = img1s_z_g.squeeze(0).numpy()
                # img2s = img2s_z_g.squeeze(0).numpy()
                img1s = img1s_w.squeeze(0).numpy()
                img2s = img2s_w.squeeze(0).numpy()
                xs1s = xs1s.squeeze(0).numpy()
                xs2s = xs2s.squeeze(0).numpy()

                import patch_extractor
                patch_extractor = patch_extractor.PatchExtractor()
                # import patch_extractor_new
                # patch_extractor = patch_extractor_new.PatchExtractor()

                # img1s, location0 = patch_extractor.get_patches(img1s, xs1s)
                # img2s, location1 = patch_extractor.get_patches(img2s, xs2s)

                img1s = patch_extractor.get_patches(img1s, xs1s)
                img2s = patch_extractor.get_patches(img2s, xs2s)

                # print("img1s, 2s", img1s.shape, img2s.shape)

                img1s = torch.tensor(img1s).unsqueeze(1)
                img2s = torch.tensor(img2s).unsqueeze(1)

            img1s = img1s.cuda()
            img2s = img2s.cuda()
            xs_4 = xs_4.cuda()
            label = label.cuda()
        else:
            pass

        with torch.no_grad():
            # output = output.squeeze(1)
            if 1:
                # print("img1s", img1s.shape, img2s.shape)
                output, weight = Network(img1s, img2s, xs_4, epoch)
                l = loss_func(output, label)
                loss_list_hehe += [l]

                label = label.type(torch.FloatTensor)
                mask = (weight > 0).type(torch.FloatTensor)

                p = torch.sum(mask * label) / torch.sum(mask)
                if math.isnan(p):
                    p = torch.Tensor([0])
                r = torch.sum(mask * label) / torch.sum(label)
                if math.isnan(r):
                    r = torch.Tensor([0])
                f = 2 * p * r / (p + r)
                if math.isnan(f):
                    f = torch.Tensor([0])

                P.append(p.cpu().numpy())
                R.append(r.cpu().numpy())
                F.append(f.cpu().numpy())
                # print(p.cpu().numpy(), r.cpu().numpy(), f.cpu().numpy())

        # if i == 14:
        #     continue
        if display:
            # print(11111, img1s_z_g.shape, img2s_z_g.shape)
            print(img1s_w.shape, img2s_w.shape)
            wanzheng_img0 = img1s_w.numpy().squeeze(0)
            wanzheng_img1 = img2s_w.numpy().squeeze(0)

            kpts0 = np.stack(
                [xs_4.cpu().numpy().squeeze(0).squeeze(0)[:, 0], xs_4.cpu().numpy().squeeze(0).squeeze(0)[:, 1]],
                axis=-1)
            kpts1 = np.stack(
                [xs_4.cpu().numpy().squeeze(0).squeeze(0)[:, 2], xs_4.cpu().numpy().squeeze(0).squeeze(0)[:, 3]],
                axis=-1)

            img_size0 = np.array((wanzheng_img0.shape[1], wanzheng_img0.shape[0]))
            img_size1 = np.array((wanzheng_img1.shape[1], wanzheng_img1.shape[0]))
            kpts0 = kpts0 * img_size0 / 2 + img_size0 / 2
            kpts1 = kpts1 * img_size1 / 2 + img_size1 / 2
            match_num = kpts0.shape[0]
            match_idx = np.tile(np.array(range(match_num))[..., None], [1, 2])  # match_num   range(2)

            label = label.cpu().numpy().squeeze(0).astype(np.int16)
            # mask = mask.numpy().squeeze(0).astype(np.int16)

            print(wanzheng_img0.shape, wanzheng_img1.shape)
            print("第几个:", i)
            # if i == 6:

            display1 = draw_matches_hengxiang(wanzheng_img0, wanzheng_img1, kpts0, kpts1, match_idx, label, mask,
                                    downscale_ratio=1.0)  # 1.0

            """
            display3 = draw_vector_field(wanzheng_img0, kpts0, kpts1, match_idx, label, mask,
                                    downscale_ratio=1.0)
            """

            print(len(location0))
            label_temp = label.reshape(-1)
            for idx in range(2000):
                # 只显示正确匹配
                if int(label_temp[idx]) == 1:
                    print("第几个idx", idx)
                    display2 = draw_location2(wanzheng_img0, wanzheng_img1, location0[idx], location1[idx], idx, downscale_ratio=1.0)

                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(display2)
                    plt.show()
                    break

            """
            display4 = draw_matches(wanzheng_img0, wanzheng_img1, kpts0, kpts1, match_idx, label, mask,
                                    downscale_ratio=1.0)  # 1.0
            """
            """
            print(555555, location0[0].shape)
            display5 = draw_patch(wanzheng_img0, wanzheng_img1, kpts0, kpts1, match_idx, label, i + 1, downscale_ratio=1.0)
            """
            """
            plt.xticks([])
            plt.yticks([])
            plt.imshow(display1)
            plt.show()
            """

            """
            plt.xticks([])
            plt.yticks([])
            plt.imshow(display2)
            plt.show()
            """
            """
            plt.xticks([])
            plt.yticks([])
            plt.imshow(display3)
            plt.show()

            plt.xticks([])
            plt.yticks([])
            plt.imshow(display4)
            plt.show()
            """

            plt.xticks([])
            plt.yticks([])
            plt.imshow(display2)
            plt.show()

            # continue

    p_ = np.expand_dims(np.mean(np.array(P)), axis=0)
    r_ = np.expand_dims(np.mean(np.array(R)), axis=0)
    f_ = np.expand_dims(np.mean(np.array(F)), axis=0)
    print(p_, r_, f_)

    log_path = os.path.join(save_file_cur, mode)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    np.savetxt(os.path.join(log_path, "Precision.txt"), p_ * 100)
    np.savetxt(os.path.join(log_path, "Recall.txt"), r_ * 100)
    np.savetxt(os.path.join(log_path, "F-measure.txt"), f_ * 100)

    loss_list = torch.stack(loss_list_hehe).view(-1)
    return f_, loss_list.mean()




