from __future__ import print_function
import pickle
import numpy as np
import torch
import os
import sys
import multiprocessing as mp
import time
import random
from tqdm import trange
from torch.utils.data import Dataset
import glob
from config import get_config, print_usage
from torchvision import transforms
from PIL import Image
import cv2
import torch.utils.data as Data
import tqdm
import joblib

max_kp_num = 2000
margin = 1e-3


def skew_symmetric(v):
    zero = np.zeros((len(v), 1))
    # print(v.shape)
    M = np.hstack((zero, -v[:, 2, :], v[:, 1, :],
                   v[:, 2, :], zero, -v[:, 0, :],
                   -v[:, 1, :], v[:, 0, :], zero))
    return M


def parallel(config, x, affine, y):
    c_num = x.shape[1]

    prj = []
    ys = np.zeros(y.shape, dtype=np.float32)
    for i in range(c_num):
        try:
            affine_x = np.linalg.inv(affine[i][:9].reshape(3, 3))
            affine_y = affine[i][9:].reshape(3, 3)
            H = np.dot(affine_y, affine_x)
            y_prj = np.dot(H, x)
            y_prj = (y_prj / np.expand_dims(y_prj[2, :], axis=0))[:2, :].transpose(1, 0)
            prj += [y_prj]
            ys[i] = y[i]

        except np.linalg.linalg.LinAlgError:
            ys[i] = [1, 1]
            continue

    prj_ = np.array(prj)

    prj_dis = np.zeros([len(prj), len(prj)], dtype=np.float32)

    for i in range(len(prj)):
        prj_dis[i, :] = np.sum(np.abs(prj_[:, i, :] - np.expand_dims(prj_[i, i, :], axis=0)), axis=-1)
    prj_dis = prj_dis + prj_dis.transpose(1, 0)

    index = np.zeros((len(prj), config.knn_num)).astype(np.int64)
    score = np.zeros((len(prj), config.knn_num)).astype(np.float32)

    for j in range(len(prj)):
        index[j, :] = np.argsort(prj_dis[j])[:config.knn_num]
        score[j, :] = np.exp(-np.sort(prj_dis[j])[:config.knn_num] * margin)

    if len(prj) < c_num:
        zeros = np.zeros([c_num - len(prj), config.knn_num], dtype=np.float32)
        score = np.concatenate([score, zeros], axis=0)
        pad_index = np.arange(len(prj), c_num)
        pad_index = np.expand_dims(pad_index, axis=-1).repeat(config.knn_num, axis=-1)
        index = np.concatenate([index, pad_index], axis=0)
    return index, score, ys


def local_score(config, xs_initial, affine, ys):
    x = xs_initial[0, :, 0:2]
    c_num = x.shape[0]
    ones = np.ones((c_num, 1))
    x = np.concatenate((x, ones), axis=-1).transpose(1, 0).astype(np.float32)

    index, score, y = parallel(config, x, affine, ys)
    return index, score, y


def data_initialization(config, database, data_list, score_idx=False):
    # Now load data.
    var_name_list = [
        "xs", "xs_initial", "ys", "Rs", "ts", "affine"
    ]
    data_folder = config.data_dump_prefix

    # Let's unpickle and save data
    data_name = []
    for data in data_list:
        data_name += [os.path.join(data_folder, data, "numkp-{}".format(config.obj_num_kp))]

    data = {}
    for cur_folder in data_name:

        ready_file = os.path.join(cur_folder, "ready")
        if not os.path.exists(ready_file):
            raise RuntimeError("Data is not prepared!")

        for var_name in var_name_list:
            cur_var_name = var_name + "_tr"
            in_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"

            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] = pickle.load(ifp)

    #  e_gt_unnorm = skew_symmetric(np.expand_dims(np.array(data["ts"]), axis=-1))
    e_gt_unnorm = np.reshape(np.matmul(
        np.reshape(skew_symmetric(np.expand_dims(np.array(data["ts"]), axis=-1)), (len(data["ts"]), 3, 3)),
        np.reshape(np.array(data["Rs"]), (len(data["ts"]), 3, 3))), (len(data["ts"]), 9))
    e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm, ord=2, axis=1, keepdims=True)
    data["Es"] = e_gt

    xs = []
    xs_initial = []
    ys = []
    Rs = []
    ts = []
    affine = []
    Es = []
    index = []
    score = []

    for i in trange(len(data["xs"])):
        xs += [data["xs"][i]]
        xs_initial += [data["xs_initial"][i]]
        Rs += [data["Rs"][i]]
        ts += [data["ts"][i]]
        affine += [data["affine"][i]]
        Es += [data["Es"][i]]

        if score_idx == True:
            idx, sco, y = local_score(config, data["xs_initial"][i], data["affine"][i], data["ys"][i])
            ys += [y]
            index += [idx]
            score += [sco]

    shuffle_list = list(zip(xs, xs_initial, ys, Rs, ts, affine, Es, index, score))
    random.shuffle(shuffle_list)

    xs, xs_initial, ys, Rs, ts, affine, Es, index, score = zip(*shuffle_list)

    var_name_list = ["xs", "xs_initial", "ys", "Rs", "ts", "affine", "Es", "index", "score"]

    data = {}
    data["xs"] = xs[:int(0.7 * len(xs))]
    data["xs_initial"] = xs_initial[:int(0.7 * len(xs))]
    data["ys"] = ys[:int(0.7 * len(xs))]
    data["Rs"] = Rs[:int(0.7 * len(xs))]
    data["ts"] = ts[:int(0.7 * len(xs))]
    data["affine"] = affine[:int(0.7 * len(xs))]
    data["Es"] = Es[:int(0.7 * len(xs))]
    data["index"] = index[:int(0.7 * len(xs))]
    data["score"] = score[:int(0.7 * len(xs))]

    print('Size of training data', len(data["xs"]))

    train_data_path = os.path.join(data_folder, database, "train")
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    for var_name in var_name_list:
        in_file_name = os.path.join(train_data_path, var_name) + ".pkl"
        with open(in_file_name, "wb") as ofp:
            pickle.dump(data[var_name], ofp)

    data = {}
    data["xs"] = xs[int(0.7 * len(xs)): int(0.85 * len(xs))]
    data["xs_initial"] = xs_initial[int(0.7 * len(xs)): int(0.85 * len(xs))]
    data["ys"] = ys[int(0.7 * len(xs)): int(0.85 * len(xs))]
    data["Rs"] = Rs[int(0.7 * len(xs)): int(0.85 * len(xs))]
    data["ts"] = ts[int(0.7 * len(xs)): int(0.85 * len(xs))]
    data["affine"] = affine[int(0.7 * len(xs)): int(0.85 * len(xs))]
    data["Es"] = Es[int(0.7 * len(xs)): int(0.85 * len(xs))]
    data["index"] = index[int(0.7 * len(xs)): int(0.85 * len(xs))]
    data["score"] = score[int(0.7 * len(xs)): int(0.85 * len(xs))]

    print('Size of validation data', len(data["xs"]))

    valid_data_path = os.path.join(data_folder, database, "valid")
    if not os.path.exists(valid_data_path):
        os.makedirs(valid_data_path)

    for var_name in var_name_list:
        in_file_name = os.path.join(valid_data_path, var_name) + ".pkl"
        with open(in_file_name, "wb") as ofp:
            pickle.dump(data[var_name], ofp)

    data = {}
    data["xs"] = xs[int(0.85 * len(xs)): len(xs)]
    data["xs_initial"] = xs_initial[int(0.85 * len(xs)): len(xs)]
    data["ys"] = ys[int(0.85 * len(xs)): len(xs)]
    data["Rs"] = Rs[int(0.85 * len(xs)): len(xs)]
    data["ts"] = ts[int(0.85 * len(xs)): len(xs)]
    data["affine"] = affine[int(0.85 * len(xs)): len(xs)]
    data["Es"] = Es[int(0.85 * len(xs)): len(xs)]
    data["index"] = index[int(0.85 * len(xs)): len(xs)]
    data["score"] = score[int(0.85 * len(xs)): len(xs)]

    print('Size of testing data', len(data["xs"]))

    test_data_path = os.path.join(data_folder, database, "test")
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    for var_name in var_name_list:
        in_file_name = os.path.join(test_data_path, var_name) + ".pkl"
        with open(in_file_name, "wb") as ofp:
            pickle.dump(data[var_name], ofp)


def load_data(config, data_name, var_mode):
    print("Loading {} data".format(var_mode))
    data_folder = config.data_dump_prefix
    data_folder = "E:/NM-net-xiexie/datasets"
    data_path = os.path.join(data_folder, data_name, var_mode)
    print("111111", data_path)
    # var_name_list = ["img1s", "xs_initial", "ys", "Rs", "ts", "Es", "affine", "index", "score"]
    # var_name_list = ["img1s", "img2s", "xs_4", "label", "xs_12", "others"]
    if config.use_which_network == 1:
        var_name_list = ["xs_4", "label", "index", "others"]
    elif config.use_which_network == 2:
        var_name_list = ["xs_4", "label", "others"]
    else:
        var_name_list = ["merge_data", "xs_4", "label", "xs_12", "others"]
    data = {}

    for var_name in var_name_list:
        in_file_name = os.path.join(data_path, var_name) + ".pkl"
        with open(in_file_name, "rb") as ifp:
            if var_name in data:
                if 0:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] += joblib.load(ifp)
            else:
                if 0:
                    data[var_name] = pickle.load(ifp)
                else:
                    data[var_name] = joblib.load(ifp)
                    print(var_name, len(data[var_name]))
    return data


def load_data_merge(config, data_name, var_mode):
    print("Loading {} data".format(var_mode))
    data_folder = config.data_dump_prefix
    data_folder = "E:/NM-net-xiexie/datasets"
    var_mode = "train"
    data_path = os.path.join(data_folder, data_name, var_mode)
    print("111111", data_path)
    var_name_list = ["img1s", "img2s", "others"]
    data = {}

    for var_name in var_name_list:
        in_file_name = os.path.join(data_path, var_name) + ".pkl"
        with open(in_file_name, "rb") as ifp:
            if var_name in data:
                if 0:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] += joblib.load(ifp)
            else:
                if 0:
                    data[var_name] = pickle.load(ifp)
                else:
                    data[var_name] = joblib.load(ifp)
                    print(var_name, len(data[var_name]))

    print("merge>>>>>>>>", len(data["img1s"]))
    merge_data = {}
    for index in trange(len(data["img1s"])):
        # print(data["others"][index])
        left_image_index = data["others"][index][0]
        right_image_index = data["others"][index][1]
        # print(left_image_index, right_image_index, type(left_image_index))

        if left_image_index in data:
            pass
        else:
            merge_data[left_image_index] = data["img1s"][index]

        if right_image_index in data:
            pass
        else:
            merge_data[right_image_index] = data["img2s"][index]

    joblib.dump(merge_data, "E:/NM-net-xiexie/datasets/" + data_name + "/" + var_mode + "/merge_data.pkl")
    print("merge finish>>>>>>>>")
    return data


class Data_Loader(Dataset):
    def __init__(self, config, database, data_list, var_mode, initialize):
        super(Data_Loader, self).__init__()

        self.config = config
        self.adjacency_num = config.knn_num
        self.var_mode = var_mode
        self.database = database
        self.data_list = data_list
        self.initialize = initialize

        if self.initialize == True:
            data_initialization(config, self.database, self.data_list, score_idx=True)

        # 合并data["img1s"] data["img2s"]以减少空间
        # load_data_merge(self.config, self.database, self.var_mode)

        self.data = load_data(self.config, self.database, self.var_mode)

    def __getitem__(self, item):
        # print(img.shape)
        # print(img)
        if self.config.use_which_network == 0:
            if 0:
                gray_img1s = cv2.cvtColor(self.data["img1s"][item], cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
                gray_img1s = gray_img1s.reshape(1000, 1000, 1)
                gray_img2s = cv2.cvtColor(self.data["img2s"][item], cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
                gray_img2s = gray_img2s.reshape(1000, 1000, 1)
                img1s = torch.from_numpy(gray_img1s).type(torch.IntTensor)
                img2s = torch.from_numpy(gray_img2s).type(torch.IntTensor)

            xs_4 = torch.from_numpy(np.asarray(self.data["xs_4"][item])).type(torch.FloatTensor)
            label = torch.from_numpy(np.asarray(self.data["label"][item])).type(torch.FloatTensor)
            xs_12 = torch.from_numpy(np.asarray(self.data["xs_12"][item])).type(torch.FloatTensor)
            others = self.data["others"][item]

            img1s_z = self.data["merge_data"][others[0]]
            img2s_z = self.data["merge_data"][others[1]]
            gray_img1s = cv2.cvtColor(self.data["merge_data"][others[0]], cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
            # gray_img1s = gray_img1s.reshape(1000, 1000, 1)
            gray_img1s = gray_img1s.reshape(others[3][0], others[3][1], 1)
            gray_img2s = cv2.cvtColor(self.data["merge_data"][others[1]], cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
            # gray_img2s = gray_img2s.reshape(1000, 1000, 1)
            gray_img2s = gray_img2s.reshape(others[4][0], others[4][1], 1)
            img1s = torch.from_numpy(gray_img1s).type(torch.FloatTensor)
            img2s = torch.from_numpy(gray_img2s).type(torch.FloatTensor)
        else:
            pass

        return img1s, img2s, xs_4, label, xs_12


    def __len__(self):
        return len(self.data["xs_4"])


if __name__ == '__main__':
    config, unparsed = get_config()
    """The main function."""
    database = "WIDE"
    database_list = []
    if database == 'COLMAP':
        database_list += ["south"]
        database_list += ["gerrard"]
        database_list += ["graham"]
        database_list += ["person"]
    elif database == 'NARROW':
        database_list += ["lib-narrow"]
        database_list += ["mao-narrow"]
        database_list += ["main-narrow"]
        database_list += ["science-narrow"]
    elif database == 'WIDE':
        database_list += ["lib-wide"]
        database_list += ["mao-wide"]
        database_list += ["main-wide"]
        database_list += ["science-wide"]
    else:
        print("Input error!!")
        exit()

    log_dir = "log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    d = Data_Loader(config, database, database_list, "train", initialize=False)
    data = Data.DataLoader(d, batch_size=5, shuffle=True, num_workers=0, drop_last=True)

    for i, (xs, label, sqe_index, image) in enumerate(data, 0):
        print(555)
        print(xs.shape, label.shape, sqe_index.shape, len(image[0]), len(image), len(image[0][0]), len(image[0][0][0]), len(image[0][0][0][0]))
        # print(image)

        break
