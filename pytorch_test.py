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


def draw_matches_yuanban(img0, img1, kpts0, kpts1, match_idx, label, mask,
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

    display = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display[:rows0, :cols0, :] = resize0
    display[:rows1, cols0:(cols0 + cols1), :] = resize1

    if 1:
        count = 0
        for idx in trange(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            val = match_idx[idx]
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))

            # 只显示正确匹配
            label_temp = label.reshape(-1)
            mask_temp = mask.reshape(-1)
            if int(label_temp[val[0]]) == 1 and int(mask_temp[val[0]]) == 1:
                count += 1
                # print("正确", val[0])
                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)
                cv2.line(display, pt0, pt1, (0, 255, 0), thickness)      # color
                # print(pt0, pt1)
                """
                if count >= 30:
                    break
                """
    display /= 255
    return display


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
                count += 1
                # print("正确", label_temp[val[0]], mask[val[0]])
                cv2.circle(display, pt0, radius, (0, 255, 0), thickness)
                cv2.circle(display, pt1, radius, (0, 255, 0), thickness)
                cv2.line(display, pt0, pt1, (0, 255, 0), thickness)      # color
                # print(pt0, pt1)

                if count >= 80:
                    break

            elif int(label_temp[val[0]]) == 0 and int(mask[val[0]]) == 0:
                count += 1
                if 1:
                    cv2.circle(display, pt0, radius, (0, 255, 0), thickness)
                    cv2.circle(display, pt1, radius, (0, 255, 0), thickness)
                    cv2.line(display, pt0, pt1, (0, 255, 0), thickness)      # color
                else:
                    pass

                if count >= 80:
                    break
            else:
                pass

            name = "E:/NM-net-xiexie/contrat_experiment/plot_diaplay/" + "display3_" + str(count) + ".png"

    cv2.imwrite(name, display)
    display /= 255
    return display




min_kp_num = 500
margin = 0.05
def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):

    # from Utils.transformations import quaternion_from_matrix

    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t

def eval_nondecompose(p1s, p2s, E_hat, dR, dt, scores):

    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    th = np.sort(scores)[::-1][num_top]
    mask = scores >= th

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    # Match types
    E_hat = E_hat.reshape(3, 3).astype(p1s.dtype)

    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E_hat, p1s_good, p2s_good)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            print("Failed in evaluation")
            print(R)
            print(t)
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    # Change mask type
    mask = mask.flatten().astype(bool)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new
    if err_q == 0:
        print(err_t)
    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated

def estimate_E(input, weight):
    E = []
    data = []
    data.append(torch.unsqueeze(input[:, 0, :, 0], -1))  # u1
    data.append(torch.unsqueeze(input[:, 0, :, 1], -1)) # v1
    data.append(torch.unsqueeze(input[:, 0, :, 2], -1))  # u2
    data.append(torch.unsqueeze(input[:, 0, :, 3], -1))
    data.append(torch.unsqueeze(torch.ones(input.size(0),input.size(2)).cuda(), -1))
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
        
def local_consistency(xs_initial, affine):  #16*1*2000*4 16*2000*18
    x = xs_initial[:, 0, :, 0:2]
    y = xs_initial[:, 0, :, 2:4]
    affine = affine.view(-1, 18)    
    affine_x = torch.stack([torch.inverse(affine[i][:9].view(3, 3)) for i in range (affine.shape[0])])
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
        
        prj_dis = torch.stack([torch.sum(torch.pow((y_prj[idx] - y_prj[idx, :, idx].unsqueeze(-1)), 2), dim=0) for idx in range(x[i].size(0))])
        prj_dis = prj_dis + prj_dis.t()
        index.append(torch.sort(prj_dis, dim=1, descending=False)[1][:, :8]) 
    return torch.stack(index)
    
def local_score(xs_initial, affine):  #16*1*2000*4 16*2000*18
    x = xs_initial[:, 0, :, 0:2]
    y = xs_initial[:, 0, :, 2:4]
    affine = affine.view(-1, 18)    
    affine_x = torch.stack([torch.inverse(affine[i][:9].view(3, 3)) for i in range (affine.shape[0])])
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
        
        prj_dis = torch.stack([torch.sum(torch.pow((y_prj[idx] - y_prj[idx, :, idx].unsqueeze(-1)), 2), dim=0) for idx in range(x[i].size(0))])
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


def test_process(mode, save_file_cur, model_name, data_name, config, f1, f2, epoch, datasets, adjacency_num=8, Image_Preprocess_XiaoRong=False, display=False):
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

    for i, (img1s_z_g, img2s_z_g, xs_4, label, xs_12, index, others, xs_6) in enumerate(data, 0):
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

            """
            img1s_z_g = cv2.cvtColor(img1s_z.numpy().squeeze(0), cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
            img2s_z_g = cv2.cvtColor(img2s_z.numpy().squeeze(0), cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B

            if datasets == "gl3d":
                img1s_z_g = img1s_z_g.reshape(1, 1000, 1000, 1)
                img2s_z_g = img2s_z_g.reshape(1, 1000, 1000, 1)
            elif datasets == "Hpatch":
                img1s_z_g = img1s_z_g.reshape(1, others[3][0], others[3][1], 1)
                img2s_z_g = img2s_z_g.reshape(1, others[4][0], others[4][1], 1)
            else:
                pass
            """

            if 1:
                """
                    现在不确定这代码的作用到底啥 反正跑步通
                """
                """
                # 第一张图片
                img1s_temp = cv2.cvtColor(img1s_z_g.numpy().astype(np.float32).squeeze(0), cv2.COLOR_BGR2GRAY)

                img1s_temp = img1s_temp.reshape(1, 1000, 1000, 1)
                # img1s_temp = img1s_temp.reshape(1, others[3][0], others[3][1], 1)

                img1s_temp = torch.from_numpy(img1s_temp).type(torch.IntTensor)
                img1s_temp = th.tensor(img1s_temp, dtype=th.float32, requires_grad=False)

                # img1s_temp = th.tensor(img1s_z_g, dtype=th.float32, requires_grad=False)
                xs1s_temp = th.tensor(xs1s, dtype=th.float32, requires_grad=False)

                # 第二张图片
                img2s_temp = cv2.cvtColor(img2s_z_g.numpy().astype(np.float32).squeeze(0), cv2.COLOR_BGR2GRAY)

                img2s_temp = img2s_temp.reshape(1, 1000, 1000, 1)
                # img2s_temp = img2s_temp.reshape(1, others[4][0], others[4][1], 1)

                img2s_temp = torch.from_numpy(img2s_temp).type(torch.IntTensor)
                img2s_temp = th.tensor(img2s_temp, dtype=th.float32, requires_grad=False)

                # img2s_temp = th.tensor(img2s_z_g, dtype=th.float32, requires_grad=False)
                xs2s_temp = th.tensor(xs2s, dtype=th.float32, requires_grad=False)

                # print(1111111111, xs1s_temp.shape)
                img1s = f1(img1s_temp, xs1s_temp)
                img2s = f2(img2s_temp, xs2s_temp)

                img1s = torch.Tensor(img1s)
                img1s = img1s.permute(0, 3, 1, 2)
                img2s = torch.Tensor(img2s)
                img2s = img2s.permute(0, 3, 1, 2)
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
                img1s = img1s_z_g.squeeze(0).numpy()
                img2s = img2s_z_g.squeeze(0).numpy()
                xs1s = xs1s.squeeze(0).numpy()
                xs2s = xs2s.squeeze(0).numpy()

                import patch_extractor
                patch_extractor = patch_extractor.PatchExtractor()
                img1s = patch_extractor.get_patches(img1s, xs1s)
                img2s = patch_extractor.get_patches(img2s, xs2s)

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

        if display:
            wanzheng_img0 = img1s_z_g.numpy().squeeze(0)
            wanzheng_img1 = img2s_z_g.numpy().squeeze(0)

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
            mask = mask.numpy().squeeze(0).astype(np.int16)

            display3 = draw_matches_yuanban(wanzheng_img0, wanzheng_img1, kpts0, kpts1, match_idx, label, mask,
                                   downscale_ratio=1.0)  # 1.0

            plt.xticks([])
            plt.yticks([])
            plt.imshow(display3)
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















































































def test_process_gai_hehehe(mode, save_file_cur, model_name, data_name, config, adjacency_num=8):
    d = Data_Loader(config, data_name, None, mode, initialize=False)

    data = Data.DataLoader(d, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    Network = torch.load(save_file_cur + model_name).cpu()
    Network.eval()

    loss_func = loss.Loss_classi()
    loss_list_hehe = []

    P = []
    R = []
    F = []
    MSE = []
    MAE = []

    # for i, (xs, Es, index, label) in enumerate(data, 0):
    # for i, (xs, label) in enumerate(data, 0):
    print(6666666)
    for i, (img1s, img2s, xs_4, label, xs_12) in enumerate(data, 0):
        if i == 5:
            break
        xs1s = xs_12[:, :, 0: 6]
        xs2s = xs_12[:, :, 6: 12]

        # 第一张图片
        img1s_temp = th.tensor(img1s, dtype=th.float32, requires_grad=False)
        xs1s_temp = th.tensor(xs1s, dtype=th.float32, requires_grad=False)
        # print("47535793", xs1s_temp)
        result1 = preprocess.get_32_32_image(img1s_temp, xs1s_temp)
        # 第二张图片
        img2s_temp = th.tensor(img2s, dtype=th.float32, requires_grad=False)
        xs2s_temp = th.tensor(xs2s, dtype=th.float32, requires_grad=False)
        # print("47535793", xs1s_temp)
        result2 = preprocess.get_32_32_image(img2s_temp, xs2s_temp)

        # print(result)
        with tf.compat.v1.Session() as sess:
            sess.run([result1, result2])
            # print(result[0].eval())
            img1s = result1[0].eval()
            img2s = result2[0].eval()

        img1s = torch.Tensor(img1s)
        img1s = img1s.permute(0, 3, 1, 2)
        img2s = torch.Tensor(img2s)
        img2s = img2s.permute(0, 3, 1, 2)

        """
        img1s = img1s
        img2s = img2s
        xs_4 = xs_4
        label = label
        """
        with torch.no_grad():
            output, weight = Network(img1s, img2s, xs_4)
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

    p_ = np.expand_dims(np.mean(np.array(P)), axis=0)
    r_ = np.expand_dims(np.mean(np.array(R)), axis=0)
    f_ = np.expand_dims(np.mean(np.array(F)), axis=0)

    print(p_, r_, f_)

    fh = open('777.txt', 'a', encoding='utf-8')
    str1 = "P:" + str(p_) + "R:" + str(r_) + "F:" + str(f_) + "\n"
    fh.write(str1)
    fh.close()
    print(77777777777777)

    log_path = os.path.join(save_file_cur, mode)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    np.savetxt(os.path.join(log_path, "Precision.txt"), p_ * 100)
    np.savetxt(os.path.join(log_path, "Recall.txt"), r_ * 100)
    np.savetxt(os.path.join(log_path, "F-measure.txt"), f_ * 100)

    if mode == 'test':
        mse = np.expand_dims(np.mean(np.array(MSE)), axis=0)
        mae = np.expand_dims(np.mean(np.array(MAE)), axis=0)
        median = np.expand_dims(np.median(np.array(MAE)), axis=0)
        Max = np.expand_dims(np.max(np.array(MAE)), axis=0)
        Min = np.expand_dims(np.min(np.array(MAE)), axis=0)
        np.savetxt(os.path.join(log_path, "MSE.txt"), mse)
        np.savetxt(os.path.join(log_path, "MAE.txt"), mae)
        np.savetxt(os.path.join(log_path, "Median.txt"), median)
        np.savetxt(os.path.join(log_path, "Max.txt"), Max)
        np.savetxt(os.path.join(log_path, "Min.txt"), Min)

    loss_list = torch.stack(loss_list_hehe).view(-1)
    return f_, loss_list.mean()






