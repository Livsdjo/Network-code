#!/usr/bin/env python3
from config import get_config, print_usage
from tqdm import trange
import numpy as np
import torch
import torch as th
import preprocess
from numba import cuda
from builtins import str
from torch.utils.tensorboard import SummaryWriter


import visdom
import loss
from torch import optim
import os
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as utils
from dataset import Data_Loader
import cv2
import matplotlib.pyplot as plt

# 之前程序跑的test process
# from pytorch_test import test_process
# 这个是用来显示图像的test process
from Attention_Yanzheng_pytorch_test import test_process

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

config = None

viz = visdom.Visdom()
np.set_printoptions(suppress=True)

"""
     1.attention权重值打印
     2.消融哪些数据敏感  共性
"""
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


def draw_matches(img0, img1, kpts0, kpts1, match_idx, im1xy, im2xy, label,
                 downscale_ratio=1, color=(0, 255, 0), radius=4, thickness=2):
    
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

    display1 = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display1[:rows0, :cols0, :] = resize0
    display1[:rows1, cols0:(cols0 + cols1), :] = resize1

    display2 = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display2[:rows0, :cols0, :] = resize0
    display2[:rows1, cols0:(cols0 + cols1), :] = resize1

    display3 = np.zeros((max(rows0, rows1), cols0 + cols1, 3))
    display3[:rows0, :cols0, :] = resize0
    display3[:rows1, cols0:(cols0 + cols1), :] = resize1

    if 1:
        for idx in trange(match_idx.shape[0]):
            # 顺序是对的 对于图像来说就是先长后高  刚好对应于0于1
            val = match_idx[idx]
            pt0 = (int(kpts0[val[0]][0]), int(kpts0[val[0]][1]))
            pt1 = (int(kpts1[val[1]][0]) + cols0, int(kpts1[val[1]][1]))

            # 只显示正确匹配
            label_temp = label.reshape(-1)
            if int(label_temp[val[0]]) == 1:
                print("正确", val[0])
                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)
                cv2.line(display, pt0, pt1, color, thickness)

                cv2.circle(display1, pt0, radius, color, thickness)
                cv2.circle(display1, pt1, radius, color, thickness)
                cv2.line(display1, pt0, pt1, color, thickness)

                cv2.circle(display2, pt0, radius, color, thickness)
                cv2.circle(display2, pt1, radius, color, thickness)
                cv2.line(display2, pt0, pt1, color, thickness)

                cv2.circle(display3, pt0, radius, color, thickness)
                cv2.circle(display3, pt1, radius, color, thickness)
                cv2.line(display3, pt0, pt1, color, thickness)

                # print(99999999, im1xy[idx].shape)
                img1_1xy_temp = im1xy[0][idx].reshape(-1, 2)
                img1_2xy_temp = im1xy[1][idx].reshape(-1, 2)
                img1_3xy_temp = im1xy[2][idx].reshape(-1, 2)
                img1_4xy_temp = im1xy[3][idx].reshape(-1, 2)

                img2_1xy_temp = im2xy[0][idx].reshape(-1, 2)
                img2_2xy_temp = im2xy[1][idx].reshape(-1, 2)
                img2_3xy_temp = im2xy[2][idx].reshape(-1, 2)
                img2_4xy_temp = im2xy[3][idx].reshape(-1, 2)
                # print(7777777, img1_2xy_temp.shape)
                radius = 2
                color = (255, 0, 0)
                thickness = 2
                # 左边
                for value in img1_1xy_temp:
                    cv2.circle(display, tuple(value), radius, color, thickness)
                for value in img1_2xy_temp:
                    cv2.circle(display1, tuple(value), radius, color, thickness)
                for value in img1_3xy_temp:
                    cv2.circle(display2, tuple(value), radius, color, thickness)
                for value in img1_4xy_temp:
                    cv2.circle(display3, tuple(value), radius, color, thickness)

                # 右边
                for value in img2_1xy_temp:
                    cv2.circle(display, tuple([value[0] + cols0, value[1]]), radius, color, thickness)
                for value in img2_2xy_temp:
                    cv2.circle(display1, tuple([value[0] + cols0, value[1]]), radius, color, thickness)
                for value in img2_3xy_temp:
                    cv2.circle(display2, tuple([value[0] + cols0, value[1]]), radius, color, thickness)
                for value in img2_4xy_temp:
                    cv2.circle(display3, tuple([value[0] + cols0, value[1]]), radius, color, thickness)
                break

    display /= 255
    display1 /= 255
    display2 /= 255
    display3 /= 255

    return display, display1, display2, display3


# 显示匹配的draw_matches
"""
def draw_matches(img0, img1, kpts0, kpts1, match_idx, label,
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
            if int(label_temp[val[0]]) == 1:
                count += 1
                print("正确", val[0])
                cv2.circle(display, pt0, radius, color, thickness)
                cv2.circle(display, pt1, radius, color, thickness)
                cv2.line(display, pt0, pt1, (0, 255, 0), thickness)      # color
                print(pt0, pt1)
                
                if count >= 30:
                    break
                
    display /= 255
    return display
"""


def main(config, datasets, write=False, Image_Preprocess_XiaoRong=False, Use_Second_Network=False):
    """The main function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    database = 'COLMAP'

    # database = 'Hpatch_Data'
    # database = 'COLMAP_hehe_j' # 'COLMAP_hehe'
    # database = "Hpatch_Data_new"
    database_list = []

    log_dir = "log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if config.restore:
        # Restore network
        # log_dir = "E:/NM-net-xiexie/datasets/COLMAP/contrast_experiment/局部训练效果不好"
        # log_dir = "E:/NM-net-xiexie/datasets/COLMAP/contrast_experiment/局部训练效果好_epcho=4"
        # log_dir = "E:/NM-net-xiexie/datasets/COLMAP/contrast_experiment/消融实验关于图像有多大的作用"
        if 1:
            # log_dir = "E:/NM-net-xiexie/hehba/log"

            # log_dir = "E:/NM-net-xiexie/hehba最终定型/log"

            log_dir = "E:/NM-net-xiexie/hehba最终定型 gl3d/结果3"
            # log_dir = "E:/NM-net-xiexie/hehba最终定型 gl3d/log"
            # log_dir = "E:/NM-net-xiexie/hehba最终定型 gl3d/最终版结果"

            # log_dir = "log/"
            # log_dir = "E:/NM-net-xiexie/datasets/Hpatch_Data"
        else:
            log_dir = "E:/NM-net-xiexie/datasets/COLMAP/参数化结果"
        print("Restoring from {}...".format(
            log_dir + '/Net_6_best_state2.pth'))         # NM-Net_6_state.pth
        # f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, session = preprocess.get_32_32_image()
        f1, f2, f3, f4, session = preprocess.get_32_32_image()
        """
        # 这个结果是之前不加attention 做对比的  不过网络架构搭的跟文章上说的不一样
        va_res, val_loss = test_process("valid", log_dir, '/NM-Net_888_state_yanzheng.pth', database, config, f1, f2,
                                       100000, datasets, config.knn_num)
        """
        """
        va_res, val_loss = test_process("test", log_dir, '/NM-Net_888_best_state6_yanzheng_hao.pth', database, config, f1, f2,
                                       100000, datasets, config.knn_num)
        """

        va_res, val_loss = test_process("test", log_dir, '/NM-Net_888_best_state1.pth', database, config, f1, f2,
                                       100000, datasets, config.knn_num)


        """
        va_res, val_loss = test_process("valid", log_dir, '/NM-Net_888_state.pth', database, config, f1, f2, # test   '/NM-Net_888_best_state1.pth'
                                        0, viz, config.knn_num)
        """
        print("测试结果：")
        print(va_res, val_loss)
        return
    else:
        # Initialize network
        print("Initializing...")
        if config.use_which_network == 0:
            """
                  用来做对比的
            """
            # import MultiModal_NetWork
            # Network = MultiModal_NetWork.Net().cuda()

            # import MultiModal_Para_Network
            # Network = MultiModal_Para_Network.NM_Net().cuda()

            # import MultiModal_netWork_New
            # Network = MultiModal_netWork_New.Net().cuda()

            """
                这个是之前用的
            """
            # 这个是将两张图片压缩在一起输入
            # import MultiModal_Network_new3_Test
            # Network = MultiModal_Network_new3_Test.Net().cuda()

            """
                用来做对比的网络 证明Attention这个模块的作用
            """
            import MultiModal_Network_new4_Test
            Network = MultiModal_Network_new4_Test.Net().cuda()

            # 最终定型网络
            # 这个是按孪生网络的思路走的
            # import MultiModal_network_new2
            # Network = MultiModal_network_new2.Net().cuda()
        elif config.use_which_network == 1:
            import nm_net_network
            Network = nm_net_network.NM_Net().cuda()
        elif config.use_which_network == 2:
            # 新版的加入了邻域全局特征的Cne网络
            # import cne_network
            # Network = cne_network.Cne_Net().cuda()
            # 原版的Cne网络
            import cne_network2
            Network = cne_network2.Cne_Net().cuda()
        elif config.use_which_network == 3:
            import oan
            Network = oan.OANet().cuda()
        elif config.use_which_network == 4:
            import acne_network
            f1, f2, session = acne_network.get_network()
        else:
            print("Input Error .......")

        # print("网络结构", Network)
        if 1:
            Network.apply(weights_init)
        else:
            print("Restoring ...... 接着训练")
            save_file_cur = "E:/NM-net-xiexie/hehba最终定型 gl3d/log"
            # model_name = '/NM-Net_6_best_state1.pth'
            # model_name = "/NM-Net_888_best_state1.pth"
            model_name = "/NM-Net_888_best_state6_yanzheng.pth"
            Network = torch.load(save_file_cur + model_name).cuda()
            # print(Network)
        Network.train()

        black = np.array([[0, 0, 0]])
        viz.line([0], [-1], win='train_loss', opts=dict(title='train_loss', linecolor=black))
        viz.line([0], [-1], win='val_loss', opts=dict(title='val_loss', linecolor=black))

        d = Data_Loader(config, database, database_list, "train", initialize=False)         # train
        data = Data.DataLoader(d, batch_size=config.train_batch_size, shuffle=True, num_workers=0, drop_last=True)
        loss_func = loss.Loss_classi().cuda()
        # loss_fn = loss_that_combine_focalloss_and_hardexample.focal_loss(alpha=0.5, num_classes=2)  # 0.35

        print("训练集个数:", len(data))

        var_list = []
        best_va_res = 0
        optimizer = optim.Adam(Network.parameters(), lr=config.train_lr)

        print("Starting from scratch...")
        print("Starting Session>>>>>>")
        f1, f2, f3, f4, session = preprocess.get_32_32_image()
        # f1 = None
        # f2 = None
        # f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, session = preprocess2.get_32_32_image()

    # ----------------------------------------
    # The training loop
    from tqdm import tqdm
    for epoch in range(config.epochs):
        loss_list = []
        for i, (img1s_z, img2s_z, img1s_z_g, img2s_z_g, xs_4, label, xs_12, index, others, xs_6) in enumerate(tqdm(data, 0)):
            if config.use_which_network == 0:
                # print(888888888888, xs_12.shape)
                if datasets == "gl3d":
                    # 原来的  gl3d
                    xs1s = xs_12[:, :, 0: 6]
                    xs2s = xs_12[:, :, 6: 12]
                elif datasets == "Hpatch":
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
                    # print("图像预处理消融实验没有开启")

                """
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
                display = draw_matches(wanzheng_img0, wanzheng_img1, kpts0, kpts1, match_idx, label, downscale_ratio=1.0)  # 1.0

                plt.xticks([])
                plt.yticks([])
                plt.imshow(display)
                plt.show()
                continue
                """

                """
                img1s_z_g = cv2.cvtColor(img1s_g.numpy().squeeze(0), cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
                img2s_z_g = cv2.cvtColor(img2s_g.numpy().squeeze(0), cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B
                
                if datasets == "gl3d":
                    img1s_z_g = img1s_z_g.reshape(1, 1000, 1000, 1)
                    img2s_z_g = img2s_z_g.reshape(1, 1000, 1000, 1)
                elif datasets == "Hpatch":
                    img1s_z_g = img1s_z_g.reshape(1, others[3][0], others[3][1], 1)
                    img2s_z_g = img2s_z_g.reshape(1, others[4][0], others[4][1], 1)
                else:
                    pass

                """

                if 0:
                    # 第一张图片
                    img1s_temp = th.tensor(img1s_z_g, dtype=th.float32, requires_grad=False)
                    xs1s_temp = th.tensor(xs1s, dtype=th.float32, requires_grad=False)
                    # 第二张图片
                    img2s_temp = th.tensor(img2s_z_g, dtype=th.float32, requires_grad=False)
                    xs2s_temp = th.tensor(xs2s, dtype=th.float32, requires_grad=False)

                    img1s = f1(img1s_temp, xs1s_temp)
                    img2s = f2(img2s_temp, xs2s_temp)
                    patch1 = f3(img1s_temp, xs1s_temp)
                    patch2 = f4(img2s_temp, xs2s_temp)

                    img1s = torch.Tensor(img1s)
                    img1s = img1s.permute(0, 3, 1, 2)
                    img2s = torch.Tensor(img2s)
                    img2s = img2s.permute(0, 3, 1, 2)

                    # print(123456, img1s.shape, patch1.shape, patch1[0].shape)
                else:
                    # 灰度图像
                    # img1s = img1s_z_g.squeeze(0).numpy()
                    # img2s = img2s_z_g.squeeze(0).numpy()
                    # 彩色图像
                    img1s = img1s_z.squeeze(0).numpy()
                    img2s = img2s_z.squeeze(0).numpy()
                    xs1s = xs1s.squeeze(0).numpy()
                    xs2s = xs2s.squeeze(0).numpy()

                    import patch_extractor
                    patch_extractor = patch_extractor.PatchExtractor()
                    # import patch_extractor_new
                    # patch_extractor = patch_extractor_new.PatchExtractor()
                    # print("输入:", img1s.shape, img2s.shape)
                    img1s = patch_extractor.get_patches(img1s, xs1s)
                    # print("结果1:", img1s.shape, type(img1s))
                    img2s = patch_extractor.get_patches(img2s, xs2s)
                    # print("结果2:", img2s.shape)
                    # print("完成")
                    """
                    for i in range(2000):
                        plt.xticks([])
                        plt.yticks([])
                        plt.imshow(img1s[i].astype(np.uint8))
                        plt.show()
                    """
                    # print("img1s", img1s[0].shape)
                    img1s = torch.tensor(img1s).unsqueeze(1)
                    img2s = torch.tensor(img2s).unsqueeze(1)
                    # print(5555555, img1s.shape, img2s.shape)


                """
                img0_x_0 = f3(img1s_temp, xs1s_temp)
                img0_x_1 = f4(img1s_temp, xs1s_temp)
                img0_y_0 = f5(img1s_temp, xs1s_temp)
                img0_y_1 = f6(img1s_temp, xs1s_temp)

                img1_x_0 = f7(img2s_temp, xs2s_temp)
                img1_x_1 = f8(img2s_temp, xs2s_temp)
                img1_y_0 = f9(img2s_temp, xs2s_temp)
                img1_y_1 = f10(img2s_temp, xs2s_temp)

                img0_1xy = torch.stack((img0_x_0, img0_y_0), dim=3).squeeze(4)
                img0_2xy = torch.stack((img0_x_0, img0_y_1), dim=3).squeeze(4)
                img0_3xy = torch.stack((img0_x_1, img0_y_0), dim=3).squeeze(4)
                img0_4xy = torch.stack((img0_x_1, img0_y_1), dim=3).squeeze(4)

                img1_1xy = torch.stack((img1_x_0, img1_y_0), dim=3).squeeze(4)
                img1_2xy = torch.stack((img1_x_0, img1_y_1), dim=3).squeeze(4)
                img1_3xy = torch.stack((img1_x_1, img1_y_0), dim=3).squeeze(4)
                img1_4xy = torch.stack((img1_x_1, img1_y_1), dim=3).squeeze(4)
                """

                img1s = img1s.cuda()
                img2s = img2s.cuda()
                xs_4 = xs_4.cuda()
                label = label.cuda()

                # 是否显示匹配和图像
                if 0:
                    # print()
                    wanzheng_img0 = img1s_z_g.numpy().squeeze(0)
                    wanzheng_img1 = img2s_z_g.numpy().squeeze(0)
                    print("hehheheheheh", xs_4.shape)
                    print(xs_4.squeeze(0).squeeze(0))
                    print(xs_4.squeeze(0).squeeze(1)[:, 0])
                    kpts0 = np.stack([xs_4.cpu().numpy().squeeze(0).squeeze(0)[:, 0], xs_4.cpu().numpy().squeeze(0).squeeze(0)[:, 1]], axis=-1)
                    kpts1 = np.stack([xs_4.cpu().numpy().squeeze(0).squeeze(0)[:, 2], xs_4.cpu().numpy().squeeze(0).squeeze(0)[:, 3]], axis=-1)

                    img_size0 = np.array((wanzheng_img0.shape[1], wanzheng_img0.shape[0]))
                    img_size1 = np.array((wanzheng_img1.shape[1], wanzheng_img1.shape[0]))
                    kpts0 = kpts0 * img_size0 / 2 + img_size0 / 2
                    kpts1 = kpts1 * img_size1 / 2 + img_size1 / 2
                    match_num = kpts0.shape[0]
                    print(1111111, match_num)
                    match_idx = np.tile(np.array(range(match_num))[..., None], [1, 2]) # match_num   range(2)
                    display, display1, display2, display3 = draw_matches(wanzheng_img0, wanzheng_img1, kpts0, kpts1, match_idx, [img0_1xy, img0_2xy, img0_3xy, img0_4xy], [img1_1xy, img1_2xy, img1_3xy, img1_4xy], label, downscale_ratio=1.0)  # 1.0
                    # print(2222222, img0_1xy.shape, img0_2xy.shape, img0_3xy.shape, img0_4xy.shape)
                    # print(img0_1xy[0, :, :, 0])
                    # print(img0_1xy[0, :, :, 1])

                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(display)
                    plt.show()

                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(display1)
                    plt.show()

                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(display2)
                    plt.show()

                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(display3)
                    plt.show()
                    # break
                if 0:
                    # print(111111111111, img1s.shape)
                    # print("lllllllllllll", patch1.shape, patch2.shape)
                    patches0 = img1s.cpu().numpy().squeeze(1).reshape(2000, 32, 32, 3)
                    patches1 = img2s.cpu().numpy().squeeze(1).reshape(2000, 32, 32, 3)
                    match_n = 2000     # 100
                    print("结果2：", patches0.shape, patches1.shape)

                    display = []
                    tmp_row = []
                    count = 0
                    label_temp = label.reshape(-1)
                    for i in range(match_n):
                        """
                        if int(label_temp[i]) == 0:
                            continue
                        print("正确2", i)
                        """
                        tmp_pair = np.concatenate((patches0[i], patches1[i]), axis=1)
                        tmp_row.append(tmp_pair)
                        # count += 1
                        if 0:
                            # if (i + 1) % 10 == 0 and i != 0:  # !=
                            if count == 5:
                                tmp_row = np.concatenate(tmp_row, axis=-1)
                                display.append(tmp_row)
                                tmp_row = []
                                break
                        else:
                            # print(i, count)
                            if (count + 1) % 10 == 0 and count != 0:  # !=   10
                                # print(tmp_row)
                                tmp_row = np.concatenate(tmp_row, axis=-1)
                                display.append(tmp_row)
                                tmp_row = []
                            count += 1

                        if i == 100:
                            break

                    # print(display)
                    display = np.concatenate(display, axis=0)
                    print(555556, display.shape)
                    # print(77777778, display)

                    plt.xticks([])
                    plt.yticks([])
                    plt.imshow(display)
                    plt.show()

                output, w = Network(img1s, img2s, xs_4, epoch)
                # print("Out", output)
            else:
                pass

            # return

            optimizer.zero_grad()

            # print("111111111", output.shape, label.shape)
            l = loss_func(output, label)
            l.backward()
            optimizer.step()
            loss_list += [l]
            # print("第几个:" + str(i) + "损失值:" + str(l))

            if write:
                fh = open('heheba_test原本网络的修改.txt', 'a', encoding='utf-8')
                str1 = "第几个:" + str(i) + "损失值:" + str(l) + "\n"
                fh.write(str1)
                fh.close()

            # viz.line([l.item()], [epoch * len(data) + i], win='train_loss', update='append')

        loss_list = torch.stack(loss_list).view(-1)
        print('Epoch: {} / {} ---- Trainning Loss : {}'.format(epoch, config.epochs, loss_list.mean()))
        viz.line([loss_list.mean().item()], [epoch], win='train_loss', update='append')

        if write:
            fh = open('heheba_test图像预处理消融.txt', 'a', encoding='utf-8')
            str1 = 'Epoch: {} / {} ---- Trainning Loss : {}'.format(epoch, config.epochs, loss_list.mean()) + "\n"
            fh.write(str1)
            fh.close()

        # Write summary and save current model
        # ----------------------------------------
        path_temp = '/NM-Net_888_state19_yanzheng_temp' + str(epoch) + '.pth'       # 9
        torch.save(Network, log_dir + path_temp)
        # torch.save(Network, log_dir + '/NM-Net_888_state9_yanzheng_temp.pth')
        # Validation
        va_res, val_loss = test_process("test", log_dir, path_temp, database, config, f1, f2,
                                       epoch, datasets, config.knn_num)
        """
        va_res, val_loss = test_process("valid", log_dir, '/NM-Net_888_state9_yanzheng_temp.pth', database, config, f1, f2,
                                       epoch, datasets, config.knn_num)
        """
        viz.line([val_loss.item()], [epoch], win='val_loss', update='append')
        print('Validation F-measure : {}'.format(va_res))
        if write:
            fh = open('heheba_test图像预处理消融.txt', 'a', encoding='utf-8')
            str1 = 'Validation F-measure : {}'.format(va_res) + "\n"
            fh.write(str1)
            fh.close()

        var_list.append(va_res)
        # np.savetxt(log_dir + './validation_list.txt', np.array(var_list))

        # Higher the better
        if va_res > best_va_res:
            print("Saving best model with va_res = {}".format(va_res))

            if write:
                fh = open('heheba_test图像预处理消融.txt', 'a', encoding='utf-8')
                str1 = "Saving best model with va_res = {}".format(va_res) + "\n"
                fh.write(str1)
                fh.close()

            best_va_res = va_res
            # Save best validation result
            np.savetxt(log_dir + "/best_results.txt", best_va_res)
            # Save best model
            if epoch <= 500:      # 15
                torch.save(Network, log_dir + '/NM-Net_888_best_state190_yanzheng_hao.pth')         # 1
            else:
                torch.save(Network, log_dir + '/NM-Net_888_best_state29_yanzheng_hao.pth')         # 2

    te_res, te_loss = test_process("test", log_dir, '/NM-Net_888_best_state10_yanzheng_hao.pth', database, config, f1, f2,       # 1
                                   epoch, datasets, config.knn_num)     # viz
    # session.close()
    print('Testing F-measure : {}'.format(te_res))

    np.savetxt(log_dir + "/test_results.txt", te_res)


if __name__ == "__main__":
    # ----------------------------------------
    config, unparsed = get_config()

    # datasets = "Hpatch"
    datasets = "gl3d"
    main(config, datasets)

































