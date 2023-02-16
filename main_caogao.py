#!/usr/bin/env python3
from config import get_config, print_usage
from pytorch_test import test_process
from tqdm import trange
import numpy as np
import torch
import torch as th
import preprocess
import MultiModal_NetWork
import MultiModal_Para_Network
from numba import cuda
from builtins import str
from torch.utils.tensorboard import SummaryWriter

# from CNN_Network import NM_Net
# from model import NM_Net       # model3     LGC架构实现
# from model6 import NM_Net       # model3
# from model import NM_Net
# from model12 import NM_Net     # NMnet架构实现

import tensorflow as tf

import visdom
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


config = None

viz = visdom.Visdom()
np.set_printoptions(suppress=True)

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


def skew_symmetric(v):
    zero = np.zeros((len(v), 1))

    M = np.hstack((zero, -v[:, 2, :], v[:, 1, :],
                   v[:, 2, :], zero, -v[:, 0, :],
                   -v[:, 1, :], v[:, 0, :], zero))
    return M

def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 500 epochs"""
    lr = max((opt.lr * (opt.decay_rate ** (epoch // opt.decay_step))), 1e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def main(config):
    """The main function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # database = sys.argv[-1]
    # database = 'COLMAP'
    # database = 'NARROW'
    # database = ""
    database = 'COLMAP'
    # database = "WIDE"
    database_list = []
    if database == 'COLMAP':
        # database_list += ["south"]         # 2360       1076 * 807
        database_list += ["gerrard"]       # 1800
        # database_list += ["graham"]        # 11000
        # database_list += ["person"]        # 6400
    elif database == 'NARROW':
        # database_list += ["lib-narrow"]
        # database_list += ["mao-narrow"]
        database_list += ["main-narrow"]        # 1900
        # database_list += ["science-narrow"]
    elif database == 'WIDE':
        # database_list += ["lib-wide"]     # 1920 1080
        # database_list += ["mao-wide"]     # 1280 720
        database_list += ["main-wide"]    # 1280 720   840
        # database_list += ["science-wide"] # 1920 1080
    else:
        print("Input error!!")
        exit()

    log_dir = "log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("Initializing Summary...")
    # 实例化Summary对象
    # tb_writer = SummaryWriter(log_dir="tb/result")
    # Initialize network
    print("Initializing...")
    Network = MultiModal_NetWork.Net().cuda()
    # Network = MultiModal_Para_Network.NM_Net().cuda()
    print(0000000, Network)
    # return
    Network.apply(weights_init)
    Network.train()

    """
    input_xs_4 = torch.zeros((1, 1, 2000, 4)).cuda()
    input_img1s = torch.zeros((2000, 1, 32, 32)).cuda()
    input_img2s = torch.zeros((2000, 1, 32, 32)).cuda()
    tb_writer.add_graph(Network, (input_xs_4, input_img1s, input_img2s))
    """

    viz.line([0], [-1], win='train_loss', opts=dict(title='train_loss'))
    viz.line([0], [-1], win='val_loss', opts=dict(title='val_loss'))

    # log_dir = "E:/"
    """
       input_xs_4 = torch.zeros((1, 1, 2000, 4))
       input_img1s = torch.zeros((2000, 1, 32, 32))
       input_img2s = torch.zeros((2000, 1, 32, 32))
       tb_writer.add_graph(Network, input)
    """
    """
    f1, f2, session = preprocess.get_32_32_image()
    # te_res = test_process("test", log_dir, '/NM-Net_2_state.pth', database, config, f1, f2, config.knn_num)
    va_res, val_loss = test_process("test", log_dir, '/NM-Net_2_best_state.pth', database, config, f1, f2, config.knn_num)
    return
    """

    d = Data_Loader(config, database, database_list, "train", initialize=False)
    # print(d)
    data = Data.DataLoader(d, batch_size=config.train_batch_size, shuffle=True, num_workers=0, drop_last=True)
    loss_func = loss.Loss_classi().cuda()
    print("训练集个数:", len(data))
    # return

    loss_his = []
    var_list = []

    optimizer = optim.Adam(Network.parameters(), lr=config.train_lr)
    scheduler = CosineAnnealingLR(optimizer, config.epochs, eta_min=config.train_lr*0.01)
    
    step = 0
    best_va_res = 0

    print("Starting Session>>>>>>")
    f1, f2, session = preprocess.get_32_32_image()

    # ----------------------------------------
    # The training loop       
    for epoch in range(config.epochs):   
        loss_list = []
        for i, (img1s, img2s, xs_4, label, xs_12) in enumerate(data, 0):
            xs1s = xs_12[:, :, 0: 6]
            xs2s = xs_12[:, :, 6: 12]

            # 第一张图片
            img1s_temp = th.tensor(img1s, dtype=th.float32, requires_grad=False)
            xs1s_temp = th.tensor(xs1s, dtype=th.float32, requires_grad=False)
            # print("47535793", xs1s_temp)
            # result1 = preprocess.get_32_32_image(img1s_temp, xs1s_temp)
            # 第二张图片
            img2s_temp = th.tensor(img2s, dtype=th.float32, requires_grad=False)
            xs2s_temp = th.tensor(xs2s, dtype=th.float32, requires_grad=False)
            # print("47535793", xs1s_temp)
            if 0:
                result = preprocess.get_32_32_image(img1s_temp, xs1s_temp, img2s_temp, xs2s_temp)
                print(88888)
                # print(result)
                with tf.compat.v1.Session() as sess:
                    print("eeeeeeeeee")
                    jieguo = sess.run(result)
                    # print(result[0].eval())
                    print("fffffffffff")
                    # img1s = result[0].eval()
                    # img2s = result[3].eval()
                    img1s = jieguo[0]
                    img2s = jieguo[3]
            else:
                img1s = f1(img1s_temp, xs1s_temp)
                img2s = f2(img2s_temp, xs2s_temp)
                # session.close()

            img1s = torch.Tensor(img1s)
            img1s = img1s.permute(0, 3, 1, 2)
            img2s = torch.Tensor(img2s)
            img2s = img2s.permute(0, 3, 1, 2)

            img1s = img1s.cuda()
            img2s = img2s.cuda()
            xs_4 = xs_4.cuda()
            label = label.cuda()

            output, w = Network(img1s, img2s, xs_4)
            # print("hehehhehe", xs_4.shape, img1s.shape, img2s.shape)
            # output, w = Network(xs_4, img1s, img2s)
            # print("输出", output)
            optimizer.zero_grad()
            l = loss_func(output, label)
            # print("损失值:", l)
            """
            for name, parms in Network.named_parameters():
                print('-->name:', name)
                # print('-->para:', parms)
                # print('-->grad_requirs:', parms.requires_grad)
                # print('-->grad_value:', parms.grad)
                print("===")
            """
            l.backward()
            optimizer.step()

            """
            print("优化 >>>>>>>>")
            for name, parms in Network.named_parameters():
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:', parms.requires_grad)
                print('-->grad_value:', parms.grad)
                print("===")
            """
            # print("损失", Network.cnn_network.fc5.weight.grad)
            # print("损失", Network.layer2.weight.grad)
            """
            代试的东西
                for name, parms in testnet.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)
                print("===")
            loss.backward() 
            optimizer.step()
            print("=============更新之后===========")
            for name, parms in testnet.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)
                print("===")
            print(optimizer)
            input("=====迭代结束=====")
            """

            loss_list += [l]
            print("第几个:", i, "损失值:", l)

            fh = open('heheba2.txt', 'a', encoding='utf-8')
            str1 = "第几个:" + str(i) + "损失值:" + str(l) + "\n"
            fh.write(str1)
            fh.close()
            # cuda.close()

        loss_list = torch.stack(loss_list).view(-1)
        print('Epoch: {} / {} ---- Trainning Loss : {}'.format(epoch, config.epochs, loss_list.mean()))
        viz.line([loss_list.mean().item()], [epoch], win='train_loss', update='append')

        fh = open('heheba2.txt', 'a', encoding='utf-8')
        str1 = 'Epoch: {} / {} ---- Trainning Loss : {}'.format(epoch, config.epochs, loss_list.mean()) + "\n"
        fh.write(str1)
        fh.close()

        # Write summary and save current model
        # ----------------------------------------
        torch.save(Network, log_dir + '/NM-Net_2_state.pth')
        # Validation
        va_res,  val_loss = test_process("valid", log_dir, '/NM-Net_2_state.pth', database, config, f1, f2, config.knn_num)
        viz.line([val_loss.item()], [epoch], win='val_loss', update='append')

        print('Validation F-measure : {}'.format(va_res))
        fh = open('heheba2.txt', 'a', encoding='utf-8')
        str1 = 'Validation F-measure : {}'.format(va_res) + "\n"
        fh.write(str1)
        fh.close()

        var_list.append(va_res)

        # np.savetxt(log_dir + './validation_list.txt', np.array(var_list))

        # Higher the better
        if va_res > best_va_res:
            print("Saving best model with va_res = {}".format(va_res))

            fh = open('heheba2.txt', 'a', encoding='utf-8')
            str1 = "Saving best model with va_res = {}".format(va_res) + "\n"
            fh.write(str1)
            fh.close()

            best_va_res = va_res
            # Save best validation result
            np.savetxt(log_dir + "/best_results.txt", best_va_res)
            # Save best model
            torch.save(Network, log_dir + '/NM-Net_2_best_state.pth')

    te_res, te_loss = test_process("test", log_dir, '/NM-Net_2_best_state.pth', database, config, f1, f2, config.knn_num)
    session.close()
    print('Testing F-measure : {}'.format(te_res))

    np.savetxt(log_dir + "/test_results.txt", te_res)

if __name__ == "__main__":
    # ----------------------------------------
    config, unparsed = get_config()

    main(config)






    if b_resume:
        # Restore network
        print("Restoring from {}...".format(
            self.res_dir_tr))
        self.saver_cur.restore(
            self.sess,
            latest_checkpoint
        )
        # restore number of steps so far
        step = self.sess.run(self.global_step)
        # restore best validation result
        if os.path.exists(self.va_res_file):
            with open(self.va_res_file, "r") as ifp:
                dump_res = ifp.read()
            dump_res = parse(
                "{best_va_res:e}\n", dump_res)
            best_va_res = dump_res["best_va_res"]
        if os.path.exists(self.va_res_file_ours_ransac):
            with open(self.va_res_file_ours_ransac, "r") as ifp:
                dump_res = ifp.read()
            dump_res = parse(
                "{best_va_res:e}\n", dump_res)
            best_va_res_ours_ransac = dump_res["best_va_res"]
    else:
        print("Starting from scratch...")
        step = 0
        best_va_res = -1
        best_va_res_ours_ransac = -1









#
# main.py ends here
