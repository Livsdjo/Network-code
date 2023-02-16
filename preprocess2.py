import torch

import tensorflow as tf
# from code.image_preprocess.tf_utils import apply_patch_pert
import code
import tf_utils
# from code.tf_utils import apply_patch_pert, photometric_augmentation
import spatial_transformer
# from code.spatial_transformer import transformer_crop
import loss


import torch as th
import numpy as np
from torch import optim
from math import cos, sin, pi
import tfpyth
# from config import get_config, print_usage
# from dataset import Data_Loader
import os
import torch.utils.data as Data
# import code.image_preprocess.spatial_transformer as st
# from code.config import get_config
# from code.dataset import Data_Loader
import config, dataset
import MultiModal_NetWork

batch_size_set = 1
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def get_rnd_homography(tower_num, batch_size, pert_ratio=0.25):
    import cv2
    all_homo = []
    corners = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1]], dtype=np.float32)
    for _ in range(tower_num):
        one_tower_homo = []
        for _ in range(batch_size):
            rnd_pert = np.random.uniform(-2 * pert_ratio, 2 * pert_ratio, (4, 2)).astype(np.float32)
            pert_corners = corners + rnd_pert
            M = cv2.getPerspectiveTransform(corners, pert_corners)
            one_tower_homo.append(M)
        one_tower_homo = np.stack(one_tower_homo, axis=0)
        all_homo.append(one_tower_homo)
    all_homo = np.stack(all_homo, axis=0)
    return all_homo.astype(np.float32)


def get_rnd_affine(tower_num, batch_size, num_corr, sync=True, distribution='uniform',
                   crop_scale=0.5, rng_angle=5, rng_scale=0.3, rng_anis=0.4):
    """In order to enhance rotation invariance, applying
    random affine transformation (3x3) on matching patches.
    Args:
        batch_size: Training batch size (number of data bags).
        num_corr: Number of correspondences in a data bag.
        crop_scale: The ratio to apply central cropping.
        rng_angle: Range of random rotation angle.
        rng_scale: Range of random scale.
        rng_anis: Range of random anis.
    Returns:
        all_pert_mat: Transformation matrices.
    """
    num_patches = batch_size * num_corr

    if sync:
        sync_angle = np.random.uniform(-90, 90, (num_patches,))
    else:
        sync_angle = 0

    all_pert_affine = []
    # two feature towers
    for _ in range(tower_num):
        if distribution == 'uniform':
            rnd_scale = np.random.uniform(2 ** -rng_scale, 2 ** rng_scale, (num_patches,))
            rnd_anis = np.random.uniform(np.sqrt(2 ** -rng_anis), np.sqrt(2 ** rng_anis), (num_patches,))
            rnd_angle = np.random.uniform(-rng_angle, rng_angle, (num_patches,))
        elif distribution == 'normal':
            rnd_scale = 1 + np.random.normal(0, rng_scale / 2 / 3, (num_patches,))
            rnd_anis = 1 + np.random.normal(0, rng_anis / 2 / 3, (num_patches,))
            rnd_angle = np.random.normal(0, rng_angle / 3, (num_patches,))

        rnd_scale *= crop_scale
        rnd_angle = (rnd_angle + sync_angle) / 180. * pi

        pert_affine = np.zeros((num_patches, 9), dtype=np.float32)
        pert_affine[:, 0] = np.cos(rnd_angle) * rnd_scale * rnd_anis
        pert_affine[:, 1] = np.sin(rnd_angle) * rnd_scale * rnd_anis
        pert_affine[:, 2] = 0
        pert_affine[:, 3] = -np.sin(rnd_angle) * rnd_scale / rnd_anis
        pert_affine[:, 4] = np.cos(rnd_angle) * rnd_scale / rnd_anis
        pert_affine[:, 5] = 0
        pert_affine[:, 8] = np.ones((num_patches,), dtype=np.float32)
        pert_affine = np.reshape(pert_affine, (batch_size, num_corr, 3, 3))
        all_pert_affine.append(pert_affine)
    all_pert_affine = np.stack(all_pert_affine, axis=0)
    return all_pert_affine.astype(np.float32)


def pre_handle(img, kpt_coeff, spec, num_corr, photaug, pert_homo, pert_affine, dense_desc, name):
    """
    Data Preprocess.
    """
    with tf.name_scope(name):  # pylint: disable=not-context-manager
        print(5555555555)
        img = tf.cast(img, tf.float32)
        img.set_shape((batch_size_set,
                       img.get_shape()[1].value,
                       img.get_shape()[2].value,
                       img.get_shape()[3].value))
        print(6666666666666666666666)
        if True and photaug:
            # print('Applying photometric augmentation.')
            img = tf.map_fn(tf_utils.photometric_augmentation, img, back_prop=False)
        img = tf.clip_by_value(img, 0, 255)
        # perturb patches and coordinates.
        pert_kpt_affine, kpt_ncoords = tf_utils.apply_patch_pert(
            kpt_coeff, pert_affine, batch_size_set, num_corr, adjust_ratio=1. if dense_desc else 5. / 6.)
        # print(2222, pert_kpt_affine.shape)
        if dense_desc:
            # image standardization
            mean, variance = tf.nn.moments(
                tf.cast(img, tf.float32), axes=[1, 2], keep_dims=True)
            out = tf.nn.batch_normalization(
                img, mean, variance, None, None, 1e-5)
        else:
            # patch sampler.
            patch = spatial_transformer.transformer_crop(
                img, pert_kpt_affine, (32, 32), True)
            # patch standardization
            mean, variance = tf.nn.moments(patch, axes=[1, 2], keep_dims=True)
            out = tf.nn.batch_normalization(
                patch, mean, variance, None, None, 1e-5)
            print(3333)
        out = tf.stop_gradient(out)
        return out, kpt_ncoords, pert_homo


def get_32_32_image():
    if 0:
        print("aaaaaaaa")
        pert_homo = tf.numpy_function(
            get_rnd_homography, [2, batch_size, 0.15], tf.float32)
        pert_homo = tf.reshape(pert_homo, (2, batch_size, 3, 3))
        print("bbbbbbbb")
        pert_affine = tf.numpy_function(
            get_rnd_affine, [2, batch_size, 2000], tf.float32)
        pert_affine = tf.reshape(pert_affine, (2, batch_size, 2000, 3, 3))
        print("ccccccccc")
        net_input0, kpt_ncoords0, pert_homo0 = pre_handle(
            img0, kpt_coeff0, None, 2000, True,
            pert_homo[0], pert_affine[0], False, name='input0')
        net_input1, kpt_ncoords1, pert_homo1 = pre_handle(
            img1, kpt_coeff1, None, 2000, True,
            pert_homo[1], pert_affine[1], False, name='input1')
        # print(55555, net_input0.shape)
        print("dddddddddd")
        return net_input0, kpt_ncoords0, pert_homo0, net_input1, kpt_ncoords1, pert_homo1
    else:
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        tf_config.log_device_placement = False

        session = tf.Session(config=tf_config)
        with tf.device("GPU:1"):
            img0 = tf.placeholder(tf.float32, name="img0", shape=(1, None, None, 1))      # shape=(1, 1000, 1000, 1)
            kpt_coeff0 = tf.placeholder(tf.float32, name="kpt_coeff0", shape=(1, 2000, 6))
            img1 = tf.placeholder(tf.float32, name="img1", shape=(1, None, None, 1))      # shape=(1, 1000, 1000, 1)
            kpt_coeff1 = tf.placeholder(tf.float32, name="kpt_coeff1", shape=(1, 2000, 6))

            pert_homo = tf.numpy_function(
                get_rnd_homography, [2, batch_size_set, 0.15], tf.float32)
            pert_homo = tf.reshape(pert_homo, (2, batch_size_set, 3, 3))
            pert_affine = tf.numpy_function(
                get_rnd_affine, [2, batch_size_set, 2000], tf.float32)
            pert_affine = tf.reshape(pert_affine, (2, batch_size_set, 2000, 3, 3))

            print(11111111111111)
            net_input0, kpt_ncoords0, pert_homo = pre_handle(
                img0, kpt_coeff0, None, 2000, True,
                pert_homo[0], pert_affine[0], False, name='input0')
            print(22222222222222)
            net_input1, kpt_ncoords1, pert_homo1 = pre_handle(
                img1, kpt_coeff1, None, 2000, True,
                pert_homo[1], pert_affine[1], False, name='input1')
            print(3333333333333333333)


        f1 = tfpyth.torch_from_tensorflow(session, [img0, kpt_coeff0], net_input0).apply
        f2 = tfpyth.torch_from_tensorflow(session, [img1, kpt_coeff1], net_input1).apply


        f3 = tfpyth.torch_from_tensorflow(session, [img0, kpt_coeff0], img0_x_0).apply
        f4 = tfpyth.torch_from_tensorflow(session, [img0, kpt_coeff0], img0_x_1).apply
        f5 = tfpyth.torch_from_tensorflow(session, [img0, kpt_coeff0], img0_y_0).apply
        f6 = tfpyth.torch_from_tensorflow(session, [img0, kpt_coeff0], img0_y_1).apply

        f7 = tfpyth.torch_from_tensorflow(session, [img1, kpt_coeff1], img1_x_0).apply
        f8 = tfpyth.torch_from_tensorflow(session, [img1, kpt_coeff1], img1_x_1).apply
        f9 = tfpyth.torch_from_tensorflow(session, [img1, kpt_coeff1], img1_y_0).apply
        f10 = tfpyth.torch_from_tensorflow(session, [img1, kpt_coeff1], img1_y_1).apply

        tf.get_default_graph().finalize()

        return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, session


if __name__ == '__main__':
    config, unparsed = config.get_config()
    """The main function."""
    database = "COLMAP"
    database_list = []
    if database == 'COLMAP':
        # database_list += ["south"]         # 2360       1076 * 807
        database_list += ["gerrard"]  # 1800
        # database_list += ["graham"]        # 11000
        # database_list += ["person"]

    # CNN_Network = MultiModal_NetWork.Net().cuda()
    # Point_Network = MultiModal_NetWork.NM_Net()
    Network = MultiModal_NetWork.Net().cuda()
    optimizer = optim.Adam(Network.parameters(), lr=config.train_lr)
    loss_func = loss.Loss_classi().cuda()

    log_dir = "log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    d = dataset.Data_Loader(config, database, database_list, "train", initialize=False)
    data = Data.DataLoader(d, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    for i, (img1s, img2s, xs_4, label, xs_12, others) in enumerate(data, 0):
        xs1s = xs_12[:, :, 0: 6]
        xs2s = xs_12[:, :, 6: 12]
        # print("2872397429", xs2s.shape, xs1s.shape, img1s.shape, img2s.shape)

        if 0:
            f = get_32_32_image()
            print(999999)
            img1s_temp = th.tensor(img1s, dtype=th.float32, requires_grad=False)
            xs1s_temp = th.tensor(xs1s, dtype=th.float32, requires_grad=False)
            x = f(img1s_temp, xs1s_temp)
            print(1010110)
            print(x)
            print(888888)
        else:
            # 第一张图片
            img1s_temp = th.tensor(img1s, dtype=th.float32, requires_grad=False)
            xs1s_temp = th.tensor(xs1s, dtype=th.float32, requires_grad=False)
            # print("47535793", xs1s_temp)
            result1 = get_32_32_image(img1s_temp, xs1s_temp)
            # 第二张图片
            img2s_temp = th.tensor(img2s, dtype=th.float32, requires_grad=False)
            xs2s_temp = th.tensor(xs2s, dtype=th.float32, requires_grad=False)
            # print("47535793", xs1s_temp)
            result2 = get_32_32_image(img2s_temp, xs2s_temp)

            # print(result)
            """
            with tf.compat.v1.Session() as sess:
                sess.run([result1, result2])
                # print(result[0].eval())
                img1s = result1[0].eval()
                img2s = result2[0].eval()
            """

        img1s = torch.Tensor(img1s)
        img1s = img1s.permute(0, 3, 1, 2)
        img2s = torch.Tensor(img2s)
        img2s = img2s.permute(0, 3, 1, 2)

        img1s = img1s.cuda()
        img2s = img2s.cuda()
        xs_4 = xs_4.cuda()
        label = label.cuda()
        output, w = Network(img1s, img2s, xs_4)

        optimizer.zero_grad()
        l = loss_func(output, label)
        print("损失值:", l)
        l.backward()
        optimizer.step()

        loss_list += [l]

        # print(image)
        if i == 1:
            break
