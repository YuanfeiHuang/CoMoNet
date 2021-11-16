import random

from skimage.transform import resize
import tifffile
import torch

from data import common
import imageio
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import matplotlib.pyplot as plt

class DF2K(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self._set_filesystem(args.dir_data)

        def _scan():
            list_gt = []
            for i in range(0, self.args.n_train):
                filename = self._make_filename(i)
                list_gt.append(self._name_file(filename))

            return list_gt

        self.images_gt = _scan()

    def _set_filesystem(self, dir_data):
        self.dir_gt = dir_data + 'Train/DF2K/DF2K_train_HR'

    def _make_filename(self, idx):
        file = os.listdir(self.dir_gt)
        file.sort()
        return file[idx]

    def _name_file(self, filename):
        return os.path.join(self.dir_gt, filename)

    def _name_bin(self):
        return os.path.join(self.dir_gt, '{}_bin_GT.npy'.format(self.split))

    def __getitem__(self, idx):
        img_gt = self._load_file(idx)
        img_gt = common.set_channel(img_gt, self.args.n_colors)
        img_gt = self._get_patch(img_gt)
        img_gt = common.np2Tensor(img_gt, self.args.value_range)
        return img_gt

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_gt)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        img_gt = self.images_gt[idx]
        img_gt = imageio.imread(img_gt)
        return img_gt

    def _get_patch(self, img_gt):
        if self.train:
            # if random.random() > 0.5:
            #     scale = np.random.uniform(0.7, 1)
            #     h_, w_ = int(img_gt.shape[0]*scale), int(img_gt.shape[1]*scale)
            #     img_gt = resize(img_gt, output_shape=(h_, w_))
            img_gt = common.get_patch(img_gt, self.args.patch_size*self.args.scale)
            img_gt = common.augment(img_gt)
        return img_gt
