import random

# from skimage.transform import resize
import tifffile
import torch, cv2

from data import common
import imageio
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import os, time
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import img_as_float
import src.bicubic_python as resize

class dataloader(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self._set_filesystem()

        if self.args.store_in_ram:
            self.img_hr, self.img_lr = [], []
            # sz = 10000
            with tqdm(total=len(self.filepath_hr), ncols=140) as pbar:
                for idx in range(len(self.filepath_hr)):
                    img_hr, img_lr = imageio.imread(self.filepath_hr[idx]), imageio.imread(self.filepath_lr[idx])
                    # img_hr = imageio.imread(self.filepath_hr[idx])
                    # H, W, C = img_hr.shape
                    # img_lr = resize.imresize(img_as_float(img_hr), scalar_scale=1 / args.scale, method='bicubic')
                    # img_lr = resize.convertDouble2Byte(img_lr)
                    h, w, c = img_lr.shape
                    if min(h, w) > self.args.patch_size:
                        self.img_hr.append(img_hr)
                        self.img_lr.append(img_lr)
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(name=self.filepath_hr[idx].split('/')[-1], number=len(self.img_hr))
            # self.n_train = len(self.img_hr)

    def _set_filesystem(self):
        self.filepath_hr, self.filepath_lr = np.array([]), np.array([])
        for idx_dataset in range(len(self.args.data_train_hr)):
            if self.args.n_train[idx_dataset] > 0:
                path_hr = os.path.join(self.args.dir_data, self.args.data_train_hr[idx_dataset])
                path_lr = os.path.join(self.args.dir_data, self.args.data_train_lr[idx_dataset])
                names_hr = os.listdir(path_hr)
                names_lr = os.listdir(path_lr)

                names_hr.sort()
                names_lr.sort()
                filepath_hr, filepath_lr = np.array([]), np.array([])

                for idx_image in range(len(names_hr)):
                    filepath_hr = np.append(filepath_hr, os.path.join(path_hr, names_hr[idx_image]))
                    filepath_lr = np.append(filepath_lr, os.path.join(path_lr, names_lr[idx_image]))

                data_lenhrh = len(filepath_hr)
                idx = np.arange(0, data_lenhrh)
                if self.args.n_train[idx_dataset] < data_lenhrh:
                    if self.args.shuffle:
                        idx = np.random.choice(idx, size=self.args.n_train[idx_dataset], replace=False)
                    else:
                        idx = np.arange(0, self.args.n_train[idx_dataset])

                self.filepath_hr = np.append(self.filepath_hr, filepath_hr[idx])
                self.filepath_lr = np.append(self.filepath_lr, filepath_lr[idx])

    def __getitem__(self, idx):
        # if random.random() > 0.25:
        idx = idx % self.args.n_train[0]
        img_hr, img_lr = self.img_hr[idx], self.img_lr[idx]
        # else:
        #     idx = idx % (len(self.img_hr) - self.args.n_train[0])
        #     img_hr, img_lr = self.img_hr[self.args.n_train[0] + idx], self.img_lr[self.args.n_train[0] + idx]

        img_lr, img_hr = common.set_channel([img_lr, img_hr], self.args.n_colors)
        img_lr, img_hr = common.get_patch([img_lr, img_hr], self.args.patch_size, self.args.scale)
        flag_aug = random.randint(0, 7)
        img_lr, img_hr = common.augment(img_lr, flag_aug), common.augment(img_hr, flag_aug)
        img_lr = common.np2Tensor(img_lr, self.args.value_range)
        img_hr = common.np2Tensor(img_hr, self.args.value_range)

        return img_lr, img_hr

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size
