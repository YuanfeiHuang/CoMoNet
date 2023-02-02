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
                    # if random.random() > 0.5:
                    #     H, W, C = img_hr.shape
                    #     scaleH, scaleW = np.random.uniform(0.5, 1, 2)
                    #     outH = max(int(H*scaleH), self.args.patch_size*self.args.scale)
                    #     outW = max(int(W*scaleW), self.args.patch_size*self.args.scale)
                    #     outH = outH - outH % self.args.scale
                    #     outW = outW - outW % self.args.scale
                    #     img_hr = cv2.resize(img_hr, dsize=(outW, outH), interpolation=cv2.INTER_NEAREST)
                    #     self.img_hr.append(img_hr)
                    #     # img_hr = resize.imresize(img_as_float(img_hr), output_shape=(outH, outW), method='bicubic')
                    #     img_lr = resize.imresize(img_as_float(img_hr), scalar_scale=1 / self.args.scale, method='bicubic')
                    #     self.img_lr.append(resize.convertDouble2Byte(img_lr))
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
        # if random.random() > 0.5:
        #     img_lr, img_hr = common.get_patch([img_lr, img_hr], self.args.patch_size, self.args.scale)
        #
        #     # scale = np.random.uniform(0.7, 1.05, 2)
        #     # interpolation = random.choices([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC],
        #     #                                weights=[2, 1, 2], k=1)[0]
        #     factor = np.array([random.uniform(0.9, 1.1), random.uniform(0.5, 1.5), random.uniform(0.75, 1.25)]).reshape((1, 1, 3))
        #     # plt.figure(dpi=300)
        #     # plt.subplot(2, 3, 1)
        #     # plt.imshow(img_lr)
        #     # plt.title('H={:.2f}'.format(factor[0,0,0]))
        #     # plt.subplot(2, 3, 4)
        #     # plt.imshow(img_hr)
        #     # plt.title(str(interpolation))
        #     # img_lr = cv2.resize(img_lr, (int(img_lr.shape[1] * scale[0]), int(img_lr.shape[0] * scale[1])), interpolation=interpolation)
        #     # img_hr = cv2.resize(img_hr, (int(img_hr.shape[1] * scale[0]), int(img_hr.shape[0] * scale[1])), interpolation=interpolation)
        #     # h, w, c = img_lr.shape
        #     # ix = h // 2 - self.args.patch_size // 2
        #     # iy = w // 2 - self.args.patch_size // 2
        #     # img_lr = img_lr[ix:(ix+self.args.patch_size), iy:(iy+self.args.patch_size), :]
        #     # tx, ty = ix*self.args.scale, iy*self.args.scale
        #     # img_hr = img_hr[tx:(tx+self.args.scale * self.args.patch_size), ty:(ty+self.args.scale*self.args.patch_size), :]
        #
        #     # plt.subplot(2, 3, 2)
        #     # plt.imshow(img_lr)
        #     # plt.title('S={:.2f}'.format(factor[0,0,1]))
        #     # plt.subplot(2, 3, 5)
        #     # plt.imshow(img_hr)
        #     # plt.title('Scale={:.2f}x{:.2f}'.format(scale[0], scale[1]))
        #
        #     img_lr = cv2.cvtColor(img_lr, cv2.COLOR_RGB2HSV) * factor
        #     img_hr = cv2.cvtColor(img_hr, cv2.COLOR_RGB2HSV) * factor
        #     img_lr[:, :, 0] = img_lr[:, :, 0] % 180
        #     img_hr[:, :, 0] = img_hr[:, :, 0] % 180
        #     img_lr[:, :, 1:] = img_lr[:, :, 1:].clip(0, 255)
        #     img_hr[:, :, 1:] = img_hr[:, :, 1:].clip(0, 255)
        #     img_lr = cv2.cvtColor(img_lr.astype(np.uint8), cv2.COLOR_HSV2RGB)
        #     img_hr = cv2.cvtColor(img_hr.astype(np.uint8), cv2.COLOR_HSV2RGB)
        #     # plt.subplot(2, 3, 3)
        #     # plt.imshow(img_lr)
        #     # plt.title('V={:.2f}'.format(factor[0, 0, 2]))
        #     # plt.subplot(2, 3, 6)
        #     # plt.imshow(img_hr)
        #     # plt.show()
        # else:
        #     img_lr, img_hr = common.get_patch([img_lr, img_hr], self.args.patch_size, self.args.scale)

        flag_aug = random.randint(0, 7)
        img_lr, img_hr = common.augment(img_lr, flag_aug), common.augment(img_hr, flag_aug)
        img_lr = common.np2Tensor(img_lr, self.args.value_range)
        img_hr = common.np2Tensor(img_hr, self.args.value_range)

        return img_lr, img_hr

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size