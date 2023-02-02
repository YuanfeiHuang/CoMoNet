import random, cv2

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms


def get_patch(img, patch_size, scale=1):
    if isinstance(img, list):
        ih, iw, c = img[0].shape
        tp = scale * patch_size
        ip = tp // scale
        if min(iw, ih) < patch_size:
            img[0] = cv2.copyMakeBorder(img[0], 0, max(ip-ih, 0), 0, max(ip-iw, 0), cv2.BORDER_DEFAULT)
            img[1] = cv2.copyMakeBorder(img[1], 0, max(tp-scale*ih, 0), 0, max(tp-scale*iw, 0), cv2.BORDER_DEFAULT)
            ih, iw, c = img[0].shape

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy
        img_out = [img[0][iy:iy + ip, ix:ix + ip, :], img[1][ty:ty + tp, tx:tx + tp, :]]
    else:
        ih, iw, c = img.shape
        ip = scale * patch_size
        if min(iw, ih) < patch_size:
            img = cv2.copyMakeBorder(img, 0, max(ip - ih, 0), 0, max(ip - iw, 0), cv2.BORDER_DEFAULT)
            ih, iw, c = img.shape
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        img_out = img[iy:iy + ip, ix:ix + ip, :]

    return img_out


def normalization(img):

    if img.max() < 2**8:
        scale = 255
    elif img.max() < 2**10:
        scale = 1023
    elif img.max() < 2**12:
        scale = 4095
    elif img.max() < 2**14:
        scale = 16383
    else:
        scale = 65535
    img = img / scale

    return np.uint8(255 * img), scale

def set_channel(img_in, n_channel):
    def _set_channel(img):
        if len(img.shape) == 3:
            h, w, c = img.shape
            if c > 3:
                img = img[:, :, :3]  # convert RGBA to RGB
        else:
            c = 1
        if n_channel == 1 and c == 3:
            img = sc.rgb2ycbcr(img)[:, :, 0]
            img = img.clip(0, 255).round()
            img = np.expand_dims(img, 2)
        if n_channel == 1 and c == 1:
            img = np.expand_dims(img, 2)
        elif n_channel == 3 and c == 1:
            img = np.expand_dims(img, axis=2)
            img = np.concatenate([img] * n_channel, 2)

        return img
    if isinstance(img_in, list):
        return [_set_channel(img_in[i]) for i in range(len(img_in))]
    else:
        return _set_channel(img_in)

def np2Tensor(img, rgb_range):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    torch_tensor = torch.from_numpy(np_transpose.copy()).float()
    torch_tensor.div_(rgb_range)

    return torch_tensor

def augment(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out
