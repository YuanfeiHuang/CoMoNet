import random, cv2

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms


def get_patch(img, patch_size):
    ih, iw, c = img.shape
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    img = img[iy:iy + ip, ix:ix + ip, :]

    return img


def set_channel(img, n_channel):

    if len(img.shape) == 3:
        h, w, c = img.shape
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

def np2Tensor(img, rgb_range):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    torch_tensor = torch.from_numpy(np_transpose).float()
    torch_tensor.div_(rgb_range)

    return torch_tensor


def augment(img_in, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return _augment(img_in)


def JPEG_compression(img, quality=40, n_channel=1):
    if len(img.shape) == 3:
        h, w, c = img.shape
    else:
        c = 1

    if n_channel == 1 and c == 3:
        img = sc.rgb2ycbcr(img)
        result, encimg = cv2.imencode('.jpg', img[:, :, 0], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img[:, :, 0] = cv2.imdecode(encimg, 0)
        img = sc.ycbcr2rgb(img)
    elif n_channel == 1 and c == 1:
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img = cv2.imdecode(encimg, 0)
        img = np.expand_dims(img, axis=2)
    elif n_channel == 3 and c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

