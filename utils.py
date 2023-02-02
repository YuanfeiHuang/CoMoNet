import os, torch, cv2, shutil
import numpy as np
from torch.autograd import Variable
import skimage.color as sc
import torch.nn.functional as F
from datetime import datetime


def random_cropping(x, patch_size, number):
    if isinstance(x, tuple):
        if min(x[0].shape[2], x[0].shape[3]) < patch_size:
            for i in range(len(x)):
                x[i] = F.interpolate(x[i], scale_factor=0.1 + patch_size / min(x[i].shape[2], x[i].shape[3]))

        b, c, w, h = x[0].size()
        ix = np.random.choice(w - patch_size + 1, number)
        iy = np.random.choice(h - patch_size + 1, number)
        patch = [[] for _ in range(len(x))]
        for i in range(number):
            for l in range(len(x)):
                if i == 0:
                    patch[l] = x[l][:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]
                else:
                    patch[l] = torch.cat((patch[l], x[l][:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]),
                                         dim=0)
    else:
        b, c, w, h = x.size()

        ix = np.random.choice(w - patch_size + 1, number)
        iy = np.random.choice(h - patch_size + 1, number)

        for i in range(number):
            if i == 0:
                patch = x[:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]
            else:
                patch = torch.cat((patch, x[:, :, ix[i]:ix[i] + patch_size, iy[i]:iy[i] + patch_size]), dim=0)

    return patch


def crop_merge_TLSR(x_value, TLSR_Param, model, scale, shave, min_size, n_GPUs):
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = x_value.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [x_value[:, :, 0:h_size, 0:w_size], x_value[:, :, 0:h_size, (w - w_size):w],
                 x_value[:, :, (h - h_size):h, 0:w_size], x_value[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, n_GPUs):
            inputbatch = torch.cat(inputlist[i:(i + n_GPUs)], dim=0)
            degree = TLSR_Param['DM'].unsqueeze(0).repeat(inputbatch.shape[0])
            inputbatch = {'value': inputbatch, 'num_samples': TLSR_Param['num_samples'], 'DM': degree,
                          'transi_learn': TLSR_Param['transi_learn']}
            outputbatch = model(inputbatch)
            outputlist.extend(outputbatch.chunk(n_GPUs, dim=0))
    else:
        outputlist = [crop_merge_TLSR(patch, TLSR_Param, model, scale, shave, min_size, n_GPUs) for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x_value.data.new(b, c, h, w))
    output[0, :, 0:h_half, 0:w_half] = outputlist[0][0, :, 0:h_half, 0:w_half]
    output[0, :, 0:h_half, w_half:w] = outputlist[1][0, :, 0:h_half, (w_size - w + w_half):w_size]
    output[0, :, h_half:h, 0:w_half] = outputlist[2][0, :, (h_size - h + h_half):h_size, 0:w_half]
    output[0, :, h_half:h, w_half:w] = outputlist[3][0, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def quantize(img, rgb_range):
    return img.mul(255).clamp(0, 255).round().div(255)


def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0)
    yCbCr = sc.rgb2ycbcr(rgb) / 255

    return torch.Tensor(yCbCr[:, :, 0])


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_SSIM(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    c, h, w = input.size()
    if c > 1:
        input = input.mul(255).clamp(0, 255).round()
        target = target[:, 0:h, 0:w].mul(255).clamp(0, 255).round()
        input = rgb2ycbcrT(input)
        target = rgb2ycbcrT(target)
    else:
        input = input[0, 0:h, 0:w].mul(255).clamp(0, 255).round()
        target = target[0, 0:h, 0:w].mul(255).clamp(0, 255).round()
    input = input[shave:(h - shave), shave:(w - shave)]
    target = target[shave:(h - shave), shave:(w - shave)]
    return ssim(input.numpy(), target.numpy())


def calc_PSNR(input, target, rgb_range, shave):
    c, h, w = input.size()
    if c > 1:
        input = quantize(input, rgb_range)
        target = quantize(target[:, 0:h, 0:w], rgb_range)
        input_Y = rgb2ycbcrT(input)
        target_Y = rgb2ycbcrT(target)
        diff = (input_Y - target_Y).view(1, h, w)
    else:
        target = target[:, 0:h, 0:w]
        diff = input - target
    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr.data.numpy()

def test_selfensemble(model, input):
    # TODO: to be tested
    # 8 augmentations
    # modified from https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/__init__.py

    def _transform(v, op):
        # if self.precision != 'single': v = v.float()
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).type_as(v)
        # if self.precision == 'half': ret = ret.half()

        return ret

    # prepare augmented data
    lr_list = [input]
    for tf in 'v', 'h', 't':
        lr_list.extend([_transform(t, tf) for t in lr_list])

    # inference
    sr_list = [model(aug) for aug in lr_list]

    # merge results
    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], 't')
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], 'h')
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], 'v')
    output = torch.cat(sr_list, dim=0)

    output = output.mean(dim=0, keepdim=True)

    return output


def save_checkpoint(model, epoch, folder):
    model_path = folder + '/model_epoch_{:d}.pth'.format(epoch)
    torch.save(model.state_dict(), model_path)
    print('Checkpoint saved to {}'.format(model_path))


def load_checkpoint(resume, model, is_cuda, n_GPUs):
    if os.path.isfile(resume):
        new_checkpoint = model.state_dict()
        print("=> loading checkpoint '{}'".format(resume))
        # checkpoint = torch.load(resume, map_location={'cuda:1': 'cuda:0'})
        checkpoint = torch.load(resume) if is_cuda else torch.load(resume, map_location=torch.device('cpu'))
        # if isinstance(checkpoint, dict):
        #     checkpoint = checkpoint['state_dict']
        if n_GPUs > 1:
            for k, v in checkpoint.items():
                if k[:6] != 'module':
                    new_checkpoint[k] = v
                else:
                    name = k[7:]
                    new_checkpoint[name] = v
        else:
            for k, v in checkpoint.items():
                if k[:6] == 'module':
                    name = k[7:]
                    if new_checkpoint[name].shape == v.shape:
                        new_checkpoint[name] = v
                else:
                    if (k in new_checkpoint) and new_checkpoint[k].shape == v.shape:
                        new_checkpoint[k] = v
        model.load_state_dict(new_checkpoint)
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    return model


def print_args(args):
    if args.train.lower() == 'train':

        if args.use_CoMo:
            args.model_path = 'models/CoMo-L_X{}'.format(args.scale) + datetime.now().strftime("_%Y%m%d_%H%M")
        else:
            args.model_path = 'models/UHDN_X{}'.format(args.scale) + datetime.now().strftime("_%Y%m%d_%H%M")

        args.resume_SR = 'models/CoMo-L_x4.pth'
        if not os.path.exists(args.model_path + '/Checkpoints'):
            os.makedirs(args.model_path + '/Checkpoints')

        if not os.path.exists(args.model_path + '/Code'):
            os.makedirs(args.model_path + '/Code')

        files = os.listdir('.')
        for file in files:
            if file[-3:] == '.py':
                shutil.copyfile(file, args.model_path + '/Code/' + file)
        shutil.copytree('src', args.model_path + '/Code/src', dirs_exist_ok=True)
        shutil.copytree('data', args.model_path + '/Code/data', dirs_exist_ok=True)

    elif args.train.lower() == 'test':
        if args.n_channels == 48:
            args.model_path = 'models/CoMo-S_x{:d}'.format(args.scale)
        elif args.n_channels == 128:
            args.model_path = 'models/CoMo-L_x{:d}'.format(args.scale)

        if not os.path.exists(args.model_path + '/SR Results'):
            os.makedirs(args.model_path + '/SR Results')

    else:
        raise InterruptedError
    return args
