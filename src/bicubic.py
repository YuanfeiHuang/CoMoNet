import torch
import torch.nn as nn
import numpy as np


class bicubic(nn.Module):
    # https://github.com/tonyzzzt/bicubic-interpolation-pytorch-version-the-same-results-with-matlab-imresize/blob/master/bicubic.py
    def __init__(self):
        super(bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x = torch.arange(start=1, end=out_size + 1).to(torch.float32)

        u = x / scale + 0.5 * (1 - 1 / scale)

        left = torch.floor(u - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice = left.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid = u.unsqueeze(1) - indice.unsqueeze(0)

        if scale < 1:
            weight = scale * self.cubic(mid * scale)
        else:
            weight = self.cubic(mid)

        weight = weight / (torch.sum(weight, 2).unsqueeze(2))

        indice = torch.min(torch.max(torch.FloatTensor([1]), indice), torch.FloatTensor([in_size])).unsqueeze(0)

        kill = torch.eq(weight, 0)[0][0]

        weight = weight[:, :, kill == 0]

        indice = indice[:, :, kill == 0]

        return weight, indice

    def forward(self, input, scale):
        if len(scale) == 1:
            scale += scale
        [b, c, h, w] = input.shape
        if (scale[0] != 1) or (scale[1] != 1):
            weight0, indice0 = self.contribute(h, int(h * scale[0]), scale[0])
            weight1, indice1 = self.contribute(w, int(w * scale[1]), scale[1])

            weight0 = np.asarray(weight0[0], dtype=np.float32)
            weight0 = torch.from_numpy(weight0)

            indice0 = np.asarray(indice0[0], dtype=np.float32)
            indice0 = torch.from_numpy(indice0).long()

            weight1 = np.asarray(weight1[0], dtype=np.float32)
            weight1 = torch.from_numpy(weight1)

            indice1 = np.asarray(indice1[0], dtype=np.float32)
            indice1 = torch.from_numpy(indice1).long()

            if input.is_cuda:
                weight0 = weight0.cuda()
                indice0 = indice0.cuda()
                weight1 = weight1.cuda()
                indice1 = indice1.cuda()

            out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
            out = (torch.sum(out, dim=3))
            A = out.permute(0, 1, 3, 2)

            out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
            out = torch.round(255 * torch.sum(out, dim=3).permute(0, 1, 3, 2)) / 255
        else:
            out = input

        return out