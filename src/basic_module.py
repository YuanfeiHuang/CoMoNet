import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IKM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, group=1, bias=False, KA=False):
        super(IKM, self).__init__()

        self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)
        self.KA = KA
        self.threshold = 0

    def forward(self, x):
        if self.KA:
            b, cin, w, h = x.size()
            CandConv2d = self.Conv2d.weight.data
            groups = self.Conv2d.groups
            cout, cin, kw, kh = self.Conv2d.weight.data.size()
            # Principal information activation
            attention = torch.where(x < self.threshold, torch.zeros_like(x), torch.ones_like(x))
            # Average pooling
            attention = F.adaptive_avg_pool2d(attention, kw).view(b, cin, kw * kh)
            # Proportion transformation
            attention = kw*kh*F.softmax(attention, dim=2) - 1
            attention = 1 + torch.sigmoid(attention.unsqueeze(1).repeat(1, cout, 1, 1).contiguous().view(b * cout, cin, kw, kh))
            if self.Conv2d.weight.is_cuda:
                attention = attention.cuda()
            self.Conv2d.weight.data = self.Conv2d.weight.data.repeat(b, 1, 1, 1)
            self.Conv2d.weight.data = self.Conv2d.weight.data * attention
            # WEI = self.Conv2d.weight.data
            self.Conv2d.groups = b

            y = self.Conv2d(x.view(1, b * cin, w, h)).view(b, cout, w, h)
            self.Conv2d.weight.data = CandConv2d
            self.Conv2d.groups = groups

            # is_train = True
            # if not x.requires_grad: #in testing phase
            #     is_train = False
            #
            # if (random.random()>0.5 and is_train) or (not is_train):
            #     b, cin, w, h = x.size()
            #     CandConv2d = self.Conv2d.weight.data
            #     groups = self.Conv2d.groups
            #     cout, cin, kw, kh = self.Conv2d.weight.data.size()
            #     self.Conv2d.weight.requires_grad = False
            #
            #     # x_mean, x_std = calc_mean_std(x)
            #     # x_normalized = (x - x_mean.expand(x.size())) / x_std.expand(x.size())
            #     # x_normalized = self.normalization(x)
            #     #threshold = torch.mean(x.view(b, cin, w*h), dim=2).view(b, cin, 1, 1)
            #     # threshold = torch.median(x)
            #     threshold = 0.0
            #     # x_count = x
            #     # x_count[x_count<-threshold] = 0
            #     # x_count[x_count>threshold]= 1
            #     # x_count = F.hardtanh(x, min_val=-threshold, max_val=threshold)
            #     # x_count = torch.where(x_count>threshold, torch.ones_like(x), x_count)
            #     x_count = torch.where(x<threshold, torch.zeros_like(x), torch.ones_like(x))
            #     attention = F.adaptive_avg_pool2d(x_count, kw)
            #     # x_global_K=self.nonlinear(x_global_K)
            #     #x_global_K = self.normalization(x_global_K)
            #     # code_att = torch.sigmoid(x_global_K)
            #     attention = self.nonlinear(attention).view(b, cin, kw*kh)
            #     attention = F.softmax(attention, dim=2) - 1 / (kw * kh)
            #     attention = 1 + attention.view(b, cin, kw, kh).unsqueeze(0).permute([1,0,2,3,4]).contiguous().repeat(1, cout, 1, 1, 1).view(b*cout, cin, kw, kh)
            #
            #     #attention = 1. + torch.mean(code_att, 0).view(1, cin, kw, kh).repeat(cout, 1, 1, 1)
            #
            #     # x_count = x_count[1]
            #     # x = torch.mean(x_global_K, 0)
            #     # x_normalized = torch.mean(x_normalized, 1)
            #     # for idx in range(cin):
            #     #     save_img(x_count.data[1, idx, :, :], 'Visualization/' + "x_count" + str(idx + 1) + ".png")
            #     #     save_img(attention.data[1, idx, :, :], 'Visualization/' + "x_K" + str(idx + 1) + ".png")
            #
            #     if self.Conv2d.weight.is_cuda:
            #         attention = attention.cuda()
            #     self.Conv2d.weight.data = self.Conv2d.weight.data.repeat(b,1,1,1)
            #     self.Conv2d.weight.data = self.Conv2d.weight.data * attention
            #     self.Conv2d.groups = b
            #
            #     y = self.Conv2d(x.view(1, b*cin, w, h)).view(b, cout, w, h)
            #     self.Conv2d.weight.data = CandConv2d
            #     self.Conv2d.groups = groups
            #     if is_train:
            #         self.Conv2d.weight.requires_grad = True
            # else:
            #     y = self.Conv2d(x)
            #     #y.register_hook(lambda x: 0.1*x)
        else:
            y = self.Conv2d(x)
        return y

class dense_conv(nn.Module):
    def __init__(self, in_feats, grow_rate, kernel=3, activation=nn.ReLU(inplace=True),bias=True, KA=False):
        super(dense_conv, self).__init__()
        layer = []
        layer.append(IKM(in_channels=in_feats, out_channels=grow_rate, kernel_size=kernel, padding=1,bias=bias, KA=KA))
        layer.append(activation)
        self.layer = nn.Sequential(*layer)

        # self.calayer = nn.Sequential(
        #     nn.Conv2d(grow_rate, grow_rate // 16, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(grow_rate // 16, grow_rate, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # )

        # self.salayer = nn.Sequential(
        #     nn.Conv2d(2, 1, 7, 1, 3, 1, 1, True),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        output = self.layer(x)

        # chnlatt = F.adaptive_avg_pool2d(output, 1)
        # chnlatt = self.calayer(chnlatt)
        # output = output * chnlatt

        # out_avg = torch.mean(output, dim=1, keepdim=True)
        # out_max, _ = torch.max(output, dim=1, keepdim=True)
        # sptlatt = self.salayer(torch.cat((out_avg, out_max), 1))
        # output = output * sptlatt

        return torch.cat((x, output), 1)

class conv_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel=3, dilation=1, bias=False, activation=nn.ReLU(inplace=True), KA=False):
        super(conv_block, self).__init__()

        pad = int(dilation * (kernel - 1) / 2)
        block = []
        block.append(IKM(in_channels=in_feats, out_channels=out_feats, kernel_size=kernel, padding=pad,
                              dilation=dilation, bias=bias, KA=KA))
        block.append(activation)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = self.block(x)
        return output

class res_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel=3, dilation=1, bias=False, activation=nn.ReLU(inplace=True), KA=False):
        super(res_block, self).__init__()

        pad = int(dilation * (kernel - 1) / 2)
        block = []
        for i in range(2):
            if i == 0:
                block.append(IKM(in_channels=in_feats, out_channels=out_feats, kernel_size=kernel, padding=pad,
                                        dilation=dilation, bias=bias, KA=KA))
            else:
                block.append(IKM(in_channels=in_feats, out_channels=out_feats, kernel_size=kernel, padding=pad,
                                          dilation=dilation, bias=bias, KA=KA))
            if i == 0: block.append(activation)

        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = self.block(x) + x
        return output

class dense_block(nn.Module):
    def __init__(self, n_feats, grow_rate, n_units, activation, KA=False):
        super(dense_block, self).__init__()
        n_units = min(max(n_units, 4), 16)

        body = []
        for i in range(n_units):
            body.append(dense_conv(n_feats+i*grow_rate, grow_rate, bias=False, activation=activation, KA=KA))

        self.body = nn.Sequential(*body)
        self.gate = nn.Sequential(IKM(n_feats+n_units*grow_rate, n_feats, 1, padding=0, bias=False, KA=False))

    def forward(self, x):
        return self.gate(self.body(x))

class CNN(nn.Module):
    def __init__(self, n_feats, n_units, kernel, act, KA=False):
        super(CNN, self).__init__()

        body = [conv_block(n_feats, n_feats, kernel, activation=act, KA=KA) for _ in range(n_units)]

        self.body = nn.Sequential(*body)

    def forward(self, x):

        output = self.body(x) + x

        return output

class ResNet(nn.Module):
    def __init__(self, n_feats, n_units, act, KA=False):
        super(ResNet, self).__init__()

        body = [res_block(n_feats, n_feats, activation=act, KA=KA) for _ in range(n_units)]

        self.body = nn.Sequential(*body)

    def forward(self, x):

        output = self.body(x) + x

        return output

class UHDB(nn.Module):
    def __init__(self, n_feats, n_units, group_in_units, growth_rate, act, KA=False):
        super(UHDB, self).__init__()

        body = [dense_block(n_feats, growth_rate, group_in_units[i], act, KA=KA) for i in range(n_units)]

        self.body = nn.Sequential(*body)

    def forward(self, x):

        # U-Hourglass, case: n_units=6
        x1 = self.body[0](x)
        x2 = self.body[1](x1)
        x3 = self.body[2](x2)
        x4 = self.body[3](x3)+x2
        x5 = self.body[4](x4)+x1
        y = self.body[5](x5) +x

        return y

class UpScale(nn.Sequential):
    def __init__(self, type, n_feats, scale, groups=1, bn=False, act=nn.ReLU(inplace=True), bias=False):
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(np.log2(scale))):
                if type == 'DeConv':
                    layers.append(nn.ConvTranspose2d(n_feats, n_feats, 4, stride=2, padding=1, groups=1, bias=bias))
                elif type == 'SubPixel':
                    layers.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats, kernel_size=3, stride=1,
                                            padding=1, groups=groups, bias=bias))
                    layers.append(nn.PixelShuffle(2))
                else:
                    raise InterruptedError
                if bn: layers.append(nn.BatchNorm2d(n_feats))
                if act: layers.append(act)
        elif scale == 3:
            layers.append(nn.Conv2d(in_channels=n_feats, out_channels=9 * n_feats, kernel_size=3, stride=1,
                                    padding=1, groups=groups, bias=bias))
            layers.append(nn.PixelShuffle(3))
            if bn: layers.append(nn.BatchNorm2d(n_feats))
            if act: layers.append(act)
        elif scale == 1:
            layers.append(nn.Identity())
        else:
            raise InterruptedError
        super(UpScale, self).__init__(*layers)