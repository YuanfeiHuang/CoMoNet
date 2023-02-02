from src.basic_module import *


class Generator(nn.Module):
    def __init__(self, n_colors, net_type, n_channels, n_blocks, n_units, n_layers, growth_rate, act, use_CoMo=False,
                 scale=4):
        super(Generator, self).__init__()
        self.scale_idx = 0
        self.n_units = n_units

        self.input = nn.Conv2d(in_channels=n_colors, out_channels=n_channels, kernel_size=3, stride=1, padding=1,
                               bias=True)

        if net_type == 'CNN':
            body = [CNN(n_channels, n_units, act, use_CoMo) for _ in range(n_blocks)]
        elif net_type == 'ResNet':
            body = [ResNet(n_channels, n_units, act, use_CoMo) for _ in range(n_blocks)]
        elif net_type == 'UHDN':
            body = [UHDB(n_channels, n_units, n_layers, growth_rate, act, use_CoMo) for _ in range(n_blocks)]
            # body = [UHDB(n_channels, n_units, n_layers, growth_rate, act, use_CoMo) for _ in range(n_blocks)]
        else:
            raise InterruptedError
        self.body = nn.Sequential(
            *body,
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        )

        self.tail = nn.Sequential(
            UpScale('SubPixel', n_channels, scale, bn=False, act=False, bias=False),
            nn.Conv2d(n_channels, n_colors, kernel_size=3, padding=1)
        )

    def forward(self, x):
        y = self.input(x)
        y = self.body(y) + y
        y = self.tail(y)

        return y


class PSNRLoss(nn.Module):

    def __init__(self, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target + 1e-12))
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise InterruptedError


class MAPELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.abs(pred - target) / (torch.abs(target) + 0.1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise InterruptedError


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class LInf_Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        loss = torch.abs(pred - target)
        loss = torch.max(loss.view(B, C, -1), dim=2)[0]
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise InterruptedError
