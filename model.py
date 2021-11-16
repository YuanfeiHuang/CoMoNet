from src.basic_module import *

class Generator(nn.Module):
    def __init__(self, n_colors, net_type, n_channels, n_blocks, n_units, growth_rate, groups, act, use_IKM=False, scale=4):
        super(Generator, self).__init__()
        self.scale_idx = 0
        self.n_units = n_units

        self.input = nn.Conv2d(in_channels=n_colors, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=True)

        if net_type == 'CNN':
            body = [CNN(n_channels, n_units, act, use_IKM) for _ in range(n_blocks)]
        elif net_type == 'ResNet':
            body = [ResNet(n_channels, n_units, act, use_IKM) for _ in range(n_blocks)]
        elif net_type == 'UHDN':
            body = [UHDB(n_channels, n_units, [6, 5, 4, 4, 5, 6], growth_rate, act, use_IKM) for _ in range(n_blocks)]
        else:
            raise InterruptedError
        self.body = nn.Sequential(
            *body,
            nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=False)
        )

        self.tail = nn.Sequential(
            UpScale('SubPixel', n_channels, scale, groups, bn=False, act=False, bias=False),
            nn.Conv2d(n_channels, n_colors, kernel_size=3, padding=1)
        )

    def forward(self, x):
        y = self.input(x)
        y = self.body(y) + y
        y = self.tail(y)

        return y