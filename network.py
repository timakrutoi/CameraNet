import torch
import torch.nn as nn

from kornia.color import rgb_to_xyz, xyz_to_rgb


class RestoreNetBlock(nn.Module):
    def __init__(self, in_c, out_c,
                 kernel=3, stride=1,
                 padding=1, n=3):
        super(RestoreNetBlock, self).__init__()

        self.block = nn.Sequential()

        for _ in range(n):
            self.block.append(nn.Conv2d(in_c, out_c,
                                        kernel_size=kernel,
                                        stride=stride,
                                        padding=padding))
            in_c = out_c
            self.block.append(nn.ReLU())

    def forward(self, x):
        return self.block(x)


class EnhanceNetBlock(nn.Module):
    def __init__(self, in_c, out_c,
                 kernel=3, stride=1,
                 padding=1, n=3):
        super(EnhanceNetBlock, self).__init__()

        self.block = nn.Sequential()

        for _ in range(n):
            self.block.append(nn.Conv2d(in_c, out_c,
                                        kernel_size=kernel,
                                        stride=stride,
                                        padding=padding))
            in_c = out_c
            # need to add some AdaptiveBatchNorm
            # self.block.append(ABN())
            self.block.append(nn.ReLU())

    def forward(self, x):
        # skip connections
        x = self.block[0](x) + x.clone()
        return self.block[1:](x)


class GlobalComponent(nn.Module):
    def __init__(self, n=512):
        super(GlobalComponent, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lin = nn.Sequential()
        self.lin.append(nn.Linear(n, n))
        self.lin.append(nn.Linear(n, n))
        self.lin.append(nn.Linear(n, n))

    def forward(self, x):
        x = self.pool(x)
        b, c, _, _ = x.shape
        return self.lin(x.reshape(b, c))


class UNetBlock(nn.Module):
    def __init__(self, first_block, second_block,
                dilation=1):
        super(UNetBlock, self).__init__()

        self.fb = first_block
        self.sb = second_block

    def forward(self, x):
        return self.sb(self.fb(x))


class UNet(nn.Module):
    def __init__(self, block):
        super(UNet, self).__init__()
        dims = [3, 32, 64, 128, 256, 512]

        self.down = nn.Sequential()
        for i, j in zip(dims[:-1], dims[1:]):
            self.down.append(UNetBlock(first_block=block(i, j),
                                      second_block=nn.MaxPool2d(2)))

        self.mid = block(in_c=512, out_c=512)

        # GlobalComponent
        self.global_component = GlobalComponent(n=512)

        dils = [1, 2, 2, 4, 8]
        self.up = nn.Sequential()
        for i, j, k in zip(dims[:0:-1], dims[-2::-1], dils):
            self.up.append(UNetBlock(dilation=k,
                                     first_block=nn.Upsample(scale_factor=2),
                                     second_block=block(i, j)))

    def forward(self, x):
        # weird system with lists,
        # but i didnt come up with anything better
        d = [x]
        for i in self.down:
            d.append(i(d[-1]))

        m = self.global_component(d[-1])
        # ???
        m = torch.einsum('bcij,bc->bcij', self.mid(d[-1]), m)

        u = [m]
        for idx, i in enumerate(self.up):
            # skip connections
            a = i(u[-1])
            print(a.shape, d[-idx - 2].shape)
            u.append(a + d[-idx - 2])

        return u[-1]


class CameraNet(nn.Module):
    def __init__(self):
        super(CameraNet, self).__init__()

        # RGB to XYZ transform
        self.rgb2xyz = rgb_to_xyz
        self.restore = UNet(block=RestoreNetBlock)
        # XYZ to RGB transform
        self.xyz2rgb = xyz_to_rgb
        self.enhance = UNet(block=RestoreNetBlock)

    def forward(self, x):
        x = self.rgb2xyz(x)
        x = self.restore(x)

        x = self.xyz2rgb(x)
        x = self.enhance(x)

        return x


if __name__ == '__main__':
    net = CameraNet()

    print(net)
