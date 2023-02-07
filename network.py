import torch.nn as nn

from kornia.color import rgb_to_xyz, xyz_to_rgb


class RestoreNetBlock(nn.Module):
    def __init__(self, in_c, out_c,
                 kernel, stride,
                 padding, n=3):
        super(RestoreNetBlock, self).__init__()

        self.block = nn.Sequential()

        for _ in range(n):
            self.block.append(nn.Conv2d(in_c, out_c,
                                        kernel=kernel,
                                        stride=stride,
                                        padding=padding))
            self.block.append(nn.ReLU())

    def forward(self, x):
        return self.block(x)


class EnhanceNetBlock(nn.Module):
    def __init__(self, in_c, out_c,
                 kernel, stride,
                 padding, n=3):
        super(EnhanceNetBlock, self).__init__()

        self.block = nn.Sequential()

        for _ in range(n):
            self.block.append(nn.Conv2d(in_c, out_c,
                                        kernel=kernel,
                                        stride=stride,
                                        padding=padding))
            # need to add some AdaptiveBatchNorm
            # self.block.append(ABN())
            self.block.append(nn.ReLU())

    def forward(self, x):
        # skip connections
        x = self.block[0](x) + x.clone()
        return self.block[1:](x)


class GlobalComponent(nn.Module):
    def __init__(self):
        super(GlobalComponent, self).__init__()
        self.pool = GlobalPool2d()

        self.lin = nn.Sequential()
        self.lin.append(nn.Linear(512, 512))
        self.lin.append(nn.Linear(512, 512))
        self.lin.append(nn.Linear(512, 512))

    def forward(self, x):
        return self.lin(self.pool(x))


class UNet(nn.Module):
    def __init__(self, block):
        super(UNet, self).__init__()

        Block = block
        self.down_sample = nn.Sequential()

        # [TODO] this wont work coz forward func,
        # fix -> put both Block() ans Pool() in same seq elem
        # 1
        self.down_sample.append(Block(in_c=32, out=32))
        self.down_sample.append(nn.MaxPool2d(2))
        # 2
        self.down_sample.append(Block(in_c=64, out=64))
        self.down_sample.append(nn.MaxPool2d(2))
        # 3
        self.down_sample.append(Block(in_c=128, out=128))
        self.down_sample.append(nn.MaxPool2d(2))
        # 4
        self.down_sample.append(Block(in_c=256, out=256))
        self.down_sample.append(nn.MaxPool2d(2))
        # 5
        self.mid = Block(in_c=512, out=512)
        # self.net.append(UpSampling(2))

        # GlobalComponent
        self.global_component = GlobalComponent()
        # Add scaling
        # self.scaling = Scale()

        self.up_sample = nn.Sequential()
        # 6
        self.up_sample.append(Block(in_c=256, out=256))
        self.up_sample.append(UpSampling(2))
        # 7
        self.up_sample.append(Block(in_c=128, out=128))
        self.up_sample.append(UpSampling(2))
        # 8
        self.up_sample.append(Block(in_c=64, out=64))
        self.up_sample.append(UpSampling(2))
        # 9
        self.up_sample.append(Block(in_c=32, out=32))

    def forward(self, x):
        # weird system with lists,
        # but i didnt come up with anything better
        d = [x]
        for i in self.down_sample:
            d.append(i(d[-1]))

        m = self.global_component(d[-1])
        m = self.scaling(d[-1], m)

        u = [m]
        for idx, i in enumerate(self.up_sample):
            # skip connections
            u.append(i(d[-1]) + d[-idx + 1])

        return u[-1]


# [TODO] fix "Given input size: (64x1x1).
# Calculated output size: (64x0x0). Output size is too small"
# when nhl is too big for img
class GlobalPool2d(nn.Module):
    def __init__(self):
        super(GlobalPool2d, self).__init__()

    def forward(self, x):
        b, c, w, h = x.shape
        return nn.functional.avg_pool2d(x, kernel_size=(h, w)).reshape((b, c))


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
