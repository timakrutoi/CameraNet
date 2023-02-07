import torch
import torch.nn as nn


class RestoreNetBlock(nn.Module):
    def __init__(self, in_c, out_c,
                 kernel=kernel,
                 stride=stride,
                 padding=padding, n=3):
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
                 kernel=kernel,
                 stride=stride,
                 padding=padding, n=3):
        super(EnhanceNetBlock, self).__init__()

        self.block = nn.Sequential()

        for _ in range(n):
            self.block.append(nn.Conv2d(in_c, out_c,
                                        kernel=kernel,
                                        stride=stride,
                                        padding=padding))
            # need to add some ActivatedBatchNorm
            # self.block.append(ABN())
            self.block.append(nn.ReLU())

    def forward(self, x):
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
        self.net = nn.Sequential()
        
        # 1
        self.net.append(Block(in_c=32, out=))
        self.net.append(nn.MaxPool2d(2))
        # 2
        self.net.append(Block(in_c=64, out=))
        self.net.append(nn.MaxPool2d(2))
        # 3
        self.net.append(Block(in_c=128, out=))
        self.net.append(nn.MaxPool2d(2))
        # 4
        self.net.append(Block(in_c=256, out=))
        self.net.append(nn.MaxPool2d(2))
        # 5
        self.net.append(Block(in_c=512, out=))
        # self.net.append(UpSampling(2))
        
        # GlobalComponent
        self.net.append(GlobalComponent())
        # Add scaling
        # self.net.append(Scale())
        
        # 6
        self.net.append(Block(in_c=256, out=))
        # self.net.append(UpSampling(2))
        # 7
        self.net.append(Block(in_c=128, out=))
        # self.net.append(UpSampling(2))
        # 8
        self.net.append(Block(in_c=64, out=))
        # self.net.append(UpSampling(2))
        # 9
        self.net.append(Block(in_c=32, out=))

    def forward(self, x):

        return x

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
        # self.rgb2xyz = ...
        self.restore = UNet(block=RestoreNetBlock)
        # XYZ to RGB transform
        # self.xyz2rgb = ...
        self.enhance = UNet(block=RestoreNetBlock)

    def forward(self, x):
        return x


if __name__ == '__main__':
    net = DeepISP(2, 2)

    print(net.lowlevel)
    print(net.highlevel)
