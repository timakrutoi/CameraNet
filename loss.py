import torch
import torch.nn as nn

from kornia.color import rgb_to_lab
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM


class RestoreNetLoss():
    def __init__(self, alpha=0.5, device='cuda'):
        self.alpha = alpha
        self.eps = 1e-3
        self.l1 = nn.L1Loss()
        self.MSSSIM = MSSSIM().to(torch.device(device))

    def __call__(self, x, t):
        
        l = self.l1(x, t)
        l += ((x.max() + self.eps).log() - (t.max() + self.eps).log()).abs()
#         l *= self.alpha
#         l += (1 - self.alpha) *  (self.MSSSIM(x, t))
        return l


class EnhanceNetLoss():
    def __init__(self):
        self.l1 = nn.L1Loss()

    def __call__(self, x, target):
        return self.l1(x, target)


def rgb2lab(rgb):
#    rgb += 0.5 # assumin input is in [-0.5,0.5]
    rgb = rgb.clip(0, 1)
#    rgb = T.switch(T.gt(rgb,0.04045), ((rgb+0.055)/1.055)**2.4, rgb/12.92)
    rgb *= 100

    x = rgb[:,0:1,:,:] * 0.4124 + rgb[:,1:2,:,:] * 0.3576 + rgb[:,2:3,:,:] * 0.1805
    y = rgb[:,0:1,:,:] * 0.2126 + rgb[:,1:2,:,:] * 0.7152 + rgb[:,2:3,:,:] * 0.0722
    z = rgb[:,0:1,:,:] * 0.0193 + rgb[:,1:2,:,:] * 0.1192 + rgb[:,2:3,:,:] * 0.9505

    x /= 95.047
    y /= 100.0
    z /= 108.883


    def f(x):
        return torch.where(x > 0.008856, x**(1/3), 7.787*x + 16/116)

    x = f(x)
    y = f(y)
    z = f(z)

    L = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return torch.cat([L,a,b], dim=1)


class deepISPloss():
    def __init__(self, device='cpu', alpha=0.5):
        self.alpha = alpha
        self.l1 = torch.nn.L1Loss()
        self.MSSSIM = MSSSIM().to(torch.device(device))

    def __call__(self, x, t):
        lab_x = rgb_to_lab(x)
        lab_t = rgb_to_lab(t)
#         print('\t', x.min(), x.max())
#         print('\t', t.min(), t.max())
#         print(lab_x.min())
#         print(lab_tar.min())
#         print(x.isnan().sum())
#         print(t.isnan().sum())
#         print(lab_x.isnan().sum())
#         print(lab_t.isnan().sum())
#         print('-'*30)

        res = (1 - self.alpha) * self.l1(lab_x, lab_t)
        # take only first channel to MS-SSIM
        # turned ssim off coz it doesnt work
#         res +=     self.alpha  * (self.MSSSIM(lab_x[:, :1, :, :],
#                                               lab_tar[:, :1, :, :]))

        return res
    