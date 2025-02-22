import random

import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F


def CutOut(length=16, img_shape=(112, 112)):
    h, w = img_shape
    mask = np.ones((h, w), np.float32)
    x, y = np.random.randint(w), np.random.randint(h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x - length // 2, 0, w)
    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    mask[y1:y2, x1:x2] = 0.
    return mask


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def GaussianSmooth(x, kernel_size, nsig=3, device='cuda'):
    if kernel_size == 1:
        return x
    kernel = gkern(kernel_size, nsig).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    weight = torch.from_numpy(stack_kernel).permute([2, 3, 0, 1]).to(device)
    padv = (kernel_size - 1) // 2
    x = F.pad(x, pad=(padv, padv, padv, padv), mode='replicate')
    src = F.conv2d(
        x, weight, bias=None, stride=1, padding=0,
        dilation=1, groups=3)
    return src


class GaussianSmoothConv(torch.nn.Module):
    def __init__(self, kernel_size, nsig, device='cuda', channel=3):
        super(GaussianSmoothConv, self).__init__()
        self.kernel_size = kernel_size
        kernel = gkern(kernel_size, nsig).astype(np.float32)
        stack_kernel = np.stack([kernel]*channel).swapaxes(2, 0)
        self.channel = channel
        stack_kernel = np.expand_dims(stack_kernel, -1)
        self.weight = torch.from_numpy(stack_kernel).permute([2, 3, 0, 1]).to(device)

    def forward(self, x):
        if self.kernel_size == 1:
            return x
        return F.conv2d(x, self.weight.data, padding='same', groups=self.channel)


def affine(x, vgrid, device='cuda'):
    output = F.grid_sample(x, vgrid)
    # mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    # mask = F.grid_sample(mask, vgrid)
    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1
    return output


def warp(x, flo, device='cuda'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = flo.size()
    H_in, W_in = x.size()[-2:]
    vgrid = torch.rand((B, 2, H, W)).to(device)
    # mesh grid

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * flo[:, 0, :, :].clone() / max(W_in - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * flo[:, 1, :, :].clone() / max(H_in - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    return affine(x, vgrid)


def WarpPerspective(x, tmatrix, out_H=None, out_W=None, dstsize=None, device='cuda', inverse=False):
    '''
    formulation: http://www.cnblogs.com/zipeilu/p/6138423.html
    input:
        x(torch.tensor): NxCxHxW
        tmatrix(numpy.array or list): 3x3
    output:
        warp_res(torch.tensor): NxCxHxW
    '''

    assert (len(x.size()) == 4)

    if inverse:
        tmatrix = np.linalg.inv(tmatrix)
    H, W = x.size()[2:]
    if out_H is None and out_W is None:
        out_H, out_W = H, W
    if dstsize is not None:
        out_H, out_W = dstsize

    flow = torch.zeros(2, out_H, out_W).to(device)
    identity = torch.ones(out_H, out_W).to(device)
    xx = torch.arange(0, out_W).view(1, -1).repeat(out_H, 1).type_as(identity).to(device)
    yy = torch.arange(0, out_H).view(-1, 1).repeat(1, out_W).type_as(identity).to(device)
    _A = (tmatrix[1][1] - tmatrix[2][1] * yy)
    _B = (tmatrix[2][2] * xx - tmatrix[0][2])
    _C = (tmatrix[0][1] - tmatrix[2][1] * xx)
    _D = (tmatrix[2][2] * yy - tmatrix[1][2])
    _E = (tmatrix[0][0] - tmatrix[2][0] * xx)
    _F = (tmatrix[1][0] - tmatrix[2][0] * yy)
    xa = _A * _B - _C * _D
    xb = _A * _E - _C * _F
    ya = _F * _B - _E * _D
    yb = _F * _C - _E * _A
    flow[0] = xa / xb
    flow[1] = ya / yb
    flow = flow.view(1, 2, out_H, out_W).repeat(x.size(0), 1, 1, 1)
    return warp(x, flow, device=device)


class WarpFunction(Function):

    @staticmethod
    def forward(ctx, input, matrix, dstsize=None):
        ctx.save_for_backward(input, torch.from_numpy(matrix))
        return WarpPerspective(input, matrix, dstsize=dstsize)

    @staticmethod
    def backward(ctx, grad_output):
        input, matrix = ctx.saved_variables
        dstsize = input.size()[-2:]
        return WarpPerspective(grad_output, matrix.cpu().numpy(), dstsize=dstsize, inverse=True), None, None


def Resize(x, device='cuda'):
    '''
    input:
        x: (N, 299, 299)
    output:
        (N, 224, 224)
    '''
    scale_factor = 2.0 / 223
    N = x.size(0)
    grid = torch.zeros((N, 224, 224, 2))
    grid[:, :, :, 0] = torch.arange(0, 224, dtype=torch.float32).view((1, 1, 224)).repeat(N, 224, 1) * scale_factor - 1
    grid[:, :, :, 1] = torch.arange(0, 224, dtype=torch.float32).view((1, 224, 1)).repeat(N, 1, 224) * scale_factor - 1
    grid = grid.to(device)
    x = x.to(device)
    return affine(x, grid, device=device)


def RandomCrop(x, device='cuda'):
    '''
    input:
        x: (N, 299, 299)
    output:
        (N, 224, 224)
    '''
    scale_factor = 2.0 / 223
    N = x.size(0)
    grid = torch.zeros((N, 224, 224, 2))
    start = torch.randint(0, (299 - 224) / 2, (N, 2))
    sx = start[:, 0].view(N, 1, 1).float()
    sy = start[:, 1].view(N, 1, 1).float()
    grid[:, :, :, 0] = (sx + torch.arange(0, 224, dtype=torch.float32).view((1, 1, 224)).repeat(N, 224,
                                                                                                1)) * scale_factor - 1
    grid[:, :, :, 1] = (sy + torch.arange(0, 224, dtype=torch.float32).view((1, 224, 1)).repeat(N, 1,
                                                                                                224)) * scale_factor - 1
    grid = grid.to(device)
    x = x.to(device)
    return affine(x, grid, device=device)


def Resize_and_padding(x, scale_factor):
    ori_size = x.size()[-2:]
    x = F.interpolate(x, scale_factor=scale_factor)
    new_size = x.size()[-2:]

    delta_w = ori_size[1] - new_size[1]
    delta_h = ori_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    x = F.pad(x, pad=(left, right, top, bottom), value=1.)
    return x


def DI_diversity(image, low=1., high=1.1071, div_prob=0.5, constant=None):
    """
    :param constant:
    :param low:
    :param image:
    :param high: default : floor(331 / 299)
    :param div_prob:
    :return:
    """
    if random.random() > div_prob:
        return image
    if constant is None:
        # keep same as tensorflow implementation
        # pad with gray
        constant = torch.mean(image).item()
    low = int(image.size(-1) * low)
    high = int(image.size(-1) * high)

    rnd = random.randint(low, high)
    rescaled = F.interpolate(image, size=[rnd, rnd])
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', constant)
    return F.interpolate(padded, size=[low, low])


def Rotate(x, theta, device='cuda'):
    rotation = np.zeros((2, 3, x.size(0)))
    cos = np.cos(theta).ravel()
    sin = np.sin(theta).ravel()
    rotation[0, 0] = cos
    rotation[0, 1] = sin
    rotation[1, 0] = -sin
    rotation[1, 1] = cos
    rotation = torch.Tensor(rotation.transpose((2, 0, 1))).to(device)
    grid = F.affine_grid(rotation, size=x.size())
    return affine(x, grid, device)


def mask_diversity(mask, noise, dropout=None, shift=None, keep_prob=0.5):
    if shift is None or shift == 0:
        return mask, noise
    if dropout is not None and 0 < dropout < 1:
        # failed ! Do not use this
        keeps = torch.rand(*mask.shape)
        mask[keeps < dropout] = 0

    if shift is not None:
        top = np.random.randint(low=-shift, high=shift)
        left = np.random.randint(low=-shift, high=shift)

        orgi_w, orig_h = mask.shape[2], mask.shape[3]
        _mask = F.pad(mask, (shift, shift, shift, shift), mode='constant', value=0)
        _noise = F.pad(noise, (shift, shift, shift, shift), mode='constant', value=0)

        if left >= 0:
            _noise = _noise[:, :, :, left:left + orig_h]
            _mask = _mask[:, :, :, left:left + orig_h]
        else:
            _noise = _noise[:, :, :, left - orig_h:left]
            _mask = _mask[:, :, :, left - orig_h:left]
        if top >= 0:
            _noise = _noise[:, :, top:top + orgi_w, :]
            _mask = _mask[:, :, top:top + orgi_w, :]
        else:
            _noise = _noise[:, :, top - orgi_w:top, :]
            _mask = _mask[:, :, top - orgi_w:top, :]
        mask, noise = (_mask, _noise) if np.random.uniform(size=()) > keep_prob else (mask, noise)

    return mask, noise


def input_diversity(x, std_proj=None, std_rotate=None, scale=1.0, cutout=False, shift=None, keep_prob=0.5,
                    device='cuda', ):
    """

    :param x:
    :param std_proj:
    :param std_rotate:
    :param scale:
    :param cutout:
    :param shift:
    :param keep_prob:
    :param device:
    :return:
    """

    if std_proj is not None and np.random.uniform(size=()) > keep_prob:
        n = x.size(0)
        M = np.tile(np.array([[1, 0, 0], [0, 1, 0]]), (n, 1, 1)) + np.random.normal(scale=std_proj, size=(n, 2, 3))
        M = torch.Tensor(M)
        grid = F.affine_grid(M, x.size()).to(device)
        x = affine(x, grid, device=device)

    if std_rotate is not None and np.random.uniform(size=()) > keep_prob:
        n = x.size(0)
        theta = np.random.normal(scale=std_rotate, size=(n, 1))
        _x = Rotate(x, theta, device=device)
        x = _x

    if shift is not None and np.random.uniform(size=()) > keep_prob:
        left = np.random.randint(0, 2 * shift)
        top = np.random.randint(0, 2 * shift)
        H, W = x.shape[-2:]
        x = F.pad(x, (shift, shift), mode='constant', value=0)
        x = x[:, :, top:top + H, left:left + W]

    x = x * scale if np.random.uniform(size=()) > keep_prob else x
    if cutout:
        mask = torch.from_numpy(CutOut(img_shape=x.shape[-2:])).to(device).expand_as(x)
        x = mask * x
    return x


def GammaTransform(x, r, keep_prob=0.5):
    """

    :param x: 图片batch， 像素范围0～255
    :param r: gamma变换的ratio。 1不变，大于1变暗，小于1变亮。 参考：0.3
    :param keep_prob: 以多大概率变。 1 不变换 ； 0 绝对变换
    :return:
    """
    x = x
    if np.random.uniform(size=()) > keep_prob:
        idx = torch.where(x > 1. / 255)
        x[idx] = torch.pow(x[idx], r)  # avoid gradient exploding or vanishing
        idx2 = torch.where(x <= 1. / 255)
        x[idx2] = x[idx2] / r ** 2
    return x


def random_warp(x, r, keep_prob=0.5):
    """

    :param x:
    :param r: 参考值 0.1， 不建议大于0.15
    :param keep_prob:
    :return:
    """
    if np.random.uniform(size=()) > keep_prob:
        n = x.size(0)
        M = np.tile(np.array([[1, 0, 0], [0, 1, 0]]), (n, 1, 1)) + np.random.normal(scale=r, size=(n, 2, 3))
        M = torch.Tensor(M)
        grid = F.affine_grid(M, x.size()).to(x.device)
        x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)
    return x


def input_diversity_v2(x, rotate=0, shearx=0, sheary=0, translatex=0, translatey=0, scale=0,
                       adjust_contrast=0, adjust_brightness=0, keep_prob=0.5):
    from utils.data_augmentation.transformations_sp import Rotate, ShearX, ShearY, \
        TranslateX, TranslateY,\
        Scale, AdjustContrast, AdjustBrightness

    x = Rotate(x, rotate) if rotate != 0 and random.random() > keep_prob else x
    x = ShearX(x, shearx) if shearx != 0 and random.random() > keep_prob else x
    x = ShearY(x, sheary) if sheary != 0 and random.random() > keep_prob else x
    x = TranslateX(x, translatex) if translatex != 0 and random.random() > keep_prob else x
    x = TranslateY(x, translatey) if translatey != 0 and random.random() > keep_prob else x
    x = Scale(x, scale) if scale != 0 and random.random() > keep_prob else x
    x = AdjustContrast(x, adjust_contrast) if adjust_contrast != 0 and random.random() > keep_prob else x
    x = AdjustBrightness(x, adjust_brightness) if adjust_brightness != 0 and random.random() > keep_prob else x
    x = torch.clamp(x, 0., 1.)
    return x

if __name__ == "__main__":
    import os, cv2

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    img = cv2.imread('ant_aligned/source/0.JPG')[:, :, ::-1].copy().astype(np.float32)
    h, w = img.shape[:2]
    pad = 10
    show_img = torch.zeros((1, 3, h, w * 3 + pad * 2))
    img = img[None, :].transpose((0, 3, 1, 2))
    img = torch.from_numpy(img).requires_grad_(True)
    show_img[:, :, :, :w] = img.clone()
    # M = np.tile(np.array([[1, 0, 0], [0, 1, 0]]), (1, 1, 1)) + np.random.normal(scale=0.05, size=(1, 2, 3))
    # M = torch.Tensor(M)
    # grid = F.affine_grid(M, img.size())
    # transform_img = affine(img, grid, device='cpu')

    transform_img = input_diversity(img, std_proj=0.1, device='cpu')
    show_img[:, :, :, w + pad: 2 * w + pad] = transform_img.clone()
    loss = torch.sum(transform_img ** 2)
    loss.backward()
    show_img[:, :, :, w * 2 + pad * 2: w * 3 + pad * 2] = img.grad.data.clone()
    show_img = torch.clamp(show_img, 0, 255)
    show_img = show_img[0].data.numpy().transpose(1, 2, 0).astype(np.uint8)
    imsave('show.png', show_img)


