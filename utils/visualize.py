import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.help_func import print_var_detail
from utils.data_utils import img_augment



def run_one_image_mae(
        dataloader,
        idx_sample,
        net,
        device,
        mask_ratio,
):
    net = net.to(device)
    net.eval()

    for idx, data in enumerate(dataloader):
        if idx != idx_sample:
            continue
        img, _ = data
        img = img.to(device).float()
        _, pred, mask = net(img, mask_ratio)

        y = net.unpatchify(pred)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, net.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3(ch))
        mask = net.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', img).detach().cpu()

        # masked image
        img_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        img_paste = x * (1 - mask) + y * mask

        return x, y, img_masked, img_paste  # target, reconstruction, masked img, pasted img

def run_one_image_mae_gan(
        dataloader,
        idx_sample,
        net,
        device,
        mask_ratio,
):
    net = net.to(device)
    net.eval()

    for idx, data in enumerate(dataloader):
        if idx != idx_sample:
            continue
        img, _ = data
        img = img.to(device).float()
        _, pred, mask, _, _, _ = net(img, mask_ratio)

        y = net.unpatchify(pred)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, net.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3(ch))
        mask = net.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', img).detach().cpu()

        # masked image
        img_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        img_paste = x * (1 - mask) + y * mask

        return x, y, img_masked, img_paste  # target, reconstruction, masked img, pasted img
