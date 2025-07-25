import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import torch
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

ssim_tensor = StructuralSimilarityIndexMeasure()
psnr_tensor = PeakSignalNoiseRatio()


def get_error_map(target, pred):
    error = abs(target - pred)
    return error


def calc_nmse_tensor(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    '''
    tensor, [N,H,W]
    '''

    return torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2


def calc_psnr_tensor(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    '''
    tensor, [N,H,W]
    '''

    return psnr_tensor(pred, gt)


def calc_ssim_tensor(gt, pred):
    """Compute Structural Similarity Index Metric (SSIM)"""
    '''
    tensor, [Nc,H,W]
    '''

    if not gt.dim() == pred.dim():
        raise ValueError("Ground truth dimensions does not match pred.")

    ssim = ssim_tensor(pred, gt)  # .unsqueeze(0) to [B,Nc,H,W]

    return ssim


def volume_nmse_tensor(gt, pred):
    """Volume Normalized Mean Squared Error (NMSE)"""
    '''
    tensor, [N,H,W]
    '''

    return torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2


def volume_psnr_tensor(gt, pred, maxval=None):
    """Volume Peak Signal to Noise Ratio metric (PSNR)"""
    '''
    tensor, [N,H,W]
    '''
    if maxval is None:
        maxval = gt.max() - gt.min()

    return psnr_tensor(pred, gt, data_range=maxval)


def volume_ssim_tensor(gt, pred, maxval=None):
    """Volume Structural Similarity Index Metric (SSIM)"""
    '''
    tensor, [N,H,W]
    '''
    if not gt.dim() == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.dim() == pred.dim():
        raise ValueError("Ground truth dimensions does not match pred.")

    if maxval is None:
        maxval = gt.max() - gt.min()

    ssim = ssim_tensor(pred.unsqueeze(0), gt.unsqueeze(0), data_range=maxval)  # .unsqueeze(0) to [N,1,H,W]

    return ssim


def calc_psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max() - gt.min()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def calc_ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() - gt.min() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]


def binaryMaskIOU(gt_01: torch.tensor, pred_01: torch.tensor):
    mask1_area = np.count_nonzero(gt_01 == 1)  # I assume this is faster as mask1 == 1 is a bool array
    mask2_area = np.count_nonzero(pred_01 == 1)
    intersection = np.count_nonzero(np.logical_and(gt_01, pred_01))
    if mask1_area == 0:
        iou = None
    else:
        iou = intersection / (mask1_area + mask2_area - intersection)
    return iou
