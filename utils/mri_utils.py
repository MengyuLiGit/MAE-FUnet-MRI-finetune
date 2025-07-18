import nibabel as nib
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from .general_utils import (fixPath, load_sorted_directory, sort_list_natural_keys, get_time_interval,
                            check_if_dir_list, check_if_dir, clear_string_char, get_file_extension)
def conbine_mag_phase_swi(mag_file, phase_file):
    ### combine the mag and phase of swi mri
    # Load the magnitude and phase images
    mag_nii = nib.load(mag_file)  # Replace with your magnitude image file
    phase_nii = nib.load(phase_file)  # Replace with your phase image file

    # Convert to numpy arrays
    mag_data = mag_nii.get_fdata()
    phase_data = phase_nii.get_fdata()

    # Check if dimensions match
    if mag_data.shape != phase_data.shape:
        raise ValueError("Magnitude and phase images must have the same dimensions.")

    # Step 1: Normalize phase to [-π, π] (critical for SWI)
    # (Assuming phase is stored in radians; adjust scaling if necessary)
    phase_normalized = (phase_data - np.min(phase_data)) / (np.max(phase_data) - np.min(phase_data))
    phase_normalized = phase_normalized * 2 * np.pi - np.pi  # Scale to [-π, π]

    # Step 2: Create a phase mask (high-pass filtering to remove low-frequency artifacts)
    # High-pass filter using a Gaussian kernel subtraction
    sigma = 3  # Adjust based on your data (larger sigma = stronger filtering)
    low_pass_phase = gaussian_filter(phase_normalized, sigma=sigma)
    high_pass_phase = phase_normalized - low_pass_phase

    # Step 3: Generate SWI phase mask (enhance susceptibility-weighted contrast)
    # Thresholding and exponentiation (common in SWI pipelines)
    phase_mask = np.clip(high_pass_phase, -np.pi, np.pi)  # Ensure phase is within valid range
    phase_mask = (np.cos(phase_mask) + 1) ** 4  # Amplify phase effects (adjust exponent as needed)

    # Step 4: Combine magnitude and phase mask to create SWI
    swi_data = mag_data * phase_mask

    return swi_data

def load_nifti(path):
    """
    Load nifti given path
    :param path: file path string
    :return: nifti datasets, return None if no nifti file is loaded
    """
    filepath_temp = fixPath(path)

    if os.path.exists(filepath_temp):
        try:
            ds = nib.load(filepath_temp)
            return ds
        except Exception:
            return None  # or you could use 'continue'
    else:
        return None


import numpy as np
from scipy.ndimage import zoom


def resize_3d_mask(mask, target_shape):
    # Compute zoom factors per dimension
    zoom_factors = [t / s for t, s in zip(target_shape, mask.shape)]

    # Resize using nearest-neighbor interpolation (order=0)
    resized = zoom(mask, zoom=zoom_factors, order=0)
    return resized


import numpy as np


def zscore_normalize_with_mask(image, mask, clip_range=[0, 99.9]):
    """
    Z-score normalize the image using the values inside the mask.

    Parameters:
    - image: 3D numpy array (MRI)
    - mask:  3D numpy array (same shape), where brain voxels == 1
    - clip_range: tuple (min, max), optional intensity range to clip after normalization

    Returns:
    - normalized image (same shape)
    """
    # Extract brain region voxels
    brain_voxels = image[mask > 0]
    # Compute mean and std from brain voxels only
    mean = np.mean(brain_voxels)
    std = np.std(brain_voxels)

    # Normalize entire image using brain stats
    normalized = (image - mean) / std

    # Optional clipping
    if clip_range is not None:
        lower_percentile, upper_percentile = np.percentile(normalized, clip_range)
        normalized = np.clip(normalized, lower_percentile, upper_percentile)

    return normalized


def max_min_normalize(image, clip_range=[0, 99.9]):
    """
    Z-score normalize the image using the values inside the mask.

    Parameters:
    - image: 3D numpy array (MRI)
    - mask:  3D numpy array (same shape), where brain voxels == 1
    - clip_range: tuple (min, max), optional intensity range to clip after normalization

    Returns:
    - normalized image (same shape)
    """

    # Optional clipping
    if clip_range is not None:
        lower_percentile, upper_percentile = np.percentile(image, clip_range)
        image = np.clip(image, lower_percentile, upper_percentile)

    minimum = image.min()
    maximum = image.max()
    scale = maximum - minimum
    if scale > 0:
        scale_co = 1. / scale
    else:
        scale_co = 0
    normalized = (image - minimum) * scale_co

    return normalized


import numpy as np
from scipy.ndimage import zoom





def zoom_to_shape(image, target_shape, order=3):
    """
    Resize an N-dimensional image to a specific shape using zoom.

    Parameters:
    - image: np.ndarray, 3D or 4D MRI data
    - target_shape: tuple of ints, desired output shape
    - order: int, interpolation order (3 = cubic, 0 = nearest for masks)

    Returns:
    - Resized image with shape matching target_shape
    """
    current_shape = image.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    resized = zoom(image, zoom_factors, order=order)
    return resized


import torch
import torch.nn.functional as F


import numpy as np
import torch
import torch.nn.functional as F
def resize_3d_torch(volume: np.ndarray, target_shape: tuple):
    dtype_map = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.float32,   # Force int → float32
        np.int64: torch.float32,   # Force int → float32
        np.uint8: torch.float32    # Force uint8 → float32
    }

    np_dtype = volume.dtype.type
    torch_dtype = dtype_map.get(np_dtype, torch.float32)

    tensor = torch.tensor(volume, dtype=torch_dtype).unsqueeze(0).unsqueeze(0)

    mode = 'trilinear' if torch_dtype.is_floating_point else 'nearest'

    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode='nearest',  # Always nearest for mask
        align_corners=None
    )

    resized_np = resized[0, 0].cpu().numpy()

    if np.issubdtype(volume.dtype, np.integer):
        resized_np = np.round(resized_np).astype(volume.dtype)  # 🔥 round and cast back
    else:
        resized_np = resized_np.astype(volume.dtype)

    return resized_np


def resize_and_pad_3d(image, target_shape):
    current_shape = np.array(image.shape)
    target_shape = np.array(target_shape)

    scale = np.min(target_shape / current_shape)
    new_shape = np.round(current_shape * scale).astype(int)

    resized = resize_3d_torch(image, tuple(new_shape))

    pad_total = target_shape - new_shape
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    pad_width = list(zip(pad_before, pad_after))

    padded = np.pad(resized, pad_width, mode='constant', constant_values=0)

    return padded


import numpy as np
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np

def resize_2d_torch(image: np.ndarray, target_shape: tuple):
    """
    Resize a 2D numpy array using PyTorch interpolation, preserving original dtype.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {image.shape}")

    is_integer = np.issubdtype(image.dtype, np.integer)

    # Always cast to float32 for interpolation
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    mode = 'bilinear' if not is_integer else 'nearest'

    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode=mode,
        align_corners=False if mode == 'bilinear' else None
    )

    resized_np = resized[0, 0].cpu().numpy()

    if is_integer:
        resized_np = np.round(resized_np).astype(image.dtype)
    else:
        resized_np = resized_np.astype(image.dtype)

    return resized_np


def resize_and_pad_xy_only(image: np.ndarray, target_shape_xy: tuple):
    """
    Resize and pad the x, y plane of a 2D or 3D image to target_shape_xy,
    preserving aspect ratio and centering result with zero padding.
    """
    if image.ndim == 2:
        x, y = image.shape
        z = None
    elif image.ndim == 3:
        x, y, z = image.shape
    else:
        raise ValueError("Input must be a 2D or 3D array")

    target_x, target_y = target_shape_xy
    scale = min(target_x / x, target_y / y)
    new_x, new_y = int(round(x * scale)), int(round(y * scale))

    if image.ndim == 2:
        resized = resize_2d_torch(image, (new_x, new_y))
        pad_x = target_x - new_x
        pad_y = target_y - new_y
        pad_before = (pad_x // 2, pad_y // 2)
        pad_after = (pad_x - pad_before[0], pad_y - pad_before[1])
        pad_width = list(zip(pad_before, pad_after))
    else:
        resized = resize_3d_torch(image, (new_x, new_y, z))
        pad_x = target_x - new_x
        pad_y = target_y - new_y
        pad_before = (pad_x // 2, pad_y // 2, 0)
        pad_after = (pad_x - pad_before[0], pad_y - pad_before[1], 0)
        pad_width = list(zip(pad_before, pad_after))

    padded = np.pad(resized, pad_width, mode='constant', constant_values=0)
    return padded


import numpy as np
import torch
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F

def resize_with_zoom_factor_torch(volume: np.ndarray, zoom_factors: tuple):
    """
    Resize a 3D or 4D volume using PyTorch with zoom factors, preserving original dtype.

    Parameters:
    - volume: np.ndarray of shape (D, H, W) or (D, H, W, T)
    - zoom_factors: tuple of zoom factors, e.g., (0.5, 0.5, 1.0) for 3D, or (0.5, 0.5, 1.0, 1.0) for 4D

    Returns:
    - Resized np.ndarray with original dtype
    """
    if volume.ndim not in [3, 4]:
        raise ValueError(f"Only 3D or 4D input supported, got shape {volume.shape}")

    original_shape = volume.shape
    if len(zoom_factors) != len(original_shape):
        raise ValueError("zoom_factors must match the number of dimensions of the input volume")

    target_shape = tuple(int(round(s * z)) for s, z in zip(original_shape, zoom_factors))

    is_integer = np.issubdtype(volume.dtype, np.integer)

    # Always convert to float32 for interpolation
    tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W] or [1,1,D,H,W,T]

    if volume.ndim == 3:
        interp_shape = target_shape  # (D, H, W)
        resized = F.interpolate(
            tensor,
            size=interp_shape,
            mode='nearest',
            align_corners=None
        )
        resized_np = resized[0, 0].cpu().numpy()

    elif volume.ndim == 4:
        d, h, w, t = volume.shape
        resized_np_list = []
        for i in range(t):
            resized_t = F.interpolate(
                tensor[..., i],
                size=target_shape[:3],
                mode='nearest',
                align_corners=None
            )
            resized_np_list.append(resized_t[0, 0].cpu().numpy())
        resized_np = np.stack(resized_np_list, axis=-1)

    # Important: cast back
    if is_integer:
        resized_np = np.round(resized_np).astype(volume.dtype)
    else:
        resized_np = resized_np.astype(volume.dtype)

    return resized_np



def cast_to_255_png(input_array: np.ndarray):
    scale = input_array.max() - input_array.min()
    if scale > 1e-14:
        return ((input_array[:, :,input_array.shape[2] // 2] - input_array.min()) / scale * 255).astype(np.uint8)
    else:
        return (input_array[:, :,input_array.shape[2] // 2] * 0).astype(np.uint8)
