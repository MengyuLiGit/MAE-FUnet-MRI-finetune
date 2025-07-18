from numpy.ma.core import masked_array

from utils.nacc_loader import NACCLoader
from utils.general_loader import PatientCase
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from help_func import print_var_detail
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.transform_util import rgb2gray, center_crop_with_pad_np
from utils.evaluation_utils import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor
import time
import random
import os
import copy
import pickle
import collections
from .general_utils import (fixPath, load_sorted_directory, sort_list_natural_keys, get_time_interval,
                            check_if_dir_list, check_if_dir, clear_string_char, natural_keys)

from collections import defaultdict
import os
import math
from multiprocessing import Pool, cpu_count
import skimage
from utils.transform_util import CustomRandomFlip, CustomRandomRot
import math
from scipy.ndimage import center_of_mass
from utils.transform_util import rotate_around_axis_position, find_mask_bounds, random_crop_given_bounds, \
    upscale_rotate_downscale_fast, upscale_rotate_downscale_binary
import SimpleITK as sitk
from utils.general_utils import get_file_extension
from utils.mri_utils import resize_3d_mask, zscore_normalize_with_mask, max_min_normalize, resize_and_pad_3d, \
    resize_and_pad_xy_only, zoom_to_shape, resize_with_zoom_factor_torch, resize_3d_torch, cast_to_255_png
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def remap_labels_fast(mask, label_groups):
    lookup = {}
    for class_idx, labels in enumerate(label_groups, start=1):
        for label in labels:
            lookup[label] = class_idx

    # Vectorized mapping
    vectorized_map = np.vectorize(lambda x: lookup.get(x, 0))
    new_mask = vectorized_map(mask).astype(np.int32)
    return new_mask


def get_all_pkl_paths(folder):
    pkl_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.pkl'):
                pkl_paths.append(os.path.join(root, file))
    return pkl_paths

def get_all_file_paths_with_extension(folder, extension, include_if_contains=None):
    """
    Recursively find all files with a given extension.
    Optionally filter to include only those whose full path contains any of the substrings in `include_if_contains`.

    Parameters:
        folder (str): Root folder to search
        extension (str): File extension (e.g., '.pkl')
        include_if_contains (list of str): Substrings to filter file paths (optional)

    Returns:
        List of full file paths
    """
    file_paths = []
    extension = extension.lower()

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extension):
                full_path = os.path.join(root, file)

                if include_if_contains:
                    if not any(substr in full_path for substr in include_if_contains):
                        continue  # skip if no match

                file_paths.append(full_path)

    return file_paths


import numpy as np
from skimage.transform import resize


def crop_and_resize_mri_random_buffer_auto(mri, mask_size_cord, min_buffer=10, mask_pixel_threshold=16 * 16):
    """
    Crop and resize MRI based on mask region only if mask area exceeds threshold.

    Parameters:
    - mri: 2D MRI image (H, W)
    - mask_size_cord: (mask size, y1, y2, x1, x2)
    - min_buffer: minimum padding buffer (in pixels)
    - mask_pixel_threshold: minimum number of mask pixels to trigger cropping

    Returns:
    - Resized MRI image (same shape as input)
    """
    # assert mri.shape == mask.shape, "MRI and mask must have same shape"
    H, W = mri.shape

    # mask_binary = (mask > 0)
    # num_mask_pixels = np.count_nonzero(mask_binary)

    mask_size, y1, y2, x1, x2 = mask_size_cord
    # If mask is too small, skip cropping
    if mask_size < mask_pixel_threshold:
        return mri.copy()

    # coords = np.argwhere(mask_binary)
    if y1 == -1:
        return mri.copy()

    # y1, x1 = coords.min(axis=0)
    # y2, x2 = coords.max(axis=0) + 1  # exclusive

    # Distance to image edge
    max_top    = y1
    max_bottom = H - y2
    max_left   = x1
    max_right  = W - x2

    # Sample buffer between min_buffer and distance to edge
    top_buffer    = np.random.randint(min_buffer, max(min_buffer, max_top) + 1)
    bottom_buffer = np.random.randint(min_buffer, max(min_buffer, max_bottom) + 1)
    left_buffer   = np.random.randint(min_buffer, max(min_buffer, max_left) + 1)
    right_buffer  = np.random.randint(min_buffer, max(min_buffer, max_right) + 1)

    # Apply padded crop
    y1 = max(0, y1 - top_buffer)
    y2 = min(H, y2 + bottom_buffer)
    x1 = max(0, x1 - left_buffer)
    x2 = min(W, x2 + right_buffer)
    if y2 <= y1 or x2 <= x1:
        # print(f"⚠️ After padding: Invalid crop range y:({y1}, {y2}), x:({x1}, {x2})")
        # print(f"mask_size_cord={mask_size_cord}, H={H}, W={W}")
        return mri.copy()
    cropped = mri[y1:y2, x1:x2]
    resized = resize_and_pad_xy_only(image=cropped, target_shape_xy=[H, W])

    return resized

def crop_and_resize_mri_mask_random_buffer_auto(mri, mask, mask_size_cord, min_buffer=10, mask_pixel_threshold=16*16):
    """
    Crop and resize MRI and its multi-class mask together based on mask region only if mask area exceeds threshold.

    Parameters:
    - mri: 2D MRI image (H, W)
    - mask: 2D multi-class mask (H, W)
    - mask_size_cord: (mask size, y1, y2, x1, x2)
    - min_buffer: minimum padding buffer (in pixels)
    - mask_pixel_threshold: minimum number of mask pixels to trigger cropping

    Returns:
    - Resized MRI image
    - Resized multi-class mask (aligned with MRI)
    """
    H, W = mri.shape
    mask_size, y1, y2, x1, x2 = mask_size_cord

    # If mask is too small or invalid, skip cropping
    if mask_size < mask_pixel_threshold or y1 == -1:
        return mri.copy(), mask.copy()

    # Distance to image edge
    max_top    = y1
    max_bottom = H - y2
    max_left   = x1
    max_right  = W - x2

    # Sample buffer between min_buffer and distance to edge
    top_buffer    = np.random.randint(min_buffer, max(min_buffer, max_top) + 1)
    bottom_buffer = np.random.randint(min_buffer, max(min_buffer, max_bottom) + 1)
    left_buffer   = np.random.randint(min_buffer, max(min_buffer, max_left) + 1)
    right_buffer  = np.random.randint(min_buffer, max(min_buffer, max_right) + 1)

    # Apply padded crop
    y1_crop = max(0, y1 - top_buffer)
    y2_crop = min(H, y2 + bottom_buffer)
    x1_crop = max(0, x1 - left_buffer)
    x2_crop = min(W, x2 + right_buffer)

    if y2_crop <= y1_crop or x2_crop <= x1_crop:
        return mri.copy(), mask.copy()

    # Crop both MRI and Mask
    cropped_mri = mri[y1_crop:y2_crop, x1_crop:x2_crop]
    cropped_mask = mask[y1_crop:y2_crop, x1_crop:x2_crop]

    # Resize both back to original shape
    resized_mri = resize_and_pad_xy_only(cropped_mri, target_shape_xy=[H, W])
    resized_mask = resize_and_pad_xy_only(cropped_mask, target_shape_xy=[H, W])

    return resized_mri, resized_mask

def random_flip_2d_controlled(array, p_horizontal=0.5, p_vertical=0.5):
    """
    Randomly flip a 2D array left-right and/or up-down with given probabilities.

    Parameters:
    - array: 2D NumPy array
    - p_horizontal: probability to flip left-right (0.0 to 1.0)
    - p_vertical: probability to flip up-down (0.0 to 1.0)
    """
    if random.random() < p_horizontal:
        array = np.fliplr(array)
    if random.random() < p_vertical:
        array = np.flipud(array)
    return array

def random_flip_2d_controlled_img_mask(img, mask, p_horizontal=0.5, p_vertical=0.5):
    """
    Randomly flip both image and mask left-right and/or up-down with given probabilities.

    Parameters:
    - img: 2D numpy array (MRI)
    - mask: 2D numpy array (multi-class mask)
    - p_horizontal: probability to flip horizontally
    - p_vertical: probability to flip vertically

    Returns:
    - flipped_img, flipped_mask
    """
    if random.random() < p_horizontal:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if random.random() < p_vertical:
        img = np.flipud(img)
        mask = np.flipud(mask)
    return img, mask

def random_flip_2d_controlled_torch(tensor, p_horizontal=0.5, p_vertical=0.5):
    """
    Randomly flip a 2D or 3D torch tensor along horizontal and/or vertical axes.

    Parameters:
    - tensor: torch.Tensor of shape [H, W] or [C, H, W]
    - p_horizontal: probability to flip left-right
    - p_vertical: probability to flip up-down

    Returns:
    - Flipped tensor
    """
    if tensor.ndim == 2:
        if random.random() < p_horizontal:
            tensor = torch.flip(tensor, dims=[1])  # flip left-right
        if random.random() < p_vertical:
            tensor = torch.flip(tensor, dims=[0])  # flip up-down
    elif tensor.ndim == 3:
        if random.random() < p_horizontal:
            tensor = torch.flip(tensor, dims=[2])  # flip left-right
        if random.random() < p_vertical:
            tensor = torch.flip(tensor, dims=[1])  # flip up-down
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    return tensor


def random_rotate_90_180_270(array):
    k = random.choice([1, 2, 3, -1, -2, -3])  # 90, 180, 270 degrees counterclockwise
    return np.rot90(array, k=k)

def random_rotate_90_180_270_img_mask(img, mask):
    """
    Randomly rotate both image and mask by 90, 180, or 270 degrees.

    Parameters:
    - img: 2D numpy array (MRI)
    - mask: 2D numpy array (multi-class mask)

    Returns:
    - rotated_img, rotated_mask
    """
    k = random.choice([1, 2, 3, -1, -2, -3])  # random multiple of 90 degrees
    img = np.rot90(img, k=k)
    mask = np.rot90(mask, k=k)
    return img, mask

def random_rotate_90_180_270_torch(tensor):
    """
    Randomly rotate a 2D or 3D PyTorch tensor by 90, 180, or 270 degrees (clockwise or counterclockwise).

    Parameters:
    - tensor: torch.Tensor of shape [H, W] or [C, H, W]

    Returns:
    - Rotated tensor
    """
    k = random.choice([1, 2, 3, -1, -2, -3])

    if tensor.ndim == 2:
        return torch.rot90(tensor, k=k, dims=[0, 1])
    elif tensor.ndim == 3:
        return torch.rot90(tensor, k=k, dims=[1, 2])
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def compute_mask_size_wrapper(args):
    """Unpack and call compute_mask_size (for Pool.map)"""
    return compute_mask_size(*args)

def compute_mask_size(mask_index, cache_type='pkl', img_folder = 'nifti_img', mask_folder = 'nifti_mask'):
    mask_path = mask_index[0].replace(img_folder, mask_folder)
    if os.path.exists(mask_path):
        if cache_type == 'pkl':
            with open(mask_path, 'rb') as f:
                mask = pickle.load(f)
                if len(mask.shape) != 2:
                    print('mask shape is {}'.format(mask.shape))
                    print(mask_path)

                H, W = mask.shape

                mask_binary = (mask > 0)
                # num_mask_pixels = np.count_nonzero(mask_binary)

                # # If mask is too small, skip cropping
                # if num_mask_pixels < mask_pixel_threshold:
                #     return mri.copy()

                coords = np.argwhere(mask_binary)
                if coords.size == 0:
                    return np.sum(mask > 0), -1, -1, -1, -1

                y1, x1 = coords.min(axis=0)
                y2, x2 = coords.max(axis=0) + 1  # exclusive

            return np.sum(mask > 0), y1, y2, x1, x2
    return 0, -1, -1, -1, -1

def process_chunk_parallel(index_chunk, num_workers=16, cache_type='pkl', img_folder = 'nifti_img', mask_folder = 'nifti_mask'):
    args_list = [(idx, cache_type, img_folder, mask_folder) for idx in index_chunk]
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(compute_mask_size_wrapper, args_list),
            total=len(index_chunk),
            desc="Computing mask sizes"
        ))
    return results

class GeneralDataset(Dataset):
    """
    General Dataset for training and validation of patient data such as ADNI and NACC.
    """

    def __init__(
            self,
            resize,
            drop_fraction,
            mri_sequence,
            patient_loader,
            labels,
            label_values,
            random_seed,
            val_split,
            is_train,
            if_latest,
            max_train_len=None,
            max_val_len=None,
            data_root='',
            cache_root='',
            cache_type='pkl',
            if_add_sequence = False,
    ):
        """
        General Dataset for training and validation of patient data such as ADNI and NACC.
        :param resize: int, resize image and keep proportion, fill the gap with black (0)
        :param drop_fraction: float, 0.1 means drop first and last 10% of the slices in a sequence
        :param mri_sequence: list of string, ['T1', 'T2'], if none, means use all sequences, otherwise use chosen sequences
        :param patient_loader: loader containing all filtered patient cases
        :param labels: list, image labels for prediction
        :param label_values: list, image label values for prediction
        :param random_seed: int, seed fix for random sample
        :param val_split: float, split fraction for validation set
        :param is_train: bool, whether form training or validation set
        :param if_latest: bool, whether to use the latest patient's data
        """
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.patient_loader = patient_loader
        self.num_patients = len(patient_loader.patients)
        self.data_root = data_root
        self.cache_root = cache_root
        self.cache_type = cache_type
        self.if_add_sequence = if_add_sequence
        self.index_list = self.get_index_list()
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len

        np.random.seed(self.random_seed)
        np.random.shuffle(self.index_list)

        if self.max_train_len is None:
            valPathsLen = int(len(self.index_list) * val_split)
            trainPathsLen = len(self.index_list) - valPathsLen
        elif self.max_train_len < len(self.index_list) - int(len(self.index_list) * val_split):
            valPathsLen = int(len(self.index_list) * val_split)
            trainPathsLen = self.max_train_len
        else:
            valPathsLen = 0
            trainPathsLen = 0
            print("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if self.max_val_len <= len(self.index_list) - trainPathsLen:
                valPathsLen = self.max_val_len
            else:
                print("assigned maximum val len exceeds total train num of slices")

        self.trainPaths = self.index_list[:trainPathsLen]
        self.valPaths = self.index_list[-valPathsLen:]
        print('train size: ' + str(len(self.trainPaths)))
        print('validation size: ' + str(len(self.valPaths)))

    # def get_index_list(self, if_latest=False):
    #     index_list_temp = []  # [[patient index, slice index per patient]], assume [0, 0, 0, 0] for slice index
    #     for i in tqdm(range(self.num_patients)):
    #         index_list_i = []  # [[0, 0, 0, 0], [0, 0, 0, 1]...[0, 0, 0, [0, 0]], [0, 0, 0, [0, 1]]]
    #         if self.mri_sequence is None:
    #             index_list_i = sum(self.patient_loader.patients[i].sequence_slice_index_lists_dict.values(), [])
    #         else:
    #             for dir_index in self.patient_loader.patients[i].dir_index_list:
    #                 for sequence_str in self.mri_sequence:
    #                     slice_list = self.patient_loader.patients[i].sequence_slice_index_lists_dict[sequence_str]
    #                     slice_list_dir = self.patient_loader.patients[i].get_slice_in_list_by_dir(dir_index, slice_list)
    #                     # remove choose fraction slices
    #                     n = int(self.drop_fraction * len(slice_list_dir))
    #                     if n > 0:
    #                         slice_list_dir = slice_list_dir[n:-n]
    #                     index_list_i = index_list_i + slice_list_dir
    #         # add patient index
    #         # print(i)
    #         # print(index_list_i)
    #         # print(index_list_temp)
    #         index_list_temp = index_list_temp + [[i] + slice_index for slice_index in
    #                                              index_list_i]  # [[0, 0, 0, 0, 0], ...]
    #     return index_list_temp
    def get_index_list(self):
        file_path_list_total = []
        for i in tqdm(range(self.num_patients)):
            file_path_list_i = []
            for dir_index in self.patient_loader.patients[i].dir_index_list:
                file_index_list = self.patient_loader.patients[i].get_file_index_list_in_dir(dir_index)
                for file_index in file_index_list:
                    # exclude the default MISC
                    sequence_current = self.patient_loader.patients[i].get_sequence_given_file_index(file_index)
                    file_path_temp = self.patient_loader.patients[i].get_file_path_given_file(file_index)
                    # slice_info = self.patient_loader.patients[i].get_slice_info_given_file(file_index)
                    # nifti_extension = get_file_extension(file_path_temp, prompt=".nii")
                    # json_path = file_path_temp[:-len(nifti_extension)] + '.json'
                    # nifti_path_stripped_temp = file_path_temp[:-len(nifti_extension)] + "_stripped.nii.gz"
                    if sequence_current in self.mri_sequence:
                        file_path_temp_cache = file_path_temp.replace(self.data_root, self.cache_root)
                        if self.cache_type == 'pkl':
                            # list of file ["E:/oasis_nifti_cache_reshape_norm_pkl/OAS30001_MR_d2430/anat5/sub-OAS30001_ses-d2430_T2star.nii.gz\nifti_img\z\slice_005.pkl",]
                            file_path_temp_pkls = get_all_file_paths_with_extension(file_path_temp_cache, ".pkl",
                                                                                    ["nifti_img"])
                            file_path_temp_pkls_index = [[item] for item in file_path_temp_pkls]
                            file_path_temp_pkls_index = [item + [i] for item in file_path_temp_pkls_index]
                            file_path_temp_pkls_index = [item + [file_index] for item in file_path_temp_pkls_index]
                            if self.if_add_sequence:
                                file_path_temp_pkls_index = [item + [sequence_current] for item in file_path_temp_pkls_index]
                            file_path_list_i = file_path_list_i + file_path_temp_pkls_index
            file_path_list_total = file_path_list_total + file_path_list_i
        return file_path_list_total  # list struct is [['path', patient_index, file_index],['path', patient_index, file_index], ...]

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)

        else:
            return len(self.valPaths)

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            slice_file_path = self.trainPaths[idx][0]
            patient_index = self.trainPaths[idx][1]
            file_index = self.trainPaths[idx][2]
        else:
            slice_file_path = self.valPaths[idx][0]
            patient_index = self.valPaths[idx][1]
            file_index = self.valPaths[idx][2]
        try:
            if self.cache_type == 'pkl':
                # image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)
                with open(slice_file_path, 'rb') as f:
                    image_array = pickle.load(f)
        except:
            print(f" {slice_file_path} pixel_array corrupted")
            image_tensor = torch.zeros(self.resize, self.resize)
            image_sequence = 0
            image_label = [1, 0]
            return image_tensor.unsqueeze(0), torch.tensor(image_sequence), torch.tensor(image_label), slice_file_path

        # # if rgb image, convert into grayscale
        # if len(image_array.shape) == 3:
        #     if image_array.shape[-1] == 3:
        #         image_array = rgb2gray(image_array)

        # convert to np float
        image_array = np.vstack(image_array).astype(np.float64)

        # define sequence: 1 for T1, 2 for T2, 0 for unknown
        image_sequence = 0
        image_label = [0, 0]
        if self.labels is not None and self.label_values is not None:
            if self.patient_loader.patients[patient_index].if_exist_by_labels_or(self.labels, self.label_values):
                image_label = [0, 1]
            else:
                image_label = [1, 0]

        # resize
        shape = image_array.shape  # [H, W]

        if shape != (self.resize, self.resize):
            if shape[0] > shape[1]:  # H > W
                sizeH = self.resize
                sizeW = int(shape[1] * self.resize / shape[0])
            else:
                sizeW = self.resize
                sizeH = int(shape[0] * self.resize / shape[1])

            try:
                res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
            except:
                image_tensor = torch.zeros(self.resize, self.resize)
                image_sequence = 0
                return image_tensor.unsqueeze(0), torch.tensor(image_sequence), torch.tensor(image_label), patient_index

            # pad the given size
            extra_left = int((self.resize - sizeW) / 2)
            extra_right = self.resize - sizeW - extra_left
            extra_top = int((self.resize - sizeH) / 2)
            extra_bottom = self.resize - sizeH - extra_top
            res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
            image_array = res_pad

        image_array = np.rot90(image_array)
        image_array = np.ascontiguousarray(image_array)
        # convert to tensor
        image_tensor = torch.from_numpy(image_array).float()

        # normalize 0-1
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            self.patient_loader.patients[patient_index].print_all_mri_sessions()
            print("file_index", file_index)
            print('shape')
        return image_tensor.unsqueeze(0), torch.tensor(image_sequence), torch.tensor(image_label), patient_index


class GeneralDatasetMae(Dataset):
    """
    General Dataset for training and validation of patient data such as ADNI and NACC.
    """

    def __init__(
            self,
            resize,
            drop_fraction,
            index_list,
            dataset_size_multi,  # multiple dataset size by duplicate index_list
            labels,
            label_values,
            random_seed,
            val_split,
            is_train,
            if_latest,
            max_train_len=None,
            max_val_len=None,
            data_root='',
            cache_root='',
            cache_type='pkl',
            random_crop_chance = 0.5,
            random_flip_chance = 0.5,
            random_rotate_chance=0.5,
            **kwargs

    ):
        """
        General Dataset for training and validation of patient data such as ADNI and NACC.
        :param resize: int, resize image and keep proportion, fill the gap with black (0)
        :param drop_fraction: float, 0.1 means drop first and last 10% of the slices in a sequence
        :param mri_sequence: list of string, ['T1', 'T2'], if none, means use all sequences, otherwise use chosen sequences
        :param patient_loader: loader containing all filtered patient cases
        :param labels: list, image labels for prediction
        :param label_values: list, image label values for prediction
        :param random_seed: int, seed fix for random sample
        :param val_split: float, split fraction for validation set
        :param is_train: bool, whether form training or validation set
        :param if_latest: bool, whether to use the latest patient's data
        """
        # super().__init__(resize, drop_fraction, mri_sequence, patient_loader, labels, label_values, random_seed,
        #                  val_split, is_train, if_latest, max_train_len, max_val_len, data_root, cache_root, cache_type)
        self.resize = resize
        self.drop_fraction = drop_fraction
        # self.mri_sequence = mri_sequence
        # self.patient_loader = patient_loader
        # self.num_patients = len(patient_loader.patients)
        self.data_root = data_root
        self.cache_root = cache_root
        self.cache_type = cache_type
        # self.index_list = self.get_index_list()
        self.index_list = index_list
        self.mask_sizes_cords = self._compute_mask_sizes_in_parallel_chunks()
        self.sample_weights = self._compute_sample_weights()
        self.index_list = self.index_list * dataset_size_multi
        self.sample_weights = self.sample_weights.tolist() * dataset_size_multi
        self.mask_sizes_cords = self.mask_sizes_cords.tolist() * dataset_size_multi
        print("len(self.index_list) " + str(len(self.index_list)))
        print("len(self.sample_weights) " + str(len(self.sample_weights)))
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.random_crop_chance = random_crop_chance
        self.random_flip_chance = random_flip_chance
        self.random_rotate_chance = random_rotate_chance

        np.random.seed(self.random_seed)
        # np.random.shuffle(self.index_list)
        combined = list(zip(self.index_list, self.sample_weights, self.mask_sizes_cords))
        random.shuffle(combined)
        list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined)
        self.index_list = list(list1_shuffled)
        self.sample_weights = torch.tensor(list(list2_shuffled), dtype=torch.float32)
        self.mask_sizes_cords = torch.tensor(list(list3_shuffled), dtype=torch.int32)
        # return list(list1_shuffled), list(list2_shuffled)
        print("len(self.index_list) " + str(len(self.index_list)))
        print("len(self.sample_weights) " + str(len(self.sample_weights)))
        print("len(self.mask_sizes_cords) " + str(len(self.mask_sizes_cords)))

        self.trainPaths = []
        self.valPaths = []
        self._update_train_val_paths()

        print('train size: ' + str(len(self.trainPaths)))
        print('validation size: ' + str(len(self.valPaths)))

    def _update_train_val_paths(self):
        if self.max_train_len is None:
            valPathsLen = int(len(self.index_list) * self.val_split)
            trainPathsLen = len(self.index_list) - valPathsLen
        elif self.max_train_len < len(self.index_list) - int(len(self.index_list) * self.val_split):
            valPathsLen = int(len(self.index_list) * self.val_split)
            trainPathsLen = self.max_train_len
        else:
            valPathsLen = 0
            trainPathsLen = 0
            print("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if self.max_val_len <= len(self.index_list) - trainPathsLen:
                valPathsLen = self.max_val_len
            else:
                print("assigned maximum val len exceeds total train num of slices")

        self.trainPaths = self.index_list[:trainPathsLen]
        if valPathsLen == 0:
            self.valPaths = []
        else:
            self.valPaths = self.index_list[-valPathsLen:]
        self.sample_weights_train = self.sample_weights[:trainPathsLen]
        self.sample_weights_val = self.sample_weights[-valPathsLen:]
        self.mask_sizes_cords_train = self.mask_sizes_cords[:trainPathsLen]
        self.mask_sizes_cords_val = self.mask_sizes_cords[-valPathsLen:]

    def update_index_list(self, index_list, sample_weights, mask_sizes_cords):
        combined = list(zip(index_list, sample_weights, mask_sizes_cords))
        random.shuffle(combined)
        list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined)
        self.index_list = list(list1_shuffled)
        self.sample_weights = torch.tensor(list(list2_shuffled), dtype=torch.float32)
        self.mask_sizes_cords = torch.tensor(list(list3_shuffled), dtype=torch.int32)
        self._update_train_val_paths()


    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)

        else:
            return len(self.valPaths)

    def _compute_mask_sizes(self, num_workers=16):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            func = partial(compute_mask_size, cache_type=self.cache_type)  # no lambda!
            sizes = list(tqdm(
                executor.map(func, self.index_list),
                total=len(self.index_list),
                desc=f"Computing mask sizes ({num_workers} workers)"
            ))
        return sizes

    def _compute_mask_sizes_in_parallel_chunks(self, chunk_size=500000, num_workers=16, cache_type='pkl'):
        """
        Computes (mask_size, y1, y2, x1, x2) tuples for each entry in self.index_list using multiprocessing.

        Returns:
            torch.IntTensor of shape (N, 5), where N is the number of items.
        """
        total = len(self.index_list)
        num_chunks = (total + chunk_size - 1) // chunk_size
        all_results = []

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, total)
            chunk = self.index_list[start:end]

            print(f"\n🟡 Processing chunk {chunk_id} ({start} → {end}) with {num_workers} workers")

            chunk_results = process_chunk_parallel(chunk, num_workers=num_workers, cache_type=cache_type, img_folder = 'nifti_img', mask_folder = 'nifti_mask')
            all_results.extend(chunk_results)

        combined = torch.tensor(all_results, dtype=torch.int32)  # Shape: [N, 5]
        print(f"✅ Done. Total entries: {combined.shape[0]} with shape {combined.shape}")
        return combined

    def convert_root_cache_folder(self, target_folder):
        for file_idx in tqdm(range(len(self.index_list)), desc="Computing mask sizes"):
            file_path = self.index_list[file_idx][0].replace(self.cache_root, target_folder)
            file_path = file_path.replace("\\", "/")
            self.index_list[file_idx][0] = file_path

        self._update_train_val_paths()
        self.cache_root = target_folder

    def _compute_sample_weights(self, large_mask_threshold=8000, baseline_weight=3000):
        mask_sizes_array = np.array(self.mask_sizes_cords, dtype=np.int32)
        sizes = mask_sizes_array[:, 0].astype(np.float32)

        # Create a binary mask: large vs not
        is_large = sizes >= large_mask_threshold

        weights = np.zeros_like(sizes)

        # For smaller masks, add a baseline and apply some scaling
        small_sizes = sizes[~is_large]
        small_weights = (small_sizes + baseline_weight) ** 1.2  # Or tweak exponent

        weights[~is_large] = small_weights

        # Normalize all weights so max weight is 1.0
        max_small = np.max(small_weights)
        small_weights /= max_small
        weights[~is_large] = small_weights
        # Set same weight for large masks
        weights[is_large] = 1.0  # All large masks get the same weight

        return torch.tensor(weights, dtype=torch.float32)

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            slice_file_path = self.trainPaths[idx][0]
            patient_index = self.trainPaths[idx][1]
            file_index = self.trainPaths[idx][2]
            sample_weight = self.sample_weights_train[idx]
            mask_size_cord = self.mask_sizes_cords_train[idx]
        else:
            slice_file_path = self.valPaths[idx][0]
            patient_index = self.valPaths[idx][1]
            file_index = self.valPaths[idx][2]
            sample_weight = self.sample_weights_val[idx]
            mask_size_cord = self.mask_sizes_cords_val[idx]
        try:
            if self.cache_type == 'pkl':
                # image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)
                with open(slice_file_path, 'rb') as f:
                    image_array = pickle.load(f)
                if np.issubdtype(image_array.dtype, np.unsignedinteger):
                    image_array = image_array.astype(np.float32) / 255.0
        except:
            print(f" {slice_file_path} pixel_array corrupted")
            image_tensor = torch.zeros(self.resize, self.resize)
            return image_tensor.unsqueeze(0).repeat(3, 1, 1), sample_weight

        # resize
        shape = image_array.shape  # [H, W]

        if shape != (self.resize, self.resize):
            print(f'resize shape {shape} to {self.resize}')
            print(slice_file_path)
            if shape[0] > shape[1]:  # H > W
                sizeH = self.resize
                sizeW = int(shape[1] * self.resize / shape[0])
            else:
                sizeW = self.resize
                sizeH = int(shape[0] * self.resize / shape[1])

            try:
                res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
            except:
                print(f"cannot resize {slice_file_path}")
                image_tensor = torch.zeros(self.resize, self.resize).to(dtype=torch.float16)
                return image_tensor.unsqueeze(0).repeat(3, 1, 1), sample_weight.to(dtype=torch.float16)

            # pad the given size
            extra_left = max(0, int((self.resize - sizeW) / 2))
            extra_right = max(0, self.resize - sizeW - extra_left)
            extra_top = max(0, int((self.resize - sizeH) / 2))
            extra_bottom = max(0, self.resize - sizeH - extra_top)
            res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
            image_array = res_pad

        image_array = np.rot90(image_array)  # start with 90 degree clockwise, which makes most MRI facing upward
        image_array = np.ascontiguousarray(image_array)
        # crop by mask
        if random.random() < self.random_crop_chance:
            # slice_mask_file_path = slice_file_path.replace('nifti_img', 'nifti_mask')
            # if os.path.exists(slice_mask_file_path):
            #     if self.cache_type == 'pkl':
            #         with open(slice_mask_file_path, 'rb') as f:
            #             image_array_mask = pickle.load(f)
            image_array = crop_and_resize_mri_random_buffer_auto(image_array,
                                                                         mask_size_cord, min_buffer=25, mask_pixel_threshold= 32 * 32)

        # random flip
        image_array = random_flip_2d_controlled(image_array, self.random_flip_chance, self.random_flip_chance)

        # random rotate
        if random.random() < self.random_rotate_chance:
            image_array = random_rotate_90_180_270(image_array)

        # convert to tensor
        image_array = np.ascontiguousarray(image_array)  # Ensures positive strides
        image_tensor = torch.from_numpy(image_array).to(dtype=torch.float16)

        # # normalize 0-1
        # maximum = torch.max(image_tensor)
        # minimum = torch.min(image_tensor)
        # scale = maximum - minimum
        # if scale > 0:
        #     scale_coeff = 1. / scale
        # else:
        #     scale_coeff = 0
        # image_tensor = (image_tensor - minimum) * scale_coeff

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            print("file_index", file_index)
            print('shape')
        return image_tensor.unsqueeze(0).repeat(3, 1, 1), sample_weight.to(dtype=torch.float16)

class GeneralDataLoaderSequenceClass2DSlice(Dataset):
    """
    General Dataset for training and validation of patient data such as ADNI and NACC.
    """

    def __init__(
            self,
            resize,
            drop_fraction,
            mri_sequence,
            index_list,
            dataset_size_multi,  # multiple dataset size by duplicate index_list
            labels,
            label_values,
            random_seed,
            val_split,
            is_train,
            if_latest,
            max_train_len=None,
            max_val_len=None,
            data_root='',
            cache_root='',
            cache_type='pkl',
            random_crop_chance = 0.5,
            random_flip_chance = 0.5,
            random_rotate_chance = 0.5,
            detect_sequence = [],
            **kwargs

    ):
        """
        General Dataset for training and validation of patient data such as ADNI and NACC.
        :param resize: int, resize image and keep proportion, fill the gap with black (0)
        :param drop_fraction: float, 0.1 means drop first and last 10% of the slices in a sequence
        :param mri_sequence: list of string, ['T1', 'T2'], if none, means use all sequences, otherwise use chosen sequences
        :param patient_loader: loader containing all filtered patient cases
        :param labels: list, image labels for prediction
        :param label_values: list, image label values for prediction
        :param random_seed: int, seed fix for random sample
        :param val_split: float, split fraction for validation set
        :param is_train: bool, whether form training or validation set
        :param if_latest: bool, whether to use the latest patient's data
        """
        # super().__init__(resize, drop_fraction, mri_sequence, patient_loader, labels, label_values, random_seed,
        #                  val_split, is_train, if_latest, max_train_len, max_val_len, data_root, cache_root, cache_type)
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.detect_sequence = detect_sequence
        # self.patient_loader = patient_loader
        # self.num_patients = len(patient_loader.patients)
        self.data_root = data_root
        self.cache_root = cache_root
        self.cache_type = cache_type
        # self.index_list = self.get_index_list()
        self.index_list = index_list
        self.mask_sizes_cords = self._compute_mask_sizes_in_parallel_chunks()
        self.sample_weights = self._compute_sample_weights()
        self.index_list = self.index_list * dataset_size_multi
        self.sample_weights = self.sample_weights.tolist() * dataset_size_multi
        self.mask_sizes_cords = self.mask_sizes_cords.tolist() * dataset_size_multi
        print("len(self.index_list) " + str(len(self.index_list)))
        print("len(self.sample_weights) " + str(len(self.sample_weights)))
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.random_crop_chance = random_crop_chance
        self.random_flip_chance = random_flip_chance
        self.random_rotate_chance = random_rotate_chance

        np.random.seed(self.random_seed)
        # np.random.shuffle(self.index_list)
        combined = list(zip(self.index_list, self.sample_weights, self.mask_sizes_cords))
        random.shuffle(combined)
        list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined)
        self.index_list = list(list1_shuffled)
        self.sample_weights = torch.tensor(list(list2_shuffled), dtype=torch.float32)
        self.mask_sizes_cords = torch.tensor(list(list3_shuffled), dtype=torch.int32)
        # return list(list1_shuffled), list(list2_shuffled)
        print("len(self.index_list) " + str(len(self.index_list)))
        print("len(self.sample_weights) " + str(len(self.sample_weights)))
        print("len(self.mask_sizes_cords) " + str(len(self.mask_sizes_cords)))

        self.trainPaths = []
        self.valPaths = []
        self._update_train_val_paths()

        print('train size: ' + str(len(self.trainPaths)))
        print('validation size: ' + str(len(self.valPaths)))

    def _update_train_val_paths(self):
        if self.max_train_len is None:
            valPathsLen = int(len(self.index_list) * self.val_split)
            trainPathsLen = len(self.index_list) - valPathsLen
        elif self.max_train_len < len(self.index_list) - int(len(self.index_list) * self.val_split):
            valPathsLen = int(len(self.index_list) * self.val_split)
            trainPathsLen = self.max_train_len
        else:
            valPathsLen = 0
            trainPathsLen = 0
            print("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if self.max_val_len <= len(self.index_list) - trainPathsLen:
                valPathsLen = self.max_val_len
            else:
                print("assigned maximum val len exceeds total train num of slices")

        self.trainPaths = self.index_list[:trainPathsLen]
        if valPathsLen == 0:
            self.valPaths = []
        else:
            self.valPaths = self.index_list[-valPathsLen:]
        self.sample_weights_train = self.sample_weights[:trainPathsLen]
        self.sample_weights_val = self.sample_weights[-valPathsLen:]
        self.mask_sizes_cords_train = self.mask_sizes_cords[:trainPathsLen]
        self.mask_sizes_cords_val = self.mask_sizes_cords[-valPathsLen:]

    def update_index_list(self, index_list, sample_weights, mask_sizes_cords):
        np.random.seed(self.random_seed)
        combined = list(zip(index_list, sample_weights, mask_sizes_cords))
        random.shuffle(combined)
        list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined)
        self.index_list = list(list1_shuffled)
        self.sample_weights = torch.tensor(list(list2_shuffled), dtype=torch.float32)
        self.mask_sizes_cords = torch.tensor(list(list3_shuffled), dtype=torch.int32)
        self._update_train_val_paths()

    def update_index_list_by_mask_sizes(self, target_size = 64*64):
        if isinstance(self.mask_sizes_cords, torch.Tensor):
            self.mask_sizes_cords = self.mask_sizes_cords.tolist()
        if isinstance(self.sample_weights, torch.Tensor):
            self.sample_weights = self.sample_weights.tolist()

        new_mask_sizes_cords = []
        new_index_path = []
        new_sample_weights = []

        for msc, idx, sample_weight in tqdm(zip(self.mask_sizes_cords, self.index_list, self.sample_weights), total=len(self.mask_sizes_cords), desc="Filtering by mask size"):
            mask_size = msc[0]  # First element is mask size
            if mask_size >= target_size:
                new_mask_sizes_cords.append(msc)
                new_index_path.append(idx)
                new_sample_weights.append(sample_weight)
        self.mask_sizes_cords = new_mask_sizes_cords
        self.index_list = new_index_path
        self.sample_weights = new_sample_weights
        self.update_index_list(self.index_list, self.sample_weights, self.mask_sizes_cords)

    def drop_fraction_from_each_sample_folder(self, drop_frac=0.1, drop_from="last"):
        """
        Drop a fraction of .pkl files from the beginning, end, or both ends of each folder in self.index_list.

        Args:
            drop_frac (float): Fraction (0–1) of files to drop per folder.
            drop_from (str): One of "first", "last", "both".
        """
        if isinstance(self.mask_sizes_cords, torch.Tensor):
            self.mask_sizes_cords = self.mask_sizes_cords.tolist()
        if isinstance(self.sample_weights, torch.Tensor):
            self.sample_weights = self.sample_weights.tolist()

        assert drop_from in ["first", "last", "both"], "drop_from must be 'first', 'last', or 'both'"

        # === Step 1: Group by folder ===
        folder_dict = defaultdict(list)
        print("Grouping .pkl files by folder...")
        for idx, entry in tqdm(enumerate(self.index_list), total=len(self.index_list), desc="Index scan"):
            pkl_path = entry[0]
            folder = os.path.dirname(pkl_path)
            folder_dict[folder].append(idx)

        # === Step 2: Filter per-folder ===
        kept_indices = set()
        print(f"Filtering each folder (drop {drop_from} {drop_frac * 100:.1f}%)...")
        for folder, indices in tqdm(folder_dict.items(), desc="Filtering folders"):
            indices_sorted = sorted(indices, key=lambda i: self.index_list[i][0])
            n = len(indices_sorted)

            if n <= 1:
                keep = indices_sorted
            else:
                drop_n = math.floor(n * drop_frac)
                if drop_from == "first":
                    keep = indices_sorted[drop_n:]
                elif drop_from == "last":
                    keep = indices_sorted[:n - drop_n] if drop_n > 0 else indices_sorted
                elif drop_from == "both":
                    half_drop = drop_n // 2
                    keep = indices_sorted[half_drop:n - (drop_n - half_drop)] if drop_n > 0 else indices_sorted

            kept_indices.update(keep)

        # === Step 3: Rebuild dataset ===
        print("Updating filtered dataset...")
        new_index_list = []
        new_mask_sizes_cords = []
        new_sample_weights = []

        for i in tqdm(range(len(self.index_list)), desc="Rebuilding dataset"):
            if i in kept_indices:
                new_index_list.append(self.index_list[i])
                new_mask_sizes_cords.append(self.mask_sizes_cords[i])
                new_sample_weights.append(self.sample_weights[i])

        self.index_list = new_index_list
        self.mask_sizes_cords = new_mask_sizes_cords
        self.sample_weights = new_sample_weights
        self.update_index_list(self.index_list, self.sample_weights, self.mask_sizes_cords)

    def filter_by_sequence_labels(self, allowed_sequences):
        """
        Filter index list by allowed sequence labels.

        Parameters:
            allowed_sequences (set or list): e.g., {'T2', 'DTI_DWI'}
        """
        if isinstance(self.mask_sizes_cords, torch.Tensor):
            self.mask_sizes_cords = self.mask_sizes_cords.tolist()
        if isinstance(self.sample_weights, torch.Tensor):
            self.sample_weights = self.sample_weights.tolist()

        new_mask_sizes_cords = []
        new_index_path = []
        new_sample_weights = []

        for msc, idx, sample_weight in tqdm(zip(self.mask_sizes_cords, self.index_list, self.sample_weights),
                                            total=len(self.index_list),
                                            desc="Filtering by sequence labels"):
            sequence_label = idx[3]
            if sequence_label in allowed_sequences:
                new_mask_sizes_cords.append(msc)
                new_index_path.append(idx)
                new_sample_weights.append(sample_weight)

        self.mask_sizes_cords = new_mask_sizes_cords
        self.index_list = new_index_path
        self.sample_weights = new_sample_weights
        self.update_index_list(self.index_list, self.sample_weights, self.mask_sizes_cords)
        
    def shuffle_train_paths(self):
        if isinstance(self.mask_sizes_cords_train, torch.Tensor):
            self.mask_sizes_cords_train = self.mask_sizes_cords_train.tolist()
        if isinstance(self.sample_weights_train, torch.Tensor):
            self.sample_weights_train = self.sample_weights_train.tolist()

        np.random.seed(self.random_seed)
        combined = list(zip(self.trainPaths, self.sample_weights_train, self.mask_sizes_cords_train))
        random.shuffle(combined)
        list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined)
        self.trainPaths = list(list1_shuffled)
        self.sample_weights_train = torch.tensor(list(list2_shuffled), dtype=torch.float32)
        self.mask_sizes_cords_train = torch.tensor(list(list3_shuffled), dtype=torch.int32)
        
    def shuffle_val_paths(self):
        if isinstance(self.mask_sizes_cords_val, torch.Tensor):
            self.mask_sizes_cords_val = self.mask_sizes_cords_val.tolist()
        if isinstance(self.sample_weights_val, torch.Tensor):
            self.sample_weights_val = self.sample_weights_val.tolist()

        np.random.seed(self.random_seed)
        combined = list(zip(self.valPaths, self.sample_weights_val, self.mask_sizes_cords_val))
        random.shuffle(combined)
        list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined)
        self.valPaths = list(list1_shuffled)
        self.sample_weights_val = torch.tensor(list(list2_shuffled), dtype=torch.float32)
        self.mask_sizes_cords_val = torch.tensor(list(list3_shuffled), dtype=torch.int32)

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)

        else:
            return len(self.valPaths)

    def _compute_mask_sizes(self, num_workers=16):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            func = partial(compute_mask_size, cache_type=self.cache_type)  # no lambda!
            sizes = list(tqdm(
                executor.map(func, self.index_list),
                total=len(self.index_list),
                desc=f"Computing mask sizes ({num_workers} workers)"
            ))
        return sizes

    def _compute_mask_sizes_in_parallel_chunks(self, chunk_size=500000, num_workers=16, cache_type='pkl'):
        """
        Computes (mask_size, y1, y2, x1, x2) tuples for each entry in self.index_list using multiprocessing.

        Returns:
            torch.IntTensor of shape (N, 5), where N is the number of items.
        """
        total = len(self.index_list)
        num_chunks = (total + chunk_size - 1) // chunk_size
        all_results = []

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, total)
            chunk = self.index_list[start:end]

            print(f"\n🟡 Processing chunk {chunk_id} ({start} → {end}) with {num_workers} workers")

            chunk_results = process_chunk_parallel(chunk, num_workers=num_workers, cache_type=cache_type, img_folder = 'nifti_img', mask_folder = 'nifti_mask')
            all_results.extend(chunk_results)

        combined = torch.tensor(all_results, dtype=torch.int32)  # Shape: [N, 5]
        print(f"✅ Done. Total entries: {combined.shape[0]} with shape {combined.shape}")
        return combined

    def convert_root_cache_folder(self, target_folder):
        for file_idx in tqdm(range(len(self.index_list)), desc="Computing mask sizes"):
            file_path = self.index_list[file_idx][0].replace(self.cache_root, target_folder)
            file_path = file_path.replace("\\", "/")
            self.index_list[file_idx][0] = file_path

        self._update_train_val_paths()
        self.cache_root = target_folder

    def _compute_sample_weights(self, large_mask_threshold=8000, baseline_weight=3000):
        mask_sizes_array = np.array(self.mask_sizes_cords, dtype=np.int32)
        sizes = mask_sizes_array[:, 0].astype(np.float32)

        # Create a binary mask: large vs not
        is_large = sizes >= large_mask_threshold

        weights = np.zeros_like(sizes)

        # For smaller masks, add a baseline and apply some scaling
        small_sizes = sizes[~is_large]
        small_weights = (small_sizes + baseline_weight) ** 1.2  # Or tweak exponent

        weights[~is_large] = small_weights

        # Normalize all weights so max weight is 1.0
        max_small = np.max(small_weights)
        small_weights /= max_small
        weights[~is_large] = small_weights
        # Set same weight for large masks
        weights[is_large] = 1.0  # All large masks get the same weight

        return torch.tensor(weights, dtype=torch.float32)

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            slice_file_path = self.trainPaths[idx][0]
            patient_index = self.trainPaths[idx][1]
            file_index = self.trainPaths[idx][2]
            sample_weight = self.sample_weights_train[idx]
            mask_size_cord = self.mask_sizes_cords_train[idx]
            sequence_tag = self.trainPaths[idx][3]
        else:
            slice_file_path = self.valPaths[idx][0]
            patient_index = self.valPaths[idx][1]
            file_index = self.valPaths[idx][2]
            sample_weight = self.sample_weights_val[idx]
            mask_size_cord = self.mask_sizes_cords_val[idx]
            sequence_tag = self.valPaths[idx][3]

        image_label = torch.zeros(len(self.detect_sequence) + 1)  # size of label = number of sequence + 1
        image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class
        try:
            if self.cache_type == 'pkl':
                # image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)
                with open(slice_file_path, 'rb') as f:
                    image_array = pickle.load(f)
                if np.issubdtype(image_array.dtype, np.unsignedinteger):
                    image_array = image_array.astype(np.float32) / 255.0
        except:
            print(f" {slice_file_path} pixel_array corrupted")
            image_tensor = torch.zeros(self.resize, self.resize)
            return image_tensor.unsqueeze(0).repeat(3, 1, 1), image_label

        # resize
        shape = image_array.shape  # [H, W]

        if shape != (self.resize, self.resize):
            print(f'resize shape {shape} to {self.resize}')
            print(slice_file_path)
            if shape[0] > shape[1]:  # H > W
                sizeH = self.resize
                sizeW = int(shape[1] * self.resize / shape[0])
            else:
                sizeW = self.resize
                sizeH = int(shape[0] * self.resize / shape[1])

            try:
                res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
            except:
                print(f"cannot resize {slice_file_path}")
                image_tensor = torch.zeros(self.resize, self.resize).to(dtype=torch.float16)
                return image_tensor.unsqueeze(0).repeat(3, 1, 1), image_label

            # pad the given size
            extra_left = max(0, int((self.resize - sizeW) / 2))
            extra_right = max(0, self.resize - sizeW - extra_left)
            extra_top = max(0, int((self.resize - sizeH) / 2))
            extra_bottom = max(0, self.resize - sizeH - extra_top)
            res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
            image_array = res_pad

        image_array = np.rot90(image_array)  # start with 90 degree clockwise, which makes most MRI facing upward
        image_array = np.ascontiguousarray(image_array)
        # crop by mask
        if random.random() < self.random_crop_chance:
            # slice_mask_file_path = slice_file_path.replace('nifti_img', 'nifti_mask')
            # if os.path.exists(slice_mask_file_path):
            #     if self.cache_type == 'pkl':
            #         with open(slice_mask_file_path, 'rb') as f:
            #             image_array_mask = pickle.load(f)
            image_array = crop_and_resize_mri_random_buffer_auto(image_array,
                                                                         mask_size_cord, min_buffer=25, mask_pixel_threshold= 32 * 32)

        # random flip
        image_array = random_flip_2d_controlled(image_array, self.random_flip_chance, self.random_flip_chance)

        # random rotate
        if random.random() < self.random_rotate_chance:
            image_array = random_rotate_90_180_270(image_array)

        # convert to tensor
        image_array = np.ascontiguousarray(image_array)  # Ensures positive strides
        image_tensor = torch.from_numpy(image_array).to(dtype=torch.float16)

        # # normalize 0-1
        # maximum = torch.max(image_tensor)
        # minimum = torch.min(image_tensor)
        # scale = maximum - minimum
        # if scale > 0:
        #     scale_coeff = 1. / scale
        # else:
        #     scale_coeff = 0
        # image_tensor = (image_tensor - minimum) * scale_coeff

        # 1-hot classify
        if sequence_tag == 'DTI_DWI_500':
            sequence_tag = 'DTI_DWI'

        if sequence_tag in self.detect_sequence:
            label_idx = self.detect_sequence.index(sequence_tag)
            image_label[-1] = 0.0
            image_label[label_idx] = 1.0

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            print("file_index", file_index)
            print('shape')
        return image_tensor.unsqueeze(0).repeat(3, 1, 1), image_label

import json
class PklDatasetWithLabel(Dataset):
    def __init__(self, json_path, label_list,random_crop_chance = 0.5,
            random_flip_chance = 0.5,
            random_rotate_chance = 0.5, resize=224):
        """
        Args:
            json_path (str): Path to the JSON file storing {label: [list of pkl paths]}.
            label_list (list): List of all possible labels (for one-hot encoding).
        """
        self.label_list = label_list
        self.label_to_index = {label: idx for idx, label in enumerate(label_list)}
        self.random_crop_chance = random_crop_chance
        self.random_flip_chance = random_flip_chance
        self.random_rotate_chance = random_rotate_chance
        self.resize = resize

        # Load dict from JSON
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        # Flatten and index
        self.samples = []
        for label, pkl_paths in self.data_dict.items():
            for path in pkl_paths:
                self.samples.append((path, label))

        self.mask_sizes_cords = self._compute_mask_sizes_in_parallel_chunks()

    def _compute_mask_sizes_in_parallel_chunks(self, chunk_size=500000, num_workers=16, cache_type='pkl'):
        """
        Computes (mask_size, y1, y2, x1, x2) tuples for each entry in self.index_list using multiprocessing.

        Returns:
            torch.IntTensor of shape (N, 5), where N is the number of items.
        """
        total = len(self.samples)
        num_chunks = (total + chunk_size - 1) // chunk_size
        all_results = []

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, total)
            chunk = self.samples[start:end]

            print(f"\n🟡 Processing chunk {chunk_id} ({start} → {end}) with {num_workers} workers")

            chunk_results = process_chunk_parallel(chunk, num_workers=num_workers, cache_type=cache_type, img_folder = 'nifti_img', mask_folder = 'nifti_seg')
            all_results.extend(chunk_results)

        combined = torch.tensor(all_results, dtype=torch.int32)  # Shape: [N, 5]
        print(f"✅ Done. Total entries: {combined.shape[0]} with shape {combined.shape}")
        return combined

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mask_size_cord = self.mask_sizes_cords[idx]
        # Load .pkl file
        with open(path, 'rb') as f:
            image_array = pickle.load(f)
            if np.issubdtype(image_array.dtype, np.unsignedinteger):
                image_array = image_array.astype(np.float32) / 255.0

        # resize
        shape = image_array.shape  # [H, W]

        if shape != (self.resize, self.resize):
            print(f'resize shape {shape} to {self.resize}')
            if shape[0] > shape[1]:  # H > W
                sizeH = self.resize
                sizeW = int(shape[1] * self.resize / shape[0])
            else:
                sizeW = self.resize
                sizeH = int(shape[0] * self.resize / shape[1])

            try:
                res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
            except:
                image_tensor = torch.zeros(self.resize, self.resize).to(dtype=torch.float16)
                label_tensor = torch.zeros(len(self.label_list) + 1, dtype=torch.float32)
                label_tensor[-1] = 1.0
                return image_tensor.unsqueeze(0).repeat(3, 1, 1), label_tensor

            # pad the given size
            extra_left = max(0, int((self.resize - sizeW) / 2))
            extra_right = max(0, self.resize - sizeW - extra_left)
            extra_top = max(0, int((self.resize - sizeH) / 2))
            extra_bottom = max(0, self.resize - sizeH - extra_top)
            res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
            image_array = res_pad

        image_array = np.rot90(image_array)  # start with 90 degree clockwise, which makes most MRI facing upward
        image_array = np.ascontiguousarray(image_array)
        # crop by mask
        if random.random() < self.random_crop_chance:
            # slice_mask_file_path = slice_file_path.replace('nifti_img', 'nifti_mask')
            # if os.path.exists(slice_mask_file_path):
            #     if self.cache_type == 'pkl':
            #         with open(slice_mask_file_path, 'rb') as f:
            #             image_array_mask = pickle.load(f)
            image_array = crop_and_resize_mri_random_buffer_auto(image_array,
                                                                 mask_size_cord, min_buffer=25,
                                                                 mask_pixel_threshold=32 * 32)

        # random flip
        image_array = random_flip_2d_controlled(image_array, self.random_flip_chance, self.random_flip_chance)

        # random rotate
        if random.random() < self.random_rotate_chance:
            image_array = random_rotate_90_180_270(image_array)

        # convert to tensor
        image_array = np.ascontiguousarray(image_array)  # Ensures positive strides
        image_tensor = torch.from_numpy(image_array).to(dtype=torch.float16)

        # Convert label to one-hot tensor
        label_idx = self.label_to_index[label]
        label_tensor = torch.zeros(len(self.label_list) + 1, dtype=torch.float32)
        label_tensor[label_idx] = 1.0

        return image_tensor.unsqueeze(0).repeat(3, 1, 1), label_tensor

class PklDatasetMultiSeg(Dataset):
    def __init__(self, json_path, label_groups, nii_root = "F:/nacc_nifti", pkl_root = "E:/nifti_seg_cache_pkl",
                 img_folder = 'nifti_img', mask_folder = 'nifti_seg', random_crop_chance = 0.5,
            random_flip_chance = 0.5,
            random_rotate_chance = 0.5, resize=224):
        """
        Args:
            json_path (str): Path to the JSON file storing {label: [list of pkl paths]}.
        """
        self.label_groups = label_groups
        self.random_crop_chance = random_crop_chance
        self.random_flip_chance = random_flip_chance
        self.random_rotate_chance = random_rotate_chance
        self.resize = resize
        self.img_folder = img_folder
        self.mask_folder = mask_folder

        # Load dict from JSON
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        # Flatten and index
        self.samples = []
        for file_idx in tqdm(range(len(self.data_dict["paths"])), desc="add pkl fels"):
            nii_path = self.data_dict["paths"][file_idx]
            file_index = self.data_dict["file_indices"][file_idx]
            patient_id = self.data_dict["patient_id"][file_idx]
            nii_path = nii_path.replace(nii_root, pkl_root)
            nii_path = nii_path + '/' + img_folder
            pkl_files = get_all_pkl_paths(nii_path)
            for pkl_file in pkl_files:
                self.samples.append((pkl_file, file_index, patient_id))

        self.mask_sizes_cords = self._compute_mask_sizes_in_parallel_chunks()

    def _compute_mask_sizes_in_parallel_chunks(self, chunk_size=500000, num_workers=16, cache_type='pkl'):
        """
        Computes (mask_size, y1, y2, x1, x2) tuples for each entry in self.index_list using multiprocessing.

        Returns:
            torch.IntTensor of shape (N, 5), where N is the number of items.
        """
        total = len(self.samples)
        num_chunks = (total + chunk_size - 1) // chunk_size
        all_results = []

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, total)
            chunk = self.samples[start:end]

            print(f"\n🟡 Processing chunk {chunk_id} ({start} → {end}) with {num_workers} workers")

            chunk_results = process_chunk_parallel(chunk, num_workers=num_workers, cache_type=cache_type, img_folder = self.img_folder, mask_folder = self.mask_folder)
            all_results.extend(chunk_results)

        combined = torch.tensor(all_results, dtype=torch.int32)  # Shape: [N, 5]
        print(f"✅ Done. Total entries: {combined.shape[0]} with shape {combined.shape}")
        return combined

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, file_index, patient_id = self.samples[idx]
        mask_size_cord = self.mask_sizes_cords[idx]
        # Load .pkl file
        with open(path, 'rb') as f:
            image_array = pickle.load(f)
            if np.issubdtype(image_array.dtype, np.unsignedinteger):
                image_array = image_array.astype(np.float32) / 255.0
        with open(path.replace(self.img_folder, self.mask_folder), 'rb') as f:
            mask_array = pickle.load(f)

        # resize
        shape = image_array.shape  # [H, W]

        if shape != (self.resize, self.resize):
            print(f'resize shape {shape} to {self.resize}')
            if shape[0] > shape[1]:  # H > W
                sizeH = self.resize
                sizeW = int(shape[1] * self.resize / shape[0])
            else:
                sizeW = self.resize
                sizeH = int(shape[0] * self.resize / shape[1])

            try:
                res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
                res_mask = cv2.resize(mask_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_NEAREST)
            except:
                image_tensor = torch.zeros(self.resize, self.resize).to(dtype=torch.float16)
                mask_tensor = torch.zeros(self.resize, self.resize).to(dtype=torch.int16)
                return image_tensor.unsqueeze(0).repeat(3, 1, 1), mask_tensor.unsqueeze(0).repeat(1, 1, 1)

            # pad the given size
            extra_left = max(0, int((self.resize - sizeW) / 2))
            extra_right = max(0, self.resize - sizeW - extra_left)
            extra_top = max(0, int((self.resize - sizeH) / 2))
            extra_bottom = max(0, self.resize - sizeH - extra_top)
            res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
            image_array = res_pad
            res_pad_mask = np.pad(res_mask, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
            mask_array = res_pad_mask

        image_array = np.rot90(image_array)  # start with 90 degree clockwise, which makes most MRI facing upward
        image_array = np.ascontiguousarray(image_array)
        mask_array = np.rot90(mask_array)  # start with 90 degree clockwise, which makes most MRI facing upward
        mask_array = np.ascontiguousarray(mask_array)
        # crop by mask
        if random.random() < self.random_crop_chance:
            # slice_mask_file_path = slice_file_path.replace('nifti_img', 'nifti_mask')
            # if os.path.exists(slice_mask_file_path):
            #     if self.cache_type == 'pkl':
            #         with open(slice_mask_file_path, 'rb') as f:
            #             image_array_mask = pickle.load(f)
            image_array, mask_array = crop_and_resize_mri_mask_random_buffer_auto(image_array,mask_array,
                                                                 mask_size_cord, min_buffer=25,
                                                                 mask_pixel_threshold=32 * 32)

        # random flip
        image_array, mask_array = random_flip_2d_controlled_img_mask(image_array, mask_array, self.random_flip_chance, self.random_flip_chance)

        # random rotate
        if random.random() < self.random_rotate_chance:
            image_array, mask_array = random_rotate_90_180_270_img_mask(image_array, mask_array)

        # convert to tensor
        image_array = np.ascontiguousarray(image_array)  # Ensures positive strides
        image_tensor = torch.from_numpy(image_array).to(dtype=torch.float16)

        mask_array = np.ascontiguousarray(mask_array)  # Ensures positive strides
        mask_array =  remap_labels_fast(mask_array, self.label_groups)
        mask_tensor = torch.from_numpy(mask_array).to(dtype=torch.int16)


        return image_tensor.unsqueeze(0).repeat(3, 1, 1), mask_tensor.unsqueeze(0).repeat(1, 1, 1)