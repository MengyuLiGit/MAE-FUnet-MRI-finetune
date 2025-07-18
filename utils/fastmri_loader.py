import h5py
import os
import pickle
import torch
import fastmri
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.transform_util import pad_to_pool, normalize_zero_to_one
from utils.transform_util import *
from fastmri.data import transforms, mri_data, subsample
from help_func import print_var_detail
import torchvision
import numpy as np
import time
from utils.general_dataloader_cache import random_rotate_90_180_270_torch, random_flip_2d_controlled_torch
# number of slices to be cut at the head and tail of each volume
NUM_CUT_SLICES = 0


def create_fastmri_dataloader(
        data_dir,
        data_info_list_path,
        batch_size,
        random_flip=False,
        is_distributed=False,
        is_train=False,
        mask_func=None,
        post_process=None,
        center_crop_size=320,
        num_layer=4,
        step_scale=2,
        target_mode='single',
        output_mode='img',
        norm_mode='unit_norm',
        num_workers=0,
):
    """
    Create a dataloader for fastmri dataset

    :param data_dir: str, the directory saving data
    :param data_info_list_path: str, the .pkl file containing data list info
    :param batch_size: int, batch size for dataloader
    :param random_flip: bool, whether to flip image, default False
    :param is_distributed: bool, whether to use distributed sampler
    :param is_train: bool, whether used for training
    :param mask_func: mask function applied on kspace, normally use fastmri.subsample.RandomMaskFunc
    :param post_process: function, used to post-process image, image_zf, kspace, kspace_zf and mask
    :param center_crop_size: int, determine the center crop size of image and reconvert to kspace if necessary, default 320
    :param num_layer: int, number of layers for model to up/down sample, default 4
    :param step_scale: int, scale of each layer for model to up/down sample, default 2
    :param target_mode: str, target mri, 'single' for single-coil (reconstruction_esc), or 'multi' for multi-coil (reconstruction_rss). default 'single'
    :param output_mode: str, 'img' or 'kspace'. Whether input and target are recovered mri img or raw k-space data. default 'img'
    :param norm_mode: str, 'unit_norm' or None. Whether to normalize output. default 'unit_norm'
    :param num_workers: int, number of workers if using multiple GPU, default 0
    :return: fastmri loader, each element is a dict
    """

    if not data_dir:
        raise ValueError("unspecified dta directory.")

    # read data information which is saved in a list.
    with open(data_info_list_path, "rb") as f:
        data_info_list = pickle.load(f)

    # create dataset
    dataset = FastmriDataset(
        data_dir,
        data_info_list,
        random_flip=random_flip,
        mask_func=mask_func,
        post_process=post_process,
        center_crop_size=center_crop_size,
        num_layer=num_layer,
        step_scale=step_scale,
        target_mode=target_mode,
        output_mode=output_mode,
        norm_mode=norm_mode,
    )

    data_sampler = None
    if is_distributed:
        data_sampler = DistributedSampler(dataset)

    # create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(data_sampler is None) and is_train,
        sampler=data_sampler,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=True,
    )

    # return loader, note all np in FastmriDataset will be converted to tensor via dataloader
    return loader


class FastmriDataset(Dataset):
    """
    A dataset class used to create and preprocess fastmri data.
    Output element is a customized dict

    Args:
    ----------
    data_dir : str
        the directory saving data.
    data_info_list : list
        a list containing elements like (file_name, index). A .h5 file contains multi slices.
    random_flip : bool
        whether to random flip image.
    mask_func : function
        mask function applied on kspace, normally use fastmri.subsample.RandomMaskFunc.
    post_process : function
        used to post-process image, image_zf, kspace, kspace_zf and mask.
    center_crop_size : int
        determine the center crop size of image and reconvert to kspace if necessary
    num_layer: int,
        number of layers for model to up/down sample
    step_scale: int,
        scale of each layer for model to up/down sample
    target_mode: str,
        target mri, 'single' for single-coil (reconstruction_esc), or 'multi' for multi-coil
        (reconstruction_rss).
    output_mode: str,
        'img' or 'kspace'. Whether input and target are recovered mri img or raw k-space data.
    norm_mode: str,
        'unit_norm' or None. Whether to normalize output.
    """

    def __init__(
            self,
            data_dir, data_info_list, random_flip, mask_func,
            post_process, center_crop_size, num_layer, step_scale,
            target_mode, output_mode, norm_mode,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.random_flip = random_flip
        self.mask_func = mask_func
        self.post_process = post_process
        self.data_info_list = data_info_list
        self.center_crop_size = center_crop_size
        self.target_mode = target_mode
        self.output_mode = output_mode
        self.num_layer = num_layer
        self.step_scale = step_scale
        self.norm_mode = norm_mode

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        file_name, index = self.data_info_list[idx]
        acquisition, kspace_raw, image_rss, image_esc \
            = read_datafile(self.data_dir, file_name, index, self.target_mode)

        if self.output_mode == 'img':
            # set target mri image
            if self.target_mode == 'multi':
                target = transforms.to_tensor(image_rss)
            else:
                target = transforms.to_tensor(image_esc)

            kspace = transforms.to_tensor(kspace_raw)  # [H,W,2]

            # apply mask
            if self.mask_func is None:
                mask = 1 - torch.zeros_like(target)
                kspace_masked = kspace
            else:
                kspace_masked, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # apply mask in kspace
                mask = mask.squeeze(2).repeat(kspace_masked.shape[0], 1)  # [H,W]

            # IFFT
            image_masked = fastmri.ifft2c(kspace_masked)

            # center crop
            image_masked = transforms.complex_center_crop(image_masked, (self.center_crop_size, self.center_crop_size))

            # pad for pool
            image_masked = pad_to_pool(image_masked.unsqueeze(0), num_layer=self.num_layer,
                                       step_scale=self.step_scale).squeeze(0)
            self.center_crop_size = image_masked.shape[1]

            # re-crop target and mask
            target = transforms.center_crop(target, (self.center_crop_size, self.center_crop_size))
            mask = transforms.center_crop(mask, (self.center_crop_size, self.center_crop_size))

            image_masked_abs = fastmri.complex_abs(image_masked)
            # input = img_abs.numpy()
            input_k = fastmri.fft2c(image_masked)

        # normalization
        # if self.norm_mode == 'unit_norm':
        #     target, _, _ = normalize_zero_to_one(target)
        #     img_abs, _, _ = normalize_zero_to_one(img_abs)
        #     img, _, _ = normalize_zero_to_one(img)
        max = torch.max(image_masked_abs)
        scale_coeff = 1. / max
        image_masked_abs = image_masked_abs * scale_coeff
        target = target * scale_coeff
        input_k = input_k * scale_coeff
        image_masked = image_masked * scale_coeff

        # print_var_detail(image_masked_abs,'image_masked_abs')
        # print_var_detail(target, 'target')
        # print_var_detail(input_k, 'input_k')
        # print_var_detail(image_masked, 'image_masked')
        # print_var_detail(mask, 'mask')

        if self.target_mode == 'multi':
            input_k = input_k.permute(0, -1, -3, -2)
            image_masked = image_masked.permute(0, -1, -3, -2)
        else:
            input_k = input_k.permute(-1, -3, -2)
            image_masked = image_masked.permute(-1, -3, -2)
        args_dict = {
            "image_masked_abs": image_masked_abs.unsqueeze(0),  # [1, H, W]
            "target": target.unsqueeze(0),  # [1, H, W]
            "mask": mask.unsqueeze(0),  # [1, H, W]
            "input_k": input_k,  # [2, H, W]
            "image_masked": image_masked,  # [2, H, W]
            "file_name": file_name,  # str
            "slice_index": index,  # int
        }

        return args_dict


class FastmriDataSetMae(Dataset):
    """
    A dataset class used to create and preprocess fastmri data.
    Output element is a customized dict

    Args:
    ----------
    data_dir : str
        the directory saving data.
    data_info_list : list
        a list containing elements like (file_name, index). A .h5 file contains multi slices.
    random_flip : bool
        whether to random flip image.
    mask_func : function
        mask function applied on kspace, normally use fastmri.subsample.RandomMaskFunc.
    post_process : function
        used to post-process image, image_zf, kspace, kspace_zf and mask.
    center_crop_size : int
        determine the center crop size of image and reconvert to kspace if necessary
    num_layer: int,
        number of layers for model to up/down sample
    step_scale: int,
        scale of each layer for model to up/down sample
    target_mode: str,
        target mri, 'single' for single-coil (reconstruction_esc), or 'multi' for multi-coil
        (reconstruction_rss).
    output_mode: str,
        'img' or 'kspace'. Whether input and target are recovered mri img or raw k-space data.
    norm_mode: str,
        'unit_norm' or None. Whether to normalize output.
    """

    def __init__(
            self,
            data_dir, data_info_list_path,
            center_crop_size, image_size,
            target_mode, if_cache_image=False, cache_path='image_cache/fastmri/'
    ):
        super().__init__()
        self.data_dir = data_dir
        with open(data_info_list_path, "rb") as f:
            self.data_info_list = pickle.load(f)
        self.center_crop_size = center_crop_size
        self.image_size = image_size
        self.target_mode = target_mode
        self.if_cache_image = if_cache_image
        self.cache_path = cache_path

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):

        file_name, index = self.data_info_list[idx]
        # time0 = time.time()
        acquisition, kspace_raw, image_rss, image_esc \
            = read_datafile(self.data_dir, file_name, index, self.target_mode, self.if_cache_image, self.cache_path)
        # time1 = time.time()
        # print('1-0: ' + str(time1 - time0))

        # set target mri image
        if self.target_mode == 'multi':
            target = transforms.to_tensor(image_rss)
        else:
            target = transforms.to_tensor(image_esc)
        # time2 = time.time()
        # print('2-1: ' + str(time2 - time1))
        # re-crop target and mask
        target = target.unsqueeze(0)  # [1,H,W]
        target = center_crop_with_pad(target, self.center_crop_size, self.center_crop_size)
        target = target.squeeze(0)

        # # re-crop target and mask
        # if (0 < self.center_crop_size <= target.shape[-2] and 0 < self.center_crop_size <= target.shape[-1]):
        #     target = transforms.center_crop(target, (self.center_crop_size, self.center_crop_size))
        #     if self.center_crop_size != self.image_size:
        #         target = torchvision.transforms.functional.resize(img=target.unsqueeze(0),
        #                                                           size=(self.image_size, self.image_size)).squeeze(0)
        # else:
        target = torchvision.transforms.functional.resize(img=target.unsqueeze(0),
                                                          size=(self.image_size, self.image_size)).squeeze(0)

        # time3 = time.time()
        # print('3-2: ' + str(time3 - time2))
        max = torch.max(target)

        if max > 1e-12:
            scale_coeff = 1. / max
        else:
            scale_coeff = 0.0

        target = target * scale_coeff
        # max = torch.max(target)
        # scale_coeff = 1. / max
        # target = target * scale_coeff

        image_tensor = target
        # mark image
        image_tensor_masked = image_tensor  # [H, W]
        # image_tensor_masked, mask = MAE_mask(image_tensor)
        # time4 = time.time()
        # print('4-3: ' + str(time4 - time3))
        # print('4-0: ' + str(time4 - time0))
        return image_tensor.unsqueeze(0).repeat(3, 1, 1), image_tensor_masked.unsqueeze(0).repeat(3, 1, 1)


class FastmriDataSetMaeSampleWeights(Dataset):
    """
    A dataset class used to create and preprocess fastmri data.
    Output element is a customized dict

    Args:
    ----------
    data_dir : str
        the directory saving data.
    data_info_list : list
        a list containing elements like (file_name, index). A .h5 file contains multi slices.
    random_flip : bool
        whether to random flip image.
    mask_func : function
        mask function applied on kspace, normally use fastmri.subsample.RandomMaskFunc.
    post_process : function
        used to post-process image, image_zf, kspace, kspace_zf and mask.
    center_crop_size : int
        determine the center crop size of image and reconvert to kspace if necessary
    num_layer: int,
        number of layers for model to up/down sample
    step_scale: int,
        scale of each layer for model to up/down sample
    target_mode: str,
        target mri, 'single' for single-coil (reconstruction_esc), or 'multi' for multi-coil
        (reconstruction_rss).
    output_mode: str,
        'img' or 'kspace'. Whether input and target are recovered mri img or raw k-space data.
    norm_mode: str,
        'unit_norm' or None. Whether to normalize output.
    """

    def __init__(
            self,
            data_dir, data_info_list_path,
            center_crop_size, image_size,
            target_mode, if_cache_image=False, cache_path='image_cache/fastmri/', random_flip_chance = 0.5,
        random_rotate_chance = 0.5
    ):
        super().__init__()
        self.data_dir = data_dir
        with open(data_info_list_path, "rb") as f:
            self.data_info_list = pickle.load(f)
        self.center_crop_size = center_crop_size
        self.image_size = image_size
        self.target_mode = target_mode
        self.if_cache_image = if_cache_image
        self.cache_path = cache_path
        self.random_flip_chance = random_flip_chance
        self.random_rotate_chance = random_rotate_chance

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):

        file_name, index = self.data_info_list[idx]
        # time0 = time.time()
        acquisition, kspace_raw, image_rss, image_esc \
            = read_datafile(self.data_dir, file_name, index, self.target_mode, self.if_cache_image, self.cache_path)
        # time1 = time.time()
        # print('1-0: ' + str(time1 - time0))

        # set target mri image
        if self.target_mode == 'multi':
            target = transforms.to_tensor(image_rss)
        else:
            target = transforms.to_tensor(image_esc)
        # time2 = time.time()
        # print('2-1: ' + str(time2 - time1))
        # re-crop target and mask
        target = target.unsqueeze(0)  # [1,H,W]
        target = center_crop_with_pad(target, self.center_crop_size, self.center_crop_size)
        target = target.squeeze(0)

        # # re-crop target and mask
        # if (0 < self.center_crop_size <= target.shape[-2] and 0 < self.center_crop_size <= target.shape[-1]):
        #     target = transforms.center_crop(target, (self.center_crop_size, self.center_crop_size))
        #     if self.center_crop_size != self.image_size:
        #         target = torchvision.transforms.functional.resize(img=target.unsqueeze(0),
        #                                                           size=(self.image_size, self.image_size)).squeeze(0)
        # else:
        target = torchvision.transforms.functional.resize(img=target.unsqueeze(0),
                                                          size=(self.image_size, self.image_size)).squeeze(0)


        # random flip
        target = random_flip_2d_controlled_torch(target, self.random_flip_chance, self.random_flip_chance)

        # random rotate
        if random.random() < self.random_rotate_chance:
            target = random_rotate_90_180_270_torch(target)


        # time3 = time.time()
        # print('3-2: ' + str(time3 - time2))
        max = torch.max(target)

        if max > 1e-12:
            scale_coeff = 1. / max
        else:
            scale_coeff = 0.0

        target = target * scale_coeff
        # max = torch.max(target)
        # scale_coeff = 1. / max
        # target = target * scale_coeff

        image_tensor = target


        image_tensor = image_tensor.to(torch.float16)
        sample_weight = torch.tensor(1.0, dtype=torch.float16)

        # mark image
        # image_tensor_masked = image_tensor  # [H, W]
        # image_tensor_masked, mask = MAE_mask(image_tensor)
        # time4 = time.time()
        # print('4-3: ' + str(time4 - time3))
        # print('4-0: ' + str(time4 - time0))
        return image_tensor.unsqueeze(0).repeat(3, 1, 1), sample_weight

def read_datafile(data_dir, file_name, slice_index, target_mode, if_cache_image=False,
                  cache_path='image_cache/fastmri/'):
    """
    Read mri data of fastmri dataset from .h5 file.

    :param data_dir: str, directory saving data.
    :param file_name: str, file name of selected data.
    :param slice_index: int, index of selected slice.
    :param target_mode: str, target mri, 'single' for single-coil (reconstruction_esc), or 'multi' for multi-coil
                        (reconstruction_rss).

    :return: tuple of (str, numpy.array of complex64, numpy.array of float32),
        acquisition, raw k-space with shape larger than (320, 320), and rss image with shape of (320, 320).
    """
    if if_cache_image:
        slice_path_rss = cache_path + file_name + '/' + str(slice_index) + '_rss.pkl'
        slice_path_esc = cache_path + file_name + '/' + str(slice_index) + '_esc.pkl'
        with open(slice_path_rss, "rb") as f:
            image_rss = pickle.load(f)
        acquisition = None
        kspace_raw = None
        image_esc = None
        if target_mode == 'single':
            with open(slice_path_esc, "rb") as f:
                image_esc = pickle.load(f)

    else:
        file_path = os.path.join(data_dir, file_name)
        data = h5py.File(file_path, mode="r")
        acquisition = data.attrs["acquisition"]
        kspace_raw = data["kspace"][slice_index]
        image_rss = data["reconstruction_rss"][slice_index]
        image_esc = None
        if target_mode == 'single':
            # image_esc = np.array(data["reconstruction_esc"])[slice_index]
            image_esc = data["reconstruction_esc"][slice_index]

    # time4 = time.time()
    # print('image_esc np.array time:', time4 - time3)

    return acquisition, kspace_raw, image_rss, image_esc


def ifftc_np_from_raw_data(kspace_raw):
    """
    Inverse orthogonal FFT2 transform raw kspace data to feasible complex image, numpy.array to numpy.array.

    :param kspace_raw: numpy.array of complex with shape of (h, w), raw kspace data from .h5 file.

    :return: numpy.array of complex with shape of (h, w), transformed image, keep dtype.
    """
    transformed_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_raw), norm="ortho"))
    transformed_image = transformed_image.astype(kspace_raw.dtype)
    return transformed_image


def seed_from_file_name_slice_index(file_name, slice_index):
    return int(file_name[4:-3]) * 100 + slice_index


def create_fastmri_data_info(data_dir="E:/mri_data/knee_singlecoil_train_200",
                             data_info_dir="E:/CodeTest/Pretrained_IPT_v1/data/fastmri",
                             num_files=-1,
                             num_pd_files=-1,
                             num_pdfs_files=-1,
                             data_info_file_name="pd_train_info", if_cache_image=False,
                             cache_path='image_cache/fastmri/',
                             target_mode='single'):
    """
    given data directories to form .pkl file stores all volume and slice information

    :param data_dir:str, directory to store the mri .h5 data
    :param data_info_dir:str, directory to store the output data info file
    :param num_files:int, number of file choose to read, default -1 leads to read all file in dir
    :param num_pd_files:int, number of file to read without fat suppression, default -1
    :param num_pdfs_files:int, number of file to read with fat suppression, default -1
    :param data_info_file_name:str, output .pkl file name
    :return: a list store element (filename (str), index (int))
    """

    # check and create data_info_dir
    isExist = os.path.exists(data_info_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(data_info_dir)
        print("The data_info directory is created!")

    # load file names in a list
    file_list = []
    for entry in sorted(os.listdir(data_dir)):
        ext = entry.split(".")[-1]
        if "." in entry and ext == "h5":
            file_list.append(entry)
            if num_files == len(file_list):
                break

    # load data info in a list
    data_info_list = []
    pd_count = 0
    pdfs_count = 0
    for i in tqdm(range(len(file_list))):
        file_name = file_list[i]
        if pd_count == num_pd_files and pdfs_count == num_pdfs_files:
            break

        file_path = os.path.join(data_dir, file_name)
        data = h5py.File(file_path, mode="r")
        image_rss = np.array(data["reconstruction_rss"])
        if target_mode == 'single':
            image_esc = np.array(data["reconstruction_esc"])
        else:
            image_esc = None
        acquisition = data.attrs["acquisition"]
        num_slice = len(image_rss)

        if acquisition == "CORPD_FBK":
            if pd_count == num_pd_files:
                continue
            for j in range(NUM_CUT_SLICES, num_slice - NUM_CUT_SLICES):
                data_info_list.append((file_name, j))
            pd_count += 1
        else:
            if pdfs_count == num_pdfs_files:
                continue
            for j in range(NUM_CUT_SLICES, num_slice - NUM_CUT_SLICES):
                data_info_list.append((file_name, j))
            pdfs_count += 1

        if if_cache_image:
            slice_path = cache_path + file_name + '/'
            if not os.path.exists(slice_path):
                os.makedirs(slice_path)
            for k in range(NUM_CUT_SLICES, num_slice - NUM_CUT_SLICES):
                slice_path_rss = slice_path + str(k) + '_rss.pkl'
                with open(slice_path_rss, "wb") as f:
                    pickle.dump(image_rss[k], f)
                if image_esc is not None:
                    slice_path_esc = slice_path + str(k) + '_esc.pkl'
                    with open(slice_path_esc, "wb") as f:
                        pickle.dump(image_esc[k], f)

    with open(os.path.join(data_info_dir, f"{data_info_file_name}.pkl"), "wb") as f:
        pickle.dump(data_info_list, f)
        print(f"{data_info_file_name}, num of slices: {len(data_info_list)}")

    return data_info_list
