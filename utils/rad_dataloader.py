import h5py
import os
import pickle
import torch
import fastmri
from imutils import paths
import numpy as np
from help_func import print_var_detail
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.transform_util import pad_to_pool, normalize_zero_to_one
from utils.transform_util import *
from fastmri.data import transforms, mri_data, subsample
import random
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
import cv2
from utils.general_dataloader_cache import random_rotate_90_180_270_torch, random_flip_2d_controlled_torch
class RedDataSetMaeArray(Dataset):
    """
    A customized dataset class used to create and preprocess fastmri data for ipt training.
    get_item outputs list of input and target image pairs, list of filename and list of slice index

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
    patch_size: int
        desired patch size
    scales: list of int
        given scale size during image processing
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
    func_list: list of functions
        store functions used to preprocess image for each head-tail
    """

    def __init__(
            self,
            data_dir, random_seed, val_split, image_size, is_train
    ):

        self.data_dir = data_dir
        self.random_seed = random_seed
        self.val_split = val_split
        self.image_size = image_size
        self.is_train = is_train

        # construct training and validation dataset
        imagePaths = list(paths.list_images(self.data_dir))
        np.random.seed(self.random_seed)
        np.random.shuffle(imagePaths)

        valPathsLen = int(len(imagePaths) * val_split)
        trainPathsLen = len(imagePaths) - valPathsLen
        self.trainPaths = imagePaths[:trainPathsLen]
        self.valPaths = imagePaths[trainPathsLen:]
        print('train size: ' + str(len(self.trainPaths)))
        print('validation size: ' + str(len(self.valPaths)))

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)
        else:
            return len(self.valPaths)

    def __getitem__(self, idx):
        # load image data

        if self.is_train:
            # image = Image.open(self.trainPaths[idx])
            image = cv2.imread(self.trainPaths[idx])
            filename = self.trainPaths[idx]
        else:
            # image = Image.open(self.valPaths[idx])
            image = cv2.imread(self.valPaths[idx])
            filename = self.valPaths[idx]

        # # Define a transform to convert PIL
        # # image to a Torch tensor
        # transform_toTensor = transforms.Compose([
        #     transforms.PILToTensor()
        # ])
        #
        # # Convert the PIL image to Torch tensor
        # transform_resize = transforms.Resize(size=self.image_size)
        # image = transform_resize(image)
        # image = transform_toTensor(image)[0].unsqueeze(0)
        # image = image / 255.0 # assume input image scale 0-255
        #
        # #use the same normalize as fastmri
        # if torch.max(image) > 1e-12:
        #     image = image / torch.max(image)
        # else:
        #     image = torch.zeros(image.shape)
        #
        # image_tensor = abs(image.squeeze(0)) # [H, W]


        # return image_tensor.unsqueeze(0).repeat(3, 1, 1)
        image = np.array(image)
        image = np.stack((image,) * 3, axis=0)
        return image

class RedDataSetMaeTensor(Dataset):
    """
    A customized dataset class used to create and preprocess fastmri data for ipt training.
    get_item outputs list of input and target image pairs, list of filename and list of slice index

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
    patch_size: int
        desired patch size
    scales: list of int
        given scale size during image processing
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
    func_list: list of functions
        store functions used to preprocess image for each head-tail
    """

    def __init__(
            self,
            data_dir, random_seed, val_split, image_size, is_train
    ):

        self.data_dir = data_dir
        self.random_seed = random_seed
        self.val_split = val_split
        self.image_size = image_size
        self.is_train = is_train

        # construct training and validation dataset
        imagePaths = list(paths.list_images(self.data_dir))
        np.random.seed(self.random_seed)
        np.random.shuffle(imagePaths)

        valPathsLen = int(len(imagePaths) * val_split)
        trainPathsLen = len(imagePaths) - valPathsLen
        self.trainPaths = imagePaths[:trainPathsLen]
        self.valPaths = imagePaths[trainPathsLen:]
        print('train size: ' + str(len(self.trainPaths)))
        print('validation size: ' + str(len(self.valPaths)))

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)
        else:
            return len(self.valPaths)

    def __getitem__(self, idx):
        # load image data

        if self.is_train:
            image = Image.open(self.trainPaths[idx])
            # image = cv2.imread(self.trainPaths[idx])
            filename = self.trainPaths[idx]
        else:
            image = Image.open(self.valPaths[idx])
            # image = cv2.imread(self.valPaths[idx])
            filename = self.valPaths[idx]

        # Define a transform to convert PIL
        # image to a Torch tensor
        transform_toTensor = transforms.Compose([
            transforms.PILToTensor()
        ])

        # Convert the PIL image to Torch tensor
        transform_resize = transforms.Resize(size=self.image_size)
        image = transform_resize(image)
        image = transform_toTensor(image)
        # image = image / 255.0 # assume input image scale 0-255
        #
        # #use the same normalize as fastmri
        # if torch.max(image) > 1e-12:
        #     image = image / torch.max(image)
        # else:
        #     image = torch.zeros(image.shape)
        #
        # image_tensor = abs(image.squeeze(0)) # [H, W]

        # print_var_detail(image)
        return image.repeat(1, 1, 1)
        # return image

class RedDataSetMae(Dataset):
    """
    A customized dataset class used to create and preprocess fastmri data for ipt training.
    get_item outputs list of input and target image pairs, list of filename and list of slice index

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
    patch_size: int
        desired patch size
    scales: list of int
        given scale size during image processing
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
    func_list: list of functions
        store functions used to preprocess image for each head-tail
    """

    def __init__(
            self,
            data_dir, random_seed, val_split, image_size, is_train
    ):

        self.data_dir = data_dir
        self.random_seed = random_seed
        self.val_split = val_split
        self.image_size = image_size
        self.is_train = is_train

        # construct training and validation dataset
        imagePaths = list(paths.list_images(self.data_dir))
        np.random.seed(self.random_seed)
        np.random.shuffle(imagePaths)

        valPathsLen = int(len(imagePaths) * val_split)
        trainPathsLen = len(imagePaths) - valPathsLen
        self.trainPaths = imagePaths[:trainPathsLen]
        self.valPaths = imagePaths[trainPathsLen:]
        print('train size: ' + str(len(self.trainPaths)))
        print('validation size: ' + str(len(self.valPaths)))

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)
        else:
            return len(self.valPaths)

    def __getitem__(self, idx):
        # load image data

        if self.is_train:
            image = Image.open(self.trainPaths[idx])
            filename = self.trainPaths[idx]
        else:
            image = Image.open(self.valPaths[idx])
            filename = self.valPaths[idx]

        # Define a transform to convert PIL
        # image to a Torch tensor
        transform_toTensor = transforms.Compose([
            transforms.PILToTensor()
        ])

        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        transform_resize = transforms.Resize(size=self.image_size)
        image = transform_resize(image)
        image = transform_toTensor(image)[0].unsqueeze(0)
        image = image / 255.0 # assume input image scale 0-255

        #use the same normalize as fastmri
        if torch.max(image) > 1e-12:
            image = image / torch.max(image)
        else:
            image = torch.zeros(image.shape)

        image_tensor = abs(image.squeeze(0)) # [H, W]

        image_tensor_masked = image_tensor  # [H, W]
        # image_tensor_masked, mask = MAE_mask(image_tensor)

        return image_tensor.unsqueeze(0).repeat(3, 1, 1), image_tensor_masked.unsqueeze(0).repeat(3, 1, 1)

class RedDataSetMaeSampleWeights(Dataset):
    """
    A customized dataset class used to create and preprocess fastmri data for ipt training.
    get_item outputs list of input and target image pairs, list of filename and list of slice index

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
    patch_size: int
        desired patch size
    scales: list of int
        given scale size during image processing
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
    func_list: list of functions
        store functions used to preprocess image for each head-tail
    """

    def __init__(
            self,
            data_dir, random_seed, val_split, image_size, is_train, random_flip_chance = 0.5, random_rotate_chance = 0.5
    ):

        self.data_dir = data_dir
        self.random_seed = random_seed
        self.val_split = val_split
        self.image_size = image_size
        self.is_train = is_train
        self.random_flip_chance = random_flip_chance
        self.random_rotate_chance = random_rotate_chance

        # construct training and validation dataset
        imagePaths = list(paths.list_images(self.data_dir))
        np.random.seed(self.random_seed)
        np.random.shuffle(imagePaths)

        valPathsLen = int(len(imagePaths) * val_split)
        trainPathsLen = len(imagePaths) - valPathsLen
        self.trainPaths = imagePaths[:trainPathsLen]
        self.valPaths = imagePaths[trainPathsLen:]
        print('train size: ' + str(len(self.trainPaths)))
        print('validation size: ' + str(len(self.valPaths)))

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)
        else:
            return len(self.valPaths)

    def __getitem__(self, idx):
        # load image data

        if self.is_train:
            image = Image.open(self.trainPaths[idx])
            # filename = self.trainPaths[idx]
        else:
            image = Image.open(self.valPaths[idx])
            # filename = self.valPaths[idx]

        # Define a transform to convert PIL
        # image to a Torch tensor
        transform_toTensor = transforms.Compose([
            transforms.PILToTensor()
        ])

        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        transform_resize = transforms.Resize(size=self.image_size)
        image = transform_resize(image)
        image = transform_toTensor(image)[0].unsqueeze(0)
        image = image / 255.0 # assume input image scale 0-255

        # random flip
        image = random_flip_2d_controlled_torch(image, self.random_flip_chance, self.random_flip_chance)

        # random rotate
        if random.random() < self.random_rotate_chance:
            image = random_rotate_90_180_270_torch(image)

        #use the same normalize as fastmri
        if torch.max(image) > 1e-12:
            image = image / torch.max(image)
        else:
            image = torch.zeros(image.shape)

        image_tensor = abs(image.squeeze(0)) # [H, W]
        image_tensor = image_tensor.to(torch.float16)
        # image_tensor_masked = image_tensor  # [H, W]
        # image_tensor_masked, mask = MAE_mask(image_tensor)
        sample_weight = torch.tensor(1.0, dtype=torch.float16)

        return image_tensor.unsqueeze(0).repeat(3, 1, 1), sample_weight