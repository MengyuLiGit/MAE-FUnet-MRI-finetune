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
from multiprocessing import Pool, cpu_count
import skimage
from utils.transform_util import CustomRandomFlip, CustomRandomRot
import math
from scipy.ndimage import center_of_mass
from utils.transform_util import rotate_around_axis_position, find_mask_bounds, random_crop_given_bounds, upscale_rotate_downscale_fast, upscale_rotate_downscale_binary
import SimpleITK as sitk



def process_image_array(image_array, image_size, if_normalize = True, scale_coeff_min = None):
    # if rgb image, convert into grayscale
    if len(image_array.shape) == 3:
        if image_array.shape[-1] == 3:
            image_array = rgb2gray(image_array)

    # time2 = time.time()
    # print('2-1: ' + str(time2 - time1))

    # convert to np float
    image_array = np.vstack(image_array).astype(np.float64)
    # resize
    shape = image_array.shape  # [H, W]

    if shape[0] > shape[1]:  # H > W
        sizeH = image_size
        sizeW = int(shape[1] * image_size / shape[0])
    else:
        sizeW = image_size
        sizeH = int(shape[0] * image_size / shape[1])

    try:
        res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
    except:
        # print("cannot resize")
        # print(self.patient_loader.patients[patient_index].patient_ID)
        # print(slice_index)
        # self.patient_loader.patients[patient_index].print_all_mri_sessions()
        image_tensor = torch.zeros(image_size, image_size)

        return image_tensor

    # pad the given size
    extra_left = int((image_size - sizeW) / 2)
    extra_right = image_size - sizeW - extra_left
    extra_top = int((image_size - sizeH) / 2)
    extra_bottom = image_size - sizeH - extra_top
    image_array = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                         mode='constant', constant_values=0)

    # convert to tensor
    image_tensor = torch.from_numpy(image_array).float()

    if if_normalize:
        if scale_coeff_min is None:
            # normalize
            maximum = torch.max(image_tensor)
            minimum = torch.min(image_tensor)
            scale = maximum - minimum
            if scale > 0:
                scale_co = 1. / scale
            else:
                scale_co = 0
            image_tensor = (image_tensor - minimum) * scale_co
        else:
            image_tensor = (image_tensor - scale_coeff_min[1]) * scale_coeff_min[0]
    return image_tensor

def process_image_array3D(image_array, image_size = None, if_N4Bias = False, clip_range = [0, 99.9]):
    # if rgb image, convert into grayscale
    if len(image_array.shape) != 3:
        raise ValueError('image_array has wrong shape')


    # clip
    shape = image_array.shape  # [C, H, W]
    if image_size == None:
        image_size = max(shape[1], shape[2])

    if if_N4Bias:
        sitk_img = sitk.GetImageFromArray(image_array)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_img = corrector.Execute(sitk_img)
        # Convert back to NumPy
        image_array = sitk.GetArrayFromImage(corrected_img)

    lower_percentile, upper_percentile = np.percentile(image_array, clip_range)
    image_array = np.clip(image_array, lower_percentile, upper_percentile)

    # Normalize to [0, 1]
    minimum = image_array.min()
    maximum = image_array.max()
    scale = maximum - minimum
    if scale > 0:
        scale_co = 1. / scale
    else:
        scale_co = 0
    scale_coeff_min = (scale_co, minimum)
    image_volume = []
    for i in range(shape[0]):
        image = image_array[i, :, :]
        image_tensor = process_image_array(image, image_size, False, scale_coeff_min).float()  # [H, W]
        image_volume.append(image_tensor.numpy())
    image_volume = np.array(image_volume, np.float32)
    return torch.from_numpy(image_volume).float()


def load_all_slice_in_folder(folder_path, mask_channel=1, image_size=None, if_flip=False, if_process_image=True):
    image_array_temp_list = []
    file_paths = os.listdir(folder_path)
    file_paths.sort(key=natural_keys)
    for file in file_paths:
        path_temp = os.path.join(folder_path, file)
        if path_temp.endswith(".pkl"):
            with open(path_temp, "rb") as f:
                image_array = pickle.load(f)
                if if_flip:
                    image_array = np.flip(image_array, axis=1)
                if if_process_image:
                    if image_size is not None:
                        image_array = process_image_array(image_array, image_size)
                    else:
                        image_size_temp = image_array.shape[-1]
                        image_array = process_image_array(image_array, image_size_temp)
                    image_array_temp_list.append(image_array.numpy())
                else:
                    image_array_temp_list.append(image_array)
    image_array_volume = np.array(image_array_temp_list, np.float32)
    image_tensor_volume = torch.from_numpy(image_array_volume).unsqueeze(1).repeat(1, mask_channel, 1, 1)
    return image_tensor_volume


def predict_img(net,
                full_img,
                device,
                n_classes,
                out_threshold=0.5, **kwargs):
    net.eval()

    img = full_img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img, **kwargs)
        output = output.cpu()
        # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        # output = (output_tran + output_tran) / 2.0
        if n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask.long().numpy()


def longest_sequential_num(numbers, step_size=1):
    max, count_ = 1, 1
    start_idx, end_idx = 0, 0
    for i in range(len(numbers) - 1):
        # if difference between number and his follower is with step size,they are in sequence
        if (numbers[i + 1] - numbers[i] < step_size) and (numbers[i + 1] > numbers[i]):
            count_ = count_ + 1
        else:
            if count_ > max:
                max = count_
                end_idx = i
                start_idx = i + 1 - max
            # Reset counter
            count_ = 1
    return (start_idx, end_idx, max)


def detect_plane_direction(patient_handler, _model_extraction, _model_plane_detect, image_tensor_volume, DETECT_PLANE,
                           topk=10):
    image_tensor_volume_mask, valid_col = patient_handler.mask_generation_by_volume(_model_extraction,
                                                                                    image_tensor_volume,
                                                                                    if_exclude_empty=True)
    if len(image_tensor_volume_mask) > 0:
        image_tensor_volume_mask[image_tensor_volume_mask > 0.5] = 1
        image_tensor_volume_mask[image_tensor_volume_mask <= 0.5] = 0
        image_tensor_volume = image_tensor_volume[valid_col, :]
        area_sizes = torch.count_nonzero(image_tensor_volume_mask, dim=(1, 2, 3))
        if len(area_sizes) > 0:
            # Get the top 10 largest numbers and their indices
            topk = min(len(area_sizes), topk)
            top_values, top_indices = torch.topk(area_sizes, topk)
            image_tensor_volume_topk = image_tensor_volume[top_indices, :]
            detect_sequence, detected_confidence = patient_handler.sequence_detection_by_volume(_model_plane_detect,
                                                                                                image_tensor_volume_topk,
                                                                                                DETECT_PLANE)
            return detect_sequence, detected_confidence
    return None, None


def load_generated_mask_slice(path_temp, resize, labels, label_values):
    if path_temp.endswith(".pkl"):
        with open(path_temp, "rb") as f:
            image_array = pickle.load(f)

            shape = image_array.shape  # [H, W]

            if shape[0] > shape[1]:  # H > W
                sizeH = resize
                sizeW = int(shape[1] * resize / shape[0])
            else:
                sizeW = resize
                sizeH = int(shape[0] * resize / shape[1])

            res_mask = skimage.transform.resize(image_array, (sizeW, sizeH), order=0,
                                                preserve_range=True,
                                                anti_aliasing=False)
            # image_array = process_image_array(image_array, image_size)

            # pad the given size
            extra_left = int((resize - sizeW) / 2)
            extra_right = resize - sizeW - extra_left
            extra_top = int((resize - sizeH) / 2)
            extra_bottom = resize - sizeH - extra_top
            image_array_mask = np.pad(res_mask, ((extra_top, extra_bottom), (extra_left, extra_right)),
                                      mode='constant', constant_values=0)

            # mark image binary depends on labels given
            # if len(labels) > 1:
            output_labels = np.zeros((len(labels), resize, resize))  # [n_class, H, W]
            for i in range(len(label_values)):
                label_value = label_values[i]
                seg_value = np.isin(image_array_mask, label_value)
                output_labels[i] = seg_value

            output_labels[output_labels > 0.1] = 1
            output_labels[output_labels <= 0.1] = 0
            return output_labels
    return None


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
        self.index_list = self.get_index_list(if_latest=if_latest)
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

    def get_index_list(self, if_latest=False):
        index_list_temp = []  # [[patient index, slice index per patient]], assume [0, 0, 0, 0] for slice index
        for i in tqdm(range(self.num_patients)):
            index_list_i = []  # [[0, 0, 0, 0], [0, 0, 0, 1]...[0, 0, 0, [0, 0]], [0, 0, 0, [0, 1]]]
            if self.mri_sequence is None:
                index_list_i = sum(self.patient_loader.patients[i].sequence_slice_index_lists_dict.values(), [])
            else:
                for dir_index in self.patient_loader.patients[i].dir_index_list:
                    for sequence_str in self.mri_sequence:
                        slice_list = self.patient_loader.patients[i].sequence_slice_index_lists_dict[sequence_str]
                        slice_list_dir = self.patient_loader.patients[i].get_slice_in_list_by_dir(dir_index, slice_list)
                        # remove choose fraction slices
                        n = int(self.drop_fraction * len(slice_list_dir))
                        if n > 0:
                            slice_list_dir = slice_list_dir[n:-n]
                        index_list_i = index_list_i + slice_list_dir
            # add patient index
            # print(i)
            # print(index_list_i)
            # print(index_list_temp)
            index_list_temp = index_list_temp + [[i] + slice_index for slice_index in
                                                 index_list_i]  # [[0, 0, 0, 0, 0], ...]
        return index_list_temp

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)
        else:
            return len(self.valPaths)

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            slice_index = self.trainPaths[idx][1:]
            file_index = self.trainPaths[idx][1:-1]
            patient_index = self.trainPaths[idx][0]
        else:
            slice_index = self.valPaths[idx][1:]
            file_index = self.valPaths[idx][1:-1]
            patient_index = self.valPaths[idx][0]

        try:
            image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)
        except:
            print("dicom_data.pixel_array corrupted")
            print(patient_index)
            print(slice_index)
            image_tensor = torch.zeros(self.resize, self.resize)
            image_sequence = 0
            image_label = [1, 0]
            return image_tensor.unsqueeze(0), torch.tensor(image_sequence), torch.tensor(image_label), patient_index

        # if rgb image, convert into grayscale
        if len(image_array.shape) == 3:
            if image_array.shape[-1] == 3:
                image_array = rgb2gray(image_array)

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
            print("slice_index", slice_index)
            print('shape')
        return image_tensor.unsqueeze(0), torch.tensor(image_sequence), torch.tensor(image_label), patient_index


class GeneralDataLoaderSequenceClass(GeneralDataset):
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
            detect_sequence,
            max_train_len=None,
            max_val_len=None,
            n_color=1,
            if_rot_flip=True,
            **kwargs
    ):
        super().__init__(resize, drop_fraction, mri_sequence, patient_loader, labels, label_values, random_seed,
                         val_split, is_train, if_latest, max_train_len, max_val_len)
        self.detect_sequence = detect_sequence  # ['T1_T1flair', 'T1post', 'T2_T2star', 'T2flair_flair', 'PD', 'DTI_DWI', 'MISC']
        self.n_color = n_color
        self.if_rot_flip = if_rot_flip
        # apply random rotate and flip
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            slice_index = self.trainPaths[idx][1:]
            file_index = self.trainPaths[idx][1:-1]
            patient_index = self.trainPaths[idx][0]
        else:
            slice_index = self.valPaths[idx][1:]
            file_index = self.valPaths[idx][1:-1]
            patient_index = self.valPaths[idx][0]

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        image_label = torch.zeros(len(self.detect_sequence) + 1)  # size of label = number of sequence + 1
        image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class

        try:
            image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)
        except:
            print("dicom_data.pixel_array corrupted")
            print(patient_index)
            print(slice_index)
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label

        # if rgb image, convert into grayscale
        if len(image_array.shape) == 3:
            if image_array.shape[-1] == 3:
                image_array = rgb2gray(image_array)

        # convert to np float
        image_array = np.vstack(image_array).astype(np.float64)

        # resize
        shape = image_array.shape  # [H, W]

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
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label

        # pad the given size
        extra_left = int((self.resize - sizeW) / 2)
        extra_right = self.resize - sizeW - extra_left
        extra_top = int((self.resize - sizeH) / 2)
        extra_bottom = self.resize - sizeH - extra_top
        res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                         mode='constant', constant_values=0)
        image_array = res_pad

        # apply rot and flip
        if self.if_rot_flip:
            image_array, _ = self.custom_flip_func(image_array, image_array)
            image_array, _ = self.custom_rot_func(image_array, image_array)

        # convert to tensor
        # image_tensor = torch.from_numpy(image_array).float()
        image_tensor = torch.from_numpy(image_array.copy()).float()  # avoid neg stride
        # normalize
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff

        # 1-hot classify
        # get cooresponding sequence
        sequence_tag = [k for k, v in
                        self.patient_loader.patients[patient_index].sequence_slice_index_lists_dict.items() if
                        slice_index in v]  # list of sequence
        if len(sequence_tag) > 1:
            print(sequence_tag)
        if len(sequence_tag) > 0:
            sequence_tag = sequence_tag[0]
        else:
            sequence_tag = ''
        # reassign label to 1-hot class vector
        # for now use only the first detected sequence
        if sequence_tag in self.detect_sequence:
            if sequence_tag == 'PD':
                ds = self.patient_loader.patients[patient_index].load_dicom_mri(slice_index)
                idx_pd = self.patient_loader.patients[patient_index].sequence_dir_index_lists_dict[sequence_tag].index(
                    file_index)

                # check by echo times
                echo_times = self.patient_loader.patients[patient_index].PD_echo_time[idx_pd]
                if len(echo_times) > 1:  # if have more than one echo time in series
                    # print("check echo_time")

                    if ds.__contains__(self.patient_loader.patients[patient_index].dicom_dict['EchoTime']):
                        echo_time = ds[self.patient_loader.patients[patient_index].dicom_dict[
                            'EchoTime']].value  # get the current echo time of slice
                        if (max(echo_times) - echo_time) < (echo_time - min(echo_times)):
                            sequence_tag = 'T2_T2star'
                        # all_echo_times = echo_times + [echo_time]
                else:  # check by EchoNumbers (0018,0086)
                    if ds.__contains__([0x0018, 0x0086]):
                        echo_number = ds[[0x0018, 0x0086]].value
                        if echo_number == 2:
                            sequence_tag = 'T2_T2star'

            label_idx = self.detect_sequence.index(sequence_tag)
            image_label[-1] = 0.0
            image_label[label_idx] = 1.0

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            self.patient_loader.patients[patient_index].print_all_mri_sessions()
            print("file_index", file_index)
            print("slice_index", slice_index)
            print('shape')
        return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label


class GeneralDataLoaderPlaneClass(GeneralDataset):
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
            detect_plane,
            max_train_len=None,
            max_val_len=None,
            n_color=1,
            detect_plane_tags_paths=[],
            brain_area_threshold=None,
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.patient_loader = patient_loader
        self.num_patients = len(patient_loader.patients)
        self.detect_plane = detect_plane  # ['AX', 'COR', 'SAG']
        self.detect_plane_dic = {}
        self.n_color = n_color

        for i in range(len(detect_plane_tags_paths)):
            detect_plane_tags_path = detect_plane_tags_paths[i]
            my_file = open(detect_plane_tags_path, "r")
            data = my_file.read()
            data_into_list = data.split("\n")
            self.detect_plane_dic[self.detect_plane[i]] = data_into_list
            my_file.close()

        self.SeriesDescription_idx = self.patient_loader.patients[i].info_tags.index('SeriesDescription')
        self.index_list = self.get_index_list(if_latest=if_latest)
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

        # apply random rotate and flip
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    def get_index_list(self, if_latest=False):
        index_list_temp = []  # [[patient index, slice index per patient]], assume [0, 0, 0, 0] for slice index
        for i in tqdm(range(self.num_patients)):
            index_list_i = []  # [[0, 0, 0, 0], [0, 0, 0, 1]...[0, 0, 0, [0, 0]], [0, 0, 0, [0, 1]]]
            if self.mri_sequence is None:
                index_list_i = sum(self.patient_loader.patients[i].sequence_slice_index_lists_dict.values(), [])
            else:
                for dir_index in self.patient_loader.patients[i].dir_index_list:
                    for sequence_str in self.mri_sequence:
                        slice_list = self.patient_loader.patients[i].sequence_slice_index_lists_dict[sequence_str]
                        slice_list_dir = self.patient_loader.patients[i].get_slice_in_list_by_dir(dir_index, slice_list)
                        # remove choose fraction slices
                        n = int(self.drop_fraction * len(slice_list_dir))
                        if n > 0:
                            slice_list_dir = slice_list_dir[n:-n]
                        index_list_i = index_list_i + slice_list_dir
            # check if slice has plane dict tags
            slice_list_dir_plane = []
            for slice_index in index_list_i:
                info_tag = self.patient_loader.patients[i].get_slice_info_given_slice(slice_index)[
                    self.SeriesDescription_idx]
                for key in self.detect_plane_dic:
                    if any(clear_string_char(info_tag, [' ', '_', '/']).upper()
                           == clear_string_char(x, [' ', '_', '/']).upper() for x in
                           self.detect_plane_dic[key]):
                        slice_list_dir_plane.append(slice_index)
                        break

            index_list_i = slice_list_dir_plane

            index_list_temp = index_list_temp + [[i] + slice_index for slice_index in
                                                 index_list_i]  # [[0, 0, 0, 0, 0], ...]
        return index_list_temp

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            slice_index = self.trainPaths[idx][1:]
            file_index = self.trainPaths[idx][1:-1]
            patient_index = self.trainPaths[idx][0]
        else:
            slice_index = self.valPaths[idx][1:]
            file_index = self.valPaths[idx][1:-1]
            patient_index = self.valPaths[idx][0]

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        image_label = torch.zeros(len(self.detect_plane) + 1)  # size of label = number of sequence + 1
        image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class

        try:
            image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)
        except:
            print("dicom_data.pixel_array corrupted")
            print(patient_index)
            print(slice_index)
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label

        # if rgb image, convert into grayscale
        if len(image_array.shape) == 3:
            if image_array.shape[-1] == 3:
                image_array = rgb2gray(image_array)

        # convert to np float
        image_array = np.vstack(image_array).astype(np.float64)

        # resize
        shape = image_array.shape  # [H, W]

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
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label

        # pad the given size
        extra_left = int((self.resize - sizeW) / 2)
        extra_right = self.resize - sizeW - extra_left
        extra_top = int((self.resize - sizeH) / 2)
        extra_bottom = self.resize - sizeH - extra_top
        res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                         mode='constant', constant_values=0)
        image_array = res_pad

        # apply rot and flip
        image_array, _ = self.custom_flip_func(image_array, image_array)
        image_array, _ = self.custom_rot_func(image_array, image_array)

        # convert to tensor
        image_tensor = torch.from_numpy(image_array.copy()).float()

        # normalize
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff

        # 1-hot classify
        # get cooresponding tags

        info_tag = self.patient_loader.patients[patient_index].get_slice_info_given_slice(slice_index)[
            self.SeriesDescription_idx]
        # reassign label to 1-hot class vector
        # for now use only the first detected sequence
        plane_tag = None
        for key in self.detect_plane_dic:
            if any(clear_string_char(info_tag, [' ', '_', '/']).upper()
                   == clear_string_char(x, [' ', '_', '/']).upper() for x in self.detect_plane_dic[key]):
                plane_tag = key
                break

        if plane_tag is not None:
            label_idx = self.detect_plane.index(plane_tag)
            image_label[-1] = 0.0
            image_label[label_idx] = 1.0

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            self.patient_loader.patients[patient_index].print_all_mri_sessions()
            print("file_index", file_index)
            print("slice_index", slice_index)
            print('shape')
        return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label


class GeneralDatasetMae(GeneralDataset):
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
            **kwargs
    ):
        super().__init__(resize, drop_fraction, mri_sequence, patient_loader, labels, label_values, random_seed,
                         val_split, is_train, if_latest, max_train_len)

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            slice_index = self.trainPaths[idx][1:]
            file_index = self.trainPaths[idx][1:-1]
            patient_index = self.trainPaths[idx][0]
        else:
            slice_index = self.valPaths[idx][1:]
            file_index = self.valPaths[idx][1:-1]
            patient_index = self.valPaths[idx][0]

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        # image_label = torch.zeros(len(self.detect_sequence) + 1)  # size of label = number of sequence + 1
        # image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class

        try:
            image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)
        except:
            print("dicom_data.pixel_array corrupted")
            print(patient_index)
            print(slice_index)
            image_tensor = torch.zeros(self.resize, self.resize)
            image_tensor_masked = image_tensor
            # print("dicom_data.pixel_array corrupted")
            # print(self.patient_loader.patients[patient_index].patient_ID)
            # print(slice_index)
            # self.patient_loader.patients[patient_index].print_all_mri_sessions()
            return image_tensor.unsqueeze(0).repeat(3, 1, 1), image_tensor_masked.unsqueeze(0).repeat(3, 1, 1)

        # if rgb image, convert into grayscale
        if len(image_array.shape) == 3:
            if image_array.shape[-1] == 3:
                image_array = rgb2gray(image_array)

        # time2 = time.time()
        # print('2-1: ' + str(time2 - time1))

        # convert to np float
        image_array = np.vstack(image_array).astype(np.float64)

        # resize
        shape = image_array.shape  # [H, W]

        if shape[0] > shape[1]:  # H > W
            sizeH = self.resize
            sizeW = int(shape[1] * self.resize / shape[0])
        else:
            sizeW = self.resize
            sizeH = int(shape[0] * self.resize / shape[1])

        try:
            res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
        except:
            print("cannot resize")
            print(self.patient_loader.patients[patient_index].patient_ID)
            print(slice_index)
            self.patient_loader.patients[patient_index].print_all_mri_sessions()
            image_tensor = torch.zeros(self.resize, self.resize)
            image_tensor_masked = image_tensor
            return image_tensor.unsqueeze(0).repeat(3, 1, 1), image_tensor_masked.unsqueeze(0).repeat(3, 1, 1)

        # pad the given size
        extra_left = int((self.resize - sizeW) / 2)
        extra_right = self.resize - sizeW - extra_left
        extra_top = int((self.resize - sizeH) / 2)
        extra_bottom = self.resize - sizeH - extra_top
        res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                         mode='constant', constant_values=0)
        image_array = res_pad

        # convert to tensor
        image_tensor = torch.from_numpy(image_array).float()

        # normalize
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff

        # mark image
        image_tensor_masked = image_tensor  # [H, W]
        # image_tensor_masked, mask = MAE_mask(image_tensor)

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            self.patient_loader.patients[patient_index].print_all_mri_sessions()
            print("file_index", file_index)
            print("slice_index", slice_index)
            print('shape')

        return image_tensor.unsqueeze(0).repeat(3, 1, 1), image_tensor_masked.unsqueeze(0).repeat(3, 1, 1)


class GeneralDataLoaderSlicer(GeneralDataset):
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
            ratio_list,
            max_train_len=None,
            **kwargs
    ):
        super().__init__(resize, drop_fraction, mri_sequence, patient_loader, labels, label_values, random_seed,
                         val_split, is_train, if_latest, max_train_len)
        self.random_combined_slices = self.get_random_combined_slices()
        self.ratio_list = ratio_list
        # self.current_slice_num = 0

    def get_random_combined_slices(self):
        record_slice_num = 0
        random_combined_slices = []
        if self.is_train:
            length = len(self.trainPaths)
        else:
            length = len(self.valPaths)
        while record_slice_num < length:
            H_num = random.randint(1, 16)
            W_num = random.randint(1, 16)
            if record_slice_num + H_num * W_num > length:
                break
            else:
                random_combined_slices.append([H_num, W_num, record_slice_num])
                record_slice_num = record_slice_num + H_num * W_num
        return random_combined_slices  # [[H_num_1, W_num_1, record_slice_num_1], [H_num_2, W_num_2, record_slice_num_2], ...]

    def __len__(self):
        return len(self.random_combined_slices)

    def __getitem__(self, idx):
        # time0 = time.time()

        image_tensor_combined = torch.zeros(self.resize, self.resize)
        image_label = torch.zeros(16 * 16)  # size of label = number of sequence + 1

        # get the slices list
        H_num = self.random_combined_slices[idx][0]
        W_num = self.random_combined_slices[idx][1]

        # setup label given H and W hum
        image_label[(H_num - 1) * 16 + W_num - 1] = 1

        # random pick a ratio
        # ratio = random.choice(self.ratio_list)
        ratio = random.uniform(min(self.ratio_list), max(self.ratio_list) * 1.2)

        # check if full align on left side or top side
        full_ratio = ratio * H_num / W_num

        if full_ratio >= 1.0:
            sizeH = int(self.resize / H_num)
            sizeW = int(sizeH / ratio)
        else:
            sizeW = int(self.resize / W_num)
            sizeH = int(sizeW * ratio)

        record_slice_num = self.random_combined_slices[idx][2]
        slice_list = list(
            range(record_slice_num, record_slice_num + H_num * W_num))  # list of slices given # of stack on H and W

        # self.current_slice_num = self.current_slice_num + H_num * W_num
        #
        # if self.is_train:
        #     length = len(self.trainPaths)
        # else:
        #     length = len(self.valPaths)
        # if idx >= len(self.random_combined_slices) - 1:
        #     print('resample trigger')
        #     self.random_combined_slices = self.get_random_combined_slices()

        # process each images, do not pad! use center crop and resize to align the individual size,
        # thus avoid model to detect dark trans edges to learn slicing
        image_list = []  # np array list, store gray scale images [[H, W],[H, W],[H, W],...]

        for slice_i in slice_list:
            if self.is_train:
                slice_index = self.trainPaths[slice_i][1:]
                file_index = self.trainPaths[slice_i][1:-1]
                patient_index = self.trainPaths[slice_i][0]
            else:
                slice_index = self.valPaths[slice_i][1:]
                file_index = self.valPaths[slice_i][1:-1]
                patient_index = self.valPaths[slice_i][0]

            try:
                image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)
                # if rgb image, convert into grayscale
                if len(image_array.shape) == 3:
                    if image_array.shape[-1] == 3:
                        image_array = rgb2gray(image_array)

                # standardize array
                image_array = np.vstack(image_array).astype(np.float64)  # (H, W)
                # print_var_detail(image_array)

                # resize
                shape = image_array.shape  # [H, W]
                # print_var_detail(image_array)
                # print(shape)
                # find image expand ratio to fit in sizeH, sizeW
                expand_ratio = max(sizeH / shape[0], sizeW / shape[1])
                # print_var_detail(image_array)
                sizeH_temp = int(shape[0] * expand_ratio)
                sizeW_temp = int(shape[1] * expand_ratio)
                # print_var_detail(image_array)

                # print_var_detail(image_array)
                # resize to align side while keeping ratio
                try:
                    res = cv2.resize(image_array, dsize=(sizeW_temp, sizeH_temp), interpolation=cv2.INTER_LINEAR)
                except:
                    res = np.zeros((sizeH_temp, sizeW_temp))

                # center crop to size
                res = center_crop_with_pad_np(res, sizeH, sizeW)  # (H, W)

                # convert to tensor
                image_tensor = torch.from_numpy(res).float()
                # print_var_detail(image_tensor)
                # normalize
                maximum = torch.max(image_tensor)
                minimum = torch.min(image_tensor)
                scale = maximum - minimum
                if scale > 0:
                    scale_coeff = 1. / scale
                else:
                    scale_coeff = 0

                image_tensor = (image_tensor - minimum) * scale_coeff

            except:
                image_tensor = torch.zeros(sizeH, sizeW)

            image_list.append(image_tensor)

        # combine all image to one by H and W num
        for i in range(H_num):
            for j in range(W_num):
                # print('i: ' +  str(i)  + ' j: '+ str(j))
                image_tensor_combined[sizeH * i: sizeH * (i + 1), sizeW * j: sizeW * (j + 1)] = image_list[
                    i * W_num + j]

        return image_tensor_combined.unsqueeze(0), image_label


class GeneralDatasetExtraction(GeneralDataset):
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
            n_color=1,
            if_flip_rot=True,
            **kwargs
    ):
        super().__init__(resize, drop_fraction, mri_sequence, patient_loader, labels, label_values, random_seed,
                         val_split, is_train, if_latest, max_train_len)
        self.n_color = n_color
        self.if_flip_rot = if_flip_rot
        # apply random rotate and flip
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    def get_index_list(self, if_latest=False):
        index_list_temp = []  # [[patient index, slice index per patient]], assume [0, 0, 0, 0] for slice index
        for i in tqdm(range(self.num_patients)):
            index_list_i = []  # [[0, 0, 0, 0], [0, 0, 0, 1]...[0, 0, 0, [0, 0]], [0, 0, 0, [0, 1]]]
            if self.mri_sequence is None:
                index_list_i = sum(self.patient_loader.patients[i].sequence_slice_index_lists_dict.values(), [])
            else:
                for dir_index in self.patient_loader.patients[i].dir_index_list:
                    for sequence_str in self.mri_sequence:
                        slice_list = self.patient_loader.patients[i].sequence_slice_index_lists_dict[sequence_str]
                        slice_list_dir = self.patient_loader.patients[i].get_slice_in_list_by_dir(dir_index, slice_list)
                        # remove choose fraction slices
                        n = int(self.drop_fraction * len(slice_list_dir))
                        if n > 0:
                            slice_list_dir = slice_list_dir[n:-n]
                        index_list_i = index_list_i + slice_list_dir

            # choose only (H, W, slice_index) image with mask
            index_list_i_mask = []
            for slice_index in index_list_i:
                if len(slice_index[-1]) == 2:
                    # check by the size of the mask
                    image_mask = self.patient_loader.patients[i].load_nifti_mask_slice(slice_index)
                    if image_mask is not None:
                        image_mask[image_mask > 1e-7] = 1
                        image_mask[image_mask <= 1e-7] = 0
                        mask_area = np.count_nonzero(image_mask == 1)
                        if mask_area > 256:  # masked area
                            # print('find area')
                            index_list_i_mask.append(slice_index)

            index_list_temp = index_list_temp + [[i] + slice_index for slice_index in
                                                 index_list_i_mask]  # [[0, 0, 0, 0, 0], ...]
        return index_list_temp

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            slice_index = self.trainPaths[idx][1:]
            file_index = self.trainPaths[idx][1:-1]
            patient_index = self.trainPaths[idx][0]
        else:
            slice_index = self.valPaths[idx][1:]
            file_index = self.valPaths[idx][1:-1]
            patient_index = self.valPaths[idx][0]

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        # image_label = torch.zeros(len(self.detect_sequence) + 1)  # size of label = number of sequence + 1
        # image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class

        try:
            image_array = self.patient_loader.patients[patient_index].load_mri_slice(slice_index)  # H, W
            image_array_mask = self.patient_loader.patients[patient_index].load_nifti_mask_slice(slice_index)  # H, W
        except:
            # print("dicom_data.pixel_array corrupted")
            # print(patient_index)
            # print(slice_index)
            image_tensor = torch.zeros(self.resize, self.resize)
            image_tensor_masked = image_tensor
            image_tensor_mask_binary = image_tensor
            # print("dicom_data.pixel_array corrupted")
            # print(self.patient_loader.patients[patient_index].patient_ID)
            # print(slice_index)
            # self.patient_loader.patients[patient_index].print_all_mri_sessions()
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
                image_tensor_masked.unsqueeze(0), \
                image_tensor_mask_binary.unsqueeze(0)

        # if rgb image, convert into grayscale
        if len(image_array.shape) == 3:
            if image_array.shape[-1] == 3:
                image_array = rgb2gray(image_array)

        # time2 = time.time()
        # print('2-1: ' + str(time2 - time1))

        # convert to np float
        image_array = np.vstack(image_array).astype(np.float64)
        image_array_mask = np.vstack(image_array_mask).astype(np.float64)
        # resize
        shape = image_array.shape  # [H, W]

        if shape[0] > shape[1]:  # H > W
            sizeH = self.resize
            sizeW = int(shape[1] * self.resize / shape[0])
        else:
            sizeW = self.resize
            sizeH = int(shape[0] * self.resize / shape[1])

        try:
            res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
            res_mask = cv2.resize(image_array_mask, dsize=(sizeW, sizeH), interpolation=cv2.INTER_LINEAR)
        except:
            # print("cannot resize")
            # print(self.patient_loader.patients[patient_index].patient_ID)
            # print(slice_index)
            # self.patient_loader.patients[patient_index].print_all_mri_sessions()
            image_tensor = torch.zeros(self.resize, self.resize)
            image_tensor_masked = image_tensor
            image_tensor_mask_binary = image_tensor
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
                image_tensor_masked.unsqueeze(0), \
                image_tensor_mask_binary.unsqueeze(0)

        # pad the given size
        extra_left = int((self.resize - sizeW) / 2)
        extra_right = self.resize - sizeW - extra_left
        extra_top = int((self.resize - sizeH) / 2)
        extra_bottom = self.resize - sizeH - extra_top
        image_array = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
        image_array_mask = np.pad(res_mask, ((extra_top, extra_bottom), (extra_left, extra_right)),
                                  mode='constant', constant_values=0)

        if self.if_flip_rot:
            # apply rot and flip
            image_array, image_array_mask = self.custom_flip_func(image_array, image_array_mask)
            image_array, image_array_mask = self.custom_rot_func(image_array, image_array_mask)

        # convert to tensor
        # image_tensor = torch.from_numpy(image_array).float()
        # image_tensor_masked = torch.from_numpy(image_array_mask).float()
        image_tensor = torch.from_numpy(image_array.copy()).float()
        image_tensor_masked = torch.from_numpy(image_array_mask.copy()).float()
        # normalize
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff
        image_tensor_masked = (
                                      image_tensor_masked - minimum) * scale_coeff  # not really a normalization, multiply binary mask later

        # mark image binary
        image_tensor_mask_binary = copy.deepcopy(image_tensor_masked)
        image_tensor_mask_binary[image_tensor_mask_binary > 1e-7] = 1
        image_tensor_mask_binary[image_tensor_mask_binary <= 1e-7] = 0

        image_tensor_masked = image_tensor * image_tensor_mask_binary

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            self.patient_loader.patients[patient_index].print_all_mri_sessions()
            print("file_index", file_index)
            print("slice_index", slice_index)
            print('shape')

        return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
            image_tensor_masked.unsqueeze(0), \
            image_tensor_mask_binary.unsqueeze(0)


def check_mask_area(mask, labels, label_values, min_area_size, output_path = None):
    if isinstance(mask, str):
        file_path = mask
        if file_path.endswith(".pkl"):
            # path_seg_temp = os.path.join(file_valid_seg_i, file)
            with open(file_path, "rb") as f:
                # start10 = time.time()
                image_output_array = pickle.load(f)
                # start11 = time.time()
                # print("start11 - start10: ", start11 - start10)
                # output_labels = np.zeros((len(labels), x, y))
                #
                # for i in range(len(label_values)):
                #     label_value = label_values[i]
                #     seg_value = np.isin(image_output_array, label_value)
                #     output_labels[i] = seg_value

                # check with all masked area combined
                label_value_1d = [j for sub in label_values for j in sub]
                output_labels = np.isin(image_output_array, label_value_1d)

                # for value in label_value:
                #     seg_value = image_output_array == value
                #     output_labels[i] = output_labels[i] + seg_value
                # start2 = time.time()
                # start12 = time.time()
                # print("start12 - start11: ", start12 - start11)
                output_labels[output_labels > 0.1] = 1
                output_labels[output_labels <= 0.1] = 0
                # start13 = time.time()
                # print("start13 - start12: ", start13 - start12)
                # start3 = time.time()
                # if start3 - start2 > 0.0001:
                #     print(value)
                #     print("start3 - start2: ", start3 - start2)

                mask_area = np.count_nonzero(output_labels == 1)
                # start14 = time.time()
                # print("start14 - start13: ", start14 - start13)
                if mask_area > min_area_size:  # masked area 8 * 8
                    return file_path
    elif isinstance(mask, np.ndarray):
        image_output_array = mask
        # start11 = time.time()
        # print("start11 - start10: ", start11 - start10)
        # output_labels = np.zeros((len(labels), x, y))
        #
        # for i in range(len(label_values)):
        #     label_value = label_values[i]
        #     seg_value = np.isin(image_output_array, label_value)
        #     output_labels[i] = seg_value

        # check with all masked area combined
        label_value_1d = [j for sub in label_values for j in sub]
        output_labels = np.isin(image_output_array, label_value_1d)

        # for value in label_value:
        #     seg_value = image_output_array == value
        #     output_labels[i] = output_labels[i] + seg_value
        # start2 = time.time()
        # start12 = time.time()
        # print("start12 - start11: ", start12 - start11)
        output_labels[output_labels > 0.1] = 1
        output_labels[output_labels <= 0.1] = 0
        # start13 = time.time()
        # print("start13 - start12: ", start13 - start12)
        # start3 = time.time()
        # if start3 - start2 > 0.0001:
        #     print(value)
        #     print("start3 - start2: ", start3 - start2)

        mask_area = np.count_nonzero(output_labels == 1)
        # start14 = time.time()
        # print("start14 - start13: ", start14 - start13)
        if mask_area > min_area_size:  # masked area 8 * 8
            if output_path is not None:
                return output_path
            else:
                return True
    return None


class GeneralDatasetMultiSeg(GeneralDataset):
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
            file_path_text,
            max_train_len=None,
            max_val_len=None,
            n_color=1,
            if_multi_channel=True,
            cache_path = 'E:/',
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.patient_loader = patient_loader
        self.num_patients = len(patient_loader.patients)
        self.n_color = n_color
        self.file_path_text = file_path_text
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.if_multi_channel = if_multi_channel
        self.cache_path = cache_path
        start0 = time.time()
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        print("start1 - start0: ", start1 - start0)


        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)

    def get_index_list(self, if_latest=False):

        # opening the file in read mode
        my_file = open(self.file_path_text, "r")

        # reading the file paths into list
        data = my_file.read()
        data_into_list = data.split("\n")
        my_file.close()

        # random shuffle list
        random.shuffle(data_into_list)

        # seperate to train and test list
        if self.max_train_len is None:
            valPathsLen = int(len(data_into_list) * self.val_split)
            trainPathsLen = len(data_into_list) - valPathsLen
        elif self.max_train_len < len(data_into_list) - int(len(data_into_list) * self.val_split):
            valPathsLen = int(len(data_into_list) * self.val_split)
            trainPathsLen = self.max_train_len
        else:
            print("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if self.max_val_len <= len(data_into_list) - trainPathsLen:
                valPathsLen = self.max_val_len
            else:
                print("assigned maximum val len exceeds total train num of slices")
        data_into_list_train = data_into_list[:trainPathsLen]
        data_into_list_val = data_into_list[-valPathsLen:]

        return (self.generate_input_output_by_file(data_into_list_train),
                self.generate_input_output_by_file(data_into_list_val))

    def generate_input_output_by_file(self, data_into_list):
        file_valid_seg = []
        file_valid_confirm = []
        for data_path in data_into_list:
            data_path_seg = data_path.replace('/mnt/f/', 'F:/')
            # print(data_path_seg)
            data_path_confirm = data_path_seg.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz')
            if os.path.isfile(data_path_confirm) and os.path.isfile(data_path_seg):
                file_valid_seg.append(data_path_seg + '/x')
                file_valid_seg.append(data_path_seg + '/y')
                file_valid_seg.append(data_path_seg + '/z')
                file_valid_confirm.append(data_path_confirm + '/x')
                file_valid_confirm.append(data_path_confirm + '/y')
                file_valid_confirm.append(data_path_confirm + '/z')

        input_image_path_list = []
        output_image_path_list = []
        for i in tqdm(range(len(file_valid_seg))):
            if i > -1:

                file_valid_seg_i = file_valid_seg[i].replace('nifti_seg', 'nifti_seg_cache')
                file_valid_seg_i = file_valid_seg_i.replace('F:/', self.cache_path)
                # print(file_valid_seg_i)
                list_temp_seg_i = []
                list_temp_confirm_i = []

                # get all slice in file
                path_seg_temp_list = []
                file_paths = os.listdir(file_valid_seg_i)
                for file in file_paths:
                    path_seg_temp = os.path.join(file_valid_seg_i, file)
                    path_seg_temp_list.append(path_seg_temp)

                # print(str(i), " file_paths ", len(path_seg_temp_list))
                if len(path_seg_temp_list) < 2048:
                    start0 = time.time()
                    for path_seg_temp in path_seg_temp_list:
                        checked_path_seg_temp = check_mask_area(path_seg_temp, self.labels, self.label_values, 128)
                        if checked_path_seg_temp is not None:
                            list_temp_seg_i.append(checked_path_seg_temp)
                            list_temp_confirm_i.append(
                                checked_path_seg_temp.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz'))
                    start1 = time.time()
                    if (start1 - start0) > 2:
                        print('start1-start0 single: ', start1 - start0)
                        print(file_valid_seg_i)
                else:
                    start0 = time.time()
                    labels_list = [self.labels] * len(path_seg_temp_list)
                    label_values_list = [self.label_values] * len(path_seg_temp_list)
                    min_area_size_list = [64] * len(path_seg_temp_list)
                    with Pool(processes=16) as pool:
                        for checked_path_seg_temp in pool.starmap(
                                check_mask_area,
                                zip(path_seg_temp_list,
                                    labels_list,
                                    label_values_list,
                                    min_area_size_list)):
                            if checked_path_seg_temp is not None:
                                list_temp_seg_i.append(checked_path_seg_temp)
                                list_temp_confirm_i.append(
                                    checked_path_seg_temp.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz'))

                        start1 = time.time()
                        if (start1 - start0) > 2:
                            print('start1-start0 multi: ', start1 - start0)
                            print(file_valid_seg_i)

                list_temp_seg_i.sort(key=natural_keys)
                list_temp_confirm_i.sort(key=natural_keys)

                # check if number matches
                if len(list_temp_seg_i) != len(list_temp_confirm_i):
                    print('unmatched input and target slices num')
                    list_temp_seg_i = []
                    list_temp_confirm_i = []

                # remove choose fraction slices
                n = int(self.drop_fraction * len(list_temp_seg_i))
                if n > 0:
                    list_temp_seg_i = list_temp_seg_i[n:-n]
                    list_temp_confirm_i = list_temp_confirm_i[n:-n]

                input_image_path_list = input_image_path_list + list_temp_confirm_i
                output_image_path_list = output_image_path_list + list_temp_seg_i
        return list(zip(input_image_path_list, output_image_path_list))

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            input_path = self.trainPaths[idx][0]
            output_path = self.trainPaths[idx][1]
        else:
            input_path = self.valPaths[idx][0]
            output_path = self.valPaths[idx][1]

        if input_path[0: len(self.cache_path)] != self.cache_path:
            a_list = list(input_path)
            a_list[0: len(self.cache_path)] = self.cache_path
            input_path = "".join(a_list)

        if output_path[0: len(self.cache_path)] != self.cache_path:
            a_list = list(output_path)
            a_list[0: len(self.cache_path)] = self.cache_path
            output_path = "".join(a_list)

        try:
            with open(input_path, "rb") as f:
                image_input_array = pickle.load(f)
            with open(output_path, "rb") as f:
                image_output_array = pickle.load(f)
        except:
            image_tensor = torch.zeros(self.resize, self.resize)
            image_tensor_labels = image_tensor
            image_tensor_labels_binary = image_tensor
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
                image_tensor_labels.unsqueeze(0).repeat(len(self.labels), 1, 1), \
                image_tensor_labels_binary.unsqueeze(0).repeat(len(self.labels), 1, 1)

        shape = image_input_array.shape  # [H, W]

        if shape[0] > shape[1]:  # H > W
            sizeH = self.resize
            sizeW = int(shape[1] * self.resize / shape[0])
        else:
            sizeW = self.resize
            sizeH = int(shape[0] * self.resize / shape[1])

        try:
            res_mask = skimage.transform.resize(image_output_array, (sizeW, sizeH), order=0, preserve_range=True,
                                                anti_aliasing=False)
            res = skimage.transform.resize(image_input_array, (sizeW, sizeH), anti_aliasing=True)
        except:

            image_tensor = torch.zeros(self.resize, self.resize)
            image_tensor_labels = image_tensor
            image_tensor_labels_binary = image_tensor
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
                image_tensor_labels.unsqueeze(0).repeat(len(self.labels), 1, 1), \
                image_tensor_labels_binary.unsqueeze(0).repeat(len(self.labels), 1, 1)

        # pad the given size
        extra_left = int((self.resize - sizeW) / 2)
        extra_right = self.resize - sizeW - extra_left
        extra_top = int((self.resize - sizeH) / 2)
        extra_bottom = self.resize - sizeH - extra_top
        image_array = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
        image_array_mask = np.pad(res_mask, ((extra_top, extra_bottom), (extra_left, extra_right)),
                                  mode='constant', constant_values=0)

        # convert to tensor
        image_tensor = torch.from_numpy(image_array).float()
        # normalize
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff

        # mark image binary depends on labels given
        if self.if_multi_channel:
            output_labels = np.zeros((len(self.labels), self.resize, self.resize))  # [n_class, H, W]
            for i in range(len(self.label_values)):
                label_value = self.label_values[i]
                seg_value = np.isin(image_array_mask, label_value)
                output_labels[i] = seg_value

            output_labels[output_labels > 0.1] = 1
            output_labels[output_labels <= 0.1] = 0
        else:
            output_labels = np.zeros((self.resize, self.resize))  # [H, W]
            for i in range(len(self.label_values)):
                label_value = self.label_values[i]
                seg_value = np.isin(image_array_mask, label_value)  # [H, W]
                output_labels += (i + 1) * seg_value

        # convert to tensor
        image_tensor_mask_binary = torch.from_numpy(output_labels).float()
        image_tensor_masked = image_tensor * image_tensor_mask_binary

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            # self.patient_loader.patients[patient_index].print_all_mri_sessions()
            # print("file_index", file_index)
            # print("slice_index", slice_index)
            # print('shape')

        return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
            image_tensor_masked, \
            image_tensor_mask_binary


class GeneralDatasetMultiSegHippo(GeneralDataset):
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
            file_path_text,
            max_train_len=None,
            max_val_len=None,
            n_color=1,
            if_multi_channel=True,
            if_rot_flip=False,
            patient_list=[],
            image_cache_path='',
            label_mask_threshold=8 * 8,
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.patient_loader = patient_loader
        self.n_color = n_color
        self.file_path_text = file_path_text
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.patient_list = patient_list
        self.image_cache_path = image_cache_path
        self.label_mask_threshold = label_mask_threshold
        self.if_multi_channel = if_multi_channel
        self.if_rot_flip = if_rot_flip

        start0 = time.time()
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        print("start1 - start0: ", start1 - start0)

        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)

        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    def get_index_list(self, if_latest=False):

        file_path_COR_list_seg = []
        for i in tqdm(range(len(self.patient_list))):
            patient = self.patient_list[i]
            for slice_index in patient.sequence_slice_index_lists_dict['T1_T1flair']:
                slice_info = patient.get_slice_info_given_slice(slice_index)
                if len(slice_info) > 4:
                    if slice_info[-1][0] == 'COR':
                        plane_idx = slice_info[-1][1]
                        plane = ['x', 'y', 'z']
                        file_path_index = patient.get_file_index_given_slice(slice_index)
                        file_path_string_seg = self.image_cache_path + patient.patient_ID + "/" + str(
                            file_path_index).replace(", ", "_").replace("[", "l").replace("]",
                                                                                          "r") + '/subject/aseg.auto_noCCseg.mgz/' + \
                                               plane[plane_idx]
                        if os.path.exists(file_path_string_seg) and (
                        file_path_string_seg, slice_info) not in file_path_COR_list_seg:
                            file_path_COR_list_seg.append((file_path_string_seg, slice_info))

        # random shuffle list
        #[('D:/nifti_seg_cache/T1_T1flair/NACC568769/l2_0_5_l0rr/subject/aseg.auto_noCCseg.mgz/z',
        # ['t1sag_208', 0.006952, 0.00292, 1.06, ['COR', 2]])]
        random.shuffle(file_path_COR_list_seg)

        # seperate to train and test list
        if self.max_train_len is None:
            valPathsLen = int(len(file_path_COR_list_seg) * self.val_split)
            trainPathsLen = len(file_path_COR_list_seg) - valPathsLen
        elif self.max_train_len < len(file_path_COR_list_seg) - int(len(file_path_COR_list_seg) * self.val_split):
            valPathsLen = int(len(file_path_COR_list_seg) * self.val_split)
            trainPathsLen = self.max_train_len
        else:
            print("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if self.max_val_len <= len(file_path_COR_list_seg) - trainPathsLen:
                valPathsLen = self.max_val_len
            else:
                print("assigned maximum val len exceeds total train num of slices")
        data_into_list_train = file_path_COR_list_seg[:trainPathsLen]
        data_into_list_val = file_path_COR_list_seg[-valPathsLen:]

        return (self.generate_input_output_by_file(data_into_list_train),
                self.generate_input_output_by_file(data_into_list_val))

    def generate_input_output_by_file(self, data_into_list):

        input_image_path_list = []
        output_image_path_list = []
        for i in tqdm(range(len(data_into_list))):
            if i > -1:
                # [('D:/nifti_seg_cache/T1_T1flair/NACC568769/l2_0_5_l0rr/subject/aseg.auto_noCCseg.mgz/z',
                # ['t1sag_208', 0.006952, 0.00292, 1.06, ['COR', 2]])]
                file_valid_seg_i = data_into_list[i][0]
                slice_info = data_into_list[i][1]
                # print(file_valid_seg_i)
                list_temp_seg_i = []
                list_temp_confirm_i = []

                # get all slice in file
                path_seg_temp_list = []
                file_paths = os.listdir(file_valid_seg_i)
                for file in file_paths:
                    path_seg_temp = os.path.join(file_valid_seg_i, file)
                    path_seg_temp_list.append(path_seg_temp)

                # print(str(i), " file_paths ", len(path_seg_temp_list))
                if len(path_seg_temp_list) < 2048:
                    start0 = time.time()
                    for path_seg_temp in path_seg_temp_list:
                        checked_path_seg_temp = check_mask_area(path_seg_temp, self.labels, self.label_values,
                                                                self.label_mask_threshold)
                        if checked_path_seg_temp is not None:
                            list_temp_seg_i.append(checked_path_seg_temp)
                            list_temp_confirm_i.append(
                                checked_path_seg_temp.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz'))
                    start1 = time.time()
                    if (start1 - start0) > 2:
                        print('start1-start0 single: ', start1 - start0)
                        print(file_valid_seg_i)
                else:
                    start0 = time.time()
                    labels_list = [self.labels] * len(path_seg_temp_list)
                    label_values_list = [self.label_values] * len(path_seg_temp_list)
                    min_area_size_list = [64] * len(path_seg_temp_list)
                    with Pool(processes=16) as pool:
                        for checked_path_seg_temp in pool.starmap(
                                check_mask_area,
                                zip(path_seg_temp_list,
                                    labels_list,
                                    label_values_list,
                                    min_area_size_list)):
                            if checked_path_seg_temp is not None:
                                list_temp_seg_i.append(checked_path_seg_temp)
                                list_temp_confirm_i.append(
                                    checked_path_seg_temp.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz'))

                        start1 = time.time()
                        if (start1 - start0) > 2:
                            print('start1-start0 multi: ', start1 - start0)
                            print(file_valid_seg_i)

                list_temp_seg_i.sort(key=natural_keys)
                list_temp_confirm_i.sort(key=natural_keys)

                # check if number matches
                if len(list_temp_seg_i) != len(list_temp_confirm_i):
                    print('unmatched input and target slices num')
                    list_temp_seg_i = []
                    list_temp_confirm_i = []

                # remove choose fraction slices
                n = int(self.drop_fraction * len(list_temp_seg_i))
                if n > 0:
                    list_temp_seg_i = list_temp_seg_i[n:-n]
                    list_temp_confirm_i = list_temp_confirm_i[n:-n]

                # add slice info for all
                list_temp_seg_i = list(zip(list_temp_seg_i, [slice_info] * len(list_temp_seg_i)))
                list_temp_confirm_i = list(zip(list_temp_confirm_i, [slice_info] * len(list_temp_seg_i)))

                input_image_path_list = input_image_path_list + list_temp_confirm_i
                output_image_path_list = output_image_path_list + list_temp_seg_i
                # [('D:/nifti_seg_cache/T1_T1flair/NACC568769/l2_0_5_l0rr/subject/aseg.auto_noCCseg.mgz/z/19.pkl',
                # ['t1sag_208', 0.006952, 0.00292, 1.06, ['COR', 2]])]
        return list(zip(input_image_path_list, output_image_path_list))

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            input_path = self.trainPaths[idx][0][0]
            output_path = self.trainPaths[idx][1][0]
            slice_info = self.trainPaths[idx][1][1]
        else:
            input_path = self.valPaths[idx][0][0]
            output_path = self.valPaths[idx][1][0]
            slice_info = self.valPaths[idx][1][1]

        direction = None
        if len(slice_info) > 2:
            direction = slice_info[-1]

        try:
            with open(input_path, "rb") as f:
                image_input_array = pickle.load(f)
            with open(output_path, "rb") as f:
                image_output_array = pickle.load(f)
        except:
            image_tensor = torch.zeros(self.resize, self.resize)
            image_tensor_labels = image_tensor
            image_tensor_labels_binary = image_tensor
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
                image_tensor_labels.unsqueeze(0).repeat(len(self.labels), 1, 1), \
                image_tensor_labels_binary.unsqueeze(0).repeat(len(self.labels), 1, 1)

        shape = image_input_array.shape  # [H, W]

        if shape[0] > shape[1]:  # H > W
            sizeH = self.resize
            sizeW = int(shape[1] * self.resize / shape[0])
        else:
            sizeW = self.resize
            sizeH = int(shape[0] * self.resize / shape[1])

        try:
            res_mask = skimage.transform.resize(image_output_array, (sizeW, sizeH), order=0, preserve_range=True,
                                                anti_aliasing=False)
            res = skimage.transform.resize(image_input_array, (sizeW, sizeH), anti_aliasing=True)
        except:

            image_tensor = torch.zeros(self.resize, self.resize)
            image_tensor_labels = image_tensor
            image_tensor_labels_binary = image_tensor
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
                image_tensor_labels.unsqueeze(0).repeat(len(self.labels), 1, 1), \
                image_tensor_labels_binary.unsqueeze(0).repeat(len(self.labels), 1, 1)

        # pad the given size
        extra_left = int((self.resize - sizeW) / 2)
        extra_right = self.resize - sizeW - extra_left
        extra_top = int((self.resize - sizeH) / 2)
        extra_bottom = self.resize - sizeH - extra_top
        image_array = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
        image_array_mask = np.pad(res_mask, ((extra_top, extra_bottom), (extra_left, extra_right)),
                                  mode='constant', constant_values=0)

        # mark image binary depends on labels given
        if self.if_multi_channel:
            output_labels = np.zeros((len(self.labels), self.resize, self.resize))  # [n_class, H, W]
            for i in range(len(self.label_values)):
                label_value = self.label_values[i]
                seg_value = np.isin(image_array_mask, label_value)
                output_labels[i] = seg_value

            output_labels[output_labels > 0.1] = 1
            output_labels[output_labels <= 0.1] = 0
        else:
            output_labels = np.zeros((self.resize, self.resize))  # [H, W]
            for i in range(len(self.label_values)):
                label_value = self.label_values[i]
                seg_value = np.isin(image_array_mask, label_value)  # [H, W]
                output_labels += (i + 1) * seg_value

        # check if direction is right
        if direction == 'right':
            if self.if_multi_channel:
                output_labels = np.flip(output_labels, axis=2)
            else:
                output_labels = np.flip(output_labels, axis=1)
            image_array = np.flip(image_array, axis=1)

        # apply rot and flip
        if self.if_rot_flip:
            image_array, output_labels = self.custom_flip_func(image_array, output_labels)
            image_array, output_labels = self.custom_rot_func(image_array, output_labels)

        # convert to tensor
        image_tensor = torch.from_numpy(image_array.copy()).float()
        # normalize
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff

        image_tensor_mask_binary = torch.from_numpy(output_labels.copy()).float()
        image_tensor_masked = image_tensor * image_tensor_mask_binary

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            # self.patient_loader.patients[patient_index].print_all_mri_sessions()
            # print("file_index", file_index)
            # print("slice_index", slice_index)
            # print('shape')

        return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
            image_tensor_masked, \
            image_tensor_mask_binary


class GeneralDatasetPathBinaryClass(GeneralDataset):
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
            n_color=1,
            if_multi_channel=True,
            if_rot_flip=False,
            patient_pos=[],
            patient_neg=[],
            _model_extraction=None,
            _model_plane_detect=None,
            _model_multi_seg=None,
            mask_channel=1,
            mask_model=None,
            mask_label=[],
            mask_label_idx=[],
            mask_area_threshold=64 * 64,
            patient_handler=None,
            image_cache_path='',  # "F:/nifti_seg/T1_T1flair/"
            detect_plane=['AX', 'COR', 'SAG'],
            target_plane='COR',
            pos_neg_multiple_size=[1, 1],
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.n_color = n_color
        self.mask_channel = mask_channel
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.if_multi_channel = if_multi_channel
        self.if_rot_flip = if_rot_flip
        self.patient_pos = patient_pos
        self.patient_neg = patient_neg
        self.mask_model = mask_model
        self.patient_handler = patient_handler
        self._model_extraction = _model_extraction
        self._model_plane_detect = _model_plane_detect
        self._model_multi_seg = _model_multi_seg
        self.mask_label = mask_label
        self.mask_label_idx = mask_label_idx
        self.mask_area_threshold = mask_area_threshold
        self.image_cache_path = image_cache_path
        self.detect_plane = detect_plane
        self.target_plane = target_plane
        start0 = time.time()
        self.data_into_list_train = []
        self.data_into_list_val = []
        self.pos_neg_multiple_size = pos_neg_multiple_size
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        print("start1 - start0: ", start1 - start0)

        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    # function to return plane detection of a given image volume

    def detect_plane_direction_given_volume_folder(self, filepathconformed, detect_plane, target_plane):

        detect_sequences = []
        detected_confidences = []
        for i in range(3):
            image_tensor_volume = load_all_slice_in_folder(filepathconformed[i], self.mask_channel, self.resize)
            # print_var_detail(image_tensor_volume)
            detect_sequence, detected_confidence = detect_plane_direction(self.patient_handler, self._model_extraction,
                                                                          self._model_plane_detect, image_tensor_volume,
                                                                          detect_plane)
            detect_sequences.append(detect_sequence)
            detected_confidences.append(detected_confidence)
        indices = [i for i, x in enumerate(detect_sequences) if x == target_plane]
        # print(filepathconformed)
        # print(detect_sequences)
        # print(detected_confidences)
        # print(indices)

        # if all is None, return -1 and 0 confidence
        if all(v is None for v in detect_sequences):
            return -1, 0  # idx, confidence

        if len(indices) > 0:
            idx = indices[0]
            # if more than one detected
            if len(indices) > 1:
                my_list_cor = map(detect_sequences.__getitem__, indices)
                my_list_confidences_cor = map(detected_confidences.__getitem__, indices)
                # find maximum confidence
                m = max(my_list_confidences_cor)
                idx = detected_confidences.index(m)
        # if none detected, find duplicate detection and choose the smallest confidence as target plane
        else:
            duplicates = [item for item, count in collections.Counter(detect_sequences).items() if count > 1]
            duplicate = duplicates[0]  # unlikely for multiple duplicates in a size 3 list
            duplicate_indices = [i for i, x in enumerate(detect_sequences) if x == duplicate]
            my_list_duplicate = map(detect_sequences.__getitem__, duplicate_indices)
            my_list_confidences_duplicate = map(detected_confidences.__getitem__, duplicate_indices)
            # find maximum confidence
            m = min(my_list_confidences_duplicate)
            idx = detected_confidences.index(m)

        # print(detected_confidences[idx])
        # if (detected_confidences[idx] < 1.0):
        #     print('dump the entire sample file, its not useful, probably blur plane')
        # print(idx)
        return idx, detected_confidences[idx]

    def filter_patients(self, patient, if_latest=False):
        if self.mri_sequence is None:
            sequence_names = list(patient.sequence_slice_index_lists_dict.keys())
        else:
            sequence_names = self.mri_sequence

        # get list of file index in patient given sequence
        file_out_path_list = []
        for sequence_name in sequence_names:
            if patient.sequence_slice_index_lists_dict[sequence_name]:
                slice_index_list = patient.sequence_slice_index_lists_dict[sequence_name]
                if len(slice_index_list) < 200000:  # more than 200000 mri slices for one series? doesn't sound right
                    for slice_index in slice_index_list:

                        # exclude the slices that are not in the latest dir
                        if if_latest:
                            dir_index = slice_index[0:-1]
                            if dir_index not in patient.dir_index_list_latest:
                                continue

                        file_path_index = patient.get_file_index_given_slice(slice_index)
                        file_path_string = self.image_cache_path + patient.patient_ID + "/" + str(
                            file_path_index).replace(", ", "_").replace("[", "l").replace("]", "r") + '/subject'
                        # file_path = file_path[:Index] + file_path[Index + 1:]
                        if os.path.exists(file_path_string):
                            if file_path_string not in file_out_path_list:
                                file_out_path_list.append(
                                    file_path_string)  # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject

        # filter by give plane direction
        file_out_path_list_plane = []
        for file_out_path in file_out_path_list:
            # load x, y, z plane
            filepathconformedx = file_out_path + '/conformed.mgz/x'
            filepathconformedy = file_out_path + '/conformed.mgz/y'
            filepathconformedz = file_out_path + '/conformed.mgz/z'
            filepathconformed = [filepathconformedx, filepathconformedy, filepathconformedz]
            idx, detected_confidence = self.detect_plane_direction_given_volume_folder(
                filepathconformed, self.detect_plane, self.target_plane)
            if idx > -0.5:
                if detected_confidence >= 1.0:
                    file_out_path_list_plane.append(filepathconformed[idx])

        # print_var_detail(image_tensor_volume)
        # filter by given mask labels area size
        file_out_path_list_plane_labels = []
        for file_out_path_plane in file_out_path_list_plane:  # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x
            image_tensor_volume = load_all_slice_in_folder(file_out_path_plane, self.mask_channel, self.resize)
            image_tensor_volume_mask_multi, valid_col_multi = self.patient_handler.mask_generation_by_volume(
                self._model_multi_seg,
                image_tensor_volume,
                if_exclude_empty=True)
            image_tensor_volume_mask_multi[image_tensor_volume_mask_multi > 0.5] = 1
            image_tensor_volume_mask_multi[image_tensor_volume_mask_multi <= 0.5] = 0
            valid_cols = []
            for label_idx in self.mask_label_idx:
                image_tensor_volume_mask_single = image_tensor_volume_mask_multi[:, label_idx, :, :]
                masked_area = torch.count_nonzero(image_tensor_volume_mask_single, dim=(1, 2))
                b = masked_area > self.mask_area_threshold
                indices = b.nonzero()
                valid_col_mask = [valid_col_multi[i] for i in indices]
                valid_cols += valid_col_mask
            valid_cols = list(set(valid_cols))
            valid_cols.sort()  # sort int no need for natural keys

            # output all valid slice
            for valid_col in valid_cols:
                #
                file_out_path_list_plane_labels.append(file_out_path_plane + '/' + str(valid_col) + '.pkl')

        return file_out_path_list_plane_labels  # ['D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x/19.pkl']

    def get_index_list(self, if_latest=False):
        # shuffle patients
        random.shuffle(self.patient_pos)
        random.shuffle(self.patient_neg)

        # filter patients
        patient_pos_and_file_path = []
        for i in tqdm(range(len(self.patient_pos))):
            patient = self.patient_pos[i]
            file_out_path_list_plane_labels = self.filter_patients(patient, if_latest)
            if len(file_out_path_list_plane_labels) > 0:
                patient_pos_and_file_path.append([patient, file_out_path_list_plane_labels])  # [patient, [list]]

        patient_neg_and_file_path = []
        # for patient in self.patient_neg:
        for i in tqdm(range(len(self.patient_neg))):
            patient = self.patient_neg[i]
            file_out_path_list_plane_labels = self.filter_patients(patient)
            if len(file_out_path_list_plane_labels) > 0:
                patient_neg_and_file_path.append([patient, file_out_path_list_plane_labels])  # [patient, [list]]

        print('num patient pos: ', len(patient_pos_and_file_path))
        print('num patient neg: ', len(patient_neg_and_file_path))

        # random shuffle list
        random.shuffle(patient_pos_and_file_path)
        random.shuffle(patient_neg_and_file_path)

        # seperate to train and test list
        if self.max_train_len is None:
            valPathsLen_pos = int(len(patient_pos_and_file_path) * self.val_split)
            trainPathsLen_pos = len(patient_pos_and_file_path) - valPathsLen_pos
            valPathsLen_neg = int(len(patient_neg_and_file_path) * self.val_split)
            trainPathsLen_neg = len(patient_neg_and_file_path) - valPathsLen_neg
        elif self.max_train_len < len(patient_pos_and_file_path) - int(len(patient_pos_and_file_path) * self.val_split) \
                and self.max_train_len < len(patient_neg_and_file_path) - int(
            len(patient_neg_and_file_path) * self.val_split):
            valPathsLen_pos = int(len(patient_pos_and_file_path) * self.val_split)
            trainPathsLen_pos = self.max_train_len
            valPathsLen_neg = int(len(patient_neg_and_file_path) * self.val_split)
            trainPathsLen_neg = self.max_train_len
        else:
            assert False, ("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if (self.max_val_len <= len(patient_pos_and_file_path) - trainPathsLen_pos and
                    self.max_val_len <= len(patient_neg_and_file_path) - trainPathsLen_neg):
                valPathsLen_neg = self.max_val_len
            else:
                assert False, ("assigned maximum val len exceeds total train num of slices")
        data_into_list_train_pos = patient_pos_and_file_path[:trainPathsLen_pos]
        data_into_list_train_pos = list(zip(data_into_list_train_pos, [1] * len(data_into_list_train_pos)))  # 1 for pos
        data_into_list_val_pos = patient_pos_and_file_path[-valPathsLen_pos:]
        data_into_list_val_pos = list(zip(data_into_list_val_pos, [1] * len(data_into_list_val_pos)))  # 1 for pos
        data_into_list_train_neg = patient_neg_and_file_path[:trainPathsLen_neg]
        data_into_list_train_neg = list(zip(data_into_list_train_neg, [0] * len(data_into_list_train_neg)))  # 0 for neg
        data_into_list_val_neg = patient_neg_and_file_path[-valPathsLen_neg:]
        data_into_list_val_neg = list(zip(data_into_list_val_neg, [0] * len(data_into_list_val_neg)))  # 0 for neg

        self.data_into_list_train = (data_into_list_train_pos * self.pos_neg_multiple_size[0]
                                     + data_into_list_train_neg * self.pos_neg_multiple_size[
                                         1])  # [([patient, [list]], int), ...]
        self.data_into_list_val = (data_into_list_val_pos * self.pos_neg_multiple_size[0]
                                   + data_into_list_val_neg * self.pos_neg_multiple_size[
                                       1])  # [([patient, [list]], int), ...]

        # expand patient slice by given slice sequence

        # return list(zip(input_image_path_list, output_image_path_list))
        return (self.generate_input_output_by_file(self.data_into_list_train),
                self.generate_input_output_by_file(self.data_into_list_val))

    def generate_input_output_by_file(self, data_into_list):
        path_label_list = []  # [('D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x/19.pkl', int, ID)]
        for i in range(len(data_into_list)):
            patient = data_into_list[i][0][0]
            path_list_i = data_into_list[i][0][1]
            label = data_into_list[i][1]
            path_label_list_i = list(zip(path_list_i, [label] * len(path_list_i),
                                         [patient.patient_ID] * len(path_list_i)))  # [(path, int, ID), ...]
            path_label_list += path_label_list_i
        return path_label_list  # [(path, int, ID), ...]

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            input_path = self.trainPaths[idx][0]
            label = self.trainPaths[idx][1]
            patient_ID = self.trainPaths[idx][2]
        else:
            input_path = self.valPaths[idx][0]
            label = self.valPaths[idx][1]
            patient_ID = self.valPaths[idx][2]

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        image_label = torch.zeros(len(self.labels))  # size of label = number of sequence + 1
        # image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class

        try:
            # image_array = patient.load_mri_slice(slice_index)
            with open(input_path, "rb") as f:
                image_array = pickle.load(f)
        except:
            print("pkl.pixel_array corrupted")
            print(input_path)
            # print(slice_index)
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label, patient_ID

        # if rgb image, convert into grayscale
        if len(image_array.shape) == 3:
            if image_array.shape[-1] == 3:
                image_array = rgb2gray(image_array)

        # convert to np float
        image_array = np.vstack(image_array).astype(np.float64)

        # resize
        shape = image_array.shape  # [H, W]

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
            return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label, patient_ID

        # pad the given size
        extra_left = int((self.resize - sizeW) / 2)
        extra_right = self.resize - sizeW - extra_left
        extra_top = int((self.resize - sizeH) / 2)
        extra_bottom = self.resize - sizeH - extra_top
        res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                         mode='constant', constant_values=0)
        image_array = res_pad

        # apply rot and flip
        if self.if_rot_flip:
            image_array, _ = self.custom_flip_func(image_array, image_array)
            image_array, _ = self.custom_rot_func(image_array, image_array)

        # convert to tensor
        # image_tensor = torch.from_numpy(image_array).float()
        image_tensor = torch.from_numpy(image_array.copy()).float()  # avoid neg stride
        # normalize
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff

        # 1-hot classify
        if len(self.labels) > 1:
            label_idx = self.label_values.index(label)
            image_label[label_idx] = 1.0
        else:
            image_label[0] = label

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            print(input_path)
        return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label, patient_ID


class GeneralDatasetHippoBinaryClass(GeneralDataset):
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
            n_color=1,
            if_multi_channel=True,
            if_rot_flip=False,
            patient_pos=[],
            patient_neg=[],
            _model_extraction=None,
            _model_plane_detect=None,
            _model_multi_seg=None,
            mask_channel=1,
            mask_model=None,
            mask_label=[],
            mask_label_idx=[],
            mask_area_threshold=64 * 64,
            patient_handler=None,
            image_cache_path='',  # "F:/nifti_seg/T1_T1flair/"
            detect_plane=['AX', 'COR', 'SAG'],
            target_plane='COR',
            pos_neg_multiple_size=[1, 1],
            if_use_cached_seg=True,
            if_output_striped=True,
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.n_color = n_color
        self.mask_channel = mask_channel
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.if_multi_channel = if_multi_channel
        self.if_rot_flip = if_rot_flip
        self.patient_pos = patient_pos
        self.patient_neg = patient_neg
        self.mask_model = mask_model
        self.patient_handler = patient_handler
        self._model_extraction = _model_extraction
        self._model_plane_detect = _model_plane_detect
        self._model_multi_seg = _model_multi_seg
        self.mask_label = mask_label
        self.mask_label_idx = mask_label_idx
        self.mask_area_threshold = mask_area_threshold
        self.image_cache_path = image_cache_path
        self.detect_plane = detect_plane
        self.target_plane = target_plane
        self.if_use_cached_seg = if_use_cached_seg
        self.if_output_striped = if_output_striped
        if _model_extraction is None:
            self.if_output_striped = False

        start0 = time.time()
        self.data_into_list_train = []
        self.data_into_list_val = []
        self.pos_neg_multiple_size = pos_neg_multiple_size
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        print("start1 - start0: ", start1 - start0)

        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    def filter_patients(self, patient: PatientCase, if_latest: bool = False):
        if self.mri_sequence is None:
            sequence_names = list(patient.sequence_slice_index_lists_dict.keys())
        else:
            sequence_names = self.mri_sequence

        # get list of file index in patient given sequence
        file_out_path_list = []
        file_out_info_list = []
        for sequence_name in sequence_names:
            if patient.sequence_slice_index_lists_dict[sequence_name]:
                slice_index_list = patient.sequence_slice_index_lists_dict[sequence_name]
                if len(slice_index_list) < 200000:  # more than 200000 mri slices for one series? doesn't sound right
                    for slice_index in slice_index_list:

                        # exclude the slices that are not in the latest dir
                        if if_latest:
                            dir_index = slice_index[0:-1]
                            if dir_index not in patient.dir_index_list_latest:
                                continue

                        file_path_index = patient.get_file_index_given_slice(slice_index)
                        slice_info = patient.get_slice_info_given_file(file_path_index)
                        if len(slice_info) > 4:
                            if slice_info[-1][0] == 'COR':
                                plane_idx = slice_info[-1][1]
                                file_path_index = patient.get_file_index_given_slice(slice_index)
                                # if file_path_index not in file_index_valid:
                                #     file_index_valid.append(file_path_index)
                                file_path_string = self.image_cache_path + patient.patient_ID + "/" + str(
                                    file_path_index).replace(", ", "_").replace("[", "l").replace("]", "r") + '/subject'
                                # file_path = file_path[:Index] + file_path[Index + 1:]
                                filepathconformedx = file_path_string + '/conformed.mgz/x'
                                filepathconformedy = file_path_string + '/conformed.mgz/y'
                                filepathconformedz = file_path_string + '/conformed.mgz/z'
                                filepathconformed = [filepathconformedx, filepathconformedy, filepathconformedz]
                                file_path_string = filepathconformed[plane_idx]
                                if os.path.exists(file_path_string):
                                    if file_path_string not in file_out_path_list:
                                        file_out_path_list.append(
                                            file_path_string)  # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/z
                                        file_out_info_list.append(slice_info)

        file_out_path_list_plane = file_out_path_list

        # filter by given mask labels area size
        file_out_path_list_plane_labels = []
        file_out_info_list_labels = []
        for j, file_out_path_plane in enumerate(
                file_out_path_list_plane):  # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/z
            file_out_path_seg_mask = file_out_path_plane.replace('conformed.mgz',
                                                                 'aseg.auto_noCCseg.mgz')  # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/aseg.auto_noCCseg.mgz/z
            slice_info_path_seg_mask = file_out_info_list[j]
            # if has generated mask
            if os.path.exists(file_out_path_seg_mask) and self.if_use_cached_seg:
                file_paths = os.listdir(file_out_path_seg_mask)
                file_paths.sort(key=natural_keys)

                # drop given fraction of valid images
                n = int(self.drop_fraction * len(file_paths))
                if n > 0:
                    file_paths = file_paths[n:-n]

                file_idx = []
                for file in file_paths:
                    path_temp = os.path.join(file_out_path_seg_mask, file)
                    output_labels = load_generated_mask_slice(path_temp, self.resize, self.labels, self.label_values)
                    if output_labels is not None:
                        mask_area = np.count_nonzero(output_labels)
                        if patient.patient_ID == 'NACC502826':
                            print(mask_area)
                        if mask_area > self.mask_area_threshold:
                            file_idx.append(int(file[:-4]))

                start_idx, end_idx, length = longest_sequential_num(file_idx, step_size=2)
                # output all valid slice
                for file_i in file_idx[start_idx:end_idx + 1]:
                    # file_out_path_list_plane_labels.append(file_out_path_plane + '/' + str(valid_col) + '.pkl')
                    # file_out_info_list_labels.append(file_out_info_list[j])
                    # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/z/19.pkl
                    file_out_path_list_plane_labels.append(file_out_path_seg_mask.replace('aseg.auto_noCCseg.mgz',
                                                                                          'conformed.mgz') + '/'
                                                           + str(file_i) + '.pkl')
                    file_out_info_list_labels.append(file_out_info_list[j])
            else:
                # if has no generated mask use mask extraction model if any
                # print(patient.patient_ID + ' has no cache')
                if self._model_multi_seg:
                    if_flip = False
                    if slice_info_path_seg_mask[-1][-1] == 'right':
                        if_flip = True
                    image_tensor_volume = load_all_slice_in_folder(file_out_path_plane, self.mask_channel, self.resize,
                                                                   if_flip=if_flip)
                    image_tensor_volume_mask_multi, valid_col_multi = self.patient_handler.mask_generation_by_volume(
                        self._model_multi_seg,
                        image_tensor_volume,
                        if_exclude_empty=True)

                    # drop given fraction of valid images
                    n = int(self.drop_fraction * image_tensor_volume_mask_multi.shape[0])
                    if n > 0:
                        image_tensor_volume_mask_multi = image_tensor_volume_mask_multi[n:-n]
                        valid_col_multi = valid_col_multi[n:-n]

                    # image_tensor_volume_mask_multi[image_tensor_volume_mask_multi > 0.5] = 1
                    # image_tensor_volume_mask_multi[image_tensor_volume_mask_multi <= 0.5] = 0
                    if image_tensor_volume_mask_multi.shape[1] > 1:
                        mask = image_tensor_volume_mask_multi.argmax(dim=1)  # [B, H, W]
                    else:
                        mask = torch.sigmoid(image_tensor_volume_mask_multi) > 0.5
                        mask = mask.unsqueeze(1)  # [B, H, W]

                    valid_cols = []
                    mask_area_threshold_temp = self.mask_area_threshold

                    while len(valid_cols) < 10 and mask_area_threshold_temp > 6 * 6:
                        # print(mask_area_threshold_temp)
                        valid_cols = []
                        for label_idx in self.mask_label_idx:
                            # image_tensor_volume_mask_single = image_tensor_volume_mask_multi[:, label_idx, :, :]
                            image_tensor_volume_mask_single = (mask == label_idx)
                            masked_area = torch.count_nonzero(image_tensor_volume_mask_single, dim=(1, 2))
                            b = masked_area > mask_area_threshold_temp
                            indices = b.nonzero()

                            # # if less than 10 slices selected choose top ten
                            # if len(indices) < 10:
                            #     _, indices = torch.topk(masked_area, 10)
                            valid_col_mask = [valid_col_multi[i] for i in indices]
                            valid_cols += valid_col_mask
                        valid_cols = list(set(valid_cols))
                        valid_cols.sort()  # sort int no need for natural keys

                        # find the longest sequential files with step size of 2
                        start_idx, end_idx, length = longest_sequential_num(valid_cols, step_size=2)
                        valid_cols = valid_cols[start_idx:end_idx + 1]
                        # print(len(valid_cols))
                        mask_area_threshold_temp -= 4  # gradually decrease area check by 2X2

                    # output all valid slice
                    for valid_col in valid_cols:
                        file_out_path_list_plane_labels.append(file_out_path_plane + '/' + str(valid_col) + '.pkl')
                        file_out_info_list_labels.append(file_out_info_list[j])

        return file_out_path_list_plane_labels, file_out_info_list_labels  # ['D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x/19.pkl']

    def get_index_list(self, if_latest=False):
        # shuffle patients
        random.shuffle(self.patient_pos)
        random.shuffle(self.patient_neg)

        # filter patients
        patient_pos_and_file_path = []
        for i in tqdm(range(len(self.patient_pos))):
            patient = self.patient_pos[i]
            file_out_path_list_plane_labels, file_out_info_list_labels = self.filter_patients(patient, if_latest)
            if len(file_out_path_list_plane_labels) > 0:
                patient_pos_and_file_path.append(
                    [patient, file_out_path_list_plane_labels, file_out_info_list_labels])  # [patient, [list],[ info]]

        patient_neg_and_file_path = []
        # for patient in self.patient_neg:
        for i in tqdm(range(len(self.patient_neg))):
            patient = self.patient_neg[i]
            file_out_path_list_plane_labels, file_out_info_list_labels = self.filter_patients(patient, if_latest)
            if len(file_out_path_list_plane_labels) > 0:
                patient_neg_and_file_path.append(
                    [patient, file_out_path_list_plane_labels, file_out_info_list_labels])  # [patient, [list], [info]]

        print('num patient pos: ', len(patient_pos_and_file_path))
        print('num patient neg: ', len(patient_neg_and_file_path))

        # random shuffle list
        random.shuffle(patient_pos_and_file_path)
        random.shuffle(patient_neg_and_file_path)

        # seperate to train and test list
        if self.max_train_len is None:
            valPathsLen_pos = int(len(patient_pos_and_file_path) * self.val_split)
            trainPathsLen_pos = len(patient_pos_and_file_path) - valPathsLen_pos
            valPathsLen_neg = int(len(patient_neg_and_file_path) * self.val_split)
            trainPathsLen_neg = len(patient_neg_and_file_path) - valPathsLen_neg
        elif self.max_train_len < len(patient_pos_and_file_path) - int(len(patient_pos_and_file_path) * self.val_split) \
                and self.max_train_len < len(patient_neg_and_file_path) - int(
            len(patient_neg_and_file_path) * self.val_split):
            valPathsLen_pos = int(len(patient_pos_and_file_path) * self.val_split)
            trainPathsLen_pos = self.max_train_len
            valPathsLen_neg = int(len(patient_neg_and_file_path) * self.val_split)
            trainPathsLen_neg = self.max_train_len
        else:
            assert False, ("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if (self.max_val_len <= len(patient_pos_and_file_path) - trainPathsLen_pos and
                    self.max_val_len <= len(patient_neg_and_file_path) - trainPathsLen_neg):
                valPathsLen_neg = self.max_val_len
            else:
                assert False, ("assigned maximum val len exceeds total train num of slices")
        data_into_list_train_pos = patient_pos_and_file_path[:trainPathsLen_pos]
        data_into_list_train_pos = list(zip(data_into_list_train_pos, [1] * len(data_into_list_train_pos)))  # 1 for pos
        data_into_list_val_pos = patient_pos_and_file_path[-valPathsLen_pos:]
        data_into_list_val_pos = list(zip(data_into_list_val_pos, [1] * len(data_into_list_val_pos)))  # 1 for pos
        data_into_list_train_neg = patient_neg_and_file_path[:trainPathsLen_neg]
        data_into_list_train_neg = list(zip(data_into_list_train_neg, [0] * len(data_into_list_train_neg)))  # 0 for neg
        data_into_list_val_neg = patient_neg_and_file_path[-valPathsLen_neg:]
        data_into_list_val_neg = list(zip(data_into_list_val_neg, [0] * len(data_into_list_val_neg)))  # 0 for neg

        self.data_into_list_train = (data_into_list_train_pos * self.pos_neg_multiple_size[0]
                                     + data_into_list_train_neg * self.pos_neg_multiple_size[
                                         1])  # [([patient, [lists], [info]], int), ...]
        self.data_into_list_val = (data_into_list_val_pos * self.pos_neg_multiple_size[0]
                                   + data_into_list_val_neg * self.pos_neg_multiple_size[
                                       1])  # [([patient, [list], [info]], int), ...]

        # expand patient slice by given slice sequence

        # return list(zip(input_image_path_list, output_image_path_list))
        return (self.generate_input_output_by_file(self.data_into_list_train),
                self.generate_input_output_by_file(self.data_into_list_val))

    def generate_input_output_by_file(self, data_into_list):
        path_label_list = []  # [('D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x/19.pkl', int, ID)]
        for i in range(len(data_into_list)):
            patient = data_into_list[i][0][0]
            path_list_i = data_into_list[i][0][1]
            slice_info = data_into_list[i][0][2]
            label = data_into_list[i][1]
            path_label_list_i = list(
                zip(path_list_i, [label] * len(path_list_i), [patient.patient_ID] * len(path_list_i),
                    slice_info))  # [(path, int, ID), ...]
            path_label_list += path_label_list_i
        return path_label_list  # [(path, int, ID, info), ...]

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            input_path = self.trainPaths[idx][0]
            label = self.trainPaths[idx][1]
            patient_ID = self.trainPaths[idx][2]
            slice_info = self.trainPaths[idx][3]
        else:
            input_path = self.valPaths[idx][0]
            label = self.valPaths[idx][1]
            patient_ID = self.valPaths[idx][2]
            slice_info = self.valPaths[idx][3]

        direction = None
        if len(slice_info) > 2:
            direction = slice_info[-1]

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        image_label = torch.zeros(len(self.labels))  # size of label = number of sequence + 1
        image_tensor_mask_strip = torch.zeros(self.resize, self.resize)
        # image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class

        try:
            # image_array = patient.load_mri_slice(slice_index)
            with open(input_path, "rb") as f:
                image_array = pickle.load(f)
        except:
            print("pkl.pixel_array corrupted")
            print(input_path)
            # print(slice_index)
            return (image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label, patient_ID,
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.n_color, 1, 1))

        # if rgb image, convert into grayscale
        if len(image_array.shape) == 3:
            if image_array.shape[-1] == 3:
                image_array = rgb2gray(image_array)

        # convert to np float
        image_array = np.vstack(image_array).astype(np.float64)

        # resize
        shape = image_array.shape  # [H, W]

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
            return (image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label, patient_ID,
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.n_color, 1, 1))

        # pad the given size
        extra_left = int((self.resize - sizeW) / 2)
        extra_right = self.resize - sizeW - extra_left
        extra_top = int((self.resize - sizeH) / 2)
        extra_bottom = self.resize - sizeH - extra_top
        res_pad = np.pad(res, ((extra_top, extra_bottom), (extra_left, extra_right)),
                         mode='constant', constant_values=0)
        image_array = res_pad

        # apply rot and flip
        if self.if_rot_flip:
            image_array, _ = self.custom_flip_func(image_array, image_array)
            image_array, _ = self.custom_rot_func(image_array, image_array)
        # check if direction is right
        if direction == 'right':
            image_array = np.flip(image_array, axis=1)

        # convert to tensor
        # image_tensor = torch.from_numpy(image_array).float()
        image_tensor = torch.from_numpy(image_array.copy()).float()  # avoid neg stride
        # normalize
        maximum = torch.max(image_tensor)
        minimum = torch.min(image_tensor)
        scale = maximum - minimum
        if scale > 0:
            scale_coeff = 1. / scale
        else:
            scale_coeff = 0
        image_tensor = (image_tensor - minimum) * scale_coeff

        # 1-hot classify
        if len(self.labels) > 1:
            label_idx = self.label_values.index(label)
            image_label[label_idx] = 1.0
        else:
            image_label[0] = label

        # extraction

        if self._model_extraction:
            # image_tensor_mask = predict_img(self._model_extraction, image_tensor.unsqueeze(0).unsqueeze(0),
            #                                 device=self.patient_handler.device, n_classes=1, out_threshold=0.5)
            #
            # image_tensor_mask_strip = image_tensor * image_tensor_mask.squeeze(0).squeeze(0)
            image_tensor_mask, _ = self.patient_handler.mask_generation_by_volume(
                self._model_extraction,
                image_tensor.unsqueeze(0).unsqueeze(0),
                if_exclude_empty=False)
            if image_tensor_mask.shape[1] > 1:
                mask = image_tensor_mask.argmax(dim=1)  # [B, H, W]
                image_tensor_mask_strip = image_tensor * mask.squeeze(0)
            else:
                mask = torch.sigmoid(image_tensor_mask) > 0.5  # [B, 1, H, W]
                image_tensor_mask_strip = image_tensor * mask.squeeze(0).squeeze(0)

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            print(input_path)

        if self.if_output_striped:
            return (image_tensor_mask_strip.unsqueeze(0).repeat(self.n_color, 1, 1), image_label, patient_ID,
                    )
        else:
            return (image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), image_label, patient_ID,
                    )


class GeneralDatasetHippoBinaryClassManual(GeneralDataset):
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
            n_color=1,
            if_multi_channel=True,
            if_rot_flip=False,
            patients=[],
            _model_extraction=None,
            _model_plane_detect=None,
            _model_multi_seg=None,
            mask_channel=1,
            mask_model=None,
            mask_label=[],
            mask_label_idx=[],
            mask_area_threshold=64 * 64,
            patient_handler=None,
            image_cache_path='',  # "F:/nifti_seg/T1_T1flair/"
            detect_plane=['AX', 'COR', 'SAG'],
            target_plane='COR',
            pos_neg_multiple_size=[1, 1],
            if_use_cached_seg=True,
            if_output_striped=True,
            frac_val_level=[0.5, 0.25, 0.25],
            target_channel = 40,
            random_extra_margin_slice = 5,
            train_level = ['level1', 'level2', 'level3'],
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.n_color = n_color
        self.mask_channel = mask_channel
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.if_multi_channel = if_multi_channel
        self.if_rot_flip = if_rot_flip
        self.patients = patients
        self.mask_model = mask_model
        self.patient_handler = patient_handler
        self._model_extraction = _model_extraction
        self._model_plane_detect = _model_plane_detect
        self._model_multi_seg = _model_multi_seg
        self.mask_label = mask_label
        self.mask_label_idx = mask_label_idx
        self.mask_area_threshold = mask_area_threshold
        self.image_cache_path = image_cache_path
        self.detect_plane = detect_plane
        self.target_plane = target_plane
        self.if_use_cached_seg = if_use_cached_seg
        self.if_output_striped = if_output_striped
        self.frac_val_level = frac_val_level
        self.target_channel = target_channel
        self.random_extra_margin_slice = random_extra_margin_slice
        self.train_level = train_level
        if _model_extraction is None:
            self.if_output_striped = False

        start0 = time.time()
        self.data_into_list_train = []
        self.data_into_list_val = []
        self.pos_neg_multiple_size = pos_neg_multiple_size
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        print("start1 - start0: ", start1 - start0)

        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    def filter_patients(self, patient: PatientCase, if_latest: bool = False):
        if self.mri_sequence is None:
            sequence_names = list(patient.sequence_slice_index_lists_dict.keys())
        else:
            sequence_names = self.mri_sequence

        # get list of file index in patient given sequence
        file_out_path_list = []
        file_out_info_list = []
        for sequence_name in sequence_names:
            if patient.sequence_slice_index_lists_dict[sequence_name]:
                slice_index_list = patient.sequence_slice_index_lists_dict[sequence_name]
                if len(slice_index_list) < 200000:  # more than 200000 mri slices for one series? doesn't sound right
                    for slice_index in slice_index_list:

                        # exclude the slices that are not in the latest dir
                        if if_latest:
                            dir_index = slice_index[0:-1]
                            if dir_index not in patient.dir_index_list_latest:
                                continue

                        file_path_index = patient.get_file_index_given_slice(slice_index)
                        slice_info = patient.get_slice_info_given_file(file_path_index)
                        if len(slice_info) > 5:
                            slice_info_panel = slice_info[4]
                            slice_info_index = slice_info[5]
                            if isinstance(slice_info_panel, list) and isinstance(slice_info_index, list):
                                if slice_info_panel[0] == 'COR' or slice_info_panel[0] == 'COR_bright':
                                    plane_idx = slice_info_panel[1]
                                    file_path_index = patient.get_file_index_given_slice(slice_index)
                                    # if file_path_index not in file_index_valid:
                                    #     file_index_valid.append(file_path_index)
                                    file_path_string = self.image_cache_path + patient.patient_ID + "/" + str(
                                        file_path_index).replace(", ", "_").replace("[", "l").replace("]",
                                                                                                      "r") + '/subject'
                                    # file_path = file_path[:Index] + file_path[Index + 1:]
                                    filepathconformedx = file_path_string + '/conformed.mgz/x'
                                    filepathconformedy = file_path_string + '/conformed.mgz/y'
                                    filepathconformedz = file_path_string + '/conformed.mgz/z'
                                    filepathconformed = [filepathconformedx, filepathconformedy, filepathconformedz]
                                    file_path_string = filepathconformed[plane_idx]
                                    if os.path.exists(file_path_string):
                                        if file_path_string not in file_out_path_list:
                                            file_out_path_list.append(
                                                file_path_string)  # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/z
                                            file_out_info_list.append(
                                                slice_info)  # [None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]

        return file_out_path_list, file_out_info_list  # ['D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/z'], [[None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]]

    def get_index_list(self, if_latest=False):
        # shuffle patients
        random.shuffle(self.patients)
        # random.shuffle(self.patient_pos)
        # random.shuffle(self.patient_neg)

        # filter patients
        patient_and_file_path = []
        for i in tqdm(range(len(self.patients))):
            patient = self.patients[i]
            file_out_path_list, file_out_info_list = self.filter_patients(patient, if_latest)
            if len(file_out_path_list) > 0:
                patient_and_file_path.append(
                    [patient, file_out_path_list, file_out_info_list])  # [patient, [list],[ info]]

        print('total num patient: ', len(patient_and_file_path))

        # get patients for validation dataset
        patient_and_file_path_train = []
        patient_and_file_path_val_level1 = []
        patient_and_file_path_val_level2 = []
        patient_and_file_path_val_level3 = []
        # set val num fraction for different atrophy levels
        if self.max_train_len is None:
            num_val = int(len(patient_and_file_path) * self.val_split)
            num_train = len(patient_and_file_path) - num_val
        else:
            num_val = len(patient_and_file_path) - self.max_train_len
            num_train = self.max_train_len
        num_val_level2 = int(num_val * self.frac_val_level[1])
        num_val_level3 = int(num_val * self.frac_val_level[2])
        num_val_level1 = num_val - num_val_level2 - num_val_level3

        for i in range(len(patient_and_file_path)):
            patient_and_file_path_i = patient_and_file_path[i]
            patient_i = patient_and_file_path_i[0]
            file_out_path_i = patient_and_file_path_i[1]
            file_out_info_i = patient_and_file_path_i[2]  # [None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]

            # get all possible level in this patient
            level_i = []
            for info in file_out_info_i:
                level_info = info[5][2]
                if level_info not in level_i and level_info != 'level4':
                    level_i.append(level_info)

            if len(level_i) == 1:  # if only one possible level for this patient, add validation to target size
                if level_i[0] == 'level1' and len(patient_and_file_path_val_level1) < num_val_level1:
                    patient_and_file_path_val_level1.append(patient_and_file_path_i)  # [patient, [list],[ info]]
                elif level_i[0] == 'level2' and len(patient_and_file_path_val_level2) < num_val_level2:
                    patient_and_file_path_val_level2.append(patient_and_file_path_i)  # [patient, [list],[ info]]
                elif level_i[0] == 'level3' and len(patient_and_file_path_val_level3) < num_val_level3:
                    patient_and_file_path_val_level3.append(patient_and_file_path_i)  # [patient, [list],[ info]]
                elif len(patient_and_file_path_train) < num_train:
                    patient_and_file_path_train.append(patient_and_file_path_i)
            elif len(level_i) > 1 and len(patient_and_file_path_train) < num_train:
                patient_and_file_path_train.append(patient_and_file_path_i)

        patient_and_file_path_val = (patient_and_file_path_val_level1 + patient_and_file_path_val_level2
                                     + patient_and_file_path_val_level3)

        print('level1 in val: ' + str(len(patient_and_file_path_val_level1)))
        print('level2 in val: ' + str(len(patient_and_file_path_val_level2)))
        print('level3 in val: ' + str(len(patient_and_file_path_val_level3)))

        # return patient_and_file_path_train, patient_and_file_path_val # [patient, [list],[ info]]
        return (self.generate_input_output_by_file(patient_and_file_path_train),
                self.generate_input_output_by_file(patient_and_file_path_val))

    def generate_input_output_by_file(self, data_into_list):
        path_label_list = []  # [[patient_ID, 'D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x', info],...]
        for i in range(len(data_into_list)):
            patient_and_file_path_i = data_into_list[i]
            patient_i = patient_and_file_path_i[0]
            file_out_path_i = patient_and_file_path_i[1]
            file_out_info_i = patient_and_file_path_i[
                2]  # [[None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']], ...]

            for j in range(len(file_out_path_i)):
                file_out_path = file_out_path_i[j]
                file_out_info = file_out_info_i[j]
                patient_ID = patient_i.patient_ID

                if file_out_info[5][2] in self.train_level:
                    path_label_list.append([patient_ID, file_out_path, file_out_info])

        return path_label_list  # [[patient_ID, path string, info_list], ...]

    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            patient_ID = self.trainPaths[idx][0]  # patient_ID
            input_path = self.trainPaths[idx][1]  # 'D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x'
            slice_info = self.trainPaths[idx][2]  # [None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]
        else:
            patient_ID = self.valPaths[idx][0]
            input_path = self.valPaths[idx][1]
            slice_info = self.valPaths[idx][2]

        level = slice_info[5][2]
        label = 0
        if level == 'level1':
            label = 0
        elif level == 'level2' or level == 'level3':
            label = 1

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        image_label = torch.zeros(len(self.labels))  # size of label = number of sequence + 1
        image_tensor_mask_strip = torch.zeros(self.resize, self.resize)
        # image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class
        try:
            # image_array = patient.load_mri_slice(slice_index)
            # with open(input_path, "rb") as f:
            #     image_array = pickle.load(f)
            image_inco_tensor_volume = load_all_slice_in_folder(input_path, mask_channel=1, image_size=self.resize)
            image_inco_array_volume = image_inco_tensor_volume.numpy() # [C, 1, H, W]
        except:
            print("pkl.pixel_array corrupted")
            print(input_path)
            # print(slice_index)
            return (image_tensor.unsqueeze(0).repeat(self.target_channel, 1, 1), image_label, patient_ID,
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.target_channel, 1, 1))

        # rotate and flip slice volume to straight up front
        direction = 0
        if len(slice_info[4]) > 2:
            if slice_info[4][2] == "right":
                direction = 1
                rotate = 'cc'
                flip = None
            else:
                rotate = 'c'
                flip = 'hor'
        else:
            rotate = 'c'
            flip = 'hor'

        if rotate == 'c':
            image_inco_array_volume = np.rot90(image_inco_array_volume, k=1, axes=(3, 2))
        elif rotate == 'cc':
            image_inco_array_volume = np.rot90(image_inco_array_volume, k=1, axes=(2, 3))

        if flip == 'hor':
            image_inco_array_volume = np.flip(image_inco_array_volume, axis=3)
        elif flip == 'ver':
            image_inco_array_volume = np.flip(image_inco_array_volume, axis=2)

        # crop slice using target_channel and start -> end index and random margin slce
        start_idx = slice_info[5][0]
        end_idx = slice_info[5][1]
        slice_interval = end_idx - start_idx
        mid_slice = start_idx + slice_interval // 2
        if slice_interval > self.target_channel:
            slice_interval = self.target_channel
            start_idx = mid_slice - slice_interval // 2
            end_idx = start_idx + slice_interval
        elif slice_interval < self.target_channel:
            min_start = mid_slice - self.target_channel // 2
            max_end = min_start + self.target_channel
            start_random_margin = random.randint(0, self.random_extra_margin_slice)
            end_random_margin = random.randint(0, self.random_extra_margin_slice)

            start_idx = max(min_start, start_idx - start_random_margin)
            end_idx = min(max_end, end_idx + end_random_margin)

        slice_interval = end_idx - start_idx
        image_array_volume = np.zeros((self.target_channel, 1, self.resize, self.resize))
        image_array_volume[0:slice_interval] = image_inco_array_volume[start_idx:end_idx]
        image_tensor = torch.from_numpy(image_array_volume.copy()).float()  # avoid neg stride
        image_tensor = image_tensor.squeeze(1) # [C, H, W]

        # 1-hot classify
        if len(self.labels) > 1:
            label_idx = self.label_values.index(label)
            image_label[label_idx] = 1.0
        else:
            image_label[0] = label

        # print(slice_info)

        return image_tensor, image_label, patient_ID, direction


class GeneralDatasetHippoSegClassManual(GeneralDataset):
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
            n_color=1,
            if_multi_channel=True,
            if_rot_flip=False,
            patients=[],
            _model_extraction=None,
            _model_plane_detect=None,
            _model_multi_seg=None,
            mask_channel=1,
            mask_model=None,
            mask_label=[],
            mask_label_idx=[],
            mask_area_threshold=64 * 64,
            patient_handler=None,
            image_cache_path='',  # "F:/nifti_seg/T1_T1flair/"
            detect_plane=['AX', 'COR', 'SAG'],
            target_plane='COR',
            pos_neg_multiple_size=[1, 1],
            if_use_cached_seg=True,
            if_output_striped=True,
            frac_val_level=[0.5, 0.25, 0.25],
            target_channel = 40,
            random_extra_margin_slice = 5,
            train_level = ['level1', 'level2', 'level3'],
            random_tilt_angle = [15, 15, 15],
            random_argment_chance = 0.5,
            if_tilt = False,
            if_crop = False,
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.n_color = n_color
        self.mask_channel = mask_channel
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.if_multi_channel = if_multi_channel
        self.if_rot_flip = if_rot_flip
        self.patients = patients
        self.mask_model = mask_model
        self.patient_handler = patient_handler
        self._model_extraction = _model_extraction
        self._model_plane_detect = _model_plane_detect
        self._model_multi_seg = _model_multi_seg
        self.mask_label = mask_label
        self.mask_label_idx = mask_label_idx
        self.mask_area_threshold = mask_area_threshold
        self.image_cache_path = image_cache_path
        self.detect_plane = detect_plane
        self.target_plane = target_plane
        self.if_use_cached_seg = if_use_cached_seg
        self.if_output_striped = if_output_striped
        self.frac_val_level = frac_val_level
        self.target_channel = target_channel
        self.random_extra_margin_slice = random_extra_margin_slice
        self.train_level = train_level
        self.random_tilt_angle = random_tilt_angle
        self.if_tilt = if_tilt
        self.if_crop = if_crop
        self.random_argment_chance = random_argment_chance
        if _model_extraction is None:
            self.if_output_striped = False

        start0 = time.time()
        self.data_into_list_train = []
        self.data_into_list_val = []
        self.pos_neg_multiple_size = pos_neg_multiple_size
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        print("start1 - start0: ", start1 - start0)

        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    def filter_patients(self, patient: PatientCase, if_latest: bool = False):
        if self.mri_sequence is None:
            sequence_names = list(patient.sequence_slice_index_lists_dict.keys())
        else:
            sequence_names = self.mri_sequence

        # get list of file index in patient given sequence
        file_out_path_list = []
        file_out_info_list = []
        for sequence_name in sequence_names:
            if patient.sequence_slice_index_lists_dict[sequence_name]:
                slice_index_list = patient.sequence_slice_index_lists_dict[sequence_name]
                if len(slice_index_list) < 200000:  # more than 200000 mri slices for one series? doesn't sound right
                    for slice_index in slice_index_list:

                        # exclude the slices that are not in the latest dir
                        if if_latest:
                            dir_index = slice_index[0:-1]
                            if dir_index not in patient.dir_index_list_latest:
                                continue

                        file_path_index = patient.get_file_index_given_slice(slice_index)
                        slice_info = patient.get_slice_info_given_file(file_path_index)
                        if len(slice_info) > 5:
                            slice_info_panel = slice_info[4]
                            slice_info_index = slice_info[5]
                            if isinstance(slice_info_panel, list) and isinstance(slice_info_index, list):
                                if slice_info_panel[0] == 'COR' or slice_info_panel[0] == 'COR_bright':
                                    plane_idx = slice_info_panel[1]
                                    file_path_index = patient.get_file_index_given_slice(slice_index)
                                    # if file_path_index not in file_index_valid:
                                    #     file_index_valid.append(file_path_index)
                                    file_path_string = self.image_cache_path + patient.patient_ID + "/" + str(
                                        file_path_index).replace(", ", "_").replace("[", "l").replace("]",
                                                                                                      "r") + '/subject'
                                    # file_path = file_path[:Index] + file_path[Index + 1:]
                                    filepathconformedx = file_path_string + '/conformed.mgz/x'
                                    filepathconformedy = file_path_string + '/conformed.mgz/y'
                                    filepathconformedz = file_path_string + '/conformed.mgz/z'
                                    filepathconformed = [filepathconformedx, filepathconformedy, filepathconformedz]
                                    file_path_string = filepathconformed[plane_idx]
                                    if os.path.exists(file_path_string):
                                        if file_path_string not in file_out_path_list:
                                            file_out_path_list.append(
                                                file_path_string)  # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/z
                                            file_out_info_list.append(
                                                slice_info)  # [None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]

        return file_out_path_list, file_out_info_list  # ['D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/z'], [[None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]]

    def get_index_list(self, if_latest=False):
        # shuffle patients
        random.shuffle(self.patients)
        # random.shuffle(self.patient_pos)
        # random.shuffle(self.patient_neg)

        # filter patients
        patient_and_file_path = []
        for i in tqdm(range(len(self.patients))):
            patient = self.patients[i]
            file_out_path_list, file_out_info_list = self.filter_patients(patient, if_latest)
            if len(file_out_path_list) > 0:
                patient_and_file_path.append(
                    [patient, file_out_path_list, file_out_info_list])  # [patient, [list],[ info]]

        print('total num patient: ', len(patient_and_file_path))

        # get patients for validation dataset
        patient_and_file_path_train = []
        patient_and_file_path_val_level1 = []
        patient_and_file_path_val_level2 = []
        patient_and_file_path_val_level3 = []
        # set val num fraction for different atrophy levels
        if self.max_train_len is None:
            num_val = int(len(patient_and_file_path) * self.val_split)
            num_train = len(patient_and_file_path) - num_val
        else:
            num_val = len(patient_and_file_path) - self.max_train_len
            num_train = self.max_train_len
        num_val_level2 = int(num_val * self.frac_val_level[1])
        num_val_level3 = int(num_val * self.frac_val_level[2])
        num_val_level1 = num_val - num_val_level2 - num_val_level3

        for i in range(len(patient_and_file_path)):
            patient_and_file_path_i = patient_and_file_path[i]
            patient_i = patient_and_file_path_i[0]
            file_out_path_i = patient_and_file_path_i[1]
            file_out_info_i = patient_and_file_path_i[2]  # [None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]

            # get all possible level in this patient
            level_i = []
            for info in file_out_info_i:
                level_info = info[5][2]
                if level_info not in level_i and level_info != 'level4':
                    level_i.append(level_info)

            if len(level_i) == 1:  # if only one possible level for this patient, add validation to target size
                if level_i[0] == 'level1' and len(patient_and_file_path_val_level1) < num_val_level1:
                    patient_and_file_path_val_level1.append(patient_and_file_path_i)  # [patient, [list],[ info]]
                elif level_i[0] == 'level2' and len(patient_and_file_path_val_level2) < num_val_level2:
                    patient_and_file_path_val_level2.append(patient_and_file_path_i)  # [patient, [list],[ info]]
                elif level_i[0] == 'level3' and len(patient_and_file_path_val_level3) < num_val_level3:
                    patient_and_file_path_val_level3.append(patient_and_file_path_i)  # [patient, [list],[ info]]
                elif len(patient_and_file_path_train) < num_train:
                    patient_and_file_path_train.append(patient_and_file_path_i)
            elif len(level_i) > 1 and len(patient_and_file_path_train) < num_train:
                patient_and_file_path_train.append(patient_and_file_path_i)

        patient_and_file_path_val = (patient_and_file_path_val_level1 + patient_and_file_path_val_level2
                                     + patient_and_file_path_val_level3)

        print('level1 in val: ' + str(len(patient_and_file_path_val_level1)))
        print('level2 in val: ' + str(len(patient_and_file_path_val_level2)))
        print('level3 in val: ' + str(len(patient_and_file_path_val_level3)))

        # return patient_and_file_path_train, patient_and_file_path_val # [patient, [list],[ info]]
        return (self.generate_input_output_by_file(patient_and_file_path_train),
                self.generate_input_output_by_file(patient_and_file_path_val))

    def generate_input_output_by_file(self, data_into_list):
        path_label_list = []  # [[patient_ID, 'D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x', info],...]
        for i in range(len(data_into_list)):
            patient_and_file_path_i = data_into_list[i]
            patient_i = patient_and_file_path_i[0]
            file_out_path_i = patient_and_file_path_i[1]
            file_out_info_i = patient_and_file_path_i[
                2]  # [[None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']], ...]

            for j in range(len(file_out_path_i)):
                file_out_path = file_out_path_i[j]
                file_out_info = file_out_info_i[j]
                patient_ID = patient_i.patient_ID

                if file_out_info[5][2] in self.train_level:
                    path_label_list.append([patient_ID, file_out_path, file_out_info])

        return path_label_list  # [[patient_ID, path string, info_list], ...]

    def get_and_reset_slices(self, path_temp, rot='c', flip = 'hor', image_size = None,
                                  if_process_image = True):

        image_inco_tensor_volume = load_all_slice_in_folder(path_temp, mask_channel=1, image_size=image_size, if_process_image=if_process_image)
        image_inco_array_volume = image_inco_tensor_volume.numpy()

        if rot == 'c':
            image_inco_array_volume = np.rot90(image_inco_array_volume, k=1, axes=(3, 2))
        elif rot == 'cc':
            image_inco_array_volume = np.rot90(image_inco_array_volume, k=1, axes=(2, 3))

        if flip == 'hor':
            image_inco_array_volume = np.flip(image_inco_array_volume, axis=3)
        elif flip == 'ver':
            image_inco_array_volume = np.flip(image_inco_array_volume, axis=2)

        return image_inco_array_volume
    def __getitem__(self, idx):
        # time0 = time.time()
        if self.is_train:
            patient_ID = self.trainPaths[idx][0]  # patient_ID
            input_path = self.trainPaths[idx][1]  # 'D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz/x'
            slice_info = self.trainPaths[idx][2]  # [None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]
        else:
            patient_ID = self.valPaths[idx][0]
            input_path = self.valPaths[idx][1]
            slice_info = self.valPaths[idx][2]

        level = slice_info[5][2]
        label = 0
        if level == 'level1':
            label = 0
        elif level == 'level2' or level == 'level3':
            label = 1

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        image_label = torch.zeros(len(self.labels))  # size of label = number of sequence + 1
        image_tensor_mask_strip = torch.zeros(self.resize, self.resize)
        # image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class
        try:
            # rotate and flip slice volume to straight up front
            direction = 0
            if len(slice_info[4]) > 2:
                if slice_info[4][2] == "right":
                    direction = 1
                    rotate = 'cc'
                    flip = None
                else:
                    rotate = 'c'
                    flip = 'hor'
            else:
                rotate = 'c'
                flip = 'hor'
            # image_inco_tensor_volume = load_all_slice_in_folder(input_path, mask_channel=1, image_size=self.resize)
            # image_inco_array_volume = image_inco_tensor_volume.numpy() # [C, 1, H, W]
            image_inco_tensor_volume = self.get_and_reset_slices(input_path, rot=rotate, flip=flip, image_size=None)
            image_inco_array_volume = image_inco_tensor_volume[:, 0, :, :] # [C,  H, W]

            input_path_mask = input_path.replace('conformed.mgz', 'aseg.auto_noCCseg.mgz')
            image_inco_tensor_volume_mask = self.get_and_reset_slices(input_path_mask, rot=rotate,
                                                                      flip=flip, image_size=None, if_process_image=False)
            image_inco_array_volume_mask = image_inco_tensor_volume_mask[:, 0, :, :]
            image_inco_array_volume_mask_total = image_inco_array_volume_mask.copy()
            image_inco_array_volume_mask_total[image_inco_array_volume_mask_total > 0.0] = 1

            # get hippo area mask only
            image_inco_array_volume_mask_hippo = image_inco_array_volume_mask.copy()
            output_labels = np.zeros_like(image_inco_array_volume_mask_hippo)  # [H, W]
            for i in range(len(self.label_values)):
                label_value = self.label_values[i]
                seg_value = np.isin(image_inco_array_volume_mask_hippo, label_value)
                output_labels += (i + 1) * seg_value

            output_labels[output_labels > 0.1] = 1
            output_labels[output_labels <= 0.1] = 0
            image_inco_array_volume_mask_hippo = output_labels
        except:
            print("pkl.pixel_array corrupted")
            print(input_path)
            # print(slice_index)
            return (image_tensor.unsqueeze(0).repeat(self.target_channel, 1, 1), image_label, patient_ID, 0,
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.target_channel, 1, 1),
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.target_channel, 1, 1)), 0

        # print_var_detail(image_inco_array_volume, 'image_inco_array_volume_fliped')
        # print_var_detail(image_inco_array_volume_mask_total, 'image_inco_array_volume_mask_total')
        # print_var_detail(image_inco_array_volume_mask_hippo, 'image_inco_array_volume_mask_hippo')

        use_argment = 0
        if random.random() > self.random_argment_chance and self.is_train:
            use_argment =1
            # random flip
            if self.if_rot_flip:
                self.custom_flip_func.flip_prob_horizon = 0.5 # only flip with 0.5 chance on horizon direction
                self.custom_flip_func.flip_prob_vertical = 0
                image_inco_array_volume_fliped, image_inco_array_volume_mask_total_fliped = self.custom_flip_func(
                    image_inco_array_volume, image_inco_array_volume_mask_total)
                image_inco_array_volume_mask_hippo_fliped, _ = self.custom_flip_func(image_inco_array_volume_mask_hippo,
                                                                                image_inco_array_volume_mask_hippo)
            else:
                image_inco_array_volume_fliped = image_inco_array_volume
                image_inco_array_volume_mask_total_fliped = image_inco_array_volume_mask_total
                image_inco_array_volume_mask_hippo_fliped = image_inco_array_volume_mask_hippo

            # tilt
            if self.if_tilt:
                start = slice_info[5][0]
                end = slice_info[5][1]
                center_d = (start + end) // 2
                center_HW = center_of_mass(image_inco_array_volume_mask_total[center_d])
                position = [center_d, int(center_HW[0]), int(center_HW[1])]

                angle_x = random.uniform(-self.random_tilt_angle[0], self.random_tilt_angle[0])
                image_inco_array_volume_tilt = upscale_rotate_downscale_fast(image_inco_array_volume_fliped, angle=angle_x,
                                                                           axis=1, position=position, upscale_factor=3)
                image_inco_array_volume_mask_total_tilt = upscale_rotate_downscale_binary(image_inco_array_volume_mask_total_fliped,
                                                                                      angle=angle_x, axis=1, position=position,
                                                                                      upscale_factor=3)
                image_inco_array_volume_mask_hippo_tilt = upscale_rotate_downscale_binary(image_inco_array_volume_mask_hippo_fliped,
                                                                                      angle=angle_x, axis=1, position=position,
                                                                                      upscale_factor=3)
                angle_y = random.uniform(-self.random_tilt_angle[1], self.random_tilt_angle[1])
                image_inco_array_volume_tilt = upscale_rotate_downscale_fast(image_inco_array_volume_tilt, angle=angle_y, axis=2,
                                                                           position=position, upscale_factor=3)
                image_inco_array_volume_mask_total_tilt = upscale_rotate_downscale_binary(image_inco_array_volume_mask_total_tilt,
                                                                                      angle=angle_y, axis=2, position=position,
                                                                                      upscale_factor=3)
                image_inco_array_volume_mask_hippo_tilt = upscale_rotate_downscale_binary(image_inco_array_volume_mask_hippo_tilt,
                                                                                      angle=angle_y, axis=2, position=position,
                                                                                      upscale_factor=3)

                angle_z = random.uniform(-self.random_tilt_angle[2], self.random_tilt_angle[2])
                image_inco_array_volume_tilt = upscale_rotate_downscale_fast(image_inco_array_volume_tilt, angle=angle_z, axis=0,
                                                                           position=position, upscale_factor=3)
                image_inco_array_volume_mask_total_tilt = upscale_rotate_downscale_binary(image_inco_array_volume_mask_total_tilt,
                                                                                      angle=angle_z, axis=0, position=position,
                                                                                      upscale_factor=3)
                image_inco_array_volume_mask_hippo_tilt = upscale_rotate_downscale_binary(image_inco_array_volume_mask_hippo_tilt,
                                                                                      angle=angle_z, axis=0, position=position,
                                                                                      upscale_factor=3)
            else:
                image_inco_array_volume_tilt = image_inco_array_volume_fliped
                image_inco_array_volume_mask_total_tilt = image_inco_array_volume_mask_total_fliped
                image_inco_array_volume_mask_hippo_tilt = image_inco_array_volume_mask_hippo_fliped

            # print_var_detail(image_inco_array_volume_tilt, 'image_inco_array_volume_tilt')
            # print_var_detail(image_inco_array_volume_mask_total_tilt, 'image_inco_array_volume_mask_total_tilt')
            # print_var_detail(image_inco_array_volume_mask_hippo_tilt, 'image_inco_array_volume_mask_hippo_tilt')

            # random crop image
            target_size = (self.resize, self.resize)
            current_image_shape = (
                image_inco_array_volume_mask_total_tilt.shape[-2], image_inco_array_volume_mask_total_tilt.shape[-1])
            if self.if_crop:
                top, bottom, left, right = find_mask_bounds(image_inco_array_volume_mask_total_tilt)
                crop_coords = (top, bottom, left, right)
                random_crop_margin_v = (random.randint(0, max(top, current_image_shape[0] - bottom)),
                                        random.randint(0, max(top, current_image_shape[0] - bottom)))
                random_crop_margin_h = (random.randint(0, max(left, current_image_shape[1] - right)),
                                        random.randint(0, max(left, current_image_shape[1] - right)))

                image_inco_array_volume_tilt_crop = random_crop_given_bounds(image_inco_array_volume_tilt, bounds=crop_coords,
                                                                             target_size=target_size,
                                                                             random_crop_margin_v=random_crop_margin_v,
                                                                             random_crop_margin_h=random_crop_margin_h)
                image_inco_array_volume_mask_total_tilt_crop = random_crop_given_bounds(image_inco_array_volume_mask_total_tilt,
                                                                                        bounds=crop_coords,
                                                                                        target_size=target_size,
                                                                                        random_crop_margin_v=random_crop_margin_v,
                                                                                        random_crop_margin_h=random_crop_margin_h)
                image_inco_array_volume_mask_hippo_tilt_crop = random_crop_given_bounds(image_inco_array_volume_mask_hippo_tilt,
                                                                                        bounds=crop_coords,
                                                                                        target_size=target_size,
                                                                                        random_crop_margin_v=random_crop_margin_v,
                                                                                        random_crop_margin_h=random_crop_margin_h)
            else:
                top, bottom, left, right = (0, current_image_shape[0], 0, current_image_shape[1])
                crop_coords = (top, bottom, left, right)
                random_crop_margin_v = (0,0)
                random_crop_margin_h = (0,0)

                image_inco_array_volume_tilt_crop = random_crop_given_bounds(image_inco_array_volume_tilt,
                                                                             bounds=crop_coords,
                                                                             target_size=target_size,
                                                                             random_crop_margin_v=random_crop_margin_v,
                                                                             random_crop_margin_h=random_crop_margin_h)
                image_inco_array_volume_mask_total_tilt_crop = random_crop_given_bounds(
                    image_inco_array_volume_mask_total_tilt,
                    bounds=crop_coords,
                    target_size=target_size,
                    random_crop_margin_v=random_crop_margin_v,
                    random_crop_margin_h=random_crop_margin_h)
                image_inco_array_volume_mask_hippo_tilt_crop = random_crop_given_bounds(
                    image_inco_array_volume_mask_hippo_tilt,
                    bounds=crop_coords,
                    target_size=target_size,
                    random_crop_margin_v=random_crop_margin_v,
                    random_crop_margin_h=random_crop_margin_h)

        else:
            # crop all image and resize to target size
            target_size = (self.resize, self.resize)
            current_image_shape = (
                image_inco_array_volume.shape[-2], image_inco_array_volume.shape[-1])
            top, bottom, left, right = (0, current_image_shape[0], 0, current_image_shape[1])
            crop_coords = (top, bottom, left, right)
            random_crop_margin_v = (0,0)
            random_crop_margin_h = (0,0)

            image_inco_array_volume_tilt_crop = random_crop_given_bounds(image_inco_array_volume,
                                                                         bounds=crop_coords,
                                                                         target_size=target_size,
                                                                         random_crop_margin_v=random_crop_margin_v,
                                                                         random_crop_margin_h=random_crop_margin_h)
            image_inco_array_volume_mask_total_tilt_crop = random_crop_given_bounds(
                image_inco_array_volume_mask_total,
                bounds=crop_coords,
                target_size=target_size,
                random_crop_margin_v=random_crop_margin_v,
                random_crop_margin_h=random_crop_margin_h)
            image_inco_array_volume_mask_hippo_tilt_crop = random_crop_given_bounds(
                image_inco_array_volume_mask_hippo,
                bounds=crop_coords,
                target_size=target_size,
                random_crop_margin_v=random_crop_margin_v,
                random_crop_margin_h=random_crop_margin_h)

        # print_var_detail(image_inco_array_volume_tilt_crop, 'image_inco_array_volume_tilt_crop')
        # print_var_detail(image_inco_array_volume_mask_total_tilt_crop, 'image_inco_array_volume_mask_total_tilt_crop')
        # print_var_detail(image_inco_array_volume_mask_hippo_tilt_crop, 'image_inco_array_volume_mask_hippo_tilt_crop')


        # crop slice using target_channel and start -> end index and random margin slce
        start_idx = slice_info[5][0]
        end_idx = slice_info[5][1]
        slice_interval = end_idx - start_idx
        mid_slice = start_idx + slice_interval // 2
        if slice_interval > self.target_channel:
            slice_interval = self.target_channel
            start_idx = mid_slice - slice_interval // 2
            end_idx = start_idx + slice_interval
        elif slice_interval < self.target_channel:
            min_start = mid_slice - self.target_channel // 2
            max_end = min_start + self.target_channel
            start_random_margin = random.randint(0, self.random_extra_margin_slice)
            end_random_margin = random.randint(0, self.random_extra_margin_slice)

            start_idx = max(min_start, start_idx - start_random_margin)
            end_idx = min(max_end, end_idx + end_random_margin)

        slice_interval = end_idx - start_idx
        image_array_volume = np.zeros((self.target_channel, self.resize, self.resize))
        image_array_volume[0:slice_interval] = image_inco_array_volume_tilt_crop[start_idx:end_idx]
        image_tensor = torch.from_numpy(image_array_volume.copy()).float()  # avoid neg stride

        image_array_volume_mask_total = np.zeros((self.target_channel, self.resize, self.resize))
        image_array_volume_mask_total[0:slice_interval] = image_inco_array_volume_mask_total_tilt_crop[start_idx:end_idx]
        image_tensor_mask_total = torch.from_numpy(image_array_volume_mask_total.copy()).float()  # avoid neg stride

        image_array_volume_mask_hippo = np.zeros((self.target_channel, self.resize, self.resize))
        image_array_volume_mask_hippo[0:slice_interval] = image_inco_array_volume_mask_hippo_tilt_crop[start_idx:end_idx]
        image_tensor_mask_hippo = torch.from_numpy(image_array_volume_mask_hippo.copy()).float()  # avoid neg stride

        # 1-hot classify
        # if len(self.labels) > 1:
        #     label_idx = self.label_values.index(label)
        #     image_label[label_idx] = 1.0
        # else:
        image_label[0] = label

        # print(slice_info)

        return image_tensor, image_label, patient_ID, direction, image_tensor_mask_total, image_tensor_mask_hippo, use_argment

def create_combine_dataloader(
        datasets,
        batch_size,
        is_distributed=False,
        is_train=False,
        num_workers=0,
):
    """
    Create a combined dataloader for all patient datasets
    """
    combined_dataset = datasets[0]
    for i in range(len(datasets)):
        if i > 0:
            combined_dataset = torch.utils.data.ConcatDataset([combined_dataset, datasets[i]])
    data_sampler = None
    if is_distributed:
        data_sampler = DistributedSampler(combined_dataset)

    # create dataloader
    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=(data_sampler is None) and is_train,
        sampler=data_sampler,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=True,
    )

    # return loader, note all np in FastmriDataset will be converted to tensor via dataloader
    return loader
