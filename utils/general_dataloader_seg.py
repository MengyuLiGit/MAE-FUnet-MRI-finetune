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
from utils.general_dataloader import GeneralDataset, check_mask_area
from scipy.ndimage import rotate
from utils.general_dataloader3D import find_mask_bounds
import re

def extract_last_number(path):
    match = re.search(r'(\d+)$', path)  # Find the last sequence of digits
    return int(match.group(1)) if match else None  # Convert to int if found

def remove_trailing_number(path):
    return re.sub(r'/\d+$', '', path)  # Removes last `/number`


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
            max_train_len=None,
            max_val_len=None,
            n_color=1,
            if_multi_channel=True,
            if_rot_flip=False,
            if_random_crop=False,
            if_tilted=False,
            tilt_angle = [-45, 45],
            image_seg_path = 'F:/nifti_seg/T1_T1flair/',
            cache_path = 'E:/nifti_seg_cache/T1_T1flair/',
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.patient_loader = patient_loader
        self.num_patients = len(patient_loader.patients)
        self.n_color = n_color
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
        self.image_seg_path = image_seg_path
        start0 = time.time()
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        print("start1 - start0: ", start1 - start0)
        self.if_rot_flip = if_rot_flip
        self.if_random_crop = if_random_crop
        self.if_tilted = if_tilted
        self.tilt_angle = tilt_angle
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()


        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)

    def get_index_list(self, if_latest=False):
        # get patient file path that have 3D T1_T1flair sequence, list them in [[patient0_path0, patient0_path1, ...], [patient1_path0, patient1_path1,...]]
        patients_file_path_list = []
        for i in tqdm(range(len(self.patient_loader.patients))):
            # if i > 40:
            #     break
            patient_file_path_list= []
            T1_T1flair_file_index_list = self.patient_loader.patients[i].get_file_index_list_given_sequence('T1_T1flair')
            for T1_T1flair_file_index in T1_T1flair_file_index_list:
                # T1_T1flair_file_path = self.patient_loader.patients[i].get_file_path_given_file(
                #     T1_T1flair_file_index)
                image_volume = self.patient_loader.patients[i].load_mri_volume(T1_T1flair_file_index,
                                                                                           if_resize=False)
                nifti_json =self.patient_loader.patients[i].load_nifti_json(slice_index=None, file_index=T1_T1flair_file_index)
                if 'MRAcquisitionType' not in nifti_json.keys():
                    nifti_json['MRAcquisitionType'] = 'unknown'

                if len(image_volume.shape) == 3:
                    if '3D' in nifti_json['MRAcquisitionType'].upper() or image_volume.shape[-1] > 48:
                        file_path_string = self.image_seg_path + self.patient_loader.patients[i].patient_ID + "/" + str(
                            T1_T1flair_file_index).replace(", ", "_").replace("[", "l").replace("]",
                                                                                                "r") + '/subject/aseg.auto_noCCseg.mgz'
                        if os.path.isfile(file_path_string) and file_path_string not in patient_file_path_list:
                            patient_file_path_list.append(file_path_string)
            patients_file_path_list.append(patient_file_path_list)

        # random shuffle list
        random.shuffle(patients_file_path_list)
        # seperate to train and test list
        if self.max_train_len is None:
            valPathsLen = int(len(patients_file_path_list) * self.val_split)
            trainPathsLen = len(patients_file_path_list) - valPathsLen
        elif self.max_train_len < len(patients_file_path_list) - int(len(patients_file_path_list) * self.val_split):
            valPathsLen = int(len(patients_file_path_list) * self.val_split)
            trainPathsLen = self.max_train_len
        else:
            print("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if self.max_val_len <= len(patients_file_path_list) - trainPathsLen:
                valPathsLen = self.max_val_len
            else:
                print("assigned maximum val len exceeds total train num of slices")

        data_into_list_train = patients_file_path_list[:trainPathsLen]
        data_into_list_val = patients_file_path_list[-valPathsLen:]
        data_into_list_train = [item for sublist in data_into_list_train for item in sublist]
        data_into_list_val = [item for sublist in data_into_list_val for item in sublist]
        return (self.generate_input_output_by_file(data_into_list_train),
                self.generate_input_output_by_file(data_into_list_val))


    def generate_input_output_by_file(self, data_into_list):
        file_valid_seg = []
        file_valid_confirm = []
        for data_path in data_into_list:
            data_path_seg = data_path
            # print(data_path_seg)
            data_path_confirm = data_path_seg.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz')
            if os.path.isfile(data_path_confirm) and os.path.isfile(data_path_seg):
                data_path_seg = data_path_seg.replace(self.image_seg_path, self.cache_path)
                file_valid_seg.append(data_path_seg + '/x')
                file_valid_seg.append(data_path_seg + '/y')
                file_valid_seg.append(data_path_seg + '/z')
                data_path_confirm = data_path_confirm.replace(self.image_seg_path, self.cache_path)
                file_valid_confirm.append(data_path_confirm + '/x')
                file_valid_confirm.append(data_path_confirm + '/y')
                file_valid_confirm.append(data_path_confirm + '/z')

        input_image_path_list = []
        output_image_path_list = []
        for i in tqdm(range(len(file_valid_seg))):
            if i > -1:
                file_valid_seg_i = file_valid_seg[i]
                # file_valid_seg_i = file_valid_seg_i.replace(self.image_seg_path, self.cache_path)
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

        # apply rot and flip
        if self.if_rot_flip and self.is_train:
            if random.random() < 0.5:
                image_array, output_labels = self.custom_flip_func(image_array, output_labels)
                image_array, output_labels = self.custom_rot_func(image_array, output_labels)

        # if self.if_rot_flip and self.is_train:
        #     if random.random() < 0.5:
        #         image_array = np.flip(image_array, axis=1)
        #         output_labels = np.flip(output_labels, axis=1)
        #     if random.random() < 0.5:
        #         image_array = np.flip(image_array, axis=0)
        #         output_labels = np.flip(output_labels, axis=0)

            # avoid neg stride
            image_array = np.ascontiguousarray(image_array)
            output_labels = np.ascontiguousarray(output_labels)

        if self.if_tilted and self.is_train:
            if random.random() < 0.5:
                tilt_angle = random.uniform(self.tilt_angle[0], self.tilt_angle[1])
                output_labels = rotate(output_labels, angle=tilt_angle, reshape=False, order=0)
                image_array = rotate(image_array, angle=tilt_angle, reshape=False, order=1)

        if self.if_random_crop and self.is_train:
            if random.random() < 0.5:
                top, bottom, left, right = find_mask_bounds(output_labels)
                target_size = (self.resize, self.resize)
                current_image_shape = (
                    output_labels.shape[-2], output_labels.shape[-1])
                crop_coords = (top, bottom, left, right)
                random_crop_margin_v = (random.randint(0, max(top, current_image_shape[0] - bottom)),
                                        random.randint(0, max(top, current_image_shape[0] - bottom)))
                random_crop_margin_h = (random.randint(0, max(left, current_image_shape[1] - right)),
                                        random.randint(0, max(left, current_image_shape[1] - right)))
                image_array = random_crop_given_bounds(image_array, bounds=crop_coords,
                                                              target_size=target_size,
                                                              random_crop_margin_v=random_crop_margin_v,
                                                              random_crop_margin_h=random_crop_margin_h)
                output_labels = random_crop_given_bounds(output_labels,
                                                                          bounds=crop_coords,
                                                                          target_size=target_size,
                                                                          random_crop_margin_v=random_crop_margin_v,
                                                                          random_crop_margin_h=random_crop_margin_h)





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
        image_tensor_mask_binary = torch.from_numpy(output_labels).float()
        image_tensor_masked = image_tensor * (image_tensor_mask_binary > 0)

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            # self.patient_loader.patients[patient_index].print_all_mri_sessions()
            # print("file_index", file_index)
            # print("slice_index", slice_index)
            # print('shape')

        return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
            image_tensor_masked, \
            image_tensor_mask_binary

class GeneralDatasetMultiSegNPY(GeneralDataset):
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
            if_random_crop=False,
            if_tilted=False,
            tilt_angle = [-45, 45],
            image_seg_path = 'F:/nifti_seg/T1_T1flair/',
            cache_path = 'E:/nifti_seg_cache/T1_T1flair/',
            min_mask_area = 128,
            **kwargs
    ):
        self.resize = resize
        self.drop_fraction = drop_fraction
        self.mri_sequence = mri_sequence
        self.patient_loader = patient_loader
        self.num_patients = len(patient_loader.patients)
        self.n_color = n_color
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
        self.image_seg_path = image_seg_path
        self.min_mask_area = min_mask_area
        start0 = time.time()
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        print("start1 - start0: ", start1 - start0)
        self.if_rot_flip = if_rot_flip
        self.if_random_crop = if_random_crop
        self.if_tilted = if_tilted
        self.tilt_angle = tilt_angle
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()


        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)

    def get_index_list(self, if_latest=False):
        # get patient file path that have 3D T1_T1flair sequence, list them in [[patient0_path0, patient0_path1, ...], [patient1_path0, patient1_path1,...]]
        patients_file_path_list = []
        for i in tqdm(range(len(self.patient_loader.patients))):
            # if i > 40:
            #     break
            patient_file_path_list= []
            T1_T1flair_file_index_list = self.patient_loader.patients[i].get_file_index_list_given_sequence('T1_T1flair')
            for T1_T1flair_file_index in T1_T1flair_file_index_list:
                # T1_T1flair_file_path = self.patient_loader.patients[i].get_file_path_given_file(
                #     T1_T1flair_file_index)
                image_volume = self.patient_loader.patients[i].load_mri_volume(T1_T1flair_file_index,
                                                                                           if_resize=False)
                nifti_json =self.patient_loader.patients[i].load_nifti_json(slice_index=None, file_index=T1_T1flair_file_index)
                if 'MRAcquisitionType' not in nifti_json.keys():
                    nifti_json['MRAcquisitionType'] = 'unknown'

                if len(image_volume.shape) == 3:
                    if '3D' in nifti_json['MRAcquisitionType'].upper() or image_volume.shape[-1] > 48:
                        file_path_string = self.image_seg_path + self.patient_loader.patients[i].patient_ID + "/" + str(
                            T1_T1flair_file_index).replace(", ", "_").replace("[", "l").replace("]",
                                                                                                "r") + '/subject/aseg.auto_noCCseg.mgz'
                        if os.path.isfile(file_path_string) and file_path_string not in patient_file_path_list:
                            patient_file_path_list.append(file_path_string)
            patients_file_path_list.append(patient_file_path_list)

        # random shuffle list
        random.shuffle(patients_file_path_list)
        # seperate to train and test list
        if self.max_train_len is None:
            valPathsLen = int(len(patients_file_path_list) * self.val_split)
            trainPathsLen = len(patients_file_path_list) - valPathsLen
        elif self.max_train_len < len(patients_file_path_list) - int(len(patients_file_path_list) * self.val_split):
            valPathsLen = int(len(patients_file_path_list) * self.val_split)
            trainPathsLen = self.max_train_len
        else:
            print("assigned maximum train len exceeds total train num of slices")
        if self.max_val_len is not None:
            if self.max_val_len <= len(patients_file_path_list) - trainPathsLen:
                valPathsLen = self.max_val_len
            else:
                print("assigned maximum val len exceeds total train num of slices")

        data_into_list_train = patients_file_path_list[:trainPathsLen]
        data_into_list_val = patients_file_path_list[-valPathsLen:]
        data_into_list_train = [item for sublist in data_into_list_train for item in sublist]
        data_into_list_val = [item for sublist in data_into_list_val for item in sublist]
        return (self.generate_input_output_by_file(data_into_list_train),
                self.generate_input_output_by_file(data_into_list_val))


    def generate_input_output_by_file(self, data_into_list):
        file_valid_seg = []
        file_valid_confirm = []
        for data_path in data_into_list:
            data_path_seg = data_path
            # print(data_path_seg)
            data_path_confirm = data_path_seg.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz')
            if os.path.isfile(data_path_confirm) and os.path.isfile(data_path_seg):
                data_path_seg = data_path_seg.replace(self.image_seg_path, self.cache_path)
                data_path_seg = data_path_seg.replace('aseg.auto_noCCseg.mgz', 'aseg.auto_noCCseg.npy')
                file_valid_seg.append(data_path_seg + '/x')
                file_valid_seg.append(data_path_seg + '/y')
                file_valid_seg.append(data_path_seg + '/z')
                data_path_confirm = data_path_confirm.replace(self.image_seg_path, self.cache_path)
                data_path_confirm = data_path_confirm.replace('conformed.mgz', 'conformed.npy')
                file_valid_confirm.append(data_path_confirm + '/x')
                file_valid_confirm.append(data_path_confirm + '/y')
                file_valid_confirm.append(data_path_confirm + '/z')

        input_image_path_list = []
        output_image_path_list = []
        for i in tqdm(range(len(file_valid_seg))):
            if i > -1:
                file_valid_seg_i = file_valid_seg[i]
                # file_valid_seg_i = file_valid_seg_i.replace(self.image_seg_path, self.cache_path)
                # print(file_valid_seg_i)
                list_temp_seg_i = []
                list_temp_confirm_i = []

                # get all slice in file
                path_seg_temp_list = []
                array_seg_mmap = np.load(file_valid_seg_i[:-2],mmap_mode="r")
                if file_valid_seg_i[-1] == 'x':
                    for x_i in range(array_seg_mmap.shape[0]):
                        path_seg_temp_list.append(file_valid_seg_i + '/' + str(x_i))
                elif file_valid_seg_i[-1] == 'y':
                    for y_i in range(array_seg_mmap.shape[1]):
                        path_seg_temp_list.append(file_valid_seg_i + '/' + str(y_i))
                elif file_valid_seg_i[-1] == 'z':
                    for z_i in range(array_seg_mmap.shape[2]):
                        path_seg_temp_list.append(file_valid_seg_i + '/' + str(z_i))

                if len(path_seg_temp_list) < 2048:
                    start0 = time.time()
                    for path_seg_temp in path_seg_temp_list:
                        slice_num = extract_last_number(path_seg_temp)
                        if slice_num is not None:
                            mask_seg = None
                            if file_valid_seg_i[-1] == 'x':
                                mask_seg = array_seg_mmap[slice_num,:,:]
                            elif file_valid_seg_i[-1] == 'y':
                                mask_seg = array_seg_mmap[:,slice_num,:]
                            elif file_valid_seg_i[-1] == 'z':
                                mask_seg = array_seg_mmap[:,:,slice_num]
                            checked_path_seg_temp = check_mask_area(mask_seg, self.labels, self.label_values,
                                                                    self.min_mask_area, path_seg_temp)
                            if checked_path_seg_temp is not None:
                                list_temp_seg_i.append(checked_path_seg_temp)
                                list_temp_confirm_i.append(
                                    checked_path_seg_temp.replace('aseg.auto_noCCseg.npy', 'conformed.npy'))
                    start1 = time.time()
                    if (start1 - start0) > 2:
                        print('start1-start0 single: ', start1 - start0)
                        print(file_valid_seg_i)
                else:
                    start0 = time.time()
                    labels_list = [self.labels] * len(path_seg_temp_list)
                    label_values_list = [self.label_values] * len(path_seg_temp_list)
                    min_area_size_list = [self.min_mask_area] * len(path_seg_temp_list)

                    # extract slice into list
                    mask_seg_list = []
                    if file_valid_seg_i[-1] == 'x':
                        mask_seg_list = [array_seg_mmap[k,:,:] for k in range(array_seg_mmap.shape[0])]
                    elif file_valid_seg_i[-1] == 'y':
                        mask_seg_list = [array_seg_mmap[:,k,:] for k in range(array_seg_mmap.shape[1])]
                    elif file_valid_seg_i[-1] == 'z':
                        mask_seg_list= [array_seg_mmap[:,:,k] for k in range(array_seg_mmap.shape[2])]

                    with Pool(processes=16) as pool:
                        for checked_path_seg_temp in pool.starmap(
                                check_mask_area,
                                zip(mask_seg_list,
                                    labels_list,
                                    label_values_list,
                                    min_area_size_list, path_seg_temp_list)):
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
        #
        #         ##########################
        #         file_paths = os.listdir(file_valid_seg_i)
        #         for file in file_paths:
        #             path_seg_temp = os.path.join(file_valid_seg_i, file)
        #             path_seg_temp_list.append(path_seg_temp)
        #
        #         # print(str(i), " file_paths ", len(path_seg_temp_list))
        #         if len(path_seg_temp_list) < 2048:
        #             start0 = time.time()
        #             for path_seg_temp in path_seg_temp_list:
        #                 checked_path_seg_temp = check_mask_area(path_seg_temp, self.labels, self.label_values, self.min_mask_area)
        #                 if checked_path_seg_temp is not None:
        #                     list_temp_seg_i.append(checked_path_seg_temp)
        #                     list_temp_confirm_i.append(
        #                         checked_path_seg_temp.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz'))
        #             start1 = time.time()
        #             if (start1 - start0) > 2:
        #                 print('start1-start0 single: ', start1 - start0)
        #                 print(file_valid_seg_i)
        #         else:
        #             start0 = time.time()
        #             labels_list = [self.labels] * len(path_seg_temp_list)
        #             label_values_list = [self.label_values] * len(path_seg_temp_list)
        #             min_area_size_list = [self.min_mask_area] * len(path_seg_temp_list)
        #             with Pool(processes=16) as pool:
        #                 for checked_path_seg_temp in pool.starmap(
        #                         check_mask_area,
        #                         zip(path_seg_temp_list,
        #                             labels_list,
        #                             label_values_list,
        #                             min_area_size_list)):
        #                     if checked_path_seg_temp is not None:
        #                         list_temp_seg_i.append(checked_path_seg_temp)
        #                         list_temp_confirm_i.append(
        #                             checked_path_seg_temp.replace('aseg.auto_noCCseg.mgz', 'conformed.mgz'))
        #
        #                 start1 = time.time()
        #                 if (start1 - start0) > 2:
        #                     print('start1-start0 multi: ', start1 - start0)
        #                     print(file_valid_seg_i)
        #
        #         list_temp_seg_i.sort(key=natural_keys)
        #         list_temp_confirm_i.sort(key=natural_keys)
        #
        #         # check if number matches
        #         if len(list_temp_seg_i) != len(list_temp_confirm_i):
        #             print('unmatched input and target slices num')
        #             list_temp_seg_i = []
        #             list_temp_confirm_i = []
        #
        #         # remove choose fraction slices
        #         n = int(self.drop_fraction * len(list_temp_seg_i))
        #         if n > 0:
        #             list_temp_seg_i = list_temp_seg_i[n:-n]
        #             list_temp_confirm_i = list_temp_confirm_i[n:-n]
        #
        #         input_image_path_list = input_image_path_list + list_temp_confirm_i
        #         output_image_path_list = output_image_path_list + list_temp_seg_i
        # return list(zip(input_image_path_list, output_image_path_list))

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
            if input_path[-4:] == '.pkl':
                with open(input_path, "rb") as f:
                    image_input_array = pickle.load(f)
            else:
                slice_num = extract_last_number(input_path)
                input_path_npy = remove_trailing_number(input_path)
                image_input_mmap = np.load(input_path_npy[:-2], mmap_mode="r")
                if input_path_npy[-1] == 'x':
                    image_input_array = image_input_mmap[slice_num, :, :]
                elif input_path_npy[-1] == 'y':
                    image_input_array = image_input_mmap[:, slice_num, :]
                elif input_path_npy[-1] == 'z':
                    image_input_array = image_input_mmap[:, :, slice_num]
            if output_path[-4:] == '.pkl':
                with open(output_path, "rb") as f:
                    image_output_array = pickle.load(f)
            else:
                slice_num = extract_last_number(output_path)
                output_path_npy = remove_trailing_number(output_path)
                image_output_mmap = np.load(output_path_npy[:-2], mmap_mode="r")
                if output_path_npy[-1] == 'x':
                    image_output_array = image_output_mmap[slice_num, :, :]
                elif output_path_npy[-1] == 'y':
                    image_output_array = image_output_mmap[:, slice_num, :]
                elif output_path_npy[-1] == 'z':
                    image_output_array = image_output_mmap[:, :, slice_num]

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

        # apply rot and flip
        if self.if_rot_flip and self.is_train:
            if random.random() < 0.5:
                image_array, output_labels = self.custom_flip_func(image_array, output_labels)
                image_array, output_labels = self.custom_rot_func(image_array, output_labels)

        # if self.if_rot_flip and self.is_train:
        #     if random.random() < 0.5:
        #         image_array = np.flip(image_array, axis=1)
        #         output_labels = np.flip(output_labels, axis=1)
        #     if random.random() < 0.5:
        #         image_array = np.flip(image_array, axis=0)
        #         output_labels = np.flip(output_labels, axis=0)

            # avoid neg stride
            image_array = np.ascontiguousarray(image_array)
            output_labels = np.ascontiguousarray(output_labels)

        if self.if_tilted and self.is_train:
            if random.random() < 0.5:
                tilt_angle = random.uniform(self.tilt_angle[0], self.tilt_angle[1])
                output_labels = rotate(output_labels, angle=tilt_angle, reshape=False, order=0)
                image_array = rotate(image_array, angle=tilt_angle, reshape=False, order=1)

        if self.if_random_crop and self.is_train:
            if random.random() < 0.5:
                top, bottom, left, right = find_mask_bounds(output_labels)
                target_size = (self.resize, self.resize)
                current_image_shape = (
                    output_labels.shape[-2], output_labels.shape[-1])
                crop_coords = (top, bottom, left, right)
                random_crop_margin_v = (random.randint(0, max(top, current_image_shape[0] - bottom)),
                                        random.randint(0, max(top, current_image_shape[0] - bottom)))
                random_crop_margin_h = (random.randint(0, max(left, current_image_shape[1] - right)),
                                        random.randint(0, max(left, current_image_shape[1] - right)))
                image_array = random_crop_given_bounds(image_array, bounds=crop_coords,
                                                              target_size=target_size,
                                                              random_crop_margin_v=random_crop_margin_v,
                                                              random_crop_margin_h=random_crop_margin_h)
                output_labels = random_crop_given_bounds(output_labels,
                                                                          bounds=crop_coords,
                                                                          target_size=target_size,
                                                                          random_crop_margin_v=random_crop_margin_v,
                                                                          random_crop_margin_h=random_crop_margin_h)





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
        image_tensor_mask_binary = torch.from_numpy(output_labels).float()
        image_tensor_masked = image_tensor * (image_tensor_mask_binary > 0)

        if torch.isnan(image_tensor).any():
            print('find nan in image tensor')
            # self.patient_loader.patients[patient_index].print_all_mri_sessions()
            # print("file_index", file_index)
            # print("slice_index", slice_index)
            # print('shape')

        return image_tensor.unsqueeze(0).repeat(self.n_color, 1, 1), \
            image_tensor_masked, \
            image_tensor_mask_binary