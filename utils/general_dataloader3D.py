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
from utils.transform_util import rotate_around_axis_position, find_mask_bounds, random_crop_given_bounds, sequential_rotate_volume_pytorch,sequential_rotate_volume, upscale_rotate_downscale_fast, upscale_rotate_downscale_binary
import SimpleITK as sitk
from utils.general_dataloader import process_image_array, process_image_array3D, load_all_slice_in_folder, predict_img, longest_sequential_num,detect_plane_direction, load_generated_mask_slice, GeneralDataset
import nibabel as nib
from utils.general_utils import running_window

def reset_slices(image_volume, slice_info, default = 'left'):
    # image_volume [C, H, W]
    if default == 'left':
        if len(slice_info[4]) > 2:
            if slice_info[4][2] == "right":
                image_volume = np.rot90(image_volume, k=1, axes=(1, 2))
            elif slice_info[4][2] == "down":
                image_volume = np.flip(image_volume, axis=1)
                image_volume = np.flip(image_volume, axis=2)
            elif slice_info[4][2] == "left":
                image_volume = np.rot90(image_volume, k=1, axes=(2, 1))
                image_volume = np.flip(image_volume, axis=2)
        else:
            image_volume = np.rot90(image_volume, k=1, axes=(2, 1))
            image_volume = np.flip(image_volume, axis=2)
    elif default == 'right':
        if len(slice_info[4]) > 2:
            if slice_info[4][2] == "left":
                image_volume = np.rot90(image_volume, k=1, axes=(2, 1))
            elif slice_info[4][2] == "down":
                image_volume = np.flip(image_volume, axis=1)
                image_volume = np.flip(image_volume, axis=2)
            elif slice_info[4][2] == "right":
                image_volume = np.rot90(image_volume, k=1, axes=(1, 2))
                image_volume = np.flip(image_volume, axis=2)
        else:
            image_volume = np.rot90(image_volume, k=1, axes=(1, 2))
            image_volume = np.flip(image_volume, axis=2)
    return image_volume

class GeneralDatasetHippoSegClassManual(GeneralDataset):
    def __init__(
            self,
            resize,
            zero_filled_chance,
            mri_sequence,
            labels,
            label_values,
            random_seed,
            val_split,
            is_train,
            if_latest,
            max_train_len=None,
            max_val_len=None,
            if_rot_flip=False,
            patients=[],
            mask_area_threshold=64 * 64,
            image_cache_path='',  # "F:/nifti_seg/T1_T1flair/"
            detect_plane=['AX', 'COR', 'SAG'],
            target_plane='COR',
            neg_pos_multiple_size=[1, 1],
            frac_val_level=[0.5, 0.25, 0.25],
            target_channel = 40,
            random_extra_margin_slice = 5,
            train_level = ['level1', 'level2', 'level3'],
            random_tilt_angle = [15, 15, 15],
            random_argment_chance = 0.5,
            random_full_size_chance = 0.5,
            if_tilt = False,
            if_crop = False,
            slice_range = [15, 30],
            max_mask_size_threshold = 0 * 0,
            if_rot = False,
            if_flip = False,
            use_mask_index = True,
            neg_pos_levels = [['level1', 'level11'],['level2', 'level22', 'level3', 'level32']],
            **kwargs
    ):
        self.resize = resize
        self.zero_filled_chance = zero_filled_chance
        self.mri_sequence = mri_sequence
        self.labels = labels
        self.label_values = label_values
        self.random_seed = random_seed
        self.val_split = val_split
        self.is_train = is_train
        self.if_latest = if_latest
        self.max_train_len = max_train_len
        self.max_val_len = max_val_len
        self.if_rot_flip = if_rot_flip
        self.patients = patients
        self.mask_area_threshold = mask_area_threshold
        self.image_cache_path = image_cache_path
        self.detect_plane = detect_plane
        self.target_plane = target_plane
        self.frac_val_level = frac_val_level
        self.target_channel = target_channel
        self.random_extra_margin_slice = random_extra_margin_slice
        self.train_level = train_level
        self.neg_pos_levels = neg_pos_levels
        self.random_tilt_angle = random_tilt_angle
        self.if_tilt = if_tilt
        self.if_crop = if_crop
        self.if_rot = if_rot
        self.if_flip = if_flip
        self.use_mask_index = use_mask_index

        self.slice_range = slice_range
        self.random_argment_chance = random_argment_chance
        self.random_full_size_chance = random_full_size_chance
        self.max_mask_size_threshold = max_mask_size_threshold
        self.filtered_patients = []

        start0 = time.time()
        self.data_into_list_train = []
        self.data_into_list_val = []
        self.neg_pos_multiple_size = neg_pos_multiple_size
        self.trainPaths, self.valPaths = self.get_index_list(if_latest=if_latest)  # [(input path, target path),()]
        start1 = time.time()
        # print("start1 - start0: ", start1 - start0)

        np.random.seed(self.random_seed)
        np.random.shuffle(self.trainPaths)
        np.random.shuffle(self.valPaths)
        self.custom_flip_func = CustomRandomFlip()
        self.custom_rot_func = CustomRandomRot()

    def filter_patients(self, patient: PatientCase, if_latest: bool = False, max_mask_size_threshold = 0 * 0):
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

                                    file_path_string = file_path_string + '/conformed.mgz'
                                    if os.path.exists(file_path_string):
                                        if file_path_string not in file_out_path_list:
                                            file_out_path_list.append(
                                                file_path_string)  # D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz
                                            file_out_info_list.append(
                                                slice_info)  # [None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]

        # filter by max_mask_size_threshold, meanings the max mask size in slices should no less than a given number
        file_out_path_list_mask_filter = []
        file_out_info_list_mask_filter = []
        for i in range(len(file_out_path_list)):
            # load the volume mask
            file_path_mask = file_out_path_list[i].replace('conformed.mgz', 'aseg.auto_noCCseg.mgz')
            file_slice_info = file_out_info_list[i]
            target_plane_idx = file_slice_info[4][1]
            if os.path.exists(file_path_mask):
                mri_data_mask = nib.load(file_path_mask)
                data_mask = mri_data_mask.get_fdata()
                if target_plane_idx == 1:
                    data_mask = np.transpose(data_mask, axes=(1, 0, 2))
                elif target_plane_idx == 2:
                    data_mask = np.transpose(data_mask, axes=(2, 0, 1))

                # data_mask = reset_slices(data_mask, file_slice_info, default = 'left')

                # get hippo area mask only
                image_inco_array_volume_mask_hippo = data_mask.copy()
                output_labels = np.zeros_like(image_inco_array_volume_mask_hippo)  # [C, H, W]
                for j in range(len(self.label_values)):
                    label_value = self.label_values[j]
                    seg_value = np.isin(image_inco_array_volume_mask_hippo, label_value)
                    output_labels += (j + 1) * seg_value

                output_labels[output_labels > 0.1] = 1
                output_labels[output_labels <= 0.1] = 0
                image_inco_array_volume_mask_hippo = output_labels
                mask_counts = np.sum(image_inco_array_volume_mask_hippo == 1, axis=(1, 2))
                if max(mask_counts) >= max_mask_size_threshold:
                    file_out_path_list_mask_filter.append(file_out_path_list[i])
                    file_out_info_list_mask_filter.append(file_out_info_list[i])




        return file_out_path_list_mask_filter, file_out_info_list_mask_filter  # ['D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz'], [[None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]]

    def get_index_list(self, if_latest=False):
        # shuffle patients
        np.random.seed(self.random_seed)
        random.shuffle(self.patients)
        # random.shuffle(self.patient_pos)
        # random.shuffle(self.patient_neg)

        # filter patients
        patient_and_file_path = []
        for i in tqdm(range(len(self.patients))):
            patient = self.patients[i]
            file_out_path_list, file_out_info_list = self.filter_patients(patient, if_latest, self.max_mask_size_threshold)
            if len(file_out_path_list) > 0:
                patient_and_file_path.append(
                    [patient, file_out_path_list, file_out_info_list])  # [patient, [list],[ info]]
                self.filtered_patients.append(patient.patient_ID)
        print('total num patient: ', len(patient_and_file_path))

        # get patients for validation dataset
        patient_and_file_path_train_pos = []
        patient_and_file_path_train_neg = []
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
                elif (len(patient_and_file_path_train_pos) + len(patient_and_file_path_train_neg)) < num_train:
                    if level_i[0] in self.neg_pos_levels[0]:
                        patient_and_file_path_train_neg.append(patient_and_file_path_i)
                    elif level_i[0] in self.neg_pos_levels[1]:
                        patient_and_file_path_train_pos.append(patient_and_file_path_i)
            elif len(level_i) > 1 and (len(patient_and_file_path_train_pos) + len(patient_and_file_path_train_neg)) < num_train:
                if level_i[-1] in self.neg_pos_levels[0]:
                    patient_and_file_path_train_neg.append(patient_and_file_path_i)
                elif level_i[-1] in self.neg_pos_levels[1]:
                    patient_and_file_path_train_pos.append(patient_and_file_path_i)

        patient_and_file_path_val = (patient_and_file_path_val_level1 + patient_and_file_path_val_level2
                                     + patient_and_file_path_val_level3)

        patient_and_file_path_train = (patient_and_file_path_train_neg * self.neg_pos_multiple_size[0]
         + patient_and_file_path_train_pos * self.neg_pos_multiple_size[
             1])

        print('level1 in val: ' + str(len(patient_and_file_path_val_level1)))
        print('level2 in val: ' + str(len(patient_and_file_path_val_level2)))
        print('level3 in val: ' + str(len(patient_and_file_path_val_level3)))
        print('neg level in train: ' + str(len(patient_and_file_path_train_neg)))
        print('pos level in train: ' + str(len(patient_and_file_path_train_pos)))



        # return patient_and_file_path_train, patient_and_file_path_val # [patient, [list],[ info]]
        return (self.generate_input_output_by_file(patient_and_file_path_train),
                self.generate_input_output_by_file(patient_and_file_path_val))

    def generate_input_output_by_file(self, data_into_list):
        path_label_list = []  # [[patient_ID, 'D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz', info],...]
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
            input_path = self.trainPaths[idx][1]  # 'D:\nifti_seg_cache\T1_T1flair\NACC000034\l0_1_l0rr\subject/conformed.mgz'
            slice_info = self.trainPaths[idx][2]  # [None, 0.035, 0.002, None, ['COR', 2], [176, 136, 'level4']]
        else:
            patient_ID = self.valPaths[idx][0]
            input_path = self.valPaths[idx][1]
            slice_info = self.valPaths[idx][2]

        level = slice_info[5][2]
        label = 0
        if level in self.neg_pos_levels[0]:
            label = 0
        elif level in self.neg_pos_levels[1]:
            label = 1

        # init image and labels
        image_tensor = torch.zeros(self.resize, self.resize)
        image_label = torch.zeros(len(self.labels))  # size of label = number of sequence + 1
        image_tensor_mask_strip = torch.zeros(self.resize, self.resize)
        # image_label[-1] = 1.0  # initialize last label to 1 as unclassified, for now pure 0 images are in this class

        if random.random() < self.zero_filled_chance and self.is_train:
            return (image_tensor.unsqueeze(0).repeat(self.target_channel, 1, 1), image_label, patient_ID, 0,
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.target_channel, 1, 1),
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.target_channel, 1, 1), 0, 0)

        try:
            mri_data = nib.load(input_path)
            input_path_mask = input_path.replace('conformed.mgz', 'aseg.auto_noCCseg.mgz')
            mri_data_mask = nib.load(input_path_mask)

            # Get the data as a numpy array
            start0 = time.time()


            data = mri_data.get_fdata()
            data_mask = mri_data_mask.get_fdata()
            target_plane_idx = slice_info[4][1]
            if target_plane_idx == 1:
                data = np.transpose(data, axes=(1, 0, 2))
                data_mask = np.transpose(data_mask, axes=(1, 0, 2))
            elif target_plane_idx == 2:
                data = np.transpose(data, axes=(2, 0, 1))
                data_mask = np.transpose(data_mask, axes=(2, 0, 1))

            start1 = time.time()
            # print("start1 - start0 Get the data as a numpy array: ", start1 - start0)
            # reset to front view
            image_inco_array_volume = reset_slices(data, slice_info, default='left')
            image_inco_array_volume = image_inco_array_volume / 255.0 # conformed.mgz is 0-255
            image_inco_array_volume_mask = reset_slices(data_mask, slice_info, default='left')
            image_inco_array_volume_mask_total = image_inco_array_volume_mask.copy()
            image_inco_array_volume_mask_total[image_inco_array_volume_mask_total > 0.0] = 1
            start2 = time.time()
            # print("reset to front view: ", start2 - start1)

            # get hippo area mask only
            image_inco_array_volume_mask_hippo = image_inco_array_volume_mask.copy()
            output_labels = np.zeros_like(image_inco_array_volume_mask_hippo)  # [C, H, W]
            for i in range(len(self.label_values)):
                label_value = self.label_values[i]
                seg_value = np.isin(image_inco_array_volume_mask_hippo, label_value)
                output_labels += (i + 1) * seg_value

            output_labels[output_labels > 0.1] = 1
            output_labels[output_labels <= 0.1] = 0
            image_inco_array_volume_mask_hippo = output_labels
            start3 = time.time()
            # print("get hippo area mask only: ", start3 - start2)

            # apply flip back-front axis 0 and right-left axis 2
            if_flip_back_front = False
            if self.if_flip and self.is_train:
                if random.random() < 0.5:
                    image_inco_array_volume = np.flip(image_inco_array_volume, axis=2)
                    image_inco_array_volume_mask_total = np.flip(image_inco_array_volume_mask_total, axis=2)
                    image_inco_array_volume_mask_hippo = np.flip(image_inco_array_volume_mask_hippo, axis=2)
                if random.random() < 0.5:
                    if_flip_back_front = True
                    image_inco_array_volume = np.flip(image_inco_array_volume, axis=0)
                    image_inco_array_volume_mask_total = np.flip(image_inco_array_volume_mask_total, axis=0)
                    image_inco_array_volume_mask_hippo = np.flip(image_inco_array_volume_mask_hippo, axis=0)

            # find slice area
            if self.use_mask_index:
                mask_counts = np.sum(image_inco_array_volume_mask_hippo == 1, axis=(1, 2))

                random_integer = random.randint(self.slice_range[0], self.slice_range[1])
                best_range, highest_sum = running_window(mask_counts, random_integer)

                start = best_range[0]
                end = best_range[1]
            else:
                start_ori = slice_info[5][0]
                end_ori = slice_info[5][1]
                if if_flip_back_front:
                    integer_list = np.arange(len(image_inco_array_volume))
                    # Flip the list
                    flipped_list = np.flip(integer_list)
                    end = np.where(flipped_list == start_ori)[0][0]
                    start = np.where(flipped_list == end_ori)[0][0]
                else:
                    start = start_ori
                    end = end_ori


        except:
            print("pkl.pixel_array corrupted")
            print(input_path)
            # print(slice_index)
            return (image_tensor.unsqueeze(0).repeat(self.target_channel, 1, 1), image_label, patient_ID, 0,
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.target_channel, 1, 1),
                    image_tensor_mask_strip.unsqueeze(0).repeat(self.target_channel, 1, 1)), 0, 0

        # print_var_detail(image_inco_array_volume, 'image_inco_array_volume_fliped')
        # print_var_detail(image_inco_array_volume_mask_total, 'image_inco_array_volume_mask_total')
        # print_var_detail(image_inco_array_volume_mask_hippo, 'image_inco_array_volume_mask_hippo')

        use_argment = 0
        if random.random() < self.random_argment_chance and self.is_train:
            use_argment = 1

            # avoid neg stride
            image_inco_array_volume_fliped = np.ascontiguousarray(image_inco_array_volume)
            image_inco_array_volume_mask_total_fliped = np.ascontiguousarray(image_inco_array_volume_mask_total)
            image_inco_array_volume_mask_hippo_fliped = np.ascontiguousarray(image_inco_array_volume_mask_hippo)

            start4 = time.time()
            # print("random flip start4 - start3: ", start4 - start3)

            # tilt
            if self.if_tilt:
                # start = slice_info[5][0]
                # end = slice_info[5][1]
                center_d = (start + end) // 2
                center_HW = center_of_mass(image_inco_array_volume_mask_total[center_d])
                position = [center_d, int(center_HW[0]), int(center_HW[1])]

                angle_x = random.uniform(-self.random_tilt_angle[0], self.random_tilt_angle[0])
                angle_y = random.uniform(-self.random_tilt_angle[1], self.random_tilt_angle[1])
                angle_z = random.uniform(-self.random_tilt_angle[2], self.random_tilt_angle[2])
                angles = [angle_x, angle_y, angle_z]

                image_inco_array_volume_tilt = sequential_rotate_volume_pytorch(image_inco_array_volume_fliped, angles, position)
                image_inco_array_volume_mask_total_tilt = sequential_rotate_volume_pytorch(image_inco_array_volume_mask_total_fliped, angles, position, if_binary=True)
                image_inco_array_volume_mask_hippo_tilt = sequential_rotate_volume_pytorch(image_inco_array_volume_mask_hippo_fliped, angles, position, if_binary=True)

                start5 = time.time()
                # print("tilt start5 - start4: ", start5 - start4)

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
                start6 = time.time()
                # print("random crop image start6 - start5: ", start6 - start5)

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


        # crop slice using target_channel and start -> end index and random margin slice
        # random chance to either use selected range or fill all target channel

        start_idx = start
        end_idx = end
        slice_interval = end_idx - start_idx

        if_full = 0
        if random.random() < self.random_full_size_chance:
            slice_interval = self.target_channel
            if_full = 1

        mid_slice = start_idx + slice_interval // 2
        if slice_interval >= self.target_channel:
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
        image_array_volume[self.target_channel // 2 - slice_interval // 2 : self.target_channel // 2 - slice_interval // 2
        + slice_interval] = image_inco_array_volume_tilt_crop[start_idx:end_idx]
        image_tensor = torch.from_numpy(image_array_volume.copy()).float()  # avoid neg stride

        image_array_volume_mask_total = np.zeros((self.target_channel, self.resize, self.resize))
        image_array_volume_mask_total[self.target_channel // 2 - slice_interval // 2 : self.target_channel // 2 - slice_interval // 2
        + slice_interval] = image_inco_array_volume_mask_total_tilt_crop[start_idx:end_idx]
        image_tensor_mask_total = torch.from_numpy(image_array_volume_mask_total.copy()).float()  # avoid neg stride

        image_array_volume_mask_hippo = np.zeros((self.target_channel, self.resize, self.resize))
        image_array_volume_mask_hippo[self.target_channel // 2 - slice_interval // 2 : self.target_channel // 2 - slice_interval // 2
        + slice_interval] = image_inco_array_volume_mask_hippo_tilt_crop[start_idx:end_idx]
        image_tensor_mask_hippo = torch.from_numpy(image_array_volume_mask_hippo.copy()).float()  # avoid neg stride

        # 1-hot classify
        # if len(self.labels) > 1:
        #     label_idx = self.label_values.index(label)
        #     image_label[label_idx] = 1.0
        # else:
        image_label[0] = label

        # print(slice_info)

        return image_tensor, image_label, patient_ID, if_full, image_tensor_mask_total, image_tensor_mask_hippo, use_argment, 1
