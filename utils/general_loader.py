import json
import os
import re
import time
import warnings
import numpy as np
from multiprocessing import Pool, cpu_count
import math
import matplotlib.pylab as plt
import pandas as pd
import pydicom as dicom
import nibabel as nib
import json
import json5
from scipy.ndimage import zoom
from pyviz_comms import extension
from tqdm import tqdm
from copy import deepcopy
import time
import pickle
from skimage.transform import resize
from help_func import print_var_detail
from .general_utils import (fixPath, load_sorted_directory, sort_list_natural_keys, get_time_interval,
                            check_if_dir_list, check_if_dir, clear_string_char, get_file_extension, safe_json_load)
from utils.mri_utils import resize_3d_torch

def get_slice_data(slice_name, slice_index, slice_path, tag_codes=None, mode='dicom', if_cache_image=False):
    """
    get mri slice data in tuples
    :param slice_name: string, slice name
    :param slice_index: int, slice index
    :param slice_path: string, slice path
    :param tag_codes: list, tag codes to access info tags, 16 bit for dicom, strings for nifti
    :param mode: string, mode to get data from 'dicom', 'nifti'
    :param if_cache_image: boolean, whether to return cache image
    :return: tuples of (
      slice_names: [string,...]
    , slice_indexes: [int,...] or [[int, int,... ],...]
    , slice_infos: [[string, string,...],...]
    , cache_images: [np.array,...]
    )
    """
    slice_path = fixPath(slice_path)
    if os.path.exists(slice_path):
        try:
            if mode == 'dicom':  # dicom
                ds = dicom.dcmread(slice_path)
                if ds is not None:
                    try:
                        slice_info = []
                        if tag_codes is not None:
                            for tag_code in tag_codes:
                                if ds.__contains__(tag_code):
                                    slice_info.append(ds[tag_code].value)
                                else:
                                    slice_info.append(None)
                        image_array = ds.pixel_array
                        shape = image_array.shape
                        cache_image = None
                        if if_cache_image:
                            cache_image = image_array

                        if len(shape) == 2:
                            return [slice_name], [slice_index], [slice_info], [cache_image]  # single gray scale
                        elif len(shape) == 3:
                            if shape[-1] == 3:
                                return [slice_name], [slice_index], [slice_info], [cache_image]  # single rgb
                            else:
                                slice_names = []
                                slice_indexes = []
                                slice_infos = []
                                cache_images = []
                                for i in range(shape[0]):  # multiple gray scale
                                    slice_names.append([slice_name, ''])
                                    slice_indexes.append([slice_index, i])
                                    slice_infos.append(slice_info)
                                    if cache_image is not None:
                                        cache_images.append(cache_image[i])
                                    else:
                                        cache_images.append(None)
                                return slice_names, slice_indexes, slice_infos, cache_images
                        elif len(shape) == 4:
                            slice_names = []
                            slice_indexes = []
                            slice_infos = []
                            cache_images = []
                            for i in range(shape[0]):  # multiple rbg
                                slice_names.append([slice_name, ''])
                                slice_indexes.append([slice_index, i])
                                slice_infos.append(slice_info)
                                if cache_image is not None:
                                    cache_images.append(cache_image[i])
                                else:
                                    cache_images.append(None)
                            return slice_names, slice_indexes, slice_infos, cache_images
                    except Exception:
                        return [], [], [], []
            elif mode == 'nifti':
                ds = nib.load(slice_path)
                if ds is not None:
                    try:
                        nifti_extension = get_file_extension(slice_path, prompt='.nii')
                        json_path = slice_path[:-len(nifti_extension)] + '.json'
                        # f = open(json_path)
                        # data_infos = json.load(f)
                        data_infos = safe_json_load(json_path)
                        slice_info = []
                        if tag_codes is not None:
                            for tag_code in tag_codes:
                                if tag_code in data_infos.keys():
                                    slice_info.append(data_infos[tag_code])
                                elif tag_code in ds.header:
                                    slice_info.append(ds.header[tag_code])
                                else:
                                    slice_info.append(None)

                        if ds._dataobj._dtype == [('R', 'u1'), ('G', 'u1'), ('B', 'u1')]:
                            image_array = np.asanyarray(ds.dataobj)
                            image_array = image_array.copy().view(dtype=np.uint8).reshape(
                                image_array.shape + (3,))  # (160, 160, 23, 3) RGB numpy.memmap

                        else:
                            image_array = ds.get_fdata()  # a <class 'numpy.memmap'> with shape (H, W, slice_index) or (H, W, slice_index, series_index)
                            # max:  680.0 min:  0.0
                        # image_array = np.rot90(image_array)
                        shape = image_array.shape
                        cache_image = None
                        if if_cache_image:
                            cache_image = image_array

                        slice_names = []
                        slice_indexes = []
                        slice_infos = []
                        cache_images = []
                        if len(shape) == 3:  # gray scale, multiple slices (H, W, slice_index_i)
                            for i in range(shape[2]):  # multiple gray scale
                                slice_names.append([slice_name, ''])
                                slice_indexes.append([slice_index, i])  # index as [file_index, slice_index_i]
                                slice_infos.append(slice_info)
                                if cache_image is not None:
                                    cache_images.append(cache_image[:, :, i])  # (H, W)
                                else:
                                    cache_images.append(None)
                            return slice_names, slice_indexes, slice_infos, cache_images
                        elif len(shape) == 4:
                            if shape[-1] == 3:  # RGB scale, multiple slices (H, W, slice_index_i, 3)
                                for i in range(shape[2]):  # multiple gray scale
                                    slice_names.append([slice_name, ''])
                                    slice_indexes.append([slice_index, i])
                                    slice_infos.append(slice_info)
                                    if cache_image is not None:
                                        cache_images.append(cache_image[:, :, i, :])  # (H, W, 3)
                                    else:
                                        cache_images.append(None)
                                return slice_names, slice_indexes, slice_infos, cache_images
                            else:  # grey scale, multiple slices and series(timesteps) (H, W, slice_index_i, series_index_i)
                                for i in range(shape[3]):
                                    for j in range(shape[2]):
                                        slice_names.append([slice_name, ''])
                                        slice_indexes.append(
                                            [slice_index, i, j])  # index as [file_index, series_index_i, slice_index_i]
                                        slice_infos.append(slice_info)
                                        if cache_image is not None:
                                            cache_images.append(cache_image[:, :, j, i])  # (H, W)
                                        else:
                                            cache_images.append(None)
                                return slice_names, slice_indexes, slice_infos, cache_images
                    except Exception:
                        return [], [], [], []
            else:  # nifti
                return [], [], [], []
        except Exception:
            return [], [], [], []
    return [], [], [], []


class PatientCase:
    def __init__(self, patient_ID, patient_ID_label, mri_directory_label, mriDF, analysisDF, dicom_root,
                 mri_directory_postfix='', postfix_label='',
                 detect_sequence=None, mode='dicom', info_tags=None, if_cache_image=False, cache_path='image_cache/',
                 series_description_folder=None):
        """
        class that stores one patient case with all information loaded from csv files
        :param patient_ID: string, patient ID
        :param patient_ID_label: string, label stores patient ID
        :param mri_directory_label: string, label stores mri directory for each patient case
        :param mriDF: dataframe, dataframe containing all MRI information
        :param analysisDF: dataframe, dataframe containing all pathology test information
        :param dicom_root: string, root directory of all patients' DICOM files
        :param mri_directory_postfix: string, redundant label come along with end of mri_directory
        :param postfix_label: string, additional label should be add along with end of mri_directory
        :param detect_sequence: string, chosen sequence such as 'T1', 'T2', 'DTI', 'DWI', etc. use to detect when create patient case, default None
        :param mode: string, mode depends on data type, either 'dicom' or 'nifti'
        :param info_tags: list of strings, tags to be detected and stored in slice_info_list
        :param if_cache_image: boolean, whether to cache the image or not
        :param cache_path: string, path to cache the image
        """
        # initialize essential elements
        self.patient_ID = patient_ID
        self.patient_ID_label = patient_ID_label
        self.mri_directory_label = mri_directory_label
        self.dicom_dict = {
            "SeriesDescription": [0x0008, 0x103E],  #
            "SequenceName": [0x0018, 0x0024],
            "SpacingBetweenSlice": [0x0018, 0x0088],
            "SliceThickness": [0x0018, 0x0050],
            "MRAcquisitionType": [0x0018, 0x0023],
            "ScanningSequence": [0x0018, 0x0020],
            "SequenceVariant": [0x0018, 0x0021],
            "RepetitionTime": [0x0018, 0x0080],
            "EchoTime": [0x0018, 0x0081],
            "InversionTime": [0x0018, 0x0082],
        }
        self.combined_tags_dict = {}
        self.mode = mode
        self.if_cache_image = if_cache_image
        self.cache_path = None
        self.series_description_folder = series_description_folder
        if self.if_cache_image:
            self.cache_path = cache_path + self.patient_ID + '/'
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)

        self.info_tags = info_tags
        self.info_tag_codes = []
        for info_tag in self.info_tags:
            if info_tag in self.dicom_dict:
                self.info_tag_codes.append(self.dicom_dict[info_tag])
            else:
                self.info_tag_codes.append([0xFFFF, 0xFFFF]) # misc defined dummy dicom tags code

        self.mriDF = mriDF
        self.analysisDF = analysisDF
        self.dicom_root = dicom_root
        self.mri_directory_postfix = mri_directory_postfix
        self.postfix_label = postfix_label
        self.latest_mri_session, self.latest_mri_date = self.get_latest_mri_session()  # df, [year, month, day]
        self.dir_name_list, self.dir_index_list, self.dir_name_list_latest, self.dir_index_list_latest = [], [], [], []
        self.slice_name_list, self.slice_index_list, self.slice_info_list = [], [], []
        self.detect_sequence = detect_sequence
        self.PD_echo_time = []

        # update essential elements
        # store the directory hierarchy with list of names and index [[level1, level2, ...],[level1, level2, ...]]
        self.update_dir_name_list()

        # store the slice with list of names and index corresponding to dir_name_list
        # [[slice1, slice2, ...],[slice1, slice2, ...]], slice# is either int or list of ints, depend on files in dir
        # are volumes(combined slices, len(shape)>2 ) or just separate slices, if former is list, latter is int
        self.update_slice_lists()

        # check if the patient has any MRI file recorded
        self.ifHasMRI = self.check_MRI_exist()
        self.sequence_dir_index_lists_dict = {'unknown': deepcopy(self.dir_index_list)}

        # store slice index lists into sequence dictionary, the slices are stored with full index, which means it has
        # hierarchy of [dir index, slice_index]. for initialization, store all slice into unknown tag
        slice_index_list_combine = []
        for dir_index in self.dir_index_list:
            slice_index_list_combine = slice_index_list_combine + self.get_slice_index_list_in_dir(dir_index)
        self.sequence_slice_index_lists_dict = {'unknown': deepcopy(slice_index_list_combine)}

    def get_patient_ID(self):
        return self.patient_ID

    def get_DF_by_labels_and(self, labels, label_values, DF_name=None):
        """
        return dataframe by given labels and corresponding values. using and condition between each label
        :param labels: list, label names
        :param label_values: list, label values
        :param DF_name: string, dataframe name to search, 'mri', 'analysis' or None, default None
        :return: mri dataframe, analysis dataframe
        """
        result_mriDF = self.mriDF
        result_analysisDF = self.analysisDF
        if DF_name == 'mri':
            result_analysisDF = pd.DataFrame(columns=result_analysisDF.columns)
            for i in range(len(labels)):
                label = labels[i]
                result_mriDF = result_mriDF.loc[result_mriDF[label].isin(label_values[i])]
        elif DF_name == 'analysis':
            result_mriDF = pd.DataFrame(columns=result_mriDF.columns)
            for i in range(len(labels)):
                label = labels[i]
                result_analysisDF = result_analysisDF.loc[result_analysisDF[label].isin(label_values[i])]
        else:  # if dataframe name is not specified, search both mri and analysis dataframe
            has_mri_labels = False
            has_analysis_labels = False
            for i in range(len(labels)):
                # check which dataframe contains given labels
                label = labels[i]
                # save corresponding labeled columns with correct values into returned dataframes
                if label in result_mriDF.columns:
                    has_mri_labels = True
                    result_mriDF = result_mriDF.loc[result_mriDF[label].isin(label_values[i])]
                if label in result_analysisDF.columns:
                    has_analysis_labels = True
                    result_analysisDF = result_analysisDF.loc[result_analysisDF[label].isin(label_values[i])]
            # if no labels is found in mri or analysis dataframe, return empty dataframe with same columns
            if not has_mri_labels:
                result_mriDF = pd.DataFrame(columns=result_mriDF.columns)
            if not has_analysis_labels:
                result_analysisDF = pd.DataFrame(columns=result_analysisDF.columns)
        return result_mriDF, result_analysisDF

    def get_DF_by_labels_or(self, labels, label_values, DF_name=None):
        """
        return dataframe by given labels and corresponding values. using or condition between each label
        :param labels: list, label names
        :param label_values: list, label values
        :param DF_name: string, dataframe name to search, 'mri', 'analysis' or None, default None
        :return: mri dataframe, analysis dataframe
        """
        result_mriDF = pd.DataFrame(columns=self.mriDF.columns)
        result_analysisDF = pd.DataFrame(columns=self.analysisDF.columns)
        # same as get_DF_by_labels_and but concatenate all columns with given labels & values then drop duplicates
        if DF_name == 'mri':
            result_analysisDF = pd.DataFrame(columns=result_analysisDF.columns)
            for i in range(len(labels)):
                label = labels[i]
                result_mriDF = pd.concat([result_mriDF, self.mriDF.loc[self.mriDF[label].isin(label_values[i])]],
                                         axis=0)
        elif DF_name == 'analysis':
            result_mriDF = pd.DataFrame(columns=result_mriDF.columns)
            for i in range(len(labels)):
                label = labels[i]
                result_analysisDF = pd.concat(
                    [result_analysisDF, self.analysisDF.loc[self.analysisDF[label].isin(label_values[i])]], axis=0)
        else:
            has_mri_labels = False
            has_analysis_labels = False
            for i in range(len(labels)):
                label = labels[i]
                if label in self.mriDF.columns:
                    has_mri_labels = True
                    result_mriDF = pd.concat([result_mriDF, self.mriDF.loc[self.mriDF[label].isin(label_values[i])]],
                                             axis=0)
                if label in self.analysisDF.columns:
                    has_analysis_labels = True
                    result_analysisDF = pd.concat(
                        [result_analysisDF, self.analysisDF.loc[self.analysisDF[label].isin(label_values[i])]], axis=0)
            if not has_mri_labels:
                result_mriDF = pd.DataFrame(columns=result_mriDF.columns)
            if not has_analysis_labels:
                result_analysisDF = pd.DataFrame(columns=result_analysisDF.columns)
            result_mriDF = result_mriDF.drop_duplicates()
            result_analysisDF = result_analysisDF.drop_duplicates()
        return result_mriDF, result_analysisDF

    def if_exist_by_labels_and(self, labels, label_values, DF_name=None):
        """
        check if given labels using 'and' condition return any non-empty dataframe
        :param labels: list, label names
        :param label_values: list, label values
        :param DF_name: string, dataframe name to search, 'mri', 'analysis' or None, default None
        :return: bool, True if any returned dataframe is not empty, False otherwise
        """
        result_mriDF, result_analysisDF = self.get_DF_by_labels_and(labels=labels, label_values=label_values,
                                                                    DF_name=DF_name)
        if not result_mriDF.empty or not result_analysisDF.empty:
            return True
        else:
            return False

    def if_exist_by_labels_or(self, labels, label_values, DF_name=None):
        """
        check if given labels using 'or' condition return any non-empty dataframe
        :param labels: list, label names
        :param label_values: list, label values
        :param DF_name: string, dataframe name to search, 'mri', 'analysis' or None, default None
        :return: bool, True if any returned dataframe is not empty, False otherwise
        """
        result_mriDF, result_analysisDF = self.get_DF_by_labels_or(labels=labels, label_values=label_values,
                                                                   DF_name=DF_name)
        if not result_mriDF.empty or not result_analysisDF.empty:
            return True
        else:
            return False

    def get_label_values_by_DF(self, labels, DF_name=None):
        """
        find dataframes with labels with all possible values
        :param labels: list, label names
        :param DF_name: string, dataframe name to search, 'mri', 'analysis' or None, default None
        :return: mri dataframe, analysis dataframe
        """
        labels.insert(0, self.patient_ID_label)
        result_mriDF = self.mriDF
        result_analysisDF = self.analysisDF
        if DF_name == 'mri':
            result_mriDF = result_mriDF.loc[:, labels]
        elif DF_name == 'analysis':
            result_analysisDF = result_analysisDF.loc[:, labels]
        else:
            labels_mri = []
            labels_analysis = []
            for i in range(len(labels)):
                label = labels[i]
                if label in result_mriDF.columns:
                    labels_mri.append(label)
                if label in result_analysisDF.columns:
                    labels_analysis.append(label)
            result_mriDF = result_mriDF.loc[:, labels_mri]
            result_analysisDF = result_analysisDF.loc[:, labels_analysis]
        return result_mriDF, result_analysisDF

    def process_mri_path(self, path_input):
        """
        :param path_input: string, path to mri image
        :return: string, path to mri image after remove mri_directory_postfix and add postfix_label
        """
        return path_input[:len(path_input) - len(self.mri_directory_postfix)] + self.postfix_label

    def check_MRI_exist(self):
        """
        Check if the patient has MRI scan recorded
        :return: bool, True if patient has MRI scan, False otherwise
        """
        mri_filepaths = self.mriDF.loc[:, self.mri_directory_label].drop_duplicates()
        for i in range(len(mri_filepaths)):
            mri_filepath = self.process_mri_path(mri_filepaths.iloc[i])
            if os.path.exists(fixPath(self.dicom_root + "/" + mri_filepath)):
                return True
        return False

    def print_directory(self, dir_path, indexes=[]):
        """
        Print all directories in root directory, note it's recursive function
        :param dir_path: root directory path
        :param indexes: initial index for root directory
        """
        space = ' ' * 4 * (len(indexes) - 1)
        print(space + f'{[indexes[-1]]} {dir_path[-1]}')
        dir_elements = load_sorted_directory(self.dicom_root + "/" + '/'.join(dir_path))

        dirs_list = []

        for f in dir_elements:
            if os.path.isdir(fixPath(os.path.join(self.dicom_root + "/" + '/'.join(dir_path), f))):
                dirs_list.append(dir_path + [f])

        dindex = 0
        for dir in dirs_list:
            self.print_directory(dir, indexes + [dindex])
            dindex += 1

    def list_directory(self, dir_path, indexes=[]):
        """
        List all directories that contains dicom file, NOTE it's recursive function
        and will update the self.dir_name_list and self.dir_index_list
        :param dir_path: root directory path
        :param indexes: initial index for root directory
        """
        dir_elements = load_sorted_directory(self.dicom_root + "/" + '/'.join(dir_path))

        files_list = []
        dirs_list = []
        if dir_elements is None:
            print(self.dicom_root + "/" + '/'.join(dir_path))
        for f in dir_elements:
            if os.path.isdir(fixPath(os.path.join(self.dicom_root + "/" + '/'.join(dir_path), f))):
                dirs_list.append(dir_path + [f])
            else:
                files_list.append(f)
        if len(dirs_list) < 1:
            self.dir_name_list.append(dir_path)
            self.dir_index_list.append(indexes)

        dindex = 0
        for dir in dirs_list:
            self.list_directory(dir, indexes + [dindex])
            dindex += 1

    def print_all_mri_sessions(self):
        """
        Print all mri sessions hierarchy
        """
        if self.ifHasMRI:
            mri_filepaths = self.mriDF.loc[:, self.mri_directory_label].drop_duplicates()
            print("this patient has", len(mri_filepaths), "mri sessions stored in:")
            for i in range(len(mri_filepaths)):
                mri_filepath = self.process_mri_path(mri_filepaths.iloc[i])
                if os.path.exists(fixPath(self.dicom_root + "/" + mri_filepath)):
                    self.print_directory(dir_path=[mri_filepath], indexes=[i])
                else:
                    print("None file found at " + self.dicom_root + "/" + mri_filepath + "/")
        else:
            print("Patient has no valid MRI session")

    def load_dicom_mri(self, index):
        """
        Load DICOM given index
        :param index: [sequence index, slice index]
        :return: dicom datasets, return None if no dicom file is loaded
        """
        filepath_temp = fixPath(self.get_file_path_given_slice(index))

        if os.path.exists(filepath_temp):
            try:
                ds = dicom.dcmread(filepath_temp)
                return ds
            except Exception:
                return None  # or you could use 'continue'
        else:
            return None

    def load_nifti_mri(self, index):
        """
        Load DICOM given index
        :param index: [sequence index, slice index]
        :return: nifti datasets, return None if no nifti file is loaded
        """
        filepath_temp = fixPath(self.get_file_path_given_slice(index))

        if os.path.exists(filepath_temp):
            try:
                ds = nib.load(filepath_temp)
                return ds
            except Exception:
                return None  # or you could use 'continue'
        else:
            return None

    def load_nifti_mri_by_file(self, index):
        """
        Load DICOM given index
        :param index: [file index]
        :return: nifti datasets, return None if no nifti file is loaded
        """
        filepath_temp = fixPath(self.get_file_path_given_file(index))

        if os.path.exists(filepath_temp):
            try:
                ds = nib.load(filepath_temp)
                return ds
            except Exception:
                return None  # or you could use 'continue'
        else:
            return None
    def load_nifti_json(self, slice_index = None, file_index = None):
        """
        Load DICOM given index
        :param slice_index: [sequence index, slice index]
        :param file_index: [sequence index, file index]
        :return: nifti datasets, return None if no nifti file is loaded
        """
        if slice_index is None:
            json_path = fixPath(self.get_file_path_given_file(file_index))
        else:
            json_path = fixPath(self.get_file_path_given_slice(slice_index))

        nifti_extension = get_file_extension(json_path, prompt=".nii")
        json_path = json_path[:-len(nifti_extension)] + '.json'

        if os.path.exists(json_path):
            try:
                # f = open(json_path)
                # data_infos = json.load(f)
                data_infos = safe_json_load(json_path)
                return data_infos
            except Exception:
                return None  # or you could use 'continue'
        else:
            return None

    def load_nifti_mask(self, index, mask_root=None, postfix='brain.nii.gz'):
        """
        load nifti mask, assume mask with postfix '_brain.nii.gz'
        :param index: [sequence index, slice index]
        :return: nifti datasets, return None if no nifti file is loaded
        """
        mask_path = fixPath(self.get_file_path_given_slice(index))
        nifti_extension = get_file_extension(mask_path, prompt=".nii")
        mask_path = mask_path[:-len(nifti_extension)] + postfix  # [H, W, slice_index]
        if mask_root is not None:
            mask_path = mask_path.replace(self.dicom_root, mask_root)

        if os.path.exists(mask_path):
            try:
                ds = nib.load(mask_path)
                return ds
            except Exception:
                return None  # or you could use 'continue'
        else:
            return None

    def update_mask_image_cache(self, cache_path=None):
        """
        update mask image cache, only works for nifti images now
        :param cache_path: string, cache path, NACC_nifti_mask, e.g.
        """
        if self.mode != 'nifti':
            raise Exception("mode is not nifti, cannot cache mask image")

        if cache_path is None:
            cache_path = self.cache_path[:-(len(self.patient_ID)) - 2] + '_mask/' + self.patient_ID
        else:
            cache_path = cache_path + self.patient_ID

        # only store cache mask if original image is cached
        if self.if_cache_image:
            for i in range(len(self.dir_index_list)):
                dir_path = cache_path + '/' + str(i) + '/'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                image_array_current = None
                image_path_current = None
                for j in range(len(self.slice_index_list[i])):
                    # only cache mask for (H, W, slice_index) now
                    if len(self.slice_index_list[i][j]) == 2:  # [file_index, slice_index]
                        slice_index = self.dir_index_list[i] + [self.slice_index_list[i][j]]
                        image_path_temp = fixPath(self.get_file_path_given_slice(slice_index))
                        nifti_extension = get_file_extension(image_path_temp, prompt=".nii")
                        image_path_temp = image_path_temp[:-len(nifti_extension)] + '_brain.nii.gz'  # [H, W, slice_index]

                        if image_path_temp != image_path_current:
                            ds = self.load_nifti_mask(slice_index)
                            if ds is not None:
                                if ds._dataobj._dtype == [('R', 'u1'), ('G', 'u1'), ('B', 'u1')]:
                                    image_array = np.asanyarray(ds.dataobj)
                                    image_array = image_array.copy().view(dtype=np.uint8).reshape(
                                        image_array.shape + (3,))  # (160, 160, 23, 3) RGB numpy.memmap
                                else:
                                    image_array = ds.get_fdata()  # a <class 'numpy.memmap'> with shape (H, W, slice_index) or (H, W, slice_index, series_index)
                                    # max:  680.0 min:  0.0
                                # image_array = np.rot90(image_array)
                            else:
                                image_array = None
                            image_array_current = image_array
                            image_path_current = image_path_temp

                        if image_array_current is not None:
                            if len(image_array_current.shape) == 2:  # (H, W) mask for (H, W, 1) image
                                cache_image = image_array_current
                            else:
                                cache_image = image_array_current[:, :, slice_index[-1][1]]
                            self.store_cache_image(j, cache_image, dir_path)

    def load_nifti_mask_slice(self, index):
        """
        load nifti mask given slices, use pkl file if mask is cached
        :param index: [sequence index, slice index]
        :return: image_array, return None if no mask file is loaded
        """
        image_array = None
        if len(index[-1]) == 2:
            if self.if_cache_image:
                file_idx = self.dir_index_list.index(index[0:-1])
                if index[-1] in self.slice_index_list[file_idx]:
                    slice_idx = (self.slice_index_list[file_idx]).index(index[-1])
                    # slice_path = self.cache_path[:-(len(self.patient_ID)) - 2] + '_mask/' + self.patient_ID + '/' + str(
                    #     file_idx) + '/' + str(slice_idx) + '.pkl'
                    slice_path = self.cache_path.replace('image_cache', 'image_cache_mask')
                    slice_path = slice_path  + str(
                        file_idx) + '/' + str(slice_idx) + '.pkl'
                    if os.path.exists(slice_path):
                        with open(slice_path, "rb") as f:
                            image_array = pickle.load(f)
            else:
                if self.mode == 'nifti':
                    ds = self.load_nifti_mask(index)
                    try:
                        if ds._dataobj._dtype == [('R', 'u1'), ('G', 'u1'), ('B', 'u1')]:
                            image_array = np.asanyarray(ds.dataobj)
                            image_array = image_array.copy().view(dtype=np.uint8).reshape(
                                image_array.shape + (3,))  # (160, 160, 23, 3) RGB numpy.memmap
                        else:
                            image_array = ds.get_fdata()
                        # image_array = np.rot90(image_array)  # rotate nifti file to align the original dicom file
                        # if multi slices [file_index, slice_index_i] or [file_index, series_index_i, slice_index_i],
                        # read give subslice index, (H, W, slice_index, series_index)
                        image_array = image_array[:, :, index[-1][1]]
                    except Exception:
                        return None  # or you could use 'continue'
        return image_array

    def load_mri_slice(self, index):
        """
        Load image given index
        :param index: [dir index, slice index]
        :return: image numpy array, return None if no image file is loaded
        """
        image_array = None
        if self.if_cache_image:
            file_idx = self.dir_index_list.index(index[0:-1])
            if index[-1] in self.slice_index_list[file_idx]:
                slice_idx = (self.slice_index_list[file_idx]).index(index[-1])
                slice_path = self.cache_path + str(file_idx) + '/' + str(slice_idx) + '.pkl'
                with open(slice_path, "rb") as f:
                    image_array = pickle.load(f)
        else:
            if self.mode == 'dicom':
                ds = self.load_dicom_mri(index)
                if ds is not None:
                    try:
                        image_array = ds.pixel_array
                        # if multi slices, read give subslice index
                        if isinstance(index[-1], list):
                            image_array = image_array[index[-1][1]]
                    except Exception:
                        return None  # or you could use 'continue'
            elif self.mode == 'nifti':
                ds = self.load_nifti_mri(index)
                try:
                    if ds._dataobj._dtype == [('R', 'u1'), ('G', 'u1'), ('B', 'u1')]:
                        image_array = np.asanyarray(ds.dataobj)
                        image_array = image_array.copy().view(dtype=np.uint8).reshape(
                            image_array.shape + (3,))  # (160, 160, 23, 3) RGB numpy.memmap
                    else:
                        image_array = ds.get_fdata()
                    # image_array = np.rot90(image_array)
                    # if multi slices [file_index, slice_index_i] or [file_index, series_index_i, slice_index_i], read give subslice index, (H, W, slice_index, series_index)
                    if isinstance(index[-1], list):
                        if len(index[-1]) == 2:  # [file_index, slice_index_i]
                            image_array = image_array[:, :, index[-1][1]]
                        elif len(index[-1]) == 3:  # [file_index, series_index_i, slice_index_i]
                            image_array = image_array[:, :, index[-1][2], index[-1][1]]
                except Exception:
                    return None  # or you could use 'continue'
        return image_array

    def load_mri_volume(self, index, if_resize = True):
        """
        Load image given index
        :param index: [dir index, file index]
        :param if_resize: bool, if resize the volume by zoom factor given if any.
        :return: image numpy array, return None if no image file is loaded
        """
        image_array = [] # if 2D [H, W, C], if 3D [D, H, W]
        if self.if_cache_image:
            file_idx = self.dir_index_list.index(index[0:-1])
            slice_idx_list = self.get_slice_index_list_in_file(index)
            for slice_idx_i in slice_idx_list:
                if slice_idx_i[-1] in self.slice_index_list[file_idx]:
                    slice_idx = (self.slice_index_list[file_idx]).index(slice_idx_i[-1])
                    slice_path = self.cache_path + str(file_idx) + '/' + str(slice_idx) + '.pkl'
                    with open(slice_path, "rb") as f:
                        image_array_i = pickle.load(f)
                        image_array.append(image_array_i)
            return np.array(image_array)
        else:
            if self.mode == 'dicom':
                slice_idx_list = self.get_slice_index_list_in_file(index)
                for slice_idx_i in slice_idx_list:
                    ds = self.load_dicom_mri(slice_idx_i)
                    if ds is not None:
                        try:
                            image_array_i = ds.pixel_array
                            # if multi slices, read give subslice index
                            if isinstance(index[-1], list):
                                image_array_i = image_array[slice_idx_i[-1][1]]
                            image_array.append(image_array_i)
                        except Exception:
                            return None  # or you could use 'continue'
                return np.array(image_array)
            elif self.mode == 'nifti':
                ds = self.load_nifti_mri_by_file(index)

                try:
                    if ds._dataobj._dtype == [('R', 'u1'), ('G', 'u1'), ('B', 'u1')]:
                        image_array = np.asanyarray(ds.dataobj)
                        image_array = image_array.copy().view(dtype=np.uint8).reshape(
                            image_array.shape + (3,))  # (160, 160, 23, 3) RGB numpy.memmap
                    else:
                        image_array = ds.get_fdata()
                    # image_array = np.rot90(image_array)
                    # if multi slices [file_index, slice_index_i] or [file_index, series_index_i, slice_index_i], read give subslice index, (H, W, slice_index, series_index)

                    # if isinstance(index[-1], list):
                        # if len(index[-1]) == 2:  # [file_index, slice_index_i]
                        #     image_array = image_array[:, :, index[-1][1]]
                        # elif len(index[-1]) == 3:  # [file_index, slice_index_i, series_index_i]
                        #     image_array = image_array[:, :, index[-1][2], index[-1][1]]
                    if if_resize:
                        zooms = ds.header.get_zooms()
                        if len(zooms) > 3:
                            max_dim = max(image_array.shape[0:-1])
                            max_dim_index = image_array.shape.index(max_dim)
                            max_zoom = zooms[max_dim_index]
                            zooms = zooms[:-1] + (max_zoom,)
                        else:
                            max_dim = max(image_array.shape)
                            max_dim_index = image_array.shape.index(max_dim)
                            max_zoom = zooms[max_dim_index]

                        # output_shape = (image_temp.shape[0] * 1.2 / max_zoom, image_temp.shape[1], image_temp.shape[2])
                        zoom_factors = tuple(z / max_zoom  for z in zooms)
                        output_shape = tuple(a * b for a, b in zip(image_array.shape, zoom_factors))
                        output_shape = tuple(int(round(x)) for x in output_shape)
                        # Apply zoom
                        # image_array = resize(image_array, output_shape, mode='reflect', anti_aliasing=True)
                        # image_array = resize(image_array,output_shape=output_shape,order=3,mode='constant',
                        #                      cval=0,anti_aliasing=True, preserve_range=True)

                        # image_array = zoom(image_array, zoom_factors, order=3)
                        image_array = resize_3d_torch(image_array, output_shape)

                    return image_array
                except Exception:
                    return None # or you could use 'continue'
        return image_array

    def get_dir_names_given_dir_index(self, dir_index):
        """
         get dir names list for given dir index
        :param slice_index: [dir index]
        :return: list of strings
        """
        dir_idx = self.dir_index_list.index(dir_index)
        if dir_idx is not None:
            return self.dir_name_list[dir_idx]
        else:
            return None

    def get_file_path_given_slice(self, slice_index):
        """
        get file path for given slice index
        :param slice_index: [dir index, slice index]
        :return: string, file path
        """
        sequence_index = slice_index[0:-1]
        if sequence_index in self.dir_index_list:
            dir_idx = self.dir_index_list.index(sequence_index)
            if slice_index[-1] in self.slice_index_list[dir_idx]:
                slice_idx = (self.slice_index_list[dir_idx]).index(slice_index[-1])
                filepath_temp1 = self.dicom_root + "/" + "/".join(self.dir_name_list[dir_idx])
                slice_name = self.slice_name_list[dir_idx][slice_idx]
                if isinstance(slice_name, str):
                    filepath_temp1 = filepath_temp1 + '/' + slice_name
                else:
                    filepath_temp1 = filepath_temp1 + '/' + slice_name[0]
                return filepath_temp1
            else:
                return ''
        else:
            return ''

    def get_file_name_given_file(self, file_index):
        """
        get file path for given slice index
        :param slice_index: [dir index, file index]
        :return: string, file name
        """
        sequence_index = file_index[0:-1]
        if sequence_index in self.dir_index_list:
            dir_idx = self.dir_index_list.index(sequence_index)
            slice_index_list = self.get_slice_index_list_in_file(file_index)
            slice_index = slice_index_list[0]
            if slice_index[-1] in self.slice_index_list[dir_idx]:
                slice_idx = (self.slice_index_list[dir_idx]).index(slice_index[-1])
                # filepath_temp1 = self.dicom_root + "/" + "/".join(self.dir_name_list[dir_idx])
                slice_name = self.slice_name_list[dir_idx][slice_idx]
                if isinstance(slice_name, str):
                    return slice_name
                else:
                    return slice_name[0]
            else:
                return ''
        else:
            return ''

    def get_file_path_given_file(self, file_index):
        """
        get file path for given file index
        :param file_index: [dir index, file index]
        :return: string, file path
        """
        slice_index_list = self.get_slice_index_list_in_file(file_index)
        return self.get_file_path_given_slice(slice_index_list[0])

    def get_slice_info_given_slice(self, slice_index):
        """
        get slice info for given slice index
        :param slice_index: [dir index, slice index]
        :return: list [info1, info2, ...]
        """
        sequence_index = slice_index[0:-1]
        if sequence_index in self.dir_index_list:
            file_idx = self.dir_index_list.index(sequence_index)
            if slice_index[-1] in self.slice_index_list[file_idx]:
                slice_idx = (self.slice_index_list[file_idx]).index(slice_index[-1])
                slice_info = self.slice_info_list[file_idx][slice_idx]
                return slice_info
            else:
                return []
        else:
            return []

    def get_slice_info_given_file(self, file_index):
        """
        get first slice info for given file index
        :param file_index: [dir index, slice index]
        :return: list [info1, info2, ...]
        """
        slice_index_list = self.get_slice_index_list_in_file(file_index)
        return self.get_slice_info_given_slice(slice_index_list[0])

    def set_slice_info_given_slice(self, slice_index, slice_info):
        """
        set slice info for given slice index
        :param slice_index: [dir index, slice index]
        :param slice_info: [info1, info2, ...]
        :return: boolean, true if slice info set correctly, false otherwise
        """
        sequence_index = slice_index[0:-1]
        if sequence_index in self.dir_index_list:
            file_idx = self.dir_index_list.index(sequence_index)
            if slice_index[-1] in self.slice_index_list[file_idx]:
                slice_idx = (self.slice_index_list[file_idx]).index(slice_index[-1])
                self.slice_info_list[file_idx][slice_idx] = slice_info
                return True
            else:
                return False
        else:
            return False

    def add_slice_info_given_slice(self, slice_index, add_slice_info, max_info_size=None):
        """
        add slice info for given slice index
        :param slice_index: [dir index, slice index]
        :param add_slice_info: string, int, etc.
        :param max_info_size: int, max len of slice info, if reach exact maximum, replace the last info tags to new one
        :return: boolean, true if slice info set correctly, false otherwise
        """
        sequence_index = slice_index[0:-1]
        if sequence_index in self.dir_index_list:
            file_idx = self.dir_index_list.index(sequence_index)
            if slice_index[-1] in self.slice_index_list[file_idx]:
                slice_idx = (self.slice_index_list[file_idx]).index(slice_index[-1])
                if max_info_size is None:
                    self.slice_info_list[file_idx][slice_idx] = self.slice_info_list[file_idx][slice_idx] + [
                        add_slice_info]
                elif max_info_size < len(self.slice_info_list[file_idx][slice_idx]):
                    return False
                elif max_info_size == len(self.slice_info_list[file_idx][slice_idx]):
                    self.slice_info_list[file_idx][slice_idx][-1] = add_slice_info
                elif max_info_size > len(self.slice_info_list[file_idx][slice_idx]):
                    self.slice_info_list[file_idx][slice_idx] = self.slice_info_list[file_idx][slice_idx] + [
                        add_slice_info]
                return True
            else:
                return False
        else:
            return False

    def get_slice_index_list_in_dir(self, dir_index):
        """
        Get mri slice index to a list given dir index, output along dir index
        :param dir_index: directory index list [0, 0, 0]
        :return: index list [[0, 0, 0, 0], [0, 0, 0, 1], ..., [0, 0, 0, n]], [] if no file index
         is loaded
        """
        slice_index_list_in_dir = []
        if dir_index in self.dir_index_list:
            file_idx = self.dir_index_list.index(dir_index)
            slice_index_list_in_dir = self.slice_index_list[file_idx]  # [0, 1,..., n] or [[0,0], [0,1],..., [2,n]]

        return [dir_index + [slice_index] for slice_index in slice_index_list_in_dir]

    def get_file_index_list_in_dir(self, dir_index):
        """
        Get mri file index to a list given dir index, output along dir index
        this is useful when load individual mri file with multiple mri slices stored
        :param dir_index: dir index list [0, 0, 0]
        :return: file index list [[0, 0, 0, 0], [0, 0, 0, [1]], ..., [0, 0, 0, n]], last element
        is int if one mri slice for one file/list if multiple mris for one file, [] if no file index is loaded
        """
        file_index_list = []
        if dir_index in self.dir_index_list:
            file_idx = self.dir_index_list.index(dir_index)
            slice_index_list_in_dir = self.slice_index_list[file_idx]  # [0, 1,..., n] or [[0,0], [0,1],..., [2,n]]
            for slice_index in slice_index_list_in_dir:
                # if the last index for slice index is an int, then only one mri stores in file
                if isinstance(slice_index, int):  # file_index
                    if slice_index not in file_index_list:
                        file_index_list.append(slice_index)
                elif isinstance(slice_index, list):  # [file_index, series_index_i, slice_index_i]
                    if [slice_index[0]] not in file_index_list:
                        file_index_list.append([slice_index[0]])

        return [dir_index + [file_idx] for file_idx in file_index_list]

    def get_file_index_given_slice(self, slice_index):
        """
        get mri file index given slice index
        :param slice_index: directory index list [0, 0, 0, 1] or [0, 0, 0, [0, 1]]
        :return: index [0, 0, 0, 1] or [0, 0, 0, [0]], [] if no file index
         is loaded
        """
        dir_index = slice_index[0:-1]
        last_index = slice_index[-1]
        if isinstance(last_index, int):
            return dir_index + [last_index]
        elif isinstance(last_index, list):
            return dir_index + [[last_index[0]]]
        return None

    def get_file_index_list_given_sequence(self, sequence_name):
        """
        Load mri slice index to a list given dir index, output along dir index
        :param sequence_name: string for sequence: 'T1_T1flair', etc.
        :return: index list [[0, 0, 0, 0], [0, 0, 0, 1], ..., [0, 0, 0, n]], [] if no file index
         is loaded
        """
        slice_index_list = self.sequence_slice_index_lists_dict[sequence_name]
        file_index_list = []
        for slice_index in slice_index_list:
            file_index_temp = self.get_file_index_given_slice(slice_index)
            if file_index_temp is not None:
                if file_index_temp not in file_index_list:
                    file_index_list.append(file_index_temp)
        return file_index_list

    def get_dir_index_given_slice(self, slice_index):
        """
        Load mri slice index to a list given dir index, output along dir index
        :param slice_index: slice index list [0, 0, 0, [0,0]] or [0, 0, 0, 0]
        :return: dir index [0,0,0]
         is loaded
        """
        dir_index = slice_index[0:-1]
        return dir_index

    def get_slice_index_list_in_file(self, file_index):
        """
        Load mri slice index to a list given file index, output along dir index
        :param file_index: file index list [dir_index, file_idx]
        """
        dir_index = file_index[0:-1]
        last_index = file_index[-1]  # 0 or [0]
        slice_index_list_in_file = []
        if dir_index in self.dir_index_list:
            file_idx = self.dir_index_list.index(dir_index)
            slice_index_list_in_dir = self.slice_index_list[file_idx]  # [0, 1,..., n] or [[0,0], [0,1],..., [2,n]]
            if isinstance(last_index, int):
                if last_index in slice_index_list_in_dir:
                    return [dir_index + [last_index]]  # [[0,0,0,1]]
            if isinstance(last_index, list):
                for slice_index in slice_index_list_in_dir:
                    if isinstance(slice_index, list):
                        if slice_index[0] == last_index[0]:
                            slice_index_list_in_file.append(slice_index)

        return [dir_index + [slice_index] for slice_index in slice_index_list_in_file]

    def get_slice_lists_in_dir(self, index):
        """
        Load mri slice index and name to a list given dir index, without sequence index
        :param index: dir index list [0, 0, 0]
        :return: tuple: name list [0.dcm, 1.dcm, 2.dcm, ..., n.dcm], [] if no file is loaded
         index list [0, 1, 2, ..., n], [] if no file is loaded
         slice info list [info1, info2, ...]
         cache images corresponding to slices [numpy1, numpy2, ...]
        """
        file_idx = self.dir_index_list.index(index)
        filepath_temp = self.dicom_root + "/" + "/".join(self.dir_name_list[file_idx])
        slice_names_temp = []
        slice_indices_temp = []
        slice_infos_temp = []
        cache_images_temp = []
        if os.path.exists(fixPath(filepath_temp)):
            list1 = []
            if self.mode == 'dicom':
                list1 = load_sorted_directory(filepath_temp)
            elif self.mode == 'nifti':
                # nifti_extension = get_file_extension(filepath_temp, prompt=".nii")
                list1 = load_sorted_directory(filepath_temp, postfix_label=".nii")
            if list1 is not None:
                if len(list1) > 0:  # avoid empty file
                    slice_indices = range(len(list1))
                    slice_paths = [filepath_temp + "/" + slice_name for slice_name in list1]
                    if self.mode == 'dicom':
                        slice_tags = [self.info_tag_codes] * len(list1)
                    else:
                        slice_tags = [self.info_tags] * len(list1)
                    modes = [self.mode] * len(list1)
                    if_cache_images = [self.if_cache_image] * len(list1)
                    if len(list1) > 1200:
                        with Pool(processes=10) as pool:
                            for slice_name_valid, slice_index_valid, slice_info_valid, cache_image in pool.starmap(
                                    get_slice_data,
                                    zip(list1,
                                        slice_indices,
                                        slice_paths,
                                        slice_tags,
                                        modes,
                                        if_cache_images)):
                                slice_names_temp = slice_names_temp + slice_name_valid
                                slice_indices_temp = slice_indices_temp + slice_index_valid
                                slice_infos_temp = slice_infos_temp + slice_info_valid
                                cache_images_temp = cache_images_temp + cache_image

                    else:
                        for i in range(len(list1)):
                            slice_name_valid, slice_index_valid, slice_info_valid, cache_image = get_slice_data(
                                list1[i],
                                slice_indices[i],
                                slice_paths[i],
                                slice_tags[i],
                                modes[i],
                                if_cache_images[i])
                            slice_names_temp = slice_names_temp + slice_name_valid
                            slice_indices_temp = slice_indices_temp + slice_index_valid
                            slice_infos_temp = slice_infos_temp + slice_info_valid
                            cache_images_temp = cache_images_temp + cache_image
        return slice_names_temp, slice_indices_temp, slice_infos_temp, cache_images_temp

    def get_slice_series_list_given_file(self, file_index):
        slice_index_list = self.get_slice_index_list_in_file(file_index)
        slice_series_list = []
        for slice_i in slice_index_list:
            if len(slice_i[-1]) == 2 or len(slice_i[-1]) == 1:
                if slice_i[-1][0] not in slice_series_list:
                    slice_series_list.append(slice_i[-1][1])
            elif len(slice_i[-1]) == 3:
                if slice_i[-1][1] not in slice_series_list:
                    slice_series_list.append(slice_i[-1][1])
            else:
                print(f"slice {slice_i} with last slice index len '{len(slice_i[-1])}' is detected")
                break
        return slice_series_list

    def get_slice_list_in_file_given_series(self, file_index, series_list):
        slice_index_list = self.get_slice_index_list_in_file(file_index)
        slice_index_list_series = []
        for slice_i in slice_index_list:
            if len(slice_i[-1]) == 2 or len(slice_i[-1]) == 1:
                if slice_i[-1][0] in series_list:
                    slice_index_list_series.append(slice_i)
            elif len(slice_i[-1]) == 3:
                if slice_i[-1][1] in series_list:
                    slice_index_list_series.append(slice_i)
            else:
                print(f"slice {slice_i} with last slice index len '{len(slice_i[-1])}' is detected")
                break
        return slice_index_list_series
    def update_dir_name_list(self):
        """
        update the directory name list and index lists
        """
        mri_filepaths = self.mriDF.loc[:, self.mri_directory_label].drop_duplicates()
        mri_latest_filepath = self.latest_mri_session.loc[:, self.mri_directory_label].drop_duplicates()

        if self.check_MRI_exist():
            self.dir_name_list = []
            self.dir_index_list = []
            self.dir_name_list_latest = []
            self.dir_index_list_latest = []
            for i in range(len(mri_filepaths)):
                mri_filepath = self.process_mri_path(mri_filepaths.iloc[i])
                if os.path.exists(fixPath(self.dicom_root + "/" + mri_filepath)):
                    # update self.dir_name_list and self.dir_index_list
                    self.list_directory(dir_path=[mri_filepath], indexes=[i])

            # update self.dir_name_list_latest and self.dir_index_list_latest
            for i in range(len(self.dir_name_list)):
                if len(mri_latest_filepath) > 0:
                    if self.process_mri_path(mri_latest_filepath.iloc[0]) == self.dir_name_list[i][0]:
                        self.dir_name_list_latest.append(self.dir_name_list[i])
                        self.dir_index_list_latest.append(self.dir_index_list[i])

    def update_slice_lists(self):
        """
        Update the self.slice_name_list and self.slice_index_list and slice_info_list
        """
        self.slice_name_list, self.slice_index_list, self.slice_info_list = [], [], []
        cache_images_list = []
        # start0 = time.time()
        for dir_i in self.dir_index_list:
            slice_names_temp, slice_indices_temp, slice_infos_temp, cache_images_temp = self.get_slice_lists_in_dir(
                dir_i)
            self.slice_name_list.append(slice_names_temp)
            self.slice_index_list.append(slice_indices_temp)
            self.slice_info_list.append(slice_infos_temp)
            cache_images_list.append(cache_images_temp)

        slice_len = 0
        for slice_index in self.slice_index_list:
            slice_len += len(slice_index)

        if self.if_cache_image:
            for i, slice_index in enumerate(self.slice_index_list):
                # make dir for each dir
                dir_path = self.cache_path + str(i) + '/'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                slice_j_list = range(len(slice_index))
                dir_path_list = [dir_path] * len(slice_j_list)

                if len(slice_index) > 8000:
                    with Pool(processes=10) as pool:
                        for if_store in pool.starmap(self.store_cache_image, zip(slice_j_list, cache_images_list[i],
                                                                                 dir_path_list)):
                            continue
                else:
                    for j in slice_j_list:
                        self.store_cache_image(j, cache_images_list[i][j], dir_path_list[j])

    def store_cache_image(self, index_j, cache_image, dir_path_i):
        """
        save cached image into pkl format
        :param index_j: int, index for each slice
        :param cache_image: np.array, cached image
        :param dir_path_i: string path to save cached image
        :return: boolean, if saved successfully, return True else False
        """
        slice_path = dir_path_i + str(index_j) + '.pkl'
        # if nifti convert float64 to float16 to reduce file size
        # do max-min normalization before saving nifti file, this way the file can be saved in float16 format if
        # encountering large float in pixels
        if self.mode == 'nifti':
            maximum = np.max(cache_image)
            minimum = np.min(cache_image)
            scale = maximum - minimum
            if scale > 0:
                scale_coeff = 1. / scale
            else:
                scale_coeff = 0
            cache_image = (cache_image - minimum) * scale_coeff
            cache_image = np.array(cache_image, dtype='float16')
        with open(slice_path, "wb") as f:
            pickle.dump(cache_image, f)
            return True
        return False

    def update_sequence_series_tags(self):
        self.dicom_dict = {
            "SeriesDescription": [0x0008, 0x103E],  #
            "SequenceName": [0x0018, 0x0024],
            "SpacingBetweenSlice": [0x0018, 0x0088],
            "SliceThickness": [0x0018, 0x0050],
            "MRAcquisitionType": [0x0018, 0x0023],
            "ScanningSequence": [0x0018, 0x0020],
            "SequenceVariant": [0x0018, 0x0021],
            "RepetitionTime": [0x0018, 0x0080],
            "EchoTime": [0x0018, 0x0081],
            "InversionTime": [0x0018, 0x0082],
        }

        Tag_dict_T1 = {
            "SeriesDescription": ['T1', 't1', 'MPRAGE_GRAPPA_ADNI', 'MPRAGE'],  #
            "SequenceName": ['*tfl3d1', '*tfl3d1_16ns', '*tfl3d1_16ns'],
            "DirName": ['T1', 't1']
        }

        Tag_dict_T2 = {
            "SeriesDescription": ['T2'],  #
            "SequenceName": [],
            "DirName": ['T2', 't2']
        }

        Tag_dict_DTI = {
            "SeriesDescription": ['DTI'],  #
            "SequenceName": [],
            "DirName": ['DTI', 'dti']
        }

        Tag_dict_DWI = {
            "SeriesDescription": ['DWI'],  #
            "SequenceName": [],
            "DirName": ['DWI', 'dwi']
        }

        Tag_dict_FLAIR = {
            "SeriesDescription": ['FLAIR'],  #
            "SequenceName": [],
            "DirName": ['FLAIR', 'flair']
        }

        # read SeriesDescription_list provided by radiologists and added to tag_dict_T1
        with open('SeriesDescription_T1.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T1["SeriesDescription"] = Tag_dict_T1["SeriesDescription"] + lines
        Tag_dict_T1["SeriesDescription"] = list(set(Tag_dict_T1["SeriesDescription"]))

        with open('SeriesDescription_T2.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T2["SeriesDescription"] = Tag_dict_T2["SeriesDescription"] + lines
        Tag_dict_T2["SeriesDescription"] = list(set(Tag_dict_T2["SeriesDescription"]))

        with open('SeriesDescription_DTI.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_DTI["SeriesDescription"] = Tag_dict_DTI["SeriesDescription"] + lines
        Tag_dict_DTI["SeriesDescription"] = list(set(Tag_dict_DTI["SeriesDescription"]))

        with open('SeriesDescription_DWI.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_DWI["SeriesDescription"] = Tag_dict_DWI["SeriesDescription"] + lines
        Tag_dict_DWI["SeriesDescription"] = list(set(Tag_dict_DWI["SeriesDescription"]))

        with open('SeriesDescription_FLAIR.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_FLAIR["SeriesDescription"] = Tag_dict_FLAIR["SeriesDescription"] + lines
        Tag_dict_FLAIR["SeriesDescription"] = list(set(Tag_dict_FLAIR["SeriesDescription"]))

        self.combined_tags_dict = {
            "T1": Tag_dict_T1,
            "T2": Tag_dict_T2,
            "DTI": Tag_dict_DTI,
            "DWI": Tag_dict_DWI,
            "FLAIR": Tag_dict_FLAIR
        }

    def get_time_given_dir_index(self, dir_index):
        """
        Get time of dir taken
        :param dir_index: [dir_index]
        :return: [[years, months, days]]
        """
        return None
    def get_latest_mri_session(self, if_update=False):
        """
        Get latest mri session
        :param if_update: if update current mri dataframe to dataframe with only latest mri session
        :return: mri dataframe with latest mri, date in ['year', 'mouth', 'day']
        """
        return None, None

    def get_latest_analysis_DF(self, year_label, month_label, day_label, if_update=False):
        """
        Get latest mri session
        :param if_update: if update current mri dataframe to dataframe with only latest mri session
        :return: mri dataframe with latest mri, date in ['year', 'mouth', 'day']
        """
        latest_analysis_session = self.analysisDF
        latest_yr = None
        latest_mo = None
        latest_dy = None
        if year_label in self.analysisDF.columns:
            latest_yr = latest_analysis_session[year_label].max()
            latest_analysis_session = latest_analysis_session.loc[latest_analysis_session[year_label] == latest_yr]
            if month_label in self.analysisDF.columns:
                latest_mo = latest_analysis_session[month_label].max()
                latest_analysis_session = latest_analysis_session.loc[latest_analysis_session[month_label] == latest_mo]
                if day_label in self.analysisDF.columns:
                    latest_dy = latest_analysis_session[day_label].max()
                    latest_analysis_session = latest_analysis_session.loc[latest_analysis_session[day_label] == latest_dy]
            if if_update:
                self.analysisDF = latest_analysis_session
            return latest_analysis_session, [latest_yr, latest_mo, latest_dy]
        else:
            return pd.DataFrame(columns=latest_analysis_session.columns), [latest_yr, latest_mo, latest_dy]

    def get_death_date(self):
        """
        get patient death date
        :return: date
        """
        # if self.analysisDF['NACCDIED'].iloc[0] == 1:
        #     return [self.analysisDF['NACCYOD'].iloc[0], self.analysisDF['NACCMOD'].iloc[0], 0]
        # else:
        #     return None
        return None

    def get_time_from_latest_mri_to_death(self, unit='month'):
        """
        get patient time period from latest mri to death
        :param unit: time period unit, 'year', 'day' ,and 'mouth', 'default is 'month'
        :return: int, time length in given unit
        """
        death_date = self.get_death_date()
        if death_date is not None:
            return get_time_interval(self.latest_mri_date, death_date, unit=unit)
        else:
            return None

    # sequence detection
    def get_sequence_by_slice_index_info(self, slice_index, ifLatest=False, ifExact=True):
        """
        get sequence by slice index
        :param slice_index: [dir index, slice index]
        return: list of detected sequence for given slice, ['PD', 'T1_T1flair', etc]
        """
        detect_sequence = []

        # get index of series description in info_tags
        SeriesDescription_idx = self.info_tags.index('SeriesDescription')
        RepetitionTime_idx = self.info_tags.index('RepetitionTime')
        EchoTime_idx = self.info_tags.index('EchoTime')
        InversionTime_idx = self.info_tags.index('InversionTime')

        temp_dir_name_list = self.dir_name_list
        temp_dir_index_list = self.dir_index_list
        dir_index = slice_index[0:-1]
        if dir_index in temp_dir_index_list:
            file_idx = temp_dir_index_list.index(dir_index)
            dir_name = temp_dir_name_list[file_idx]

            # get sequence by dir name
            for sequence_name in self.detect_sequence:
                Tag_dict_temp = self.combined_tags_dict[sequence_name]
                for sequence_tag in Tag_dict_temp['DirName']:
                    if ifExact:
                        # if any(sequence_tag.upper() == x.upper() for x in dir_name):
                        if any(clear_string_char(sequence_tag, [' ', '_', '/']).upper()
                               == clear_string_char(x, [' ', '_', '/']).upper() for x in dir_name):
                            detect_sequence.append(sequence_name)
                            break
                    else:
                        # if any(sequence_tag.upper() in x.upper() for x in dir_name):
                        if any(clear_string_char(sequence_tag, [' ', '_', '/']).upper()
                               in clear_string_char(x, [' ', '_', '/']).upper() for x in dir_name):
                            detect_sequence.append(sequence_name)
                            break

            # get sequence by SeriesDescription
            if slice_index[-1] in self.slice_index_list[file_idx]:
                slice_idx = (self.slice_index_list[file_idx]).index(slice_index[-1])
                # print(self.slice_info_list)
                slice_info_SeriesDescription = self.slice_info_list[file_idx][slice_idx][SeriesDescription_idx]
                if slice_info_SeriesDescription is not None:
                    for sequence_name in self.detect_sequence:
                        Tag_dict_temp = self.combined_tags_dict[sequence_name]
                        if ifExact:
                            # if any(x.upper() == slice_info.upper() for x in Tag_dict_temp['SeriesDescription']):
                            if any(clear_string_char(x, [' ', '_', '/']).upper()
                                   == clear_string_char(slice_info_SeriesDescription, [' ', '_', '/']).upper()
                                   for x in Tag_dict_temp['SeriesDescription']):
                                detect_sequence.append(sequence_name)
                        else:
                            # if any(x.upper() in slice_info.upper() for x in Tag_dict_temp['SeriesDescription']):
                            if any(clear_string_char(x, [' ', '_', '/']).upper()
                                   in clear_string_char(slice_info_SeriesDescription, [' ', '_', '/']).upper()
                                   for x in Tag_dict_temp['SeriesDescription']):
                                detect_sequence.append(sequence_name)
                else:  # if has no series description check TR and TE with restricted standards
                    slice_info_RepetitionTime = self.slice_info_list[file_idx][slice_idx][RepetitionTime_idx]
                    slice_info_EchoTime = self.slice_info_list[file_idx][slice_idx][EchoTime_idx]
                    slice_info_InversionTime = self.slice_info_list[file_idx][slice_idx][InversionTime_idx]
                    coeff_multi = 1
                    if self.mode == 'nifti':
                        coeff_multi = 0.001  # nifti use s, dicom use ms
                    # T1 standard
                    if slice_info_RepetitionTime is not None and slice_info_EchoTime is not None:
                        # TR < 600 ms and TE < 30
                        if slice_info_RepetitionTime < 600 * coeff_multi and slice_info_EchoTime < 30 * coeff_multi:
                            if 'T1_T1flair' in self.detect_sequence:
                                detect_sequence.append('T1_T1flair')  # for sequence detection task only
                            if 'T1' in self.detect_sequence:
                                detect_sequence.append('T1')
                        # TR == 2550 ms +/- 20%, TE == 1.8 ms +/- 200%, Inv == 1100 ms +/- 30% is T2
                        if slice_info_InversionTime is not None:
                            if abs(slice_info_RepetitionTime - 2550 * coeff_multi) / (
                                    2550 * coeff_multi) < 0.2 and abs(
                                slice_info_EchoTime - 1.8 * coeff_multi) / (1.8 * coeff_multi) < 2 and abs(
                                slice_info_InversionTime - 1100 * coeff_multi) / (1100 * coeff_multi) < 0.3:
                                if 'T1_T1flair' in self.detect_sequence:
                                    detect_sequence.append('T1_T1flair')  # for sequence detection task only
                                if 'T1' in self.detect_sequence:
                                    detect_sequence.append('T1')

                    # T2 standard
                    if slice_info_RepetitionTime is not None and slice_info_EchoTime is not None and slice_info_InversionTime is not None:
                        # TR == 8000 ms +/- 20%, TE == 15 ms +/- 20%, Inv == 150 ms +/- 20% is T2
                        if abs(slice_info_RepetitionTime - 8000 * coeff_multi) / (8000 * coeff_multi) < 0.2 and abs(
                                slice_info_EchoTime - 15 * coeff_multi) / (15 * coeff_multi) < 0.2 and abs(
                            slice_info_InversionTime - 150 * coeff_multi) / (150 * coeff_multi) < 0.2:
                            if 'T2_T2star' in self.detect_sequence:
                                detect_sequence.append('T2_T2star')  # for sequence detection task only
                            if 'T2' in self.detect_sequence:
                                detect_sequence.append('T2')
                        # flair standard
                        # TR == 9000 ms +/- 20%, TE == 104 ms +/- 30%, Inv == 2350 ms +/- 20% is flair
                        elif abs(slice_info_RepetitionTime - 9000 * coeff_multi) / (9000 * coeff_multi) < 0.2 and abs(
                                slice_info_EchoTime - 104 * coeff_multi) / (104 * coeff_multi) < 0.3 and abs(
                            slice_info_InversionTime - 2350 * coeff_multi) / (2350 * coeff_multi) < 0.2:
                            if 'T2flair_flair' in self.detect_sequence:
                                detect_sequence.append('T2flair_flair')  # for sequence detection task only
                            if 'flair' in self.detect_sequence:
                                detect_sequence.append('flair')

        detect_sequence = list(set(detect_sequence))
        return detect_sequence

    def get_sequence_dir_slice_lists_dict(self, sequence_names, if_latest=False, ifExact=True):
        """
        get sequence dict for both dir index and slice index lists
        :param sequence_names: list of strings, sequence names corresponding to self.combined_tags_dict keys
        :return: tuple, (sequence_dir_index_lists_dict, sequence_slice_index_lists_dict)
        """
        sequence_dir_index_lists_dict, sequence_slice_index_lists_dict = {}, {}
        for sequence_name in sequence_names:
            sequence_dir_index_lists_dict[sequence_name], sequence_slice_index_lists_dict[sequence_name] = (
                [], [])

        for idx, dir_index in enumerate(self.dir_index_list):

            if len(self.slice_index_list[idx]) > 20000:
                slice_index_combine_list = [dir_index + [slice_index] for slice_index in self.slice_index_list[idx]]
                if_latest_list = [if_latest] * len(slice_index_combine_list)
                ifExact_list = [ifExact] * len(slice_index_combine_list)
                detect_sequence_list = []

                with Pool(processes=10) as pool:
                    for detect_sequence in pool.starmap(
                            self.get_sequence_by_slice_index_info,
                            zip(slice_index_combine_list,
                                if_latest_list,
                                ifExact_list, )):
                        detect_sequence_list.append(detect_sequence)

                for j in range(len(detect_sequence_list)):
                    detect_sequence = detect_sequence_list[j]
                    slice_index_combine = slice_index_combine_list[j]
                    if len(detect_sequence) > 1:
                        print(self.patient_ID + ": " + str(
                            slice_index_combine) + " has more than one sequence label of: " + str(
                            detect_sequence))
                    for sequence_name in detect_sequence:
                        if dir_index not in sequence_dir_index_lists_dict[sequence_name]:
                            sequence_dir_index_lists_dict[sequence_name].append(dir_index)
                        sequence_slice_index_lists_dict[sequence_name].append(slice_index_combine)
            else:
                for slice_index in self.slice_index_list[idx]:
                    slice_index_combine = dir_index + [slice_index]
                    detect_sequence = self.get_sequence_by_slice_index_info(slice_index_combine, if_latest, ifExact)

                    if len(detect_sequence) > 1:
                        print(self.patient_ID + ": " + str(
                            slice_index_combine) + " has more than one sequence label of: " + str(
                            detect_sequence))
                    for sequence_name in detect_sequence:
                        if dir_index not in sequence_dir_index_lists_dict[sequence_name]:
                            sequence_dir_index_lists_dict[sequence_name].append(dir_index)
                        sequence_slice_index_lists_dict[sequence_name].append(slice_index_combine)

        return sequence_dir_index_lists_dict, sequence_slice_index_lists_dict

    def get_sequence_given_dir_index(self, dir_index):
        """
        return seqeuence name given dir index
        :param dir_index: [0,0,0]
        :return: sequence name: string
        """
        for key in self.sequence_dir_index_lists_dict:
            if dir_index in self.sequence_dir_index_lists_dict[key]:
                return key
        return ''

    def get_sequence_given_file_index(self, file_index):
        """
        return seqeuence name given slice index
        :param file_index: [0,0,0,0] or [0,0,0,[0]]
        :return: sequence name: string
        """
        first_slice_index = self.get_slice_index_list_in_file(file_index)[0]
        for key in self.sequence_slice_index_lists_dict:
            if first_slice_index in self.sequence_slice_index_lists_dict[key]:
                return key
        return ''

    def get_sequence_given_slice_index(self, slice_index):
        """
        return seqeuence name given slice index
        :param slice_index: [0,0,0,0] or [0,0,0,[0,0]]
        :return: sequence name: string
        """
        for key in self.sequence_slice_index_lists_dict:
            if slice_index in self.sequence_slice_index_lists_dict[key]:
                return key
        return ''

    def get_slice_in_list_by_dir(self, dir_index, slice_list):
        """
        get all slice index in a list given dir index
        param: dir_index: [0,0,0]
        param: slice_list: [[0,0,0,0], [0,0,0,1], [1,0,0,0], [1,0,0,1],...]
        return: slices: [[0,0,0,0], [0,0,0,1]]
        """
        slices = []
        for slice_index in slice_list:
            if slice_index[:-1] == dir_index:
                slices.append(slice_index)
        return slices

    def set_sequence_to_dir(self, sequence_name, dir_index):
        """
        set input dir_index sequence with given sequence_name
        param: dir_index: [0,0,0]
        """
        if sequence_name in self.sequence_dir_index_lists_dict.keys():
            current_sequence_name = self.get_sequence_given_dir_index(dir_index)
            self.sequence_dir_index_lists_dict[current_sequence_name].remove(dir_index)
            self.sequence_dir_index_lists_dict[sequence_name].append(dir_index)

    def set_sequence_to_slice(self, sequence_name, slice_index):
        """
        set input slice_index sequence with given sequence_name
        param: slice_index: [0,0,0,0] or [0,0,0,[0,0]]
        """
        if sequence_name in self.sequence_slice_index_lists_dict.keys():
            current_sequence_name = self.get_sequence_given_slice_index(slice_index)
            self.sequence_slice_index_lists_dict[current_sequence_name].remove(slice_index)
            self.sequence_slice_index_lists_dict[sequence_name].append(slice_index)


    def update_pd_echo_time(self):
        """
        update different echo time for sequences marked as pd
        """
        self.PD_echo_time = []
        for i, dir_index in enumerate(self.sequence_dir_index_lists_dict['PD']):
            self.PD_echo_time.append([])
            idx = self.dir_index_list.index(dir_index)
            slice_index = self.slice_index_list[idx]
            for slice_i in slice_index:
                temp_index = dir_index + [slice_i]
                if self.mode == 'dicom':
                    dicom_data = self.load_dicom_mri(index=temp_index)
                    if dicom_data is not None:
                        if dicom_data.__contains__(self.dicom_dict['EchoTime']):
                            echo_time = dicom_data[self.dicom_dict['EchoTime']].value
                            if echo_time not in self.PD_echo_time[i]:
                                self.PD_echo_time[i].append(echo_time)
                elif self.mode == 'nifti':
                    data_infos = self.load_nifti_json(temp_index)
                    if 'EchoTime' in data_infos:
                        echo_time = data_infos['EchoTime']
                        if echo_time not in self.PD_echo_time[i]:
                            self.PD_echo_time[i].append(echo_time)

    def update_sequence_dir_slice_lists_dict(self):
        """
        update both self.sequence_dir_index_lists_dict and self.sequence_slice_index_lists_dict with info tag detection
        """
        if self.detect_sequence is not None:
            sequence_dir_index_lists_dict, sequence_slice_index_lists_dict = self.get_sequence_dir_slice_lists_dict(
                sequence_names=self.detect_sequence)

            # remove detected sequence from 'unknown'
            for key in sequence_dir_index_lists_dict:
                for x in sequence_dir_index_lists_dict[key]:
                    if x in self.sequence_dir_index_lists_dict['unknown']:
                        self.sequence_dir_index_lists_dict['unknown'].remove(x)
                for x in sequence_slice_index_lists_dict[key]:
                    if x in self.sequence_slice_index_lists_dict['unknown']:
                        self.sequence_slice_index_lists_dict['unknown'].remove(x)

            self.sequence_dir_index_lists_dict.update(sequence_dir_index_lists_dict)
            self.sequence_slice_index_lists_dict.update(sequence_slice_index_lists_dict)

    def check_overlap_sequence_dir(self, ):
        """
        check if one dir index marked as multiple sequence
        """
        for dir_index in self.dir_index_list:
            detect_sequence = []
            for key in self.sequence_dir_index_lists_dict:
                if dir_index in self.sequence_dir_index_lists_dict[key]:
                    detect_sequence.append(key)
                    if len(detect_sequence) > 1:
                        print(self.patient_ID + ": " + str(dir_index) + " has more than one sequence label of: " + str(
                            detect_sequence))

    def check_overlap_sequence_slice(self, ):
        """
        check if one slice index marked as multiple sequence
        """
        for i in range(len(self.slice_index_list)):
            dir_index = self.dir_index_list[i]
            for slice_i in self.slice_index_list[i]:
                detect_sequence = []
                slice_index_combine = dir_index + [slice_i]
                for key in self.sequence_slice_index_lists_dict:
                    if slice_index_combine in self.sequence_slice_index_lists_dict[key]:
                        detect_sequence.append(key)
                        if len(detect_sequence) > 1:
                            print(self.patient_ID + ": " + str(
                                slice_index_combine) + " has more than one sequence label of: " + str(
                                detect_sequence))


class GeneralLoader(object):
    """
    This class is responsible for loading dicom medical dataset
    """

    def __init__(self, mri_csv_path, analysis_csv_path, dicom_root, patient_id_label, mri_directory_label,
                 pre_filter_labels=None, mri_directory_postfix='', postfix_label='',
                 detect_sequence=None, max_num_patients=None, mode='dicom', info_tags=None, if_cache_image=False,
                 cache_path='image_cache/', series_description_folder = None):
        """
        construct general medical mri dicom loader class
        :param mri_csv_path: path to mri csv file
        :param analysis_csv_path: path to analysis csv file
        :param dicom_root: path to dicom files root folder
        :param patient_id_label: string, patient ID for each unique patient case
        :param pre_filter_labels: list, pre filtering labels for both mri and analysis csv
        :param detect_sequence:
        """
        self.mri_csv_path = mri_csv_path
        self.analysis_csv_path = analysis_csv_path
        self.dicom_root = dicom_root
        self.patient_id_label = patient_id_label
        self.mri_directory_label = mri_directory_label
        self.mri_csv = self.read_csv(mri_csv_path)
        self.analysis_csv = self.read_csv(analysis_csv_path)
        self.mri_directory_postfix = mri_directory_postfix
        self.postfix_label = postfix_label
        self.detect_sequence = detect_sequence
        self.cache_path = cache_path
        self.series_description_folder = series_description_folder

        self.max_num_patients = max_num_patients
        self.mode = mode
        self.if_cache_image = if_cache_image
        self.info_tags = info_tags
        if pre_filter_labels is not None:
            self.pre_filter_csv_by_labels(pre_filter_labels)

        self.patients = self.create_patients_data()

    def read_csv(self, filepath, target_type=str):
        """
        read csv file to data frame
        :param filepath: csv file path
        :param target_type: if cannot recognize column labels, force to read it as string
        :return:
        """
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")

            mydata = pd.read_csv(filepath, header=0)
            print("Warnings raised:", ws)
            # We have an error on specific columns, try and load them as string
            for w in ws:
                s = str(w.message)
                print("Warning message:", s)
                match = re.search(r"Columns \(([0-9,]+)\) have mixed types\.", s)
                if match:
                    columns = match.group(1).split(',')  # Get columns as a list
                    columns = [int(c) for c in columns]
                    print("Applying %s dtype to columns:" % target_type, columns)
                    mydata.iloc[:, columns] = mydata.iloc[:, columns].astype(target_type)
        return mydata

    def pre_filter_csv_by_labels(self, labels):
        """
        pre-filter csv by labels
        :param labels: given labels to only include in mri and analysis csv
        """
        labels_mri = []
        labels_analysis = []
        for i in range(len(labels)):
            label = labels[i]
            if label in self.mri_csv.columns:
                labels_mri.append(label)
            if label in self.analysis_csv.columns:
                labels_analysis.append(label)
        self.mri_csv = self.mri_csv.loc[:, labels_mri]
        self.analysis_csv = self.analysis_csv.loc[:, labels_analysis]

    def add_csv_by_labels(self, labels):
        """
        add csv by labels, labels must exist in original mri and analysis csv
        :param labels: given labels to only include in mri and analysis csv
        """
        ori_mri_csv = self.read_csv(self.mri_csv_path)
        ori_analysis_csv = self.read_csv(self.analysis_csv_path)
        for i in range(len(labels)):
            label = labels[i]
            if label not in self.mri_csv.columns and label in ori_mri_csv.columns:
                self.mri_csv[label] = ori_mri_csv[label]
            if label not in self.analysis_csv.columns and label in ori_analysis_csv.columns:
                self.analysis_csv[label] = ori_analysis_csv[label]
        # update patients
        for i in tqdm(range(len(self.patients))):
            rslt_df_mri = self.mri_csv.loc[self.mri_csv[self.patient_id_label] == self.patients[i].patient_ID]
            rslt_df_analysis = self.analysis_csv.loc[self.analysis_csv[self.patient_id_label] == self.patients[i].patient_ID]

            self.patients[i].mriDF = rslt_df_mri
            self.patients[i].analysisDF = rslt_df_analysis

    def create_patients_data(self):
        """
        create patients list
        :return: lists contains patient cases class
        """
        # print(type(self.mri_csv[self.patient_id_label]))
        # print(self.mri_csv[self.patient_id_label])
        patientIDs = self.mri_csv[self.patient_id_label].unique()
        if self.max_num_patients is not None:
            patientIDs = patientIDs[0:self.max_num_patients]
        patients = []
        for i in tqdm(range(len(patientIDs))):
            patientID = patientIDs[i]
            rslt_df_mri = self.mri_csv.loc[self.mri_csv[self.patient_id_label] == patientID]
            rslt_df_analysis = self.analysis_csv.loc[self.analysis_csv[self.patient_id_label] == patientID]
            patient = PatientCase(patientID, self.patient_id_label, self.mri_directory_label, rslt_df_mri,
                                  rslt_df_analysis, self.dicom_root, self.mri_directory_postfix, self.postfix_label,
                                  self.detect_sequence,
                                  self.mode, self.info_tags, self.if_cache_image, self.cache_path)
            patients.append(patient)
        return patients

    def filter_patients_by_labels(self, labels, label_values, DF=None, if_or=False, if_update=False):
        """
        filter patients by given labels
        :param labels: filtering labels
        :param label_values: matching label values
        :param DF: only filter from a given dataframe, otherwise None
        :param if_or: if choose or filter instead of and filter
        :param if_update: if update dataframe and patient cases
        :return: patients list
        """
        patients = []
        result_mri_csv = pd.DataFrame(columns=self.mri_csv.columns)
        result_analysis_csv = pd.DataFrame(columns=self.analysis_csv.columns)
        if not if_or:
            for i in tqdm(range(len(self.patients))):
                result_mriDF, result_analysisDF = self.patients[i].get_DF_by_labels_and(labels=labels,
                                                                                        label_values=label_values,
                                                                                        DF_name=DF)
                if not result_mriDF.empty or not result_analysisDF.empty:
                    patients.append(self.patients[i])
                    result_mri_csv = pd.concat([result_mri_csv, result_mriDF], axis=0)
                    result_analysis_csv = pd.concat([result_analysis_csv, result_analysisDF], axis=0)
        elif if_or:
            for i in tqdm(range(len(self.patients))):
                result_mriDF, result_analysisDF = self.patients[i].get_DF_by_labels_or(labels=labels,
                                                                                       label_values=label_values,
                                                                                       DF_name=DF)
                if not result_mriDF.empty or not result_analysisDF.empty:
                    patients.append(self.patients[i])
                    result_mri_csv = pd.concat([result_mri_csv, result_mriDF], axis=0)
                    result_analysis_csv = pd.concat([result_analysis_csv, result_analysisDF], axis=0)
        if if_update:
            self.update_self(mriDF=result_mri_csv, analysisDF=result_analysis_csv, patients=patients)

        return patients

    def get_patients_by_ID(self, patient_ID):
        """
        get patient case by patient_ID
        :param patient_ID: string, patient_ID
        :return: patient case
        """
        for patient in self.patients:
            if patient.get_patient_ID() == patient_ID:
                return patient

    def get_patients_index_by_ID(self, patient_ID):
        """
        get patient case by patient_ID
        :param patient_ID: string, patient_ID
        :return: patient case
        """
        for i in range(len(self.patients)):
            patient = self.patients[i]
            if patient.get_patient_ID() == patient_ID:
                return i

    def update_self(self, mriDF=None, analysisDF=None, patients=None):
        """
        update the mriDF, analysisDF, and patients
        :param mriDF: dataframe containing new mri data
        :param analysisDF: dataframe containing new analysis data
        :param patients: list of patient cases
        """
        if mriDF is not None:
            self.mri_csv = mriDF
        if analysisDF is not None:
            self.analysis_csv = mriDF
        if patients is not None:
            self.patients = patients

    def filter_patients_by_threshold_mri_death_interval(self, threshold, unit='month', if_update=False):
        """
        filter the patients by death interval
        :param threshold: int, time threshold from latest mri to death
        :param unit: time unit, 'day', 'month, 'year'
        :param if_update: if update the dataloader
        :return: patients list
        """
        patients = []
        result_mri_csv = pd.DataFrame(columns=self.mri_csv.columns)
        result_analysis_csv = pd.DataFrame(columns=self.analysis_csv.columns)
        for i in tqdm(range(len(self.patients))):
            time_interval = self.patients[i].get_time_from_latest_mri_to_death(unit=unit)
            if time_interval is not None:
                if time_interval <= threshold:
                    result_mriDF = self.patients[i].mriDF
                    result_analysisDF = self.patients[i].analysisDF
                    patients.append(self.patients[i])
                    result_mri_csv = pd.concat([result_mri_csv, result_mriDF], axis=0)
                    result_analysis_csv = pd.concat([result_analysis_csv, result_analysisDF], axis=0)
        if if_update:
            self.update_self(mriDF=result_mri_csv, analysisDF=result_analysis_csv, patients=patients)

        return patients

    def filter_patients_by_MRI_exist(self, if_update=False):
        """
        Filter patients by MRI existence
        :param if_update: if update loader
        :return: patients list with MRI marked as existed
        """
        patients = []
        result_mri_csv = pd.DataFrame(columns=self.mri_csv.columns)
        result_analysis_csv = pd.DataFrame(columns=self.analysis_csv.columns)
        for i in tqdm(range(len(self.patients))):
            if self.patients[i].ifHasMRI:
                result_mriDF = self.patients[i].mriDF
                result_analysisDF = self.patients[i].analysisDF
                result_mri_csv = pd.concat([result_mri_csv, result_mriDF], axis=0)
                result_analysis_csv = pd.concat([result_analysis_csv, result_analysisDF], axis=0)
                patients.append(self.patients[i])
        if if_update:
            self.update_self(mriDF=result_mri_csv, analysisDF=result_analysis_csv, patients=patients)
        return patients
