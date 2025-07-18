import json
import os
import re
import time
import warnings
import concurrent.futures
import multiprocessing
from multiprocessing import Pool, cpu_count
import math
import matplotlib.pylab as plt
import pandas as pd
import pydicom as dicom
from tqdm import tqdm
from copy import deepcopy
import time
from .general_utils import (fixPath, load_sorted_directory, sort_list_natural_keys, get_time_interval,
                            check_if_dir_list, check_if_dir)


def check_valid_slice(slice_name, slice_index, slice_path, tag_code=None):
    # if tag_code is None:
    #     tag_code = [0x0008, 0x103E]
    if os.path.exists(fixPath(slice_path)):
        # print('check')
        try:
            ds = dicom.dcmread(fixPath(slice_path))
            if ds is not None:
                try:
                    slice_info = None
                    if tag_code is not None:
                        if ds.__contains__(tag_code):
                            slice_info = ds[tag_code].value

                    image_array = ds.pixel_array
                    shape = image_array.shape
                    if len(shape) == 2:
                        return [slice_name], [slice_index], [slice_info]  # single gray scale
                    elif len(shape) == 3:
                        if shape[-1] == 3:
                            return [slice_name], [slice_index], [slice_info]  # single rgb
                        else:
                            slice_names = []
                            slice_indexes = []
                            slice_infos = []
                            for i in range(shape[0]):  # multiple gray scale
                                slice_names.append([slice_name, ''])
                                slice_indexes.append([slice_index, i])
                                slice_infos.append([slice_info, ''])
                            return slice_names, slice_indexes, slice_infos
                    elif len(shape) == 4:
                        slice_names = []
                        slice_indexes = []
                        slice_infos = []
                        for i in range(shape[0]):  # multiple rbg
                            slice_names.append([slice_name, ''])
                            slice_indexes.append([slice_index, i])
                            slice_infos.append([slice_info, ''])
                        return slice_names, slice_indexes, slice_infos
                except Exception:
                    return [], [], []
        except Exception:
            return [], [], []
    return [], [], []


class PatientCase():
    def __init__(self, patient_ID, patient_ID_label, mri_directory_label, mriDF, analysisDF, dicom_root,
                 mri_directory_postfix='',
                 detect_sequence=None, mode='dicom', info_tag=None):
        """
        class that stores one patient case with all information loaded from csv files
        :param patient_ID: string, patient ID
        :param patient_ID_label: string, label stores patient ID
        :param mri_directory_label: string, label stores mri directory for each patient case
        :param mriDF: dataframe, dataframe containing all MRI information
        :param analysisDF: dataframe, dataframe containing all pathology test information
        :param dicom_root: string, root directory of all patients' DICOM files
        :param detect_sequence: string, chosen sequence such as 'T1', 'T2', 'DTI', 'DWI', etc.
        to detect when create patient case, default None
        """

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
        }
        self.combined_tags_dict = {}
        self.mode = mode
        self.info_tag = info_tag
        self.info_tag_code = self.dicom_dict[self.info_tag]
        ### alter for each datset
        # print(patient_ID)
        # time0 = time.time()
        self.mriDF = mriDF
        self.analysisDF = analysisDF
        self.dicom_root = dicom_root
        self.mri_directory_postfix = mri_directory_postfix
        self.latest_mri_session, self.latest_mri_date = self.get_latest_mri_session()  # df, [year, month, day]
        self.dir_name_list, self.dir_index_list, self.dir_name_list_latest, self.dir_index_list_latest = [], [], [], []
        self.slice_name_list, self.slice_index_list, self.slice_info_list = [], [], []
        self.sample_slice_name_list, self.sample_slice_index_list = [], []

        # time1 = time.time()
        # print('time1-0: ' + str(time1 - time0))
        self.update_dir_name_list()
        self.update_slice_lists()
        self.update_sample_slice_index()
        self.ifHasMRI = self.check_MRI_exist()
        # time2 = time.time()
        # print('time2-1: ' + str(time2 - time1))

        # prestore all sequence to unknown dict class\
        # self.dicom_dict = {}
        # self.combined_tags_dict = {}
        # self.update_sequence_series_tags()
        # (
        #     self.sequence_dir_name_lists_dict, self.sequence_dir_index_lists_dict,
        #     self.latest_sequence_dir_name_lists_dict,
        #     self.latest_sequence_dir_index_lists_dict) = (
        #     {'unknown': deepcopy(self.dir_name_list)}, {'unknown': deepcopy(self.dir_index_list)},
        #     {'unknown': deepcopy(self.dir_name_list_latest)}, {'unknown': deepcopy(self.dir_index_list_latest)})

        (
            self.sequence_slice_name_lists_dict, self.sequence_slice_index_lists_dict) = (
            {'unknown': deepcopy(self.slice_name_list)}, {'unknown': deepcopy(self.slice_index_list)})

        self.detect_sequence = detect_sequence
        # time3 = time.time()
        # print('time3-2: ' + str(time3 - time2))
        # update for customize patient loader
        # self.update_sequence_series_tags()
        # self.update_sequence_dir_lists_dict()

        self.PD_echo_time = []

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
            for i in range(len(labels)):
                # check which dataframe contains given labels
                label = labels[i]
                has_mri_labels = False
                has_analysis_labels = False
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
            for i in range(len(labels)):
                label = labels[i]
                has_mri_labels = False
                has_analysis_labels = False
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

    def check_MRI_exist(self):
        """
        Check if the patient has MRI scan recorded
        :return: bool, True if patient has MRI scan, False otherwise
        """
        mri_filepaths = self.mriDF.loc[:, self.mri_directory_label].drop_duplicates()
        for i in range(len(mri_filepaths)):
            mri_filepath = mri_filepaths.iloc[i][:len(mri_filepaths.iloc[i]) - len(self.mri_directory_postfix)]
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
        # dirs_list = sort_list_natural_keys(dirs_list)
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
        # print(dirs_list)
        # dirs_list = sort_list_natural_keys(dirs_list)
        # files_list = sort_list_natural_keys(files_list)
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
                mri_filepath = mri_filepaths.iloc[i][:len(mri_filepaths.iloc[i]) - len(self.mri_directory_postfix)]
                if os.path.exists(fixPath(self.dicom_root + "/" + mri_filepath)):
                    self.print_directory(dir_path=[mri_filepath], indexes=[i])
                else:
                    print("None file found at " + self.dicom_root + "/" + mri_filepath + "/")
        else:
            print("Patient has no valid MRI session")

    # def load_dicom_mri(self, index, filename_list=None):
    #     """
    #     Load DICOM given index
    #     :param index: [sequence index, slice num]
    #     :param filename_list: listed dicom filenames if given, default None, and load all dicom file in session
    #     :return: dicom datasets, return None if no dicom file is loaded
    #     """
    #     filepath_temp = self.get_file_path_given_sequence(index[:-1])
    #
    #     if os.path.exists(fixPath(filepath_temp)):
    #         if filename_list is None:
    #             list1 = load_sorted_directory(filepath_temp)
    #         else:
    #             list1 = filename_list
    #         if list1 is not None:
    #             if len(list1) > 0:  # avoid empty file
    #                 filepath_temp = filepath_temp + '/' + list1[index[-1]]
    #             else:
    #                 return None
    #         else:
    #             return None
    #
    #         try:
    #             ds = dicom.dcmread(fixPath(filepath_temp))
    #             return ds
    #         except Exception:
    #             return None  # or you could use 'continue'
    #     else:
    #         return None

    def load_dicom_mri(self, index):
        """
        Load DICOM given index
        :param index: [sequence index, slice num]
        :param filename_list: listed dicom filenames if given, default None, and load all dicom file in session
        :return: dicom datasets, return None if no dicom file is loaded
        """
        filepath_temp = self.get_file_path_given_slice(index)

        if os.path.exists(fixPath(filepath_temp)):
            try:
                ds = dicom.dcmread(fixPath(filepath_temp))
                return ds
            except Exception:
                return None  # or you could use 'continue'
        else:
            return None

    def get_file_index_list_in_sequence(self, index):
        """
        Load mri slice index to a list given sequence index, output along index
        :param index: sequence index list [0, 0, 0]
        :return: index list [[0, 0, 0, 0], [0, 0, 0, 1], ..., [0, 0, 0, n]], None if no file index is loaded
        """
        if index in self.dir_index_list:
            file_idx = self.dir_index_list.index(index)
            slice_index_list_sequence = self.slice_index_list[file_idx]  # [0, 1,..., n]

        return [index + [slice_index] for slice_index in slice_index_list_sequence]

    def get_slice_list_in_dir(self, index):
        """
        Load mri slice index and name to a list given sequence index, without sequence index
        :param index: sequence index list [0, 0, 0]
        :return: tuple: name list [0.dcm, 1.dcm, 2.dcm, ..., n.dcm], [] if no file is loaded
         index list [0, 1, 2, ..., n], [] if no file is loaded
        """
        file_idx = self.dir_index_list.index(index)
        filepath_temp = self.dicom_root + "/" + "/".join(self.dir_name_list[file_idx])
        slice_names_temp = []
        slice_indices_temp = []
        slice_infos_temp = []
        if os.path.exists(fixPath(filepath_temp)):
            list1 = load_sorted_directory(filepath_temp)
            if list1 is not None:
                if len(list1) > 0:  # avoid empty file
                    slice_indices = range(len(list1))
                    slice_paths = [filepath_temp + "/" + slice_name for slice_name in list1]
                    slice_tags = [self.info_tag_code] * len(list1)
                    # time_get_slice_list_in_sequence0 = time.time()

                    # print(self.patient_ID + ": " + str(len(list1)))
                    # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                    #     for slice_name_valid, slice_index_valid in executor.map(check_valid_slice, list1,
                    #                                                             slice_indices, slice_paths):
                    #         slice_names_temp = slice_names_temp + slice_name_valid
                    #         slice_indices_temp = slice_indices_temp + slice_index_valid

                    # print(self.patient_ID + ": " + str(len(list1)))
                    if len(list1) > 1200:
                        with Pool(processes=10) as pool:
                            for slice_name_valid, slice_index_valid, slice_info_valid in pool.starmap(check_valid_slice,
                                                                                                      zip(list1,
                                                                                                          slice_indices,
                                                                                                          slice_paths,
                                                                                                          slice_tags)):
                                slice_names_temp = slice_names_temp + slice_name_valid
                                slice_indices_temp = slice_indices_temp + slice_index_valid
                                slice_infos_temp = slice_infos_temp + slice_info_valid

                    # n_cores = 10
                    # q_in = [multiprocessing.Queue() for i in range(n_cores - 1)]
                    # read_processes = [
                    #     multiprocessing.Process(target=check_valid_slice, args=(list1[i], q_in[i])) for i in range(n_cores - 1)]

                    # print(self.patient_ID + ": " + str(len(list1)))
                    else:
                        for i in range(len(list1)):
                            slice_name_valid, slice_index_valid, slice_info_valid = check_valid_slice(list1[i],
                                                                                                      slice_indices[i],
                                                                                                      slice_paths[i],
                                                                                                      slice_tags[i])
                            slice_names_temp = slice_names_temp + slice_name_valid
                            slice_indices_temp = slice_indices_temp + slice_index_valid
                            slice_infos_temp = slice_infos_temp + slice_info_valid

                    # for i in range(len(list1)):
                    #     slice_names_temp.append(list1[i])
                    #     slice_indices_temp.append(i)

                    # time_get_slice_list_in_sequence1 = time.time()
                    # print("time_get_slice_list_in_sequence: " + str(
                    #     time_get_slice_list_in_sequence1 - time_get_slice_list_in_sequence0))
        return slice_names_temp, slice_indices_temp, slice_infos_temp

    def update_slice_lists(self):
        """
        Update the self.slice_name_list and self.slice_index_list
        """
        self.slice_name_list, self.slice_index_list = [], []
        for dir_i in self.dir_index_list:
            slice_names_temp, slice_indices_temp, slice_infos_temp = self.get_slice_list_in_dir(dir_i)
            self.slice_name_list.append(slice_names_temp)
            self.slice_index_list.append(slice_indices_temp)
            self.slice_info_list.append(slice_infos_temp)

    def update_sample_slice_index(self):
        """
        Update the sample slice index and name to a list given sequence index, without sequence
        :return: list of names [1.dcm, 1.dcm, 2.dcm, ..., a.dcm], list of int, [1, 1, 2, ..., a],
        """
        self.sample_slice_name_list = []
        self.sample_slice_index_list = []
        for sequence in self.dir_index_list:
            slice_name_valid, slice_index_valid = self.get_first_mri_valid(sequence)
            self.sample_slice_name_list.append(slice_name_valid)
            self.sample_slice_index_list.append(slice_index_valid)

    def get_latest_mri_session(self, if_update=False):
        """
        Get latest mri session
        :param if_update: if update current mri dataframe to dataframe with only latest mri session
        :return: mri dataframe with latest mri, date in ['year', 'mouth', 'day']
        """
        return None, None

    def load_first_dicom_mri_by_session(self, index, mri_session):
        """
        Load first dicom file given a mri session dataframe
        :param index: [sequence index, slice num]
        :param mri_session: mri session dataframe
        :return: dicom datasets, return None if no dicom file is loaded
        """
        MRI_filepath = mri_session[self.mri_directory_label].iloc[0]
        filepath_temp = self.dicom_root + "/" + MRI_filepath[:len(MRI_filepath) - len(self.mri_directory_postfix)]
        if os.path.exists(fixPath(filepath_temp)):
            list1 = load_sorted_directory(filepath_temp)
            idx = 0
            for j in range(len(index)):
                if check_if_dir_list(filepath_temp, list1):
                    filepath_temp = filepath_temp + "/" + list1[index[idx]]
                    list1 = load_sorted_directory(filepath_temp)
                    idx = idx + 1
            filepath_temp = filepath_temp + "/" + list1[index[idx]]
            ds = dicom.dcmread(fixPath(filepath_temp))
            return ds
        else:
            return None

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
                mri_filepath = mri_filepaths.iloc[i][:len(mri_filepaths.iloc[i]) - len(self.mri_directory_postfix)]
                if os.path.exists(fixPath(self.dicom_root + "/" + mri_filepath)):
                    # update self.dir_name_list and self.dir_index_list
                    self.list_directory(dir_path=[mri_filepath], indexes=[i])

            # update self.dir_name_list_latest and self.dir_index_list_latest
            for i in range(len(self.dir_name_list)):
                if len(mri_latest_filepath) > 0:
                    if mri_latest_filepath.iloc[0][
                       :len(mri_latest_filepath.iloc[0]) - len(self.mri_directory_postfix)] == self.dir_name_list[i][0]:
                        self.dir_name_list_latest.append(self.dir_name_list[i])
                        self.dir_index_list_latest.append(self.dir_index_list[i])

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

    def get_sequence_dir_lists(self, sequence_name, ifLatest=False, ifExact=True):
        """
        get sequence directory lists based on sequence name recorded in anaylsis dataframe
        :param sequence_name: string, 'T1', 'T2', ''DWI'
        :param ifLatest: bool, if only check on the latest sequence directories, default is False
        :return: (dir_name_list, dir_index_list) for matching sequence directories.
        """
        if ifLatest:
            # temp_mriDF = self.latest_mri_session
            temp_dir_name_list = self.dir_name_list_latest
            temp_dir_index_list = self.dir_index_list_latest
        else:
            # temp_mriDF = self.mriDF
            temp_dir_name_list = self.dir_name_list
            temp_dir_index_list = self.dir_index_list

        # mri_filepaths = temp_mriDF.loc[:, self.mri_directory_label].drop_duplicates()

        tag_idx = []
        Tag_dict_temp = self.combined_tags_dict[sequence_name]
        # time_get_sequence_dir_lists0 = time.time()
        if ifExact:
            name_idx = [x for x in range(len(temp_dir_name_list)) for y in range(len(temp_dir_name_list[x])) for
                        sequence_tag in Tag_dict_temp['DirName'] if
                        sequence_tag.upper() == temp_dir_name_list[x][y].upper()]
        else:
            name_idx = [x for x in range(len(temp_dir_name_list)) for y in range(len(temp_dir_name_list[x])) for
                        sequence_tag in Tag_dict_temp['DirName'] if
                        sequence_tag.upper() in temp_dir_name_list[x][y].upper()]

        for i in range(len(temp_dir_index_list)):
            # time_get_sequence_dir_lists1 = time.time()
            # temp_valid_mri = self.get_first_mri_valid(temp_dir_index_list[i])
            temp_valid_mri = self.sample_slice_index_list[i]
            # time_get_sequence_dir_lists2 = time.time()
            # if (time_get_sequence_dir_lists2 - time_get_sequence_dir_lists1) > 0.1:
            #     print("time_get_sequence_dir_lists2-1: " + str(i) + "/" + str(len(temp_dir_index_list)) + " " + str(
            #         time_get_sequence_dir_lists2 - time_get_sequence_dir_lists1))
            if temp_valid_mri is not None:
                temp_index = temp_dir_index_list[i] + [temp_valid_mri]
                dicom_data = self.load_dicom_mri(index=temp_index)
                if dicom_data is not None:
                    for key in self.dicom_dict:  # check by tag dicts
                        if dicom_data.__contains__(self.dicom_dict[key]) and key in Tag_dict_temp.keys():
                            if ifExact:
                                if any(x.upper() == dicom_data[self.dicom_dict[key]].value.upper() for x in
                                       Tag_dict_temp[key]):
                                    tag_idx.append(i)
                            else:
                                if any(x.upper() in dicom_data[self.dicom_dict[key]].value.upper() for x in
                                       Tag_dict_temp[key]):
                                    tag_idx.append(i)

        final_idx = name_idx + tag_idx
        final_idx = list(set(final_idx))
        return [temp_dir_name_list[k] for k in final_idx], [temp_dir_index_list[k] for k in final_idx]

    def get_sequence_by_dir_index(self, dir_index, ifLatest=False, ifExact=True):
        detect_sequence = []
        if ifLatest:
            temp_dir_name_list = self.dir_name_list_latest
            temp_dir_index_list = self.dir_index_list_latest
        else:
            temp_dir_name_list = self.dir_name_list
            temp_dir_index_list = self.dir_index_list

            if dir_index in temp_dir_index_list:
                idx = temp_dir_index_list.index(dir_index)
                dir_name = temp_dir_name_list[idx]

                # get sequence by dir name
                for sequence_name in self.detect_sequence:
                    Tag_dict_temp = self.combined_tags_dict[sequence_name]
                    for sequence_tag in Tag_dict_temp['DirName']:
                        if ifExact:
                            if any(sequence_tag.upper() == x.upper() for x in dir_name):
                                detect_sequence.append(sequence_name)
                                break
                        else:
                            if any(sequence_tag.upper() in x.upper() for x in dir_name):
                                detect_sequence.append(sequence_name)
                                break

                # get sequence by dicom tags
                temp_valid_mri = self.sample_slice_index_list[idx]
                if temp_valid_mri is not None:
                    temp_index = dir_index + [temp_valid_mri]
                    dicom_data = self.load_dicom_mri(index=temp_index)
                    if dicom_data is not None:
                        for sequence_name in self.detect_sequence:
                            Tag_dict_temp = self.combined_tags_dict[sequence_name]
                            for key in self.dicom_dict:  # check by tag dicts
                                if dicom_data.__contains__(self.dicom_dict[key]) and key in Tag_dict_temp.keys():
                                    if ifExact:
                                        if any(x.upper() == dicom_data[self.dicom_dict[key]].value.upper() for x in
                                               Tag_dict_temp[key]):
                                            detect_sequence.append(sequence_name)
                                            break
                                    else:
                                        if any(x.upper() in dicom_data[self.dicom_dict[key]].value.upper() for x in
                                               Tag_dict_temp[key]):
                                            detect_sequence.append(sequence_name)
                                            break
        detect_sequence = list(set(detect_sequence))
        return detect_sequence

    def get_sequence_by_slice_index(self, slice_index, ifLatest=False, ifExact=True):
        detect_sequence = []
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
                        if any(sequence_tag.upper() == x.upper() for x in dir_name):
                            detect_sequence.append(sequence_name)
                            break
                    else:
                        if any(sequence_tag.upper() in x.upper() for x in dir_name):
                            detect_sequence.append(sequence_name)
                            break
            # get sequence by info tags
            if slice_index[-1] in self.slice_index_list[file_idx]:
                slice_idx = (self.slice_index_list[file_idx]).index(slice_index[-1])
                slice_info = self.slice_info_list[file_idx][slice_idx]
                if slice_info is not None:
                    for sequence_name in self.detect_sequence:
                        Tag_dict_temp = self.combined_tags_dict[sequence_name]
                        if isinstance(slice_info, str):
                            if ifExact:
                                if any(x.upper() == slice_info.upper() for x in Tag_dict_temp[self.info_tag]):
                                    detect_sequence.append(sequence_name)
                            else:
                                if any(x.upper() in slice_info.upper() for x in Tag_dict_temp[self.info_tag]):
                                    detect_sequence.append(sequence_name)
                        else:
                            if ifExact:
                                if any(x.upper() == slice_info[0].upper() for x in Tag_dict_temp[self.info_tag]):
                                    detect_sequence.append(sequence_name)
                            else:
                                if any(x.upper() in slice_info[0].upper() for x in Tag_dict_temp[self.info_tag]):
                                    detect_sequence.append(sequence_name)

        detect_sequence = list(set(detect_sequence))
        return detect_sequence

    def update_pd_echo_time(self):
        self.PD_echo_time = []
        for i, dir_index in enumerate(self.sequence_dir_index_lists_dict['PD']):
            self.PD_echo_time.append([])
            idx = self.dir_index_list.index(dir_index)
            slice_index = self.slice_index_list[idx]
            for slice_i in slice_index:
                temp_index = dir_index + [slice_i]
                dicom_data = self.load_dicom_mri(index=temp_index)
                if dicom_data is not None:
                    if dicom_data.__contains__(self.dicom_dict['EchoTime']):
                        echo_time = dicom_data[self.dicom_dict['EchoTime']].value
                        if echo_time not in self.PD_echo_time[i]:
                            self.PD_echo_time[i].append(echo_time)

    def get_sequence_dir_lists_dict(self, sequence_names, if_latest=False, ifExact=True):
        sequence_dir_name_lists_dict, sequence_dir_index_lists_dict = {}, {}

        for sequence_name in sequence_names:
            sequence_dir_name_lists_dict[sequence_name], sequence_dir_index_lists_dict[sequence_name] = (
                [], [])

        for idx, dir_index in enumerate(self.dir_index_list):
            detect_sequence = self.get_sequence_by_dir_index(dir_index, if_latest, ifExact)
            if len(detect_sequence) > 1:
                print(self.patient_ID + ": " + str(dir_index) + " has more than one sequence label of: " + str(
                    detect_sequence))
            for sequence_name in detect_sequence:
                sequence_dir_name_lists_dict[sequence_name].append(self.dir_name_list[idx])
                sequence_dir_index_lists_dict[sequence_name].append(dir_index)

        return sequence_dir_name_lists_dict, sequence_dir_index_lists_dict

    def get_sequence_slice_lists_dict(self, sequence_names, if_latest=False, ifExact=True):
        sequence_slice_name_lists_dict, sequence_slice_index_lists_dict = {}, {}

        for sequence_name in sequence_names:
            sequence_slice_name_lists_dict[sequence_name], sequence_slice_index_lists_dict[sequence_name] = (
                [], [])

        for idx, dir_index in enumerate(self.dir_index_list):
            for slice_index in self.slice_index_list[idx]:
                slice_index_combine = dir_index + [slice_index]
                detect_sequence = self.get_sequence_by_slice_index(slice_index_combine, if_latest, ifExact)
            if len(detect_sequence) > 1:
                print(self.patient_ID + ": " + str(dir_index) + " has more than one sequence label of: " + str(
                    detect_sequence))
            for sequence_name in detect_sequence:
                sequence_slice_name_lists_dict[sequence_name].append(self.dir_name_list[idx])
                sequence_slice_index_lists_dict[sequence_name].append(dir_index)

        return sequence_dir_name_lists_dict, sequence_dir_index_lists_dict

    def update_sequence_dir_lists_dict(self):
        if self.detect_sequence is not None:
            # time_update_sequence_dir_lists_dict0 = time.time()
            sequence_dir_name_lists_dict, sequence_dir_index_lists_dict = self.get_sequence_dir_lists_dict(
                sequence_names=self.detect_sequence)
            latest_sequence_dir_name_lists_dict, latest_sequence_dir_index_lists_dict = self.get_sequence_dir_lists_dict(
                sequence_names=self.detect_sequence, if_latest=True)
            # time_update_sequence_dir_lists_dict1 = time.time()
            # if (time_update_sequence_dir_lists_dict1 - time_update_sequence_dir_lists_dict0) > 5:
            #     print("time_update_sequence_dir_lists_dict0-1: ", str(time_update_sequence_dir_lists_dict1 - time_update_sequence_dir_lists_dict0))
            # remove detected sequence from 'unknown'
            for key in sequence_dir_index_lists_dict:
                for x in sequence_dir_index_lists_dict[key]:
                    if x in self.sequence_dir_index_lists_dict['unknown']:
                        self.sequence_dir_index_lists_dict['unknown'].remove(x)
                for x in sequence_dir_name_lists_dict[key]:
                    if x in self.sequence_dir_name_lists_dict['unknown']:
                        self.sequence_dir_name_lists_dict['unknown'].remove(x)
            for key in latest_sequence_dir_index_lists_dict:
                for x in latest_sequence_dir_index_lists_dict[key]:
                    if x in self.latest_sequence_dir_index_lists_dict['unknown']:
                        self.latest_sequence_dir_index_lists_dict['unknown'].remove(x)
                for x in latest_sequence_dir_name_lists_dict[key]:
                    if x in self.latest_sequence_dir_name_lists_dict['unknown']:
                        self.latest_sequence_dir_name_lists_dict['unknown'].remove(x)

            self.sequence_dir_name_lists_dict.update(sequence_dir_name_lists_dict)
            self.sequence_dir_index_lists_dict.update(sequence_dir_index_lists_dict)
            self.latest_sequence_dir_name_lists_dict.update(latest_sequence_dir_name_lists_dict)
            self.latest_sequence_dir_index_lists_dict.update(latest_sequence_dir_index_lists_dict)

    def check_overlap_sequence_dir(self, ):
        for dir_index in self.dir_index_list:
            detect_sequence = []
            for key in self.sequence_dir_index_lists_dict:
                if dir_index in self.sequence_dir_index_lists_dict[key]:
                    detect_sequence.append(key)
                    if len(detect_sequence) > 1:
                        print(self.patient_ID + ": " + str(dir_index) + " has more than one sequence label of: " + str(
                            detect_sequence))

    def get_sequence_sessions(self, sequence, ifLatest=False, ifUpadte=False):
        """
        get MRI sessions that contains at least one give sequence by checking mri data frames only
        :param sequence: given sequence name
        :param ifLatest: if only check latest MRI session
        :param ifUpadte: if update current data frame with detected sessions only
        :return: (sequence_sessions_dir_name_list, sequence_sessions_dir_index_list) for matching session directories.
        """
        latest_temp_mriDF = self.latest_mri_session
        latest_temp_dir_name_list = self.dir_name_list_latest
        temp_mriDF = self.mriDF
        temp_dir_name_list = self.dir_name_list
        temp_dir_index_list = self.dir_index_list

        # all mri session
        mri_filepaths = temp_mriDF.loc[:, self.mri_directory_label].drop_duplicates()
        # check if has MRI sequence recorded
        sequence_sessions_dir_name_list = []
        sequence_sessions_dir_index_list = []
        new_mriDF_list = []
        for j in range(len(mri_filepaths)):
            if temp_mriDF.iloc[j]['MRI' + sequence] == 1 and self.ifHasMRI:
                file_name = mri_filepaths.iloc[j][:len(mri_filepaths.iloc[j]) - len(self.mri_directory_postfix)]
                for i in range(len(temp_dir_name_list)):
                    if file_name == temp_dir_name_list[i][0]:
                        file_index = temp_dir_index_list[i][0]
                        if self.check_if_session_valid([file_index]):
                            sequence_sessions_dir_index_list.append(file_index)
                            sequence_sessions_dir_name_list.append(file_name)
                            new_mriDF_list.append(pd.DataFrame(temp_mriDF.iloc[j]).transpose())
                        break
        if len(new_mriDF_list) > 0:
            new_mriDF = pd.concat(new_mriDF_list, axis=0)

        # latest mri session
        latest_mri_filepaths = latest_temp_mriDF.loc[:, self.mri_directory_label].drop_duplicates()
        # check if has MRIT1 recorded
        latest_sequence_sessions_dir_name_list = []
        latest_sequence_sessions_dir_index_list = []
        latest_new_mriDF_list = []
        for j in range(len(latest_mri_filepaths)):
            if latest_temp_mriDF.iloc[j]['MRI' + sequence] == 1 and self.ifHasMRI:
                file_name = latest_mri_filepaths.iloc[j][
                            :len(latest_mri_filepaths.iloc[j]) - len(self.mri_directory_postfix)]
                for i in range(len(latest_temp_dir_name_list)):
                    if file_name == latest_temp_dir_name_list[i][0]:
                        file_index = temp_dir_index_list[i][0]
                        if self.check_if_session_valid([file_index]):
                            latest_sequence_sessions_dir_index_list.append(file_index)
                            latest_sequence_sessions_dir_name_list.append(file_name)
                            latest_new_mriDF_list.append(pd.DataFrame(latest_temp_mriDF.iloc[j]).transpose())
                        break

        if len(latest_new_mriDF_list) > 0:
            latest_new_mriDF = pd.concat(latest_new_mriDF_list, axis=0)

        if ifUpadte:
            self.latest_mri_session = latest_new_mriDF
            self.mriDF = new_mriDF

        if ifLatest:
            return latest_sequence_sessions_dir_name_list, latest_sequence_sessions_dir_index_list
        else:
            return sequence_sessions_dir_name_list, sequence_sessions_dir_index_list

    def check_if_index_valid(self, index):
        """
        check if index is valid for loading dicom type mri images
        :param index: [dir_index, slice index]
        :return: bool, True if index is valid for loading dicom type
        """
        ds = self.load_dicom_mri(index)
        if ds is not None:
            try:
                image = ds.pixel_array
                return True
            except:
                return False
        else:
            return False

    def check_if_sequence_valid(self, sequence_index):
        """
        check if sequence is valid for loading dicom type mri images
        :param sequence_index: [dir_index]
        :return: bool, True if any slice in directory is valid for loading dicom type
        """
        sequence_length = self.get_sequence_length(sequence_index)
        for i in range(sequence_length):
            ds = self.load_dicom_mri(sequence_index + [i])
            if ds is not None:
                try:
                    image = ds.pixel_array
                    return True
                except:
                    continue
        return False

    def get_file_path_given_sequence(self, sequence_index):
        """
        get file path for given sequence index
        :param sequence_index: [dir_index]
        :return: string, file path
        """
        if sequence_index in self.dir_index_list:
            file_idx = self.dir_index_list.index(sequence_index)
            filepath_temp1 = self.dicom_root + "/" + "/".join(self.dir_name_list[file_idx])
            return filepath_temp1
        else:
            return ''

    def get_file_path_given_slice(self, slice_index):
        """
        get file path for given slice index
        :param slice_index: [dir index, slice index]
        :return: string, file path
        """
        sequence_index = slice_index[0:-1]
        if sequence_index in self.dir_index_list:
            file_idx = self.dir_index_list.index(sequence_index)
            if slice_index[-1] in self.slice_index_list[file_idx]:
                slice_idx = (self.slice_index_list[file_idx]).index(slice_index[-1])
                filepath_temp1 = self.dicom_root + "/" + "/".join(self.dir_name_list[file_idx])
                slice_name = self.slice_name_list[file_idx][slice_idx]
                if isinstance(slice_name, str):
                    filepath_temp1 = filepath_temp1 + '/' + slice_name
                else:
                    filepath_temp1 = filepath_temp1 + '/' + slice_name[0]
                return filepath_temp1
            else:
                return ''
        else:
            return ''

    def get_first_mri_valid(self, sequence_index):
        """
        get first mri number that is valid for loading
        :param sequence_index: [dir_index]
        :return: string, int: first valid mri name, number
        """
        # filepath = self.get_file_path_given_sequence(sequence_index)
        # dir_name_list = load_sorted_directory(filepath)
        if sequence_index in self.dir_index_list:
            idx = self.dir_index_list.index(sequence_index)
            slice_names = self.slice_name_list[idx]
            slice_indices = self.slice_index_list[idx]
            if slice_indices is not None:
                for i in slice_indices:
                    ds = self.load_dicom_mri(sequence_index + [i])
                    if ds is not None:
                        try:
                            image = ds.pixel_array
                            idx_name = slice_indices.index(i)
                            return slice_names[idx_name], i
                        except:
                            continue
        return None, None

    def get_first_sequence_given_session_index(self, session_index):
        """
        get first sequence number for given session index
        :param session_index: [session index]
        :return: int, first sequence number
        """
        for i in self.dir_index_list:
            if i[0] == session_index[0]:
                return i
        return None

    def get_all_sequence_given_session_index(self, session_index):
        """
        get all sequence number for given session index
        :param session_index: [session index]
        :return: list, all sequence number of the given session index
        """
        sequence_list = []
        for i in self.dir_index_list:
            if i[0] == session_index[0]:
                sequence_list.append(i)
        return sequence_list

    def get_sequence_length(self, sequence_index):
        """
        get number of slices in a sequence
        :param sequence_index: [dir index]
        :return: int, number of slices
        """
        index = self.dir_index_list.index(sequence_index)
        filepath_temp = '/'.join(self.dir_name_list[index])
        filepath_temp = self.dicom_root + '/' + filepath_temp
        list = load_sorted_directory(filepath_temp)
        if list is not None:
            return len(list)
        else:
            return 0

    def get_session_length(self, session_index):
        """
        get number of sequences in a session
        :param session_index: [session index]
        :return: int, number of sequences
        """
        index = session_index[0]
        filepath_temp1 = self.dicom_root + '/' + self.dir_name_list[index][0]
        list1 = load_sorted_directory(filepath_temp1)
        print(list1)
        if list1 is not None:
            return len(list1)
        else:
            return 0

    def check_if_session_valid(self, session_index):
        """
        check if session has valid dicom mri
        :param session_index: [session index]
        :return: bool, True if contains any valid dicom slice
        """
        dir_index_list = self.get_all_sequence_given_session_index(session_index)
        for index in dir_index_list:
            if self.check_if_sequence_valid(index):
                return True
        return False

    def get_file_path_by_index(self, index):
        """
        get file path by index, deprecate, only use for export jpeg, use get_file_path_given_slice instead
        :param index: [dir index, slice index]
        :return: tuple, (string: filepath, list: filenames)
        """
        mri_filepaths = self.mriDF.loc[:, self.mri_directory_label].drop_duplicates()
        filepath_temp = self.dicom_root + "/" + mri_filepaths.iloc[index[0]][
                                                :len(mri_filepaths.iloc[index[0]]) - len(self.mri_directory_postfix)]
        path_list = [mri_filepaths.iloc[index[0]][:len(mri_filepaths.iloc[index[0]]) - len(self.mri_directory_postfix)]]
        if os.path.exists(fixPath(filepath_temp)):
            list = load_sorted_directory(filepath_temp)
            idx = 0
            for j in range(len(index) - 1):
                if len(list) > 0:  # avoid empty folder
                    if check_if_dir_list(filepath_temp, list):
                        idx = idx + 1
                        filepath_temp = filepath_temp + "/" + list[index[idx]]
                        path_list.append(list[index[idx]])
                        list = load_sorted_directory(filepath_temp)
                else:
                    return None
            idx = idx + 1
            if len(list) > 0:  # avoid empty file
                filepath_temp = filepath_temp + "/" + list[index[idx]]
                path_list.append(list[index[idx]])
                return filepath_temp, path_list
            else:
                return None

    def export_to_jpeg(self, index, path=None, ifSeperateFolder=True, Tags=[]):
        """
        export images to JPEG format
        :param index: [dir index, slice number]
        :param path: file path to save the images, if none, save images to same directory
        :param ifSeperateFolder: if seperate folder is true, save images in separate folders with _jpeg extension
        :param Tags: tags to save in json file
        :return: bool, true if successful save, false if not
        """
        dicom_data = self.load_dicom_mri(index=index)
        if dicom_data is not None:
            try:
                rescaled = dicom_data.pixel_array
            except:
                print("dicom_data.pixel_array corrupted")
                return False

            # load tags to dict
            dict_tag = {"filename0": "", "filename1": "", "filename2": "", "filename3": "", "filename4": ""}
            for tag in Tags:
                # print(tag)
                if dicom_data.__contains__(self.dicom_dict[tag]):
                    dict_tag_value = dicom_data[self.dicom_dict[tag]].value
                    if isinstance(dict_tag_value, dicom.multival.MultiValue):  # check if it is multivalue
                        dict_tag_value_list = [x for x in dict_tag_value]
                        dict_tag[tag] = dict_tag_value_list
                    else:
                        dict_tag[tag] = dict_tag_value

            # add auto T1 to dict
            dir_name_list_T1, dir_index_list_T1 = self.get_T1_file_index()
            if len(dir_index_list_T1) > 0:
                if index[:-1] in dir_index_list_T1:
                    dict_tag['T1_auto'] = "Yes"
            # add manual T1 to dict
            dict_tag['T1_manual'] = ""

            # save jpeg and json tags
            if path is not None:
                try:
                    plt.imsave(path + ".jpeg", rescaled, cmap='gray')
                except:
                    print(path + ".jpeg" + " cannot be saved.")
                return True
            else:
                path, path_list = self.get_file_path_by_index(index)
                if path is not None:
                    # add path name into dict
                    for i in range(len(path_list[:-1])):
                        dict_tag['filename' + str(i)] = path_list[:-1][i]

                    if ifSeperateFolder:
                        path_folder = self.dicom_root + "_jpeg"
                        path_folder = path_folder + "/" + "/".join(path_list[:-1])
                        if not os.path.exists(fixPath(path_folder)):
                            os.makedirs(path_folder)
                        # write tags to json
                        with open(path_folder + "/tag.json", "w") as outfile:
                            json.dump(dict_tag, outfile)

                        path = path_folder + "/" + path_list[-1]
                        try:
                            plt.imsave(path + ".jpeg", rescaled, cmap='gray')
                        except:
                            print(path + ".jpeg" + " cannot be saved.")
                    else:
                        path_folder = self.dicom_root + "/" + "/".join(path_list[:-1])
                        with open(path_folder + "/tag.json", "w") as outfile:
                            json.dump(dict_tag, outfile)
                        try:
                            plt.imsave(path + ".jpeg", rescaled, cmap='gray')
                        except:
                            print(path + ".jpeg" + " cannot be saved.")
                    # Convert and write JSON object to file
                    try:
                        with open(path + ".json", "w") as outfile:
                            json.dump(dict_tag, outfile)
                    except:
                        print(path + ".json" + " cannot be saved.")
                    return True
                else:
                    return False
        else:
            return False

    def seperate_dir_name_list_by_session(self, dir_name_list):
        """
        seperate dir_name_list [[filename1, filename2, filename3] (session 1), [filename4, filename5] (session 2)]
        , into list with first level is session. [[[filename1, filename2, filename3]]], [[filename4, [filename5]]]
        :param dir_name_list: list [[filename1, filename2, filename3], [filename4, filename5]]
        :return: list with first additional level seperated by session
        """
        session_name = dir_name_list[0][0]
        dir_name_list_final = []
        dir_name_list_sub = []
        for i in range(len(dir_name_list)):
            if dir_name_list[i][0] == session_name:
                dir_name_list_sub.append(dir_name_list[i])
            else:
                session_name = dir_name_list[i][0]
                dir_name_list_final.append(dir_name_list_sub)
                dir_name_list_sub = []
                dir_name_list_sub.append(dir_name_list[i])
        dir_name_list_final.append(dir_name_list_sub)
        return dir_name_list_final

    def export_all_to_jpeg(self, dir_name_list=None, dir_index_list=None, ifSeperateFolder=True, Tags=[],
                           MaxNumberOfImage=None,
                           if_latest=False):
        """
        export all given dir_name_list to jpeg
        :param dir_name_list: name list to export, same as self.dir_name_list hierarchy
        :param dir_index_list: corresponding index list to export, same as self.dir_index_list hierarchy
        :param ifSeperateFolder: if seperate folder is true, save images in separate folders with _jpeg extension
        :param Tags: tags to save in json file
        :param MaxNumberOfImage: maximum number of slices to export for each sequence
        :param if_latest: if only export latest images.
        """
        if dir_name_list is None and dir_index_list is None:
            if if_latest:
                dir_name_list = self.dir_name_list_latest
                dir_index_list = self.dir_index_list_latest
            else:
                dir_name_list = self.dir_name_list
                dir_index_list = self.dir_index_list
            # create folder to store json
            dir_name_list_seperate = self.seperate_dir_name_list_by_session(dir_name_list)
            if ifSeperateFolder:
                for i in range(len(dir_name_list_seperate)):
                    path_sequence = self.dicom_root + "_jpeg/" + dir_name_list_seperate[i][0][0]
                    if not os.path.exists(fixPath(path_sequence)):
                        os.makedirs(path_sequence)
                    # writen sequence file list to json
                    with open(path_sequence + "/sequence.json", "w") as outfile:
                        json.dump(dir_name_list_seperate[i], outfile)
            else:
                for i in range(len(dir_name_list_seperate)):
                    path_sequence = self.dicom_root + "/" + dir_name_list_seperate[i][0][0]
                    # writen sequence file list to json
                    with open(path_sequence + "/sequence.json", "w") as outfile:
                        json.dump(dir_name_list_seperate[i], outfile)

            for i in range(len(dir_name_list)):
                path = self.dicom_root + "/" + "/".join(dir_name_list[i])
                list1 = load_sorted_directory(path)
                if MaxNumberOfImage is None:
                    for j in range(len(list1)):
                        self.export_to_jpeg(index=dir_index_list[i] + [j], ifSeperateFolder=ifSeperateFolder, Tags=Tags)
                else:
                    for j in range(min(len(list1), MaxNumberOfImage)):
                        self.export_to_jpeg(index=dir_index_list[i] + [j], ifSeperateFolder=ifSeperateFolder, Tags=Tags)
        else:
            dir_name_list_seperate = self.seperate_dir_name_list_by_session(dir_name_list)
            for i in range(len(dir_name_list_seperate)):
                path_sequence = self.dicom_root + "/" + dir_name_list_seperate[i][0][0]
                # writen sequence file list to json
                with open(path_sequence + "/sequence.json", "w") as outfile:
                    json.dump(dir_name_list_seperate[i], outfile)

            for i in range(len(dir_name_list)):
                path = self.dicom_root + "/" + "/".join(dir_name_list[i])
                list1 = load_sorted_directory(path)
                if MaxNumberOfImage is None:
                    for j in range(len(list1)):
                        self.export_to_jpeg(index=dir_index_list[i] + [j], ifSeperateFolder=ifSeperateFolder, Tags=Tags)
                else:
                    for j in range(min(len(list1), MaxNumberOfImage)):
                        self.export_to_jpeg(index=dir_index_list[i] + [j], ifSeperateFolder=ifSeperateFolder, Tags=Tags)


class GeneralLoader(object):
    """
    This class is responsible for loading dicom medical dataset
    """

    def __init__(self, mri_csv_path, analysis_csv_path, dicom_root, patient_id_label, mri_directory_label,
                 pre_filter_labels=None, mri_directory_postfix='',
                 detect_sequence=None, max_num_patients=None, mode='dicom', info_tag=None):
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
        self.detect_sequence = detect_sequence

        self.max_num_patients = max_num_patients
        self.mode = mode
        self.info_tag = info_tag
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
                                  rslt_df_analysis, self.dicom_root, self.mri_directory_postfix, self.detect_sequence,
                                  self.mode, self.info_tag)
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
                                                                                        DF=DF)
                if not result_mriDF.empty or not result_analysisDF.empty:
                    patients.append(self.patients[i])
                    result_mri_csv = pd.concat([result_mri_csv, result_mriDF], axis=0)
                    result_analysis_csv = pd.concat([result_analysis_csv, result_analysisDF], axis=0)
        elif if_or:
            for i in tqdm(range(len(self.patients))):
                result_mriDF, result_analysisDF = self.patients[i].get_DF_by_labels_or(labels=labels,
                                                                                       label_values=label_values, DF=DF)
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
