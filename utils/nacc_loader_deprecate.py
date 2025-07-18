import json
import os
import re
import time
import warnings

import matplotlib.pylab as plt
import pandas as pd
import pydicom as dicom
from tqdm import tqdm

from .general_utils import (fixPath, load_sorted_directory,
                            get_time_interval, check_if_dir_list, check_if_dir)

dicom_dict = {
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
    "SeriesDescription": ['T1', 'MPRAGE_GRAPPA_ADNI', 'MPRAGE'],  #
    "SequenceName": ['*tfl3d1', '*tfl3d1_16ns', '*tfl3d1_16ns']
}

Tag_dict_T2 = {
    "SeriesDescription": ['T2'],  #
    "SequenceName": []
}

Tag_dict_DTI = {
    "SeriesDescription": ['DTI'],  #
    "SequenceName": []
}

Tag_dict_DWI = {
    "SeriesDescription": ['DWI'],  #
    "SequenceName": []
}

Tag_dict_FLAIR = {
    "SeriesDescription": ['FLAIR'],  #
    "SequenceName": []
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

combined_tags_dict = {
    "T1": Tag_dict_T1,
    "T2": Tag_dict_T2,
    "DTI": Tag_dict_DTI,
    "DWI": Tag_dict_DWI,
    "FLAIR": Tag_dict_FLAIR
}


class PatientCaseNACC():
    def __init__(self, NACCID, mriDF, analysisDF, dicom_root, detect_sequence=None):
        """
        class that stores one patient case with all information loaded from csv files
        :param NACCID: string, patient ID
        :param mriDF: dataframe, dataframe containing all MRI information
        :param analysisDF: dataframe, dataframe containing all pathology test information
        :param dicom_root: string, root directory of all patients' DICOM files
        :param detect_sequence: string, chosen sequence such as 'T1', 'T2', 'DTI', 'DWI', etc.
        to detect when create patient case, default None
        """
        self.NACCID = NACCID
        self.mriDF = mriDF
        self.analysisDF = analysisDF
        self.dicom_root = dicom_root
        self.latest_mri_session, self.latest_mri_date = self.get_latest_mri_session()  # df, [year, month, day]
        self.dir_name_list, self.dir_index_list, self.dir_name_list_latest, self.dir_index_list_latest = [], [], [], []
        self.update_dir_name_list()
        self.ifHasMRI = self.check_MRI_exist()
        if detect_sequence is not None:
            self.sequence_dir_name_list, self.sequence_dir_index_list = self.get_sequence_dir_lists(
                sequence_name=detect_sequence)
            self.latest_sequence_dir_name_list, self.latest_sequence_dir_index_list = self.get_sequence_dir_lists(
                sequence_name=detect_sequence, ifLatest=True)

    def get_NACCID(self):
        return self.NACCID

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
        check if given labels using and condition return any non-empty dataframe
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
        check if given labels using or condition return any non-empty dataframe
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
        labels.insert(0, 'NACCID')
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
        NACCMRFI_filepaths = self.mriDF.loc[:, 'NACCMRFI']
        for i in range(len(NACCMRFI_filepaths)):
            NACCMRFI_filepath = NACCMRFI_filepaths.iloc[i][:-4]
            if os.path.exists(fixPath(self.dicom_root + "/" + NACCMRFI_filepath)):
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
        for dir in sorted(dirs_list):
            self.print_directory(dir, indexes + [dindex])
            dindex += 1

    def list_directory(self, dir_path, indexes=[]):
        """
        List all directories that contains dicom file, NOTE it's recursive function
        and will update the dir_name_list and dir_index_list
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
        for dir in sorted(dirs_list):
            self.list_directory(dir, indexes + [dindex])
            dindex += 1

    def print_all_mri_sessions(self):
        """
        Print all mri sessions hierarchy
        """
        if self.ifHasMRI:
            NACCMRFI_filepaths = self.mriDF.loc[:, 'NACCMRFI']
            print("this patient has", len(NACCMRFI_filepaths), "mri sessions stored in:")
            for i in range(len(NACCMRFI_filepaths)):
                NACCMRFI_filepath = NACCMRFI_filepaths.iloc[i][:-4]
                if os.path.exists(self.dicom_root + "/" + NACCMRFI_filepath):
                    self.print_directory(dir_path=[NACCMRFI_filepath], indexes=[i])
                else:
                    print("None file found at " + NACCMRFI_filepath + "/")
        else:
            print("Patient has no valid MRI session")

    def load_dicom_mri(self, index, filename_list=None):
        """
        Load DICOM given index
        :param index: [sequence index, slice num]
        :param filename_list: listed dicom filenames if given, default None, and load all dicom file in session
        :return: dicom datasets, return None if no dicom file is loaded
        """
        filepath_temp = self.get_file_path_given_sequence(index[:-1])

        if os.path.exists(filepath_temp):
            if filename_list is None:
                list = load_sorted_directory(filepath_temp)
            else:
                list = filename_list
            if list is not None:
                if len(list) > 0:  # avoid empty file
                    filepath_temp = filepath_temp + '/' + list[index[-1]]
                else:
                    return None
            else:
                return None

            try:
                ds = dicom.dcmread(fixPath(filepath_temp))
                return ds
            except Exception:
                return None  # or you could use 'continue'
        else:
            return None

    def get_file_index_list_in_sequence(self, index):
        """
        Load file index to a list given sequence index
        :param index: sequence index list [0, 0, 0]
        :return: index list [[0, 0, 0, 0], [0, 0, 0, 1], ..., [0, 0, 0, n]], None if no file index is loaded
        """
        NACCMRFI_filepaths = self.mriDF.loc[:, 'NACCMRFI']
        filepath_temp1 = self.dicom_root + "/" + NACCMRFI_filepaths.iloc[index[0]][:-4]
        if os.path.exists(filepath_temp1):
            list1 = load_sorted_directory(filepath_temp1)
            if list1 is not None:
                idx = 0
                for j in range(len(index) - 1):
                    if len(list1) > 0 and list1 is not None:  # avoid empty folder
                        if check_if_dir_list(filepath_temp1, list1):
                            idx = idx + 1
                            filepath_temp1 = filepath_temp1 + "/" + list1[index[idx]]
                            list1 = load_sorted_directory(filepath_temp1)
                            if list1 is None:
                                return None
                    else:
                        return None
                idx = idx + 1
                if len(list1) > 0:  # avoid empty file
                    dir_index_list_temp = []
                    for i in range(len(list1)):
                        dir_index_list_temp.append(index + [i])
                    return dir_index_list_temp
                else:
                    return None
            else:
                return None
        else:
            return None

    def get_latest_mri_session(self, if_update=False):
        """
        Get latest mri session
        :param if_update: if update current mri dataframe to dataframe with only latest mri session
        :return: mri dataframe with latest mri, date in ['year', 'mouth', 'day']
        """
        latest_mri_session = self.mriDF
        latest_yr = None
        latest_mo = None
        latest_dy = None
        if 'MRIYR' in self.mriDF.columns:
            latest_yr = latest_mri_session['MRIYR'].max()
            latest_mri_session = latest_mri_session.loc[latest_mri_session['MRIYR'] == latest_yr]
            if 'MRIMO' in self.mriDF.columns:
                latest_mo = latest_mri_session['MRIMO'].max()
                latest_mri_session = latest_mri_session.loc[latest_mri_session['MRIMO'] == latest_mo]
                if 'MRIDY' in self.mriDF.columns:
                    latest_dy = latest_mri_session['MRIDY'].max()
                    latest_mri_session = latest_mri_session.loc[latest_mri_session['MRIDY'] == latest_dy]
            if if_update:
                self.mriDF = latest_mri_session
            return latest_mri_session, [latest_yr, latest_mo, latest_dy]
        else:
            return pd.DataFrame(columns=self.latest_mri_session.columns), [latest_yr, latest_mo, latest_dy]

    def load_dicom_mri_by_session(self, index, mri_session):
        """
        Load first dicom file given a mri session dataframe
        :param index: [sequence index, slice num]
        :param mri_session: mri session dataframe
        :return: dicom datasets, return None if no dicom file is loaded
        """
        NACCMRFI_filepath = mri_session['NACCMRFI'].iloc[0]
        filepath_temp1 = self.dicom_root + "/" + NACCMRFI_filepath[:-4]
        if os.path.exists(filepath_temp1):
            list1 = load_sorted_directory(filepath_temp1)
            idx = 0
            for j in range(len(index)):
                if check_if_dir_list(filepath_temp1, list1):
                    filepath_temp1 = filepath_temp1 + "/" + list1[index[idx]]
                    list1 = load_sorted_directory(filepath_temp1)
                    idx = idx + 1
            filepath_temp = filepath_temp1 + "/" + list1[index[idx]]
            ds = dicom.dcmread(fixPath(filepath_temp))
            return ds
        else:
            return None

    def get_death_date(self):
        """
        get patient death date
        :return: date
        """
        if self.analysisDF['NACCDIED'].iloc[0] == 1:
            return [self.analysisDF['NACCYOD'].iloc[0], self.analysisDF['NACCMOD'].iloc[0], 0]
        else:
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
        NACCMRFI_filepaths = self.mriDF.loc[:, 'NACCMRFI']
        NACCMRFI_latest_filepath = self.latest_mri_session.loc[:, 'NACCMRFI']

        if self.check_MRI_exist():
            self.dir_name_list = []
            self.dir_index_list = []
            self.dir_name_list_latest = []
            self.dir_index_list_latest = []
            for i in range(len(NACCMRFI_filepaths)):
                NACCMRFI_filepath = NACCMRFI_filepaths.iloc[i][:-4]
                if os.path.exists(self.dicom_root + "/" + NACCMRFI_filepath):
                    # update self.dir_name_list and self.dir_index_list
                    self.list_directory(dir_path=[NACCMRFI_filepath], indexes=[i])

            # update self.dir_name_list_latest and self.dir_index_list_latest
            for i in range(len(self.dir_name_list)):
                if NACCMRFI_latest_filepath.iloc[0][:-4] == self.dir_name_list[i][0]:
                    self.dir_name_list_latest.append(self.dir_name_list[i])
                    self.dir_index_list_latest.append(self.dir_index_list[i])

    def get_sequence_dir_lists(self, sequence_name, ifLatest=False):
        """
        get sequence directory lists based on sequence name
        :param sequence_name: string, 'T1', 'T2', ''DWI'
        :param ifLatest: bool, if only check on the latest sequence directories, default is False
        :return: (dir_name_list, dir_index_list) for matching sequence directories.
        """
        if ifLatest:
            temp_mriDF = self.latest_mri_session
            temp_dir_name_list = self.dir_name_list_latest
            temp_dir_index_list = self.dir_index_list_latest
        else:
            temp_mriDF = self.mriDF
            temp_dir_name_list = self.dir_name_list
            temp_dir_index_list = self.dir_index_list

        NACCMRFI_filepaths = temp_mriDF.loc[:, 'NACCMRFI']

        # check if has 'MRI' + sequence_name recorded
        ifHasMRISequence = False
        for j in range(len(NACCMRFI_filepaths)):
            if temp_mriDF.iloc[j]['MRI' + sequence_name] == 1 and self.ifHasMRI:
                ifHasMRISequence = True

        if ifHasMRISequence:
            # check by directory name
            name_idx = [x for x in range(len(temp_dir_name_list)) for y in range(len(temp_dir_name_list[x])) if
                        sequence_name in temp_dir_name_list[x][y]]
            # return name_idx
            # check by dictionary of dicom tags
            # load one dicom file from each session
            tag_idx = []
            Tag_dict_temp = combined_tags_dict[sequence_name]
            for i in range(len(temp_dir_index_list)):
                temp_valid_mri = self.get_first_mri_valid(temp_dir_index_list[i])
                if temp_valid_mri is not None:
                    temp_index = temp_dir_index_list[i] + [temp_valid_mri]
                    dicom_data = self.load_dicom_mri(index=temp_index)
                    if dicom_data is not None:
                        for key in dicom_dict:
                            if dicom_data.__contains__(dicom_dict[key]) and key in Tag_dict_temp.keys():
                                if any(x in dicom_data[dicom_dict[key]].value for x in Tag_dict_temp[key]):
                                    tag_idx.append(i)
            final_idx = name_idx + tag_idx
            final_idx = list(set(final_idx))
            return [temp_dir_name_list[k] for k in final_idx], [temp_dir_index_list[k] for k in final_idx]
        else:
            return None, None

    def get_sequence_sessions(self, sequence, ifLatest=False, ifUpadte=False):
        """

        :param sequence: get MRI sessions that contains at least one give sequence
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
        NACCMRFI_filepaths = temp_mriDF.loc[:, 'NACCMRFI']
        # check if has MRI sequence recorded
        sequence_sessions_dir_name_list = []
        sequence_sessions_dir_index_list = []
        new_mriDF_list = []
        for j in range(len(NACCMRFI_filepaths)):
            if temp_mriDF.iloc[j]['MRI' + sequence] == 1 and self.ifHasMRI:
                file_name = NACCMRFI_filepaths.iloc[j][:-4]
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
        latest_NACCMRFI_filepaths = latest_temp_mriDF.loc[:, 'NACCMRFI']
        # check if has MRIT1 recorded
        latest_sequence_sessions_dir_name_list = []
        latest_sequence_sessions_dir_index_list = []
        latest_new_mriDF_list = []
        for j in range(len(latest_NACCMRFI_filepaths)):
            if latest_temp_mriDF.iloc[j]['MRI' + sequence] == 1 and self.ifHasMRI:
                file_name = latest_NACCMRFI_filepaths.iloc[j][:-4]
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

    def get_first_mri_valid(self, sequence_index):
        """
        get first mri number that is valid for loading
        :param sequence_index: [dir_index]
        :return: int, first valid mri number
        """
        filepath = self.get_file_path_given_sequence(sequence_index)
        dir_name_list = load_sorted_directory(filepath)
        if dir_name_list is not None:
            for i in range(len(dir_name_list)):
                ds = self.load_dicom_mri(sequence_index + [i], filename_list=dir_name_list)

                if ds is not None:
                    try:
                        image = ds.pixel_array
                        return i
                    except:
                        continue
        return None

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
        get file path by index
        :param index: [dir index, slice index]
        :return: tuple, (string: filepath, list: filenames)
        """
        NACCMRFI_filepaths = self.mriDF.loc[:, 'NACCMRFI']
        filepath_temp = self.dicom_root + "/" + NACCMRFI_filepaths.iloc[index[0]][:-4]
        path_list = [NACCMRFI_filepaths.iloc[index[0]][:-4]]
        if os.path.exists(filepath_temp):
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
                if dicom_data.__contains__(dicom_dict[tag]):
                    dict_tag_value = dicom_data[dicom_dict[tag]].value
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
                        if not os.path.exists(path_folder):
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
                    if not os.path.exists(path_sequence):
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


class NACCLoader(object):
    """
    This class is responsible for loading NACC dataset
    """
    def __init__(self, mri_csv_path, analysis_csv_path, dicom_root, pre_filter_labels=None, detect_sequence=None):
        """
        construct NACC loader class
        :param mri_csv_path: path to mri csv file
        :param analysis_csv_path: path to analysis csv file
        :param dicom_root: path to dicom files root folder
        :param pre_filter_labels: list, pre filtering labels for both mri and analysis csv
        :param detect_sequence:
        """
        self.mri_csv_path = mri_csv_path
        self.analysis_csv_path = analysis_csv_path
        self.dicom_root = dicom_root
        self.mri_csv = self.read_csv(mri_csv_path)
        self.analysis_csv = self.read_csv(analysis_csv_path)
        self.detect_sequence = detect_sequence
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
        NACCIDs = self.mri_csv["NACCID"].unique()
        patients = []
        for i in tqdm(range(len(NACCIDs))):
            NACCID = NACCIDs[i]
            rslt_df_mri = self.mri_csv.loc[self.mri_csv['NACCID'] == NACCID]
            rslt_df_analysis = self.analysis_csv.loc[self.analysis_csv['NACCID'] == NACCID]
            patient = PatientCaseNACC(NACCID, rslt_df_mri, rslt_df_analysis, self.dicom_root, self.detect_sequence)
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

    def get_patients_by_ID(self, NACCID):
        """
        get patient case by NACCID
        :param NACCID: string, NACCID
        :return: patient case
        """
        for patient in self.patients:
            if patient.get_NACCID() == NACCID:
                return patient

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
        :param if_update: if update NACC loader
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
