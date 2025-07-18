import json
import os
import re
import time
import warnings

import pandas as pd
import pydicom as dicom
from tqdm import tqdm
from copy import deepcopy
from .general_utils import (fixPath, load_sorted_directory, get_time_interval, check_if_dir_list, check_if_dir)
from .general_loader import PatientCase, GeneralLoader


class PatientCase_adni(PatientCase):
    def __init__(self, patient_ID, patient_ID_label, mri_directory_label, mriDF, analysisDF, dicom_root,
                 mri_directory_postfix='', postfix_label='',
                 detect_sequence=None, mode='dicom', info_tags=None, if_cache_image=False,
                 cache_path='image_cache/ADNI/', series_description_folder=None, **kwargs):
        super().__init__(patient_ID, patient_ID_label, mri_directory_label, mriDF, analysisDF, dicom_root,
                         mri_directory_postfix, postfix_label, detect_sequence, mode, info_tags, if_cache_image,
                         cache_path,series_description_folder)

        # update for customize patient loader
        time3 = time.time()
        self.update_sequence_series_tags()
        time4 = time.time()
        if (time4 - time3) > 15:
            print(patient_ID + ' time4-3: ' + str(time4 - time3))
        self.update_sequence_dir_slice_lists_dict()
        time5 = time.time()
        if (time5 - time4) > 15:
            print(patient_ID + ' time5-4: ' + str(time5 - time4))
        if self.detect_sequence is not None:
            self.update_pd_echo_time()

    # def update_sequence_dir_lists_dict_by_TR_TE(self):
    # if tags stored in  self.combined_tags_dict['byTETR'] occurs, check TE and TR to find if PD, T1 or T2 or flair

    def get_latest_mri_session(self, if_update=False):
        """
        Get latest mri session
        :param if_update: if update current mri dataframe to dataframe with only latest mri session
        :return: mri dataframe with latest mri, date in ['year', 'mouth', 'day']
        """
        latest_mri_session = self.mriDF
        latest_yr = 0
        latest_mo = 0
        latest_dy = 0
        if 'EXAMDATE' in self.mriDF.columns:
            dates = self.mriDF['EXAMDATE']
            for i in range(len(dates)):
                date = dates.iloc[i].split("/")  # [month, day, year]
                if int(date[2]) > latest_yr:
                    latest_yr = int(date[2])
                    latest_mo = int(date[0])
                    latest_dy = int(date[1])
                    latest_mri_session = self.mriDF.iloc[[i]]
                elif int(date[2]) == latest_yr:
                    if int(date[0]) > latest_mo:
                        latest_mo = int(date[0])
                        latest_dy = int(date[1])
                        latest_mri_session = self.mriDF.iloc[[i]]
                    elif int(date[0]) == latest_mo:
                        if int(date[1]) > latest_dy:
                            latest_dy = int(date[1])
                            latest_mri_session = self.mriDF.iloc[[i]]
            if if_update:
                self.mriDF = latest_mri_session
            return latest_mri_session, [latest_yr, latest_mo, latest_dy]
        else:
            return pd.DataFrame(columns=latest_mri_session.columns), [latest_yr, latest_mo, latest_dy]

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
            "SeriesDescription": ['T1', 'MPRAGE_GRAPPA_ADNI', 'MPRAGE', 't1', 'mprage', 'Mprage'],  #
            "SequenceName": [],
            "DirName": ['T1', 't1', 'MPRAGE', 'mprage']
        }

        Tag_dict_T2 = {
            "SeriesDescription": ['T2', 't2'],  #
            "SequenceName": [],
            "DirName": ['T2', 't2']
        }

        Tag_dict_FLAIR = {
            "SeriesDescription": ['FLAIR', 'flair', 'Flair'],  #
            "SequenceName": [],
            "DirName": ['FLAIR', 'flair', 'Flair']
        }

        Tag_dict_T1_T1flair = {
            "SeriesDescription": ['T1', 'MPRAGE_GRAPPA_ADNI', 'MPRAGE', 't1', 'mprage', 'Mprage', 'MP-RAGE',
                                  'MP-RAGE REPEAT', 'SAG MP-RAGE', 'SAG MP-RAGE Repeat', 'MP-RAGE-Repeat', 'MP-RAGE-',
                                  'MP-RAGE  REPEAT', 'MP-RAGE- REPEAT', 'MP-RAGE REPEAT #3',
                                  'IR-FSPGR (replaces MP-Rage)', 'MP-RAGE - REPEAT', 'REPEAT MP-RAGE', 'MP-RAGE 24 FOV',
                                  'MP-RAGE 24 FOV REPEAT', 'MP-RAGE REPEAT (SERIES 3)', 'MP-RAGE (SERIES 2)',
                                  'MP-RAGE #3', 'MP-RAGE REPEAT #2', 'MP-RAGE rpt', 'MP-RAGE #2', 'MP-RAGE   REPEAT'],
            #
            "SequenceName": [],
            "DirName": ['T1', 'MPRAGE', 'MP-RAGE', 'MPRAGE_REPEAT', 'MP-RAGE_REPEAT']
        }

        Tag_dict_T1post = {
            "SeriesDescription": ['T1post', 'T1_post',
                                  ],  #
            "SequenceName": [],
            "DirName": []
        }

        Tag_dict_T2_T2star = {
            "SeriesDescription": ['T2', 'T2*', 'T2_star',
                                  ],  #
            "SequenceName": [],
            "DirName": []
        }

        Tag_dict_T2flair_flair = {
            "SeriesDescription": ['T2_flair', 'T2 flair', 'T2flair', 'FLAIR', 'flair', 'Flair'],  #
            "SequenceName": [],
            "DirName": ['FLAIR', 'flair', 'Flair']
        }

        Tag_dict_PD = {
            "SeriesDescription": ['pd', 'PD',
                                  ],  #
            "SequenceName": [],
            "DirName": []
        }

        Tag_dict_DTI_DWI = {
            "SeriesDescription": ['DTI', 'DWI', 'dwi', 'dti', 'diffuse', 'DIFFUSE', 'Diffuse'
                , 'diffusion', 'DIFFUSION', 'Diffusion'],  #
            "SequenceName": [],
            "DirName": ['DTI', 'DWI', 'dwi', 'dti', 'diffuse', 'DIFFUSE', 'Diffuse'
                , 'diffusion', 'DIFFUSION', 'Diffusion']
        }

        Tag_dict_MISC = {
            "SeriesDescription": ['localize', 'Localize', 'LOCALIZAE', 'B1-calibration Body', 'B1-calibration Head',
                                  'B1-calibration Body SAG', 'B1-calibration Head SAG', 'B1-Calibration PA',
                                  'B1-Calibration', 'Calibration Scan', 'SAG B1 CALIBRATION BODY',
                                  'SAG B1 CALIBRATION HEAD', 'SAG B1 CALIBRATION BODY REPEAT', 'B1 calibration Body',
                                  'B1 calibration Head', 'B1-Calibration 8hrbrn', 'B1-calibration - Body Coil',
                                  'B1-calibration - Head Coil', 'repeat SAG B1 CALIBRATION BODY',
                                  '*B1-Calibration Body', '*B1-Calibration PA', 'act_te = 6000 B1-Calibration Body',
                                  'act_te = 6000 B1-Calibration PA', '*B1-Calibration Body TE 6',
                                  '*B1-Calibration PA TE 6', 'B1-Calibration Body te 5.5',
                                  'B1-Calibration PA (TE min full)', 'B1-Calibration PA te 5.5',
                                  'B1-Calibration PA te 5.5ms', 'ASSET CALIBRATION', 'B1-Calibration Body (SERIES 5',
                                  'B1-Calibration PA (SERIES 4)', 'B1-Calibration Body TE=5.5',
                                  'B1-Calibration Body TE =5.5ms', 'B1-Calibration PA TE=5.5',
                                  'B1-Calibration Body  TE 5.5ms', 'B1-Calibration PA 5.5 TE',
                                  'B1-Calibration Body TE=5.5ms', 'B1-Calibration PA TE=5.5MS',
                                  'B1-calibration Bodycoil', 'TE 6 B1-Calibration Body', 'TE 6 B1-Calibration PA',
                                  'B1-Calibration Body TE 5.5ms', 'B1-Calibration Body te5.5ms',
                                  'B1-Calibration PA TE =5.5', 'B1-calibration 8ch', 'B1-calibration-BodyCoil',
                                  'B1-calibration-HeadCoil', 'B1-Calibration PA TE5.5ms', 'B1-Calibration BODY 5.5MS',
                                  'B1-Calibration Body TE = 5.5'],  #
            "SequenceName": [],
            "DirName": ['localize', 'Localize', 'LOCALIZAE', 'B1-calibration_Body', 'B1-calibration_Head',
                        'calibration',
                        'B1-calibration_Head_SAG',
                        'B1-calibration_PA', ]
        }

        Tag_dict_byTETR = {
            "SeriesDescription": ['ADNI-R11   Double_TSE', 'ADNI_new   Double_TSE', 'ADNI_new   Double_TSE2',
                                  'Double_TSE', 'ADNI STUDY Double_TSE2s', 'ADNI-R11-ASDouble_TSE',
                                  '           Double_TSE_', 'Double_TSE_new', 'ADNI       Double_TSE',
                                  '           Double_TSE', 'DOUBLE_TSE SENSE', 'Double_TSE_new SENSE',
                                  'ADNI       Double_TSEad', 'ADNI-R11   Double_TSE S', 'Double_TSE-3rdRepeat'],  #
            "SequenceName": [],
            "DirName": []
        }

        # read SeriesDescription_list provided by radiologists and added to tag_dict_T1
        with open('series_description/adni/SeriesDescription_T1.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T1["SeriesDescription"] = Tag_dict_T1["SeriesDescription"] + lines
        Tag_dict_T1["SeriesDescription"] = list(set(Tag_dict_T1["SeriesDescription"]))

        with open('series_description/adni/SeriesDescription_T2.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T2["SeriesDescription"] = Tag_dict_T2["SeriesDescription"] + lines
        Tag_dict_T2["SeriesDescription"] = list(set(Tag_dict_T2["SeriesDescription"]))

        with open('series_description/adni/SeriesDescription_FLAIR.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_FLAIR["SeriesDescription"] = Tag_dict_FLAIR["SeriesDescription"] + lines
        Tag_dict_FLAIR["SeriesDescription"] = list(set(Tag_dict_FLAIR["SeriesDescription"]))

        with open('series_description/adni/SeriesDescription_T1_T1flair.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T1_T1flair["SeriesDescription"] = Tag_dict_T1_T1flair["SeriesDescription"] + lines
        Tag_dict_T1_T1flair["SeriesDescription"] = list(set(Tag_dict_T1_T1flair["SeriesDescription"]))

        with open('series_description/adni/SeriesDescription_T1post.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T1post["SeriesDescription"] = Tag_dict_T1post["SeriesDescription"] + lines
        Tag_dict_T1post["SeriesDescription"] = list(set(Tag_dict_T1post["SeriesDescription"]))

        with open('series_description/adni/SeriesDescription_T2_T2star.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T2_T2star["SeriesDescription"] = Tag_dict_T2_T2star["SeriesDescription"] + lines
        Tag_dict_T2_T2star["SeriesDescription"] = list(set(Tag_dict_T2_T2star["SeriesDescription"]))

        with open('series_description/adni/SeriesDescription_T2flair_flair.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T2flair_flair["SeriesDescription"] = Tag_dict_T2flair_flair["SeriesDescription"] + lines
        Tag_dict_T2flair_flair["SeriesDescription"] = list(set(Tag_dict_T2flair_flair["SeriesDescription"]))
        #
        with open('series_description/adni/SeriesDescription_PD.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_PD["SeriesDescription"] = Tag_dict_PD["SeriesDescription"] + lines
        Tag_dict_PD["SeriesDescription"] = list(set(Tag_dict_PD["SeriesDescription"]))
        #
        with open('series_description/adni/SeriesDescription_DTI_DWI.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_DTI_DWI["SeriesDescription"] = Tag_dict_DTI_DWI["SeriesDescription"] + lines
        Tag_dict_DTI_DWI["SeriesDescription"] = list(set(Tag_dict_DTI_DWI["SeriesDescription"]))
        #
        with open('series_description/adni/SeriesDescription_MISC.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_MISC["SeriesDescription"] = Tag_dict_MISC["SeriesDescription"] + lines
        Tag_dict_MISC["SeriesDescription"] = list(set(Tag_dict_MISC["SeriesDescription"]))

        self.combined_tags_dict = {
            "T1": Tag_dict_T1,
            "T2": Tag_dict_T2,
            "FLAIR": Tag_dict_FLAIR,
            "T1_T1flair": Tag_dict_T1_T1flair,
            "T1post": Tag_dict_T1post,
            "T2_T2star": Tag_dict_T2_T2star,
            "T2flair_flair": Tag_dict_T2flair_flair,
            "PD": Tag_dict_PD,
            "DTI_DWI": Tag_dict_DTI_DWI,
            "MISC": Tag_dict_MISC,
            "byTETR": Tag_dict_byTETR,
            # "DWI": Tag_dict_DWI,
        }


class ADNILoader(GeneralLoader):
    def __init__(self, mri_csv_path, analysis_csv_path, dicom_root, patient_id_label, mri_directory_label,
                 pre_filter_labels=None, mri_directory_postfix='', postfix_label='',
                 detect_sequence=None, max_num_patients=None, mode='dicom', info_tags=None, if_cache_image=False,
                 cache_path='image_cache/ADNI/',series_description_folder=None):
        super().__init__(mri_csv_path, analysis_csv_path, dicom_root, patient_id_label, mri_directory_label,
                         pre_filter_labels, mri_directory_postfix, postfix_label,
                         detect_sequence, max_num_patients, mode, info_tags, if_cache_image, cache_path,series_description_folder)

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
            # print(patientID)
            time0 = time.time()
            rslt_df_mri = self.mri_csv.loc[self.mri_csv[self.patient_id_label] == patientID]
            rslt_df_analysis = self.analysis_csv.loc[self.analysis_csv[self.patient_id_label] == patientID]
            patient = PatientCase_adni(patientID, self.patient_id_label, self.mri_directory_label, rslt_df_mri,
                                       rslt_df_analysis, self.dicom_root, self.mri_directory_postfix,
                                       self.postfix_label,
                                       self.detect_sequence, self.mode, self.info_tags, self.if_cache_image,
                                       self.cache_path)
            time1 = time.time()
            time_interval = time1 - time0
            if time_interval > 20:
                print(patientID + ' used time:' + str(time_interval))
            patients.append(patient)
        return patients
