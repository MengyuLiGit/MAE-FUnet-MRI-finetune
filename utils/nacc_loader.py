import json
import os
import re
import time
import warnings

import matplotlib.pylab as plt
import pandas as pd
import pydicom as dicom
import time
from tqdm import tqdm
from copy import deepcopy
from .general_utils import (fixPath, load_sorted_directory, get_time_interval, check_if_dir_list, check_if_dir,
                            clear_string_char)
from .general_loader import PatientCase, GeneralLoader


class PatientCase_nacc(PatientCase):
    def __init__(self, patient_ID, patient_ID_label, mri_directory_label, mriDF, analysisDF, dicom_root,
                 mri_directory_postfix='.zip', postfix_label='',
                 detect_sequence=None, mode='dicom', info_tags=None, if_cache_image=False,
                 cache_path='image_cache/NACC/', series_description_folder=None, **kwargs):
        super().__init__(patient_ID, patient_ID_label, mri_directory_label, mriDF, analysisDF, dicom_root,
                         mri_directory_postfix, postfix_label,
                         detect_sequence, mode, info_tags, if_cache_image, cache_path,series_description_folder)

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
        if self.detect_sequence is not None and self.mode == 'dicom':
            self.update_pd_echo_time()

    def get_time_given_dir_index(self, dir_index):
        """
        Get time of dir taken
        :param dir_index: [dir_index]
        :return: [[years, months, days]]
        """
        dir_names = self.get_dir_names_given_dir_index(dir_index)
        if dir_names is None:
            return None
        dir_name_first = dir_names[0]
        dir_name_first = dir_name_first[:len(dir_name_first) - len(self.postfix_label)] + self.mri_directory_postfix
        mriDF_index_list = self.mriDF.index[self.mriDF['NACCMRFI'] == dir_name_first].tolist()
        dates = []
        for mriDF_index in mriDF_index_list:
            year = self.mriDF.loc[mriDF_index]['MRIYR']
            month = self.mriDF.loc[mriDF_index]['MRIMO']
            day = self.mriDF.loc[mriDF_index]['MRIDY']
            dates.append([year, month, day])
        return dates
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
            return pd.DataFrame(columns=latest_mri_session.columns), [latest_yr, latest_mo, latest_dy]

    def get_death_date(self):
        """
        get patient death date
        :return: date
        """
        if self.analysisDF['NACCDIED'].iloc[0] == 1:
            return [self.analysisDF['NACCYOD'].iloc[0], self.analysisDF['NACCMOD'].iloc[0], 0]
        else:
            return None
        return None

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
            "SeriesDescription": ['T1', 'MPRAGE_GRAPPA_ADNI', 'MPRAGE', '3D T1TFE', ],  #
            "SequenceName": ['*tfl3d1', '*tfl3d1_16ns', '*tfl3d1_16ns'],
            "DirName": ['T1', 'MPRAGE', ]
        }

        Tag_dict_T2 = {
            "SeriesDescription": ['T2', ],  #
            "SequenceName": [],
            "DirName": ['T2', ]
        }

        Tag_dict_FLAIR = {
            "SeriesDescription": ['FLAIR', ],  #
            "SequenceName": [],
            "DirName": ['FLAIR', ]
        }

        Tag_dict_T1_T1flair = {
            # "SeriesDescription": ['T1', 'MPRAGE_GRAPPA_ADNI', 'MPRAGE', '3D T1TFE',
            #                       'PU:STRUC FSGPR Pre','SC:STRUC BRAVO SAG3D ARC 1mm','STRUC BRAVO SAG3D ARC 1mm',
            #                       'STRUC FSGPR Pre', 'COR SPGR', 'SAGITTAL T 1', 'AXIAL T 1','AX TI','ELAN       SE     TRANS',
            #                       'ADRC       STIR/TSE$1', 'Brain      TI SAG','Ax 3D SPGR', 'SE  SAG', 'SAG 3D FFE',
            #                       'SAG FSPGR 3D'],  #
            "SeriesDescription": ['T1', 'MPRAGE_GRAPPA_ADNI', 'MPRAGE', '3D T1TFE', 'STRUC BRAVO SAG3D ARC 1mm',
                                  'STRUC FSGPR Pre', 'PU:STRUC FSGPR Pre',
                                  'SC:STRUC BRAVO SAG3D ARC 1mm', 'COR SPGR T1', 'SAGITTAL T 1', 'AXIAL T 1', 'AX TI',
                                  'ELAN       SE     TRANS',
                                  'Sag T1 FLAIR', 'Brain      TI SAG T1', 'STRUC BRAVO SAG3D ARC 1mm',
                                  'STRUC FSGPR Pre', 'PU:STRUC FSGPR Pre', 'SC:STRUC BRAVO SAG3D ARC 1mm'
                , 'SE  SAG T1', 'Ax 3D SPGR T1', 'SE  SAG T1', 'STRUC BRAVO SAG3D ARC 1mm', 'STRUC FSGPR Pre',
                                  'PU:STRUC FSGPR Pre',
                                  'SC:STRUC BRAVO SAG3D ARC 1mm', 'SE T1', '3D_T1TFE', 'Brain_T1_SE_AX',
                                  'Cor_T1+C_SE_fc_F', 'SENSE_BRAINT1W_AX', 'MEMPRAGE', 'MEMPRAGE_RMS',
                                  'mprage_coronal_mpr', 'MPRAGE_adni_ipat', 'T1_MEMPRAGE_RMS', 'T1_MEMPRAGE_P',
                                  'T1_MEMPRAGE', 'mprage_cor_mpr', 'T1_MPRAGE_iso_short', 'mprage_axial_mpr'
                                  ############ below for RAGEN only, may not be accuract
                , 'Brain_TI_SAG', 'Ax_3D_SPGR', 'SE_SAG', 'SE_TRANS', '2D_Saggital', 'mpr_axi', 'CORONAL_MPR',
                                  '3dflair_axial_mpr', '3dflair_mpr_axial',  'Sagittal_3D','COR_IR-FSPGR',
                                  # 'SAG_3D_BRAVO'
                                  ],
            # SAG_3D_BRAVO could be T1 or T1post

            # for RAGEN only, may not accuract
            "SequenceName": [],
            "DirName": ['MT1']
        }

        Tag_dict_T1post = {
            "SeriesDescription": ['T1post', 'T1_post',
                                  'PU:STRUC FSGPR Post', 'STRUC FSGPR Post', 'PU:STRUC FSGPR Post', 'POST CORONAL',
                                  '+C 3D MT SPGR', 'SE T1 post',
                                  # for RAGEN only, may not accuract
                                  'COR_POST_(OPT)', 'SPGR_post'],
            "SequenceName": [],
            "DirName": []
        }

        Tag_dict_T2_T2star = {
            "SeriesDescription": ['T2', 'T2*', 'T2_star',
                                  'tse t2 axial', 'Ax T2 FS PROPELLER', 'ADRC       STIR/TSE$1 T2', 'FSE T2',
                                  # for RAGEN only, may not accuract
                                  # 'CORONAL FMPIR', 'ADRC_STIR_TSE$1', 'AXIAL_FSE', 'GRADIENT_ECHO_AXIAL',
                                  # 'T2_TSE_Axial_320', 'T2_SPACE', 'T2_BLADE'
                                  ],
            "SequenceName": [],
            "DirName": []
        }

        Tag_dict_T2flair_flair = {
            "SeriesDescription": ['T2_flair', 'T2 flair', 'T2flair', 'FLAIR',
                                  'Ax T2 PROPELLER FLAIR', 'FSE flair', 'Axial_T2-FLAIR',
                                  # for RAGEN only, may not accuract
                                  'T2_BLADE_FLAIR_IPAT','T2_SPACE_FLAIR','Flair_Axial_Fat_Sat','FLAIR-AXIAL'],  #
            "SequenceName": [],
            "DirName": ['AFL']
        }

        Tag_dict_PD = {
            "SeriesDescription": ['pd',
                                  'AXIAL FSE2', 'Axial_PD-T2_TSE'],  #
            "SequenceName": [],
            "DirName": []
        }

        Tag_dict_DTI_DWI = {
            "SeriesDescription": ['DTI', 'DWI', 'DIFFUSE'
                , 'DIFFUSION', 'HARDI 64 b3k  max', 'EPI_2d_diff_3scan_trace_ADC','EPI_2d_diff_3scan_trace',
                                  'DIFFUSION_HighRes', 'DIFFUSION_HighRes_ADC', 'DIFFUSION_HighRes_TRACEW',
                                  'DIFFUSION_HighRes_ColorFA', 'DIFFUSION_HighRes_LOWB'],  #
            "SequenceName": [],
            "DirName": ['DTI', 'DWI', 'DIFFUSE', 'DIFFUSION']
        }

        Tag_dict_MISC = {
            "SeriesDescription": ['calib', 'localiz',
                                  ## for RAGEN only, may not accuract
                                  'AAScout','SWI_Images', 'ge_func_128_2mm_rest', 'field_mapping_128_2mm','Perfusion_Weighted',
                                  'relCBF','PASL','MoCoSeries','3_PLANE_LOC', 'LOC', 'loc',
                                  '3Plane_Loc_SSFSE', 'trufi_multiplane_loc', 'trufi_3PLANE_LOC', 'B-LOC', 'SAG_LOC', '_3_plane_loc',
                                  'Reformatted','Average', 'Average_DC', 'FLAIR_PARALLEL_TO_AC_PC','DEMENTIA_AXIALS',
                                  'DEMENTIA_SCOUT'],
            "SequenceName": [],
            "DirName": ['calib', 'localiz']
        }

        # read SeriesDescription_list provided by radiologists and added to tag_dict_T1
        with open('series_description/nacc/SeriesDescription_T1.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T1["SeriesDescription"] = Tag_dict_T1["SeriesDescription"] + lines
        Tag_dict_T1["SeriesDescription"] = list(set(Tag_dict_T1["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_T2.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T2["SeriesDescription"] = Tag_dict_T2["SeriesDescription"] + lines
        Tag_dict_T2["SeriesDescription"] = list(set(Tag_dict_T2["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_FLAIR.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_FLAIR["SeriesDescription"] = Tag_dict_FLAIR["SeriesDescription"] + lines
        Tag_dict_FLAIR["SeriesDescription"] = list(set(Tag_dict_FLAIR["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_T1_T1flair.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T1_T1flair["SeriesDescription"] = Tag_dict_T1_T1flair["SeriesDescription"] + lines
        Tag_dict_T1_T1flair["SeriesDescription"] = list(set(Tag_dict_T1_T1flair["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_T1post.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T1post["SeriesDescription"] = Tag_dict_T1post["SeriesDescription"] + lines
        Tag_dict_T1post["SeriesDescription"] = list(set(Tag_dict_T1post["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_T2_T2star.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T2_T2star["SeriesDescription"] = Tag_dict_T2_T2star["SeriesDescription"] + lines
        Tag_dict_T2_T2star["SeriesDescription"] = list(set(Tag_dict_T2_T2star["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_T2flair_flair.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_T2flair_flair["SeriesDescription"] = Tag_dict_T2flair_flair["SeriesDescription"] + lines
        Tag_dict_T2flair_flair["SeriesDescription"] = list(set(Tag_dict_T2flair_flair["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_PD.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_PD["SeriesDescription"] = Tag_dict_PD["SeriesDescription"] + lines
        Tag_dict_PD["SeriesDescription"] = list(set(Tag_dict_PD["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_DTI_DWI.txt') as f:
            lines = f.read().splitlines()
        Tag_dict_DTI_DWI["SeriesDescription"] = Tag_dict_DTI_DWI["SeriesDescription"] + lines
        Tag_dict_DTI_DWI["SeriesDescription"] = list(set(Tag_dict_DTI_DWI["SeriesDescription"]))

        with open('series_description/nacc/SeriesDescription_MISC.txt') as f:
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
            # "DWI": Tag_dict_DWI,
        }


class NACCLoader(GeneralLoader):
    def __init__(self, mri_csv_path, analysis_csv_path, dicom_root, patient_id_label, mri_directory_label,
                 pre_filter_labels=None, mri_directory_postfix='.zip', postfix_label='',
                 detect_sequence=None, max_num_patients=None, mode='dicom', info_tags=None, if_cache_image=False,
                 cache_path='image_cache/NACC/',series_description_folder=None):
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
            patient = PatientCase_nacc(patientID, self.patient_id_label, self.mri_directory_label, rslt_df_mri,
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
