# import numpy as np
# from matplotlib import pyplot as plt
# from help_func import print_var_detail
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import os
sys.path.append(os.path.abspath(".."))
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils.adni_loader import ADNILoader
from utils.nacc_loader import NACCLoader
from utils.general_dataloader import create_combine_dataloader
from utils.general_dataloader_cache import GeneralDataset, GeneralDatasetMae
from utils.fastmri_loader import FastmriDataSetMae, create_fastmri_data_info
import os
import time
import pickle
from help_func import  print_var_detail
from utils.oasis_loader import OASISLoader
import copy

from utils.help_func import create_path
from utils.data_utils import img_augment

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset, DistributedSampler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


rebuilt_OASIS = False
rebuilt_ADNI = False
rebuilt_NACC = False
RANDOM_SEED = 42
VAL_SPLIT = 0.0
IS_TRAIN = True
INPUT_SIZE = 224
nifti_root_oasis = "C:/oasis_nifti/"
cache_root_oasis_pkl = "E:/oasis_nifti_cache_reshape_norm_pkl/"
nifti_root_adni = "F:/adni_nifti/"
cache_root_adni_pkl = "E:/adni_nifti_cache_reshape_norm_pkl/"
nifti_root_nacc = "F:/nacc_nifti/"
cache_root_nacc_pkl = "E:/nacc_n_pkl/"
target_sequence_woDWI = ['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi']
target_sequence_woDWI_woT1_T1flair = [ 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi']
target_sequence_DWI = ['DTI_DWI_500', 'DTI_DWI']
target_sequence_all = ['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi', 'DTI_DWI_500', 'DTI_DWI']
mri_sequence = target_sequence_all
detect_sequence = ['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi', 'DTI_DWI']


#['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi', 'DTI_DWI', 'DTI_DWI_500']
max_val_len = 20000
rebuild_val_dataset = False
if rebuild_val_dataset:
    OASISdataset_train_all_sequence_val_T1_T1flair = copy.deepcopy(OASISdataset_train_all_sequence_val)
    OASISdataset_train_all_sequence_val_T1_T1flair.filter_by_sequence_labels(['T1_T1flair'])
    print(len(OASISdataset_train_all_sequence_val_T1_T1flair))
    OASISdataset_train_all_sequence_val_T1_T1flair.max_val_len = max_val_len
    OASISdataset_train_all_sequence_val_T1_T1flair._update_train_val_paths()
    print(len(OASISdataset_train_all_sequence_val_T1_T1flair))

    OASISdataset_train_all_sequence_val_T2 = copy.deepcopy(OASISdataset_train_all_sequence_val)
    OASISdataset_train_all_sequence_val_T2.filter_by_sequence_labels(['T2'])
    print(len(OASISdataset_train_all_sequence_val_T2))
    OASISdataset_train_all_sequence_val_T2.max_val_len = max_val_len
    OASISdataset_train_all_sequence_val_T2._update_train_val_paths()
    print(len(OASISdataset_train_all_sequence_val_T2))

    OASISdataset_train_all_sequence_val_T2flair_flair = copy.deepcopy(OASISdataset_train_all_sequence_val)
    OASISdataset_train_all_sequence_val_T2flair_flair.filter_by_sequence_labels(['T2flair_flair'])
    print(len(OASISdataset_train_all_sequence_val_T2flair_flair))
    OASISdataset_train_all_sequence_val_T2flair_flair.max_val_len = max_val_len
    OASISdataset_train_all_sequence_val_T2flair_flair._update_train_val_paths()
    print(len(OASISdataset_train_all_sequence_val_T2flair_flair))

    OASISdataset_train_all_sequence_val_PD = copy.deepcopy(OASISdataset_train_all_sequence_val)
    OASISdataset_train_all_sequence_val_PD.filter_by_sequence_labels(['PD'])
    print(len(OASISdataset_train_all_sequence_val_PD))
    OASISdataset_train_all_sequence_val_PD.max_val_len = max_val_len
    OASISdataset_train_all_sequence_val_PD._update_train_val_paths()
    print(len(OASISdataset_train_all_sequence_val_PD))

    OASISdataset_train_all_sequence_val_T2star_hemo = copy.deepcopy(OASISdataset_train_all_sequence_val)
    OASISdataset_train_all_sequence_val_T2star_hemo.filter_by_sequence_labels(['T2star_hemo'])
    print(len(OASISdataset_train_all_sequence_val_T2star_hemo))
    OASISdataset_train_all_sequence_val_T2star_hemo.max_val_len = max_val_len
    OASISdataset_train_all_sequence_val_T2star_hemo._update_train_val_paths()
    print(len(OASISdataset_train_all_sequence_val_T2star_hemo))

    OASISdataset_train_all_sequence_val_T2star_swi = copy.deepcopy(OASISdataset_train_all_sequence_val)
    OASISdataset_train_all_sequence_val_T2star_swi.filter_by_sequence_labels(['T2star_swi'])
    print(len(OASISdataset_train_all_sequence_val_T2star_swi))
    OASISdataset_train_all_sequence_val_T2star_swi.max_val_len = max_val_len
    OASISdataset_train_all_sequence_val_T2star_swi._update_train_val_paths()
    print(len(OASISdataset_train_all_sequence_val_T2star_swi))

    OASISdataset_train_all_sequence_val_DTI_DWI = copy.deepcopy(OASISdataset_train_all_sequence_val)
    OASISdataset_train_all_sequence_val_DTI_DWI.filter_by_sequence_labels(['DTI_DWI', 'DTI_DWI_500'])
    print(len(OASISdataset_train_all_sequence_val_DTI_DWI))
    OASISdataset_train_all_sequence_val_DTI_DWI.max_val_len = max_val_len
    OASISdataset_train_all_sequence_val_DTI_DWI._update_train_val_paths()
    print(len(OASISdataset_train_all_sequence_val_DTI_DWI))
    # save them
    with open('./index_list/OASISdataset_train_all_sequence_val_T1_T1flair.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_val_T1_T1flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_val_T1_T1flair))
    with open('./index_list/OASISdataset_train_all_sequence_val_T2.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_val_T2, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_val_T2))
    with open('./index_list/OASISdataset_train_all_sequence_val_T2flair_flair.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_val_T2flair_flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_val_T2flair_flair))
    with open('./index_list/OASISdataset_train_all_sequence_val_PD.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_val_PD, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_val_PD))
    with open('./index_list/OASISdataset_train_all_sequence_val_T2star_hemo.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_val_T2star_hemo, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_val_T2star_hemo))
    with open('./index_list/OASISdataset_train_all_sequence_val_T2star_swi.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_val_T2star_swi, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_val_T2star_swi))
    with open('./index_list/OASISdataset_train_all_sequence_val_DTI_DWI.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_val_DTI_DWI, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_val_DTI_DWI))
else:
    with open('./index_list/OASISdataset_train_all_sequence_val_T1_T1flair.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_val_T1_T1flair = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_val_T1_T1flair))
    with open('./index_list/OASISdataset_train_all_sequence_val_T2.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_val_T2 = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_val_T2))
    with open('./index_list/OASISdataset_train_all_sequence_val_T2flair_flair.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_val_T2flair_flair = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_val_T2flair_flair))
    with open('./index_list/OASISdataset_train_all_sequence_val_PD.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_val_PD = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_val_PD))
    with open('./index_list/OASISdataset_train_all_sequence_val_T2star_hemo.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_val_T2star_hemo = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_val_T2star_hemo))
    with open('./index_list/OASISdataset_train_all_sequence_val_T2star_swi.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_val_T2star_swi = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_val_T2star_swi))
    with open('./index_list/OASISdataset_train_all_sequence_val_DTI_DWI.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_val_DTI_DWI = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_val_DTI_DWI))

#['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi', 'DTI_DWI', 'DTI_DWI_500']
if rebuild_val_dataset:
    ADNIdataset_train_all_sequence_val_T1_T1flair = copy.deepcopy(ADNIdataset_train_all_sequence_val)
    ADNIdataset_train_all_sequence_val_T1_T1flair.filter_by_sequence_labels(['T1_T1flair'])
    print(len(ADNIdataset_train_all_sequence_val_T1_T1flair))
    ADNIdataset_train_all_sequence_val_T1_T1flair.max_val_len = max_val_len
    ADNIdataset_train_all_sequence_val_T1_T1flair._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_val_T1_T1flair))

    ADNIdataset_train_all_sequence_val_T2 = copy.deepcopy(ADNIdataset_train_all_sequence_val)
    ADNIdataset_train_all_sequence_val_T2.filter_by_sequence_labels(['T2'])
    print(len(ADNIdataset_train_all_sequence_val_T2))
    ADNIdataset_train_all_sequence_val_T2.max_val_len = max_val_len
    ADNIdataset_train_all_sequence_val_T2._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_val_T2))

    ADNIdataset_train_all_sequence_val_T2flair_flair = copy.deepcopy(ADNIdataset_train_all_sequence_val)
    ADNIdataset_train_all_sequence_val_T2flair_flair.filter_by_sequence_labels(['T2flair_flair'])
    print(len(ADNIdataset_train_all_sequence_val_T2flair_flair))
    ADNIdataset_train_all_sequence_val_T2flair_flair.max_val_len = max_val_len
    ADNIdataset_train_all_sequence_val_T2flair_flair._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_val_T2flair_flair))

    ADNIdataset_train_all_sequence_val_PD = copy.deepcopy(ADNIdataset_train_all_sequence_val)
    ADNIdataset_train_all_sequence_val_PD.filter_by_sequence_labels(['PD'])
    print(len(ADNIdataset_train_all_sequence_val_PD))
    ADNIdataset_train_all_sequence_val_PD.max_val_len = max_val_len
    ADNIdataset_train_all_sequence_val_PD._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_val_PD))

    ADNIdataset_train_all_sequence_val_T2star_hemo = copy.deepcopy(ADNIdataset_train_all_sequence_val)
    ADNIdataset_train_all_sequence_val_T2star_hemo.filter_by_sequence_labels(['T2star_hemo'])
    print(len(ADNIdataset_train_all_sequence_val_T2star_hemo))
    ADNIdataset_train_all_sequence_val_T2star_hemo.max_val_len = max_val_len
    ADNIdataset_train_all_sequence_val_T2star_hemo._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_val_T2star_hemo))

    ADNIdataset_train_all_sequence_val_T2star_swi = copy.deepcopy(ADNIdataset_train_all_sequence_val)
    ADNIdataset_train_all_sequence_val_T2star_swi.filter_by_sequence_labels(['T2star_swi'])
    print(len(ADNIdataset_train_all_sequence_val_T2star_swi))
    ADNIdataset_train_all_sequence_val_T2star_swi.max_val_len = max_val_len
    ADNIdataset_train_all_sequence_val_T2star_swi._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_val_T2star_swi))

    ADNIdataset_train_all_sequence_val_DTI_DWI = copy.deepcopy(ADNIdataset_train_all_sequence_val)
    ADNIdataset_train_all_sequence_val_DTI_DWI.filter_by_sequence_labels(['DTI_DWI', 'DTI_DWI_500'])
    print(len(ADNIdataset_train_all_sequence_val_DTI_DWI))
    ADNIdataset_train_all_sequence_val_DTI_DWI.max_val_len = max_val_len
    ADNIdataset_train_all_sequence_val_DTI_DWI._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_val_DTI_DWI))
    # save them
    with open('./index_list/ADNIdataset_train_all_sequence_val_T1_T1flair.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_val_T1_T1flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_val_T1_T1flair))
    with open('./index_list/ADNIdataset_train_all_sequence_val_T2.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_val_T2, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_val_T2))
    with open('./index_list/ADNIdataset_train_all_sequence_val_T2flair_flair.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_val_T2flair_flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_val_T2flair_flair))
    with open('./index_list/ADNIdataset_train_all_sequence_val_PD.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_val_PD, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_val_PD))
    with open('./index_list/ADNIdataset_train_all_sequence_val_T2star_hemo.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_val_T2star_hemo, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_val_T2star_hemo))
    with open('./index_list/ADNIdataset_train_all_sequence_val_T2star_swi.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_val_T2star_swi, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_val_T2star_swi))
    with open('./index_list/ADNIdataset_train_all_sequence_val_DTI_DWI.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_val_DTI_DWI, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_val_DTI_DWI))
else:
    with open('./index_list/ADNIdataset_train_all_sequence_val_T1_T1flair.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_val_T1_T1flair = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_val_T1_T1flair))
    with open('./index_list/ADNIdataset_train_all_sequence_val_T2.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_val_T2 = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_val_T2))
    with open('./index_list/ADNIdataset_train_all_sequence_val_T2flair_flair.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_val_T2flair_flair = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_val_T2flair_flair))
    with open('./index_list/ADNIdataset_train_all_sequence_val_PD.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_val_PD = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_val_PD))
    with open('./index_list/ADNIdataset_train_all_sequence_val_T2star_hemo.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_val_T2star_hemo = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_val_T2star_hemo))
    with open('./index_list/ADNIdataset_train_all_sequence_val_T2star_swi.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_val_T2star_swi = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_val_T2star_swi))
    with open('./index_list/ADNIdataset_train_all_sequence_val_DTI_DWI.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_val_DTI_DWI = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_val_DTI_DWI))

#['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi', 'DTI_DWI', 'DTI_DWI_500']
if rebuild_val_dataset:
    NACCdataset_train_all_sequence_val_T1_T1flair = copy.deepcopy(NACCdataset_train_all_sequence_val)
    NACCdataset_train_all_sequence_val_T1_T1flair.filter_by_sequence_labels(['T1_T1flair'])
    print(len(NACCdataset_train_all_sequence_val_T1_T1flair))
    NACCdataset_train_all_sequence_val_T1_T1flair.max_val_len = max_val_len
    NACCdataset_train_all_sequence_val_T1_T1flair._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_val_T1_T1flair))

    NACCdataset_train_all_sequence_val_T2 = copy.deepcopy(NACCdataset_train_all_sequence_val)
    NACCdataset_train_all_sequence_val_T2.filter_by_sequence_labels(['T2'])
    print(len(NACCdataset_train_all_sequence_val_T2))
    NACCdataset_train_all_sequence_val_T2.max_val_len = max_val_len
    NACCdataset_train_all_sequence_val_T2._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_val_T2))

    NACCdataset_train_all_sequence_val_T2flair_flair = copy.deepcopy(NACCdataset_train_all_sequence_val)
    NACCdataset_train_all_sequence_val_T2flair_flair.filter_by_sequence_labels(['T2flair_flair'])
    print(len(NACCdataset_train_all_sequence_val_T2flair_flair))
    NACCdataset_train_all_sequence_val_T2flair_flair.max_val_len = max_val_len
    NACCdataset_train_all_sequence_val_T2flair_flair._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_val_T2flair_flair))

    NACCdataset_train_all_sequence_val_PD = copy.deepcopy(NACCdataset_train_all_sequence_val)
    NACCdataset_train_all_sequence_val_PD.filter_by_sequence_labels(['PD'])
    print(len(NACCdataset_train_all_sequence_val_PD))
    NACCdataset_train_all_sequence_val_PD.max_val_len = max_val_len
    NACCdataset_train_all_sequence_val_PD._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_val_PD))

    NACCdataset_train_all_sequence_val_T2star_hemo = copy.deepcopy(NACCdataset_train_all_sequence_val)
    NACCdataset_train_all_sequence_val_T2star_hemo.filter_by_sequence_labels(['T2star_hemo'])
    print(len(NACCdataset_train_all_sequence_val_T2star_hemo))
    NACCdataset_train_all_sequence_val_T2star_hemo.max_val_len = max_val_len
    NACCdataset_train_all_sequence_val_T2star_hemo._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_val_T2star_hemo))

    NACCdataset_train_all_sequence_val_T2star_swi = copy.deepcopy(NACCdataset_train_all_sequence_val)
    NACCdataset_train_all_sequence_val_T2star_swi.filter_by_sequence_labels(['T2star_swi'])
    print(len(NACCdataset_train_all_sequence_val_T2star_swi))
    NACCdataset_train_all_sequence_val_T2star_swi.max_val_len = max_val_len
    NACCdataset_train_all_sequence_val_T2star_swi._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_val_T2star_swi))

    NACCdataset_train_all_sequence_val_DTI_DWI = copy.deepcopy(NACCdataset_train_all_sequence_val)
    NACCdataset_train_all_sequence_val_DTI_DWI.filter_by_sequence_labels(['DTI_DWI', 'DTI_DWI_500'])
    print(len(NACCdataset_train_all_sequence_val_DTI_DWI))
    NACCdataset_train_all_sequence_val_DTI_DWI.max_val_len = max_val_len
    NACCdataset_train_all_sequence_val_DTI_DWI._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_val_DTI_DWI))
    # save them
    with open('./index_list/NACCdataset_train_all_sequence_val_T1_T1flair.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_val_T1_T1flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_val_T1_T1flair))
    with open('./index_list/NACCdataset_train_all_sequence_val_T2.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_val_T2, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_val_T2))
    with open('./index_list/NACCdataset_train_all_sequence_val_T2flair_flair.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_val_T2flair_flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_val_T2flair_flair))
    with open('./index_list/NACCdataset_train_all_sequence_val_PD.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_val_PD, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_val_PD))
    with open('./index_list/NACCdataset_train_all_sequence_val_T2star_hemo.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_val_T2star_hemo, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_val_T2star_hemo))
    with open('./index_list/NACCdataset_train_all_sequence_val_T2star_swi.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_val_T2star_swi, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_val_T2star_swi))
    with open('./index_list/NACCdataset_train_all_sequence_val_DTI_DWI.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_val_DTI_DWI, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_val_DTI_DWI))
else:
    with open('./index_list/NACCdataset_train_all_sequence_val_T1_T1flair.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_val_T1_T1flair = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_val_T1_T1flair))
    with open('./index_list/NACCdataset_train_all_sequence_val_T2.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_val_T2 = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_val_T2))
    with open('./index_list/NACCdataset_train_all_sequence_val_T2flair_flair.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_val_T2flair_flair = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_val_T2flair_flair))
    with open('./index_list/NACCdataset_train_all_sequence_val_PD.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_val_PD = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_val_PD))
    with open('./index_list/NACCdataset_train_all_sequence_val_T2star_hemo.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_val_T2star_hemo = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_val_T2star_hemo))
    with open('./index_list/NACCdataset_train_all_sequence_val_T2star_swi.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_val_T2star_swi = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_val_T2star_swi))
    with open('./index_list/NACCdataset_train_all_sequence_val_DTI_DWI.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_val_DTI_DWI = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_val_DTI_DWI))

# train dataset construction
#['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi', 'DTI_DWI', 'DTI_DWI_500']
max_train_len = 100
rebuild_train_dataset= True
if rebuild_train_dataset:
    OASISdataset_train_all_sequence_train_T1_T1flair = copy.deepcopy(OASISdataset_train_all_sequence_val_T1_T1flair)
    OASISdataset_train_all_sequence_train_T1_T1flair.is_train = True
    OASISdataset_train_all_sequence_train_T1_T1flair.update_train_paths_w_shuffle(max_train_len)    # shuffle train dataset
    # OASISdataset_train_all_sequence_train_T1_T1flair.max_train_len = max_train_len
    print(len(OASISdataset_train_all_sequence_train_T1_T1flair))

    OASISdataset_train_all_sequence_train_T2 = copy.deepcopy(OASISdataset_train_all_sequence_val_T2)
    OASISdataset_train_all_sequence_train_T2.is_train = True
    OASISdataset_train_all_sequence_train_T2.update_train_paths_w_shuffle(max_train_len)    # shuffle train dataset
    # OASISdataset_train_all_sequence_train_T2.max_train_len = max_train_len
    print(len(OASISdataset_train_all_sequence_train_T2))

    OASISdataset_train_all_sequence_train_T2flair_flair = copy.deepcopy(OASISdataset_train_all_sequence_val_T2flair_flair)
    OASISdataset_train_all_sequence_train_T2flair_flair.is_train = True
    OASISdataset_train_all_sequence_train_T2flair_flair.update_train_paths_w_shuffle(max_train_len)
    # OASISdataset_train_all_sequence_train_T2flair_flair.max_train_len = max_train_len
    print(len(OASISdataset_train_all_sequence_train_T2flair_flair))

    OASISdataset_train_all_sequence_train_PD = copy.deepcopy(OASISdataset_train_all_sequence_val_PD)
    OASISdataset_train_all_sequence_train_PD.is_train = True
    OASISdataset_train_all_sequence_train_PD.update_train_paths_w_shuffle(max_train_len)
    # OASISdataset_train_all_sequence_train_PD.max_train_len = max_train_len
    print(len(OASISdataset_train_all_sequence_train_PD))

    OASISdataset_train_all_sequence_train_T2star_hemo = copy.deepcopy(OASISdataset_train_all_sequence_val_T2star_hemo)
    OASISdataset_train_all_sequence_train_T2star_hemo.is_train = True
    OASISdataset_train_all_sequence_train_T2star_hemo.update_train_paths_w_shuffle(max_train_len)
    # OASISdataset_train_all_sequence_train_T2star_hemo.max_train_len = max_train_len
    print(len(OASISdataset_train_all_sequence_train_T2star_hemo))

    OASISdataset_train_all_sequence_train_T2star_swi = copy.deepcopy(OASISdataset_train_all_sequence_val_T2star_swi)
    OASISdataset_train_all_sequence_train_T2star_swi.is_train = True
    OASISdataset_train_all_sequence_train_T2star_swi.update_train_paths_w_shuffle(max_train_len)
    # OASISdataset_train_all_sequence_train_T2star_swi.max_train_len = max_train_len
    print(len(OASISdataset_train_all_sequence_train_T2star_swi))

    OASISdataset_train_all_sequence_train_DTI_DWI = copy.deepcopy(OASISdataset_train_all_sequence_val_DTI_DWI)
    OASISdataset_train_all_sequence_train_DTI_DWI.is_train = True
    OASISdataset_train_all_sequence_train_DTI_DWI.update_train_paths_w_shuffle(max_train_len)
    # OASISdataset_train_all_sequence_train_DTI_DWI.max_train_len = max_train_len
    print(len(OASISdataset_train_all_sequence_train_DTI_DWI))
    # save them
    with open('./index_list/OASISdataset_train_all_sequence_train_T1_T1flair.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_train_T1_T1flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_train_T1_T1flair))
    with open('./index_list/OASISdataset_train_all_sequence_train_T2.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_train_T2, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_train_T2))
    with open('./index_list/OASISdataset_train_all_sequence_train_T2flair_flair.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_train_T2flair_flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_train_T2flair_flair))
    with open('./index_list/OASISdataset_train_all_sequence_train_PD.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_train_PD, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_train_PD))
    with open('./index_list/OASISdataset_train_all_sequence_train_T2star_hemo.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_train_T2star_hemo, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_train_T2star_hemo))
    with open('./index_list/OASISdataset_train_all_sequence_train_T2star_swi.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_train_T2star_swi, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_train_T2star_swi))
    with open('./index_list/OASISdataset_train_all_sequence_train_DTI_DWI.pkl', 'wb') as outp:
        pickle.dump(OASISdataset_train_all_sequence_train_DTI_DWI, outp, pickle.HIGHEST_PROTOCOL)
        print(len(OASISdataset_train_all_sequence_train_DTI_DWI))
else:
    with open('./index_list/OASISdataset_train_all_sequence_train_T1_T1flair.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_train_T1_T1flair = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_train_T1_T1flair))
    with open('./index_list/OASISdataset_train_all_sequence_train_T2.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_train_T2 = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_train_T2))
    with open('./index_list/OASISdataset_train_all_sequence_train_T2flair_flair.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_train_T2flair_flair = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_train_T2flair_flair))
    with open('./index_list/OASISdataset_train_all_sequence_train_PD.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_train_PD = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_train_PD))
    with open('./index_list/OASISdataset_train_all_sequence_train_T2star_hemo.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_train_T2star_hemo = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_train_T2star_hemo))
    with open('./index_list/OASISdataset_train_all_sequence_train_T2star_swi.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_train_T2star_swi = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_train_T2star_swi))
    with open('./index_list/OASISdataset_train_all_sequence_train_DTI_DWI.pkl', 'rb') as inp:
        OASISdataset_train_all_sequence_train_DTI_DWI = pickle.load(inp)
        print(len(OASISdataset_train_all_sequence_train_DTI_DWI))
# train dataset construction
#['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi', 'DTI_DWI', 'DTI_DWI_500']
if rebuild_train_dataset:
    ADNIdataset_train_all_sequence_train_T1_T1flair = copy.deepcopy(ADNIdataset_train_all_sequence_val_T1_T1flair)
    ADNIdataset_train_all_sequence_train_T1_T1flair.is_train = True
    ADNIdataset_train_all_sequence_train_T1_T1flair.update_train_paths_w_shuffle(max_train_len)
    # ADNIdataset_train_all_sequence_train_T1_T1flair.max_train_len = max_train_len
    ADNIdataset_train_all_sequence_train_T1_T1flair._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_train_T1_T1flair))

    ADNIdataset_train_all_sequence_train_T2 = copy.deepcopy(ADNIdataset_train_all_sequence_val_T2)
    ADNIdataset_train_all_sequence_train_T2.is_train = True
    ADNIdataset_train_all_sequence_train_T2.update_train_paths_w_shuffle(max_train_len)
    # ADNIdataset_train_all_sequence_train_T2.max_train_len = max_train_len
    ADNIdataset_train_all_sequence_train_T2._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_train_T2))

    ADNIdataset_train_all_sequence_train_T2flair_flair = copy.deepcopy(ADNIdataset_train_all_sequence_val_T2flair_flair)
    ADNIdataset_train_all_sequence_train_T2flair_flair.is_train = True
    ADNIdataset_train_all_sequence_train_T2flair_flair.update_train_paths_w_shuffle(max_train_len)
    # ADNIdataset_train_all_sequence_train_T2flair_flair.max_train_len = max_train_len
    ADNIdataset_train_all_sequence_train_T2flair_flair._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_train_T2flair_flair))

    ADNIdataset_train_all_sequence_train_PD = copy.deepcopy(ADNIdataset_train_all_sequence_val_PD)
    ADNIdataset_train_all_sequence_train_PD.is_train = True
    ADNIdataset_train_all_sequence_train_PD.update_train_paths_w_shuffle(max_train_len)
    # ADNIdataset_train_all_sequence_train_PD.max_train_len = max_train_len
    ADNIdataset_train_all_sequence_train_PD._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_train_PD))

    ADNIdataset_train_all_sequence_train_T2star_hemo = copy.deepcopy(ADNIdataset_train_all_sequence_val_T2star_hemo)
    ADNIdataset_train_all_sequence_train_T2star_hemo.is_train = True
    ADNIdataset_train_all_sequence_train_T2star_hemo.update_train_paths_w_shuffle(max_train_len)
    # ADNIdataset_train_all_sequence_train_T2star_hemo.max_train_len = max_train_len
    ADNIdataset_train_all_sequence_train_T2star_hemo._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_train_T2star_hemo))

    ADNIdataset_train_all_sequence_train_T2star_swi = copy.deepcopy(ADNIdataset_train_all_sequence_val_T2star_swi)
    ADNIdataset_train_all_sequence_train_T2star_swi.is_train = True
    ADNIdataset_train_all_sequence_train_T2star_swi.update_train_paths_w_shuffle(max_train_len)
    # ADNIdataset_train_all_sequence_train_T2star_swi.max_train_len = max_train_len
    ADNIdataset_train_all_sequence_train_T2star_swi._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_train_T2star_swi))

    ADNIdataset_train_all_sequence_train_DTI_DWI = copy.deepcopy(ADNIdataset_train_all_sequence_val_DTI_DWI)
    ADNIdataset_train_all_sequence_train_DTI_DWI.is_train = True
    ADNIdataset_train_all_sequence_train_DTI_DWI.update_train_paths_w_shuffle(max_train_len)
    # ADNIdataset_train_all_sequence_train_DTI_DWI.max_train_len = max_train_len
    ADNIdataset_train_all_sequence_train_DTI_DWI._update_train_val_paths()
    print(len(ADNIdataset_train_all_sequence_train_DTI_DWI))
    # save them
    with open('./index_list/ADNIdataset_train_all_sequence_train_T1_T1flair.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_train_T1_T1flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_train_T1_T1flair))
    with open('./index_list/ADNIdataset_train_all_sequence_train_T2.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_train_T2, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_train_T2))
    with open('./index_list/ADNIdataset_train_all_sequence_train_T2flair_flair.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_train_T2flair_flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_train_T2flair_flair))
    with open('./index_list/ADNIdataset_train_all_sequence_train_PD.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_train_PD, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_train_PD))
    with open('./index_list/ADNIdataset_train_all_sequence_train_T2star_hemo.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_train_T2star_hemo, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_train_T2star_hemo))
    with open('./index_list/ADNIdataset_train_all_sequence_train_T2star_swi.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_train_T2star_swi, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_train_T2star_swi))
    with open('./index_list/ADNIdataset_train_all_sequence_train_DTI_DWI.pkl', 'wb') as outp:
        pickle.dump(ADNIdataset_train_all_sequence_train_DTI_DWI, outp, pickle.HIGHEST_PROTOCOL)
        print(len(ADNIdataset_train_all_sequence_train_DTI_DWI))
else:
    with open('./index_list/ADNIdataset_train_all_sequence_train_T1_T1flair.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_train_T1_T1flair = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_train_T1_T1flair))
    with open('./index_list/ADNIdataset_train_all_sequence_train_T2.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_train_T2 = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_train_T2))
    with open('./index_list/ADNIdataset_train_all_sequence_train_T2flair_flair.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_train_T2flair_flair = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_train_T2flair_flair))
    with open('./index_list/ADNIdataset_train_all_sequence_train_PD.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_train_PD = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_train_PD))
    with open('./index_list/ADNIdataset_train_all_sequence_train_T2star_hemo.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_train_T2star_hemo = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_train_T2star_hemo))
    with open('./index_list/ADNIdataset_train_all_sequence_train_T2star_swi.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_train_T2star_swi = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_train_T2star_swi))
    with open('./index_list/ADNIdataset_train_all_sequence_train_DTI_DWI.pkl', 'rb') as inp:
        ADNIdataset_train_all_sequence_train_DTI_DWI = pickle.load(inp)
        print(len(ADNIdataset_train_all_sequence_train_DTI_DWI))

# train dataset construction
#['T1_T1flair', 'T2', 'T2flair_flair', 'PD', 'T2star_hemo', 'T2star_swi', 'DTI_DWI', 'DTI_DWI_500']
if rebuild_train_dataset:
    NACCdataset_train_all_sequence_train_T1_T1flair = copy.deepcopy(NACCdataset_train_all_sequence_val_T1_T1flair)
    NACCdataset_train_all_sequence_train_T1_T1flair.is_train = True
    NACCdataset_train_all_sequence_train_T1_T1flair.update_train_paths_w_shuffle(max_train_len)
    # NACCdataset_train_all_sequence_train_T1_T1flair.max_train_len = max_train_len
    NACCdataset_train_all_sequence_train_T1_T1flair._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_train_T1_T1flair))

    NACCdataset_train_all_sequence_train_T2 = copy.deepcopy(NACCdataset_train_all_sequence_val_T2)
    NACCdataset_train_all_sequence_train_T2.is_train = True
    NACCdataset_train_all_sequence_train_T2.update_train_paths_w_shuffle(max_train_len)
    # NACCdataset_train_all_sequence_train_T2.max_train_len = max_train_len
    NACCdataset_train_all_sequence_train_T2._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_train_T2))

    NACCdataset_train_all_sequence_train_T2flair_flair = copy.deepcopy(NACCdataset_train_all_sequence_val_T2flair_flair)
    NACCdataset_train_all_sequence_train_T2flair_flair.is_train = True
    NACCdataset_train_all_sequence_train_T2flair_flair.update_train_paths_w_shuffle(max_train_len)
    # NACCdataset_train_all_sequence_train_T2flair_flair.max_train_len = max_train_len
    NACCdataset_train_all_sequence_train_T2flair_flair._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_train_T2flair_flair))

    NACCdataset_train_all_sequence_train_PD = copy.deepcopy(NACCdataset_train_all_sequence_val_PD)
    NACCdataset_train_all_sequence_train_PD.is_train = True
    NACCdataset_train_all_sequence_train_PD.update_train_paths_w_shuffle(max_train_len)
    # NACCdataset_train_all_sequence_train_PD.max_train_len = max_train_len
    NACCdataset_train_all_sequence_train_PD._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_train_PD))

    NACCdataset_train_all_sequence_train_T2star_hemo = copy.deepcopy(NACCdataset_train_all_sequence_val_T2star_hemo)
    NACCdataset_train_all_sequence_train_T2star_hemo.is_train = True
    NACCdataset_train_all_sequence_train_T2star_hemo.update_train_paths_w_shuffle(max_train_len)
    # NACCdataset_train_all_sequence_train_T2star_hemo.max_train_len = max_train_len
    NACCdataset_train_all_sequence_train_T2star_hemo._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_train_T2star_hemo))

    NACCdataset_train_all_sequence_train_T2star_swi = copy.deepcopy(NACCdataset_train_all_sequence_val_T2star_swi)
    NACCdataset_train_all_sequence_train_T2star_swi.is_train = True
    NACCdataset_train_all_sequence_train_T2star_swi.update_train_paths_w_shuffle(max_train_len)
    # NACCdataset_train_all_sequence_train_T2star_swi.max_train_len = max_train_len
    NACCdataset_train_all_sequence_train_T2star_swi._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_train_T2star_swi))

    NACCdataset_train_all_sequence_train_DTI_DWI = copy.deepcopy(NACCdataset_train_all_sequence_val_DTI_DWI)
    NACCdataset_train_all_sequence_train_DTI_DWI.is_train = True
    NACCdataset_train_all_sequence_train_DTI_DWI.update_train_paths_w_shuffle(max_train_len)
    # NACCdataset_train_all_sequence_train_DTI_DWI.max_train_len = max_train_len
    NACCdataset_train_all_sequence_train_DTI_DWI._update_train_val_paths()
    print(len(NACCdataset_train_all_sequence_train_DTI_DWI))
    # save them
    with open('./index_list/NACCdataset_train_all_sequence_train_T1_T1flair.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_train_T1_T1flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_train_T1_T1flair))
    with open('./index_list/NACCdataset_train_all_sequence_train_T2.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_train_T2, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_train_T2))
    with open('./index_list/NACCdataset_train_all_sequence_train_T2flair_flair.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_train_T2flair_flair, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_train_T2flair_flair))
    with open('./index_list/NACCdataset_train_all_sequence_train_PD.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_train_PD, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_train_PD))
    with open('./index_list/NACCdataset_train_all_sequence_train_T2star_hemo.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_train_T2star_hemo, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_train_T2star_hemo))
    with open('./index_list/NACCdataset_train_all_sequence_train_T2star_swi.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_train_T2star_swi, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_train_T2star_swi))
    with open('./index_list/NACCdataset_train_all_sequence_train_DTI_DWI.pkl', 'wb') as outp:
        pickle.dump(NACCdataset_train_all_sequence_train_DTI_DWI, outp, pickle.HIGHEST_PROTOCOL)
        print(len(NACCdataset_train_all_sequence_train_DTI_DWI))
else:
    with open('./index_list/NACCdataset_train_all_sequence_train_T1_T1flair.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_train_T1_T1flair = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_train_T1_T1flair))
    with open('./index_list/NACCdataset_train_all_sequence_train_T2.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_train_T2 = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_train_T2))
    with open('./index_list/NACCdataset_train_all_sequence_train_T2flair_flair.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_train_T2flair_flair = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_train_T2flair_flair))
    with open('./index_list/NACCdataset_train_all_sequence_train_PD.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_train_PD = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_train_PD))
    with open('./index_list/NACCdataset_train_all_sequence_train_T2star_hemo.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_train_T2star_hemo = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_train_T2star_hemo))
    with open('./index_list/NACCdataset_train_all_sequence_train_T2star_swi.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_train_T2star_swi = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_train_T2star_swi))
    with open('./index_list/NACCdataset_train_all_sequence_train_DTI_DWI.pkl', 'rb') as inp:
        NACCdataset_train_all_sequence_train_DTI_DWI = pickle.load(inp)
        print(len(NACCdataset_train_all_sequence_train_DTI_DWI))

BATCH_SIZE= 128 * 4 # 128 * 4
datasets_train = [OASISdataset_train_all_sequence_train_T1_T1flair, OASISdataset_train_all_sequence_train_T2, OASISdataset_train_all_sequence_train_T2flair_flair, OASISdataset_train_all_sequence_train_PD, OASISdataset_train_all_sequence_train_T2star_hemo, OASISdataset_train_all_sequence_train_T2star_swi, OASISdataset_train_all_sequence_train_DTI_DWI,
                  ADNIdataset_train_all_sequence_train_T1_T1flair, ADNIdataset_train_all_sequence_train_T2, ADNIdataset_train_all_sequence_train_T2flair_flair, ADNIdataset_train_all_sequence_train_PD, ADNIdataset_train_all_sequence_train_T2star_hemo, ADNIdataset_train_all_sequence_train_T2star_swi, ADNIdataset_train_all_sequence_train_DTI_DWI,
                  NACCdataset_train_all_sequence_train_T1_T1flair, NACCdataset_train_all_sequence_train_T2, NACCdataset_train_all_sequence_train_T2flair_flair, NACCdataset_train_all_sequence_train_PD, NACCdataset_train_all_sequence_train_T2star_hemo, NACCdataset_train_all_sequence_train_T2star_swi, NACCdataset_train_all_sequence_train_DTI_DWI]

num_workers = 0
dataloader_train = create_combine_dataloader(
        datasets= datasets_train,
        batch_size = BATCH_SIZE,
        is_distributed=False,
        is_train=True,
        num_workers = num_workers,
)

len_datasets = 0
for dataset in datasets_train:
    len_datasets += len(dataset)
print('len(train_dataset):', len_datasets)
print('len(train_dataloader):', len(dataloader_train))

rebuild_datasets_val = True

if rebuild_datasets_val:
    datasets_val = [OASISdataset_train_all_sequence_val_T1_T1flair, OASISdataset_train_all_sequence_val_T2, OASISdataset_train_all_sequence_val_T2flair_flair, OASISdataset_train_all_sequence_val_PD, OASISdataset_train_all_sequence_val_T2star_hemo, OASISdataset_train_all_sequence_val_T2star_swi, OASISdataset_train_all_sequence_val_DTI_DWI,
                      ADNIdataset_train_all_sequence_val_T1_T1flair, ADNIdataset_train_all_sequence_val_T2, ADNIdataset_train_all_sequence_val_T2flair_flair, ADNIdataset_train_all_sequence_val_PD, ADNIdataset_train_all_sequence_val_T2star_hemo, ADNIdataset_train_all_sequence_val_T2star_swi, ADNIdataset_train_all_sequence_val_DTI_DWI,
                      NACCdataset_train_all_sequence_val_T1_T1flair, NACCdataset_train_all_sequence_val_T2, NACCdataset_train_all_sequence_val_T2flair_flair, NACCdataset_train_all_sequence_val_PD, NACCdataset_train_all_sequence_val_T2star_hemo, NACCdataset_train_all_sequence_val_T2star_swi, NACCdataset_train_all_sequence_val_DTI_DWI]

    # reset random crop flip and rot to 0 random_crop_change = 0.5# 0.5
# random_flip_chance = 0.5# 0.5
# random_rotate_chance = 0.85# 0.85


    for dataset in datasets_val:
        dataset.random_crop_change = 0.0
        dataset.random_flip_chance = 0.0
        dataset.random_rotate_chance = 0.0


    num_workers = 0
    batch_size = BATCH_SIZE
    # batch_size = 128 * 4
    for dataset in datasets_val:
        print(dataset.__class__, dataset.random_crop_change, dataset.random_flip_chance, dataset.random_rotate_chance)
    dataloader_val = create_combine_dataloader(
            datasets= datasets_val,
            batch_size = batch_size,
            is_distributed=False,
            is_train=False,
            num_workers = num_workers,
    )
    with open('./index_list/dataloader_val.pkl', 'wb') as outp:
        pickle.dump(dataloader_val, outp, pickle.HIGHEST_PROTOCOL)
        print(len(dataloader_val))
else:
    with open('./index_list/dataloader_val.pkl', 'rb') as inp:
        dataloader_val = pickle.load(inp)
        print(len(dataloader_val))

print('len(dataloader_val):', len(dataloader_val))

from sequence_detection.models_mae_finetune import MaskedAutoencoderViTClassify
from functools import partial
import torch
import torch.nn as nn
model = MaskedAutoencoderViTClassify(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=len(detect_sequence) + 1, mode='cls')

pretrain_path = 'D:/Mengyu_Li/General_Dataloader_Git_V1/saved_models/mae_vit_base_patch16_pretrain_test0.75_E30/model_E30.pt'

missing, unexpected = model.load_state_dict(torch.load(pretrain_path)['model_state_dict'], strict=False)  # strict=False ignores unmatched keys
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

from sequence_detection.train_mae_finetune import Trainer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
loss_fn = torch.nn.CrossEntropyLoss()
freeze_mae_encoder = True
TRAIN_EPOCHS = 1000
cls_strategy = 'cls_token'
if freeze_mae_encoder:
    path_save = "../saved_models/sequence_detection/max_train_len_freeze_mae_encoder_" + cls_strategy + "_e" + str(TRAIN_EPOCHS) + '_' + str(max_train_len) + '/'
else:
    path_save = "../saved_models/sequence_detection/max_train_len_nofreeze_mae_encoder_" + cls_strategy + "_e" + str(TRAIN_EPOCHS) + '_' + str(max_train_len) + '/'
create_path(path_save)
trainer = Trainer(
    loader_train=dataloader_train,
    loader_test=dataloader_val,
    my_model=model,
    my_loss=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer,
    RESUME_EPOCH=0,
    PATH_MODEL=path_save,
    device=device,
    cls_strategy = cls_strategy,  # or 'cls_token', 'mean_patch', 'mean_all', 'attn_pool',
    freeze_mae_encoder=freeze_mae_encoder,
    freeze_mae_encoder_decoder=freeze_mae_encoder, # free encoder and decoder for classification to get actual trainable params printed out
)

trainer.train(epochs=TRAIN_EPOCHS, show_step=500, show_test=True)
