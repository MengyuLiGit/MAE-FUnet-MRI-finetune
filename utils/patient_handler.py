import torch
from utils.general_loader import PatientCase
from tqdm import tqdm
import pickle
import os
from utils.general_dataloader import process_image_array
from utils.data_utils import torch_tensor_loader
import gc
import time
import numpy as np
from help_func import print_var_detail
import nibabel
from utils.general_utils import (get_root_folder, get_file_extension)
class PatientHandler():
    """
    This class handles patient cases, process them by trained model, e.g.
    """

    def __init__(self, detect_sequence_list, batch_size_volume, drop_fraction, n_color=1, image_size=None,
                 device='cpu', max_volume_size=None, min_volume_size=0):
        """
        :param max_volume_size: maximum size used for each slice volume, e.g. if the volume go beyond that, the handler
        will only use [0:max_volume_size] slices
        :param min_volume_size: assume the minimum volume that required for the handler, if volume size smaller than
        that, the handler will utilize all slices excluding any dropping or ignoring in the member functions
        :
        """
        self.n_color = n_color
        self.image_size = image_size
        self.detect_sequence_list = detect_sequence_list
        self.batch_size_volume = batch_size_volume
        self.drop_fraction = drop_fraction
        self.device = device
        self.max_volume_size = max_volume_size
        self.min_volume_size = min_volume_size

    def load_mri_slice_tensor(self, patient, slice_index):
        """
        load mri as tensor given file index
        :param slice_index: [0,0,0,[0, 0]] or [0,0,0,0, 0]
        :return mri volume tensor: [1, n_color, H, W]
        """
        image = patient.load_mri_slice(slice_index)
        if self.image_size is None:
            self.image_size = max(image.shape)
        image_tensor = process_image_array(image, self.image_size).unsqueeze(0).repeat(self.n_color, 1,
                                                                                       1).float()  # [C, H, W]
        return image_tensor.unsqueeze(0) # [1, C, H, W]
    def load_mri_file_tensor(self, patient, file_index, max_tensor_size=None, min_tensor_size=None, is_volume=False, load_plane=2):
        """
        load mri as tensor given file index
        :param file_index: [0,0,0,[0]] or [0,0,0,0]
        :return mri volume tensor: [num_slice, n_color, H, W]
        """
        if is_volume:
            mri_volume = patient.load_mri_volume(file_index)
            if mri_volume is None:
                return None
            mri_volume_shape = mri_volume.shape
            if len(mri_volume_shape) == 3:  # H, W, C
                H,W,C = mri_volume_shape
            elif len(mri_volume_shape) == 4:  # H, W, series_index_i, slice_index_i
                H,W,C,series_index_i  = mri_volume_shape

            image_volume = []
            if max_tensor_size is not None:
                volume_size = min(max_tensor_size, (H,W,C)[load_plane])
            else:
                volume_size = (H,W,C)[load_plane]
            if min_tensor_size is None:
                min_tensor_size = 0
            for i in range(volume_size):
                if len(mri_volume_shape) == 3:  # H, W, C
                    if load_plane == 0:
                        image = mri_volume[i,:,:]
                    elif load_plane == 1:
                        image = mri_volume[:,i,:]
                    elif load_plane == 2:
                        image = mri_volume[:,:,i]
                elif len(mri_volume_shape) == 4:  # H, W, series_index_i, slice_index_i
                    # only use the first series to check, assume same sequence for all series for now
                    if load_plane == 0:
                        image = mri_volume[i,:,:,0]
                    elif load_plane == 1:
                        image = mri_volume[:,i,:,0]
                    elif load_plane == 2:
                        image = mri_volume[:,:,i,0]
                if self.image_size is None:
                    self.image_size = max(image.shape)
                start0 = time.time()
                image_tensor = process_image_array(image, self.image_size).unsqueeze(0).repeat(self.n_color, 1,
                                                                                               1).float()  # [C, H, W]
                start1 = time.time()
                if (start1 - start0) > 0.1:
                    print(patient.patient_ID + " start1 - start0 ", start1 - start0)
                    print(file_index)
                    print_var_detail(image_tensor)
                image_volume.append(image_tensor.numpy())
                start2 = time.time()
                if (start2 - start1) > 0.1:
                    print("start2 - start1 ", start2 - start1)
        else:
            slice_index_list = patient.get_slice_index_list_in_file(file_index)
            # image_volume = torch.Tensor() # tensor way for concat
            # image_volume = np.array([], dtype=np.float64).reshape(0,self.n_color, self.image_size, self.image_size)# [0, C, H, W] # np way to concat
            image_volume = []  # list way to concat, using most memory but fastest
            if max_tensor_size is not None:
                volume_size = min(max_tensor_size, len(slice_index_list))
            else:
                volume_size = len(slice_index_list)
            if min_tensor_size is None:
                min_tensor_size = 0
            # for i in tqdm(range(volume_size)):
            for i in range(volume_size):
                slice_index = slice_index_list[i]
                image = patient.load_mri_slice(slice_index)
                if self.image_size is None:
                    self.image_size = max(image.shape)
                start0 = time.time()
                image_tensor = process_image_array(image, self.image_size).unsqueeze(0).repeat(self.n_color, 1,
                                                                                               1).float()  # [C, H, W]
                start1 = time.time()
                if (start1 - start0) > 0.1:
                    print(patient.patient_ID + " start1 - start0 ", start1 - start0)
                    print(slice_index)
                    print_var_detail(image_tensor)
                # image_volume = torch.cat((image_volume, image_tensor), 0)
                # image_volume = np.concatenate((image_volume, image_tensor.numpy()), axis=0)
                image_volume.append(image_tensor.numpy())
                start2 = time.time()
                if (start2 - start1) > 0.1:
                    print("start2 - start1 ", start2 - start1)
        image_volume = np.array(image_volume, np.float32)

        # remove slices given drop faction
        if min_tensor_size < len(image_volume):
            n = int(self.drop_fraction * len(image_volume))
            if n / 2 >= len(image_volume) / 2:  # if drop faction is too large, leave the middle slices unmoved
                n = int(len(image_volume) / 2) - 1
            if n > 0:
                image_volume = image_volume[n:-n]
        return torch.from_numpy(image_volume).float()  # convert to torch

    def load_mri_given_strip_tensor(self, patient, file_index, max_tensor_size=None, load_plane=2, mask_ratio_threshold=0.5,
                                    npy_cache_root=None):

        if npy_cache_root is None:
            # load mri strip volume file path
            file_path = patient.get_file_path_given_file(file_index)
            nifti_extension = get_file_extension(file_path, prompt=".nii")
            json_path = file_path[:-len(nifti_extension)] + '.json'
            nifti_path_stripped = file_path[:-len(nifti_extension)] + "_stripped.nii.gz"

            # load both volume and stripped mri
            if os.path.exists(nifti_path_stripped):
                nifti_img_stripped = nibabel.load(nifti_path_stripped)
                mri_stripped_volume = nifti_img_stripped.get_fdata()
                # load mri volume without zoom resize
                mri_volume = patient.load_mri_volume(file_index, if_resize=False)

                # check if stripped and raw has the same shape
                if mri_stripped_volume.shape != mri_volume.shape:
                    print("striped and raw has different shape for " + file_path)
                    return None, None
            else:
                return None, None
        else:
            # load mri strip volume file path
            file_path = patient.get_file_path_given_file(file_index)
            nifti_extension = get_file_extension(file_path, prompt=".nii")
            json_path = file_path[:-len(nifti_extension)] + '.json'
            nifti_path_stripped = file_path[:-len(nifti_extension)] + "_stripped.nii.gz"

            # convert to cache files
            root_folder = get_root_folder(file_path)
            root_folder = root_folder.replace("\\", "/")
            file_path = file_path.replace(root_folder, npy_cache_root) + "/nifti_img.npy"
            nifti_path_stripped = nifti_path_stripped.replace(root_folder, npy_cache_root) + "/nifti_img_stripped.npy"

            # load both volume and stripped mri
            if os.path.exists(nifti_path_stripped):
                mri_stripped_volume = np.load(nifti_path_stripped)
                # load mri volume without zoom resize
                mri_volume = np.load(file_path)

                # check if stripped and raw has the same shape
                if mri_stripped_volume.shape != mri_volume.shape:
                    print("striped and raw has different shape for " + file_path)
                    return None, None
            else:
                return None, None


        # use binary mask and threshold ratio related to maximum brain stripped area to crop series
        threshold = 1e-14  # Adjust as needed
        binary_mask = (mri_stripped_volume > threshold).astype(np.uint8)
        masked_areas = np.sum(binary_mask, axis=(0, 1))
        indices = np.where(masked_areas > int(np.max(masked_areas) * mask_ratio_threshold))[0]
        if len(indices) > 0: # make sure there is at least one slice
            end_idx = np.max(indices)+1
            start_idx = np.min(indices)
        else:
            start_idx = mri_volume.shape[2] // 2
            end_idx = start_idx + 1

        if len(mri_volume.shape) == 3:  # H, W, C
            mri_volume = mri_volume[:, :, start_idx:end_idx]
            mri_stripped_volume = mri_stripped_volume[:, :, start_idx:end_idx]
            mri_volume_shape = mri_volume.shape
            H, W, C = mri_volume_shape
        elif len(mri_volume.shape) == 4:  # H, W, slice_index_i, series_index_i
            mri_volume = mri_volume[:, :, start_idx:end_idx,:]
            mri_stripped_volume = mri_stripped_volume[:, :, start_idx:end_idx,:]
            mri_volume_shape = mri_volume.shape
            H, W, C, series_index_i = mri_volume_shape
        image_volume = []
        image_stripped_volume = []
        if max_tensor_size is not None:
            volume_size = min(max_tensor_size, (H, W, C)[load_plane])
        else:
            volume_size = (H, W, C)[load_plane]
        for i in range(volume_size):
            if len(mri_volume_shape) == 3:  # H, W, C
                if load_plane == 0:
                    image = mri_volume[i,:,:]
                    image_stripped = mri_stripped_volume[i, :, :]
                elif load_plane == 1:
                    image = mri_volume[:,i,:]
                    image_stripped = mri_stripped_volume[:, i, :]
                elif load_plane == 2:
                    image = mri_volume[:,:,i]
                    image_stripped = mri_stripped_volume[:, :, i]
            elif len(mri_volume_shape) == 4:  # H, W, series_index_i, slice_index_i
                # only use the first series to check, assume same sequence for all series for now
                if load_plane == 0:
                    image = mri_volume[i,:,:,0]
                    image_stripped = mri_stripped_volume[i, :, :, 0]
                elif load_plane == 1:
                    image = mri_volume[:,i,:,0]
                    image_stripped = mri_stripped_volume[:, i, :, 0]
                elif load_plane == 2:
                    image = mri_volume[:,:,i,0]
                    image_stripped = mri_stripped_volume[:, :, i, 0]
            if self.image_size is None:
                self.image_size = max(image.shape)
            start0 = time.time()
            image_tensor = process_image_array(image, self.image_size).unsqueeze(0).repeat(self.n_color, 1,
                                                                                           1).float()  # [C, H, W]
            image_stripped_tensor = process_image_array(image_stripped, self.image_size).unsqueeze(0).repeat(self.n_color, 1,
                                                                                           1).float()
            start1 = time.time()
            if (start1 - start0) > 0.1:
                print(patient.patient_ID + " start1 - start0 ", start1 - start0)
                print(file_index)
                print_var_detail(image_tensor)
            image_volume.append(image_tensor.numpy())
            image_stripped_volume.append(image_stripped_tensor.numpy())
            start2 = time.time()
            if (start2 - start1) > 0.1:
                print("start2 - start1 ", start2 - start1)

        image_volume = np.array(image_volume, np.float32)
        image_stripped_volume = np.array(image_stripped_volume, np.float32)
        return torch.from_numpy(image_volume).float(), torch.from_numpy(image_stripped_volume).float() # convert to torch

    def load_mri_given_path_tensor(self, file_path, max_tensor_size=None, min_tensor_size=None, is_volume=False, load_plane=2):
        """
        load mri as tensor given file path, for now only works for file extension with .mgz, .nii.gz and .nii
        :param file_index: [0,0,0,[0]] or [0,0,0,0]
        :return mri volume tensor: [num_slice, n_color, H, W]
        """
        if is_volume and (file_path[-3:] == 'nii' or file_path[-3:] == 'mgz' or file_path[-6:] == 'nii.gz'):
            # mri_volume = patient.load_mri_volume(file_index)
            ds = nibabel.load(file_path)
            mri_volume = ds.get_fdata()
            mri_volume_shape = mri_volume.shape
            if len(mri_volume_shape) == 3:  # H, W, C
                H,W,C = mri_volume_shape
            elif len(mri_volume_shape) == 4:  # H, W, series_index_i, slice_index_i
                H,W,C,series_index_i  = mri_volume_shape

            image_volume = []
            if max_tensor_size is not None:
                volume_size = min(max_tensor_size, (H,W,C)[load_plane])
            else:
                volume_size = (H,W,C)[load_plane]
            if min_tensor_size is None:
                min_tensor_size = 0
            for i in range(volume_size):
                if len(mri_volume_shape) == 3:  # H, W, C
                    if load_plane == 0:
                        image = mri_volume[i,:,:]
                    elif load_plane == 1:
                        image = mri_volume[:,i,:]
                    elif load_plane == 2:
                        image = mri_volume[:,:,i]
                elif len(mri_volume_shape) == 4:  # H, W, series_index_i, slice_index_i
                    # only use the first series to check, assume same sequence for all series for now
                    if load_plane == 0:
                        image = mri_volume[i,:,:,0]
                    elif load_plane == 1:
                        image = mri_volume[:,i,:,0]
                    elif load_plane == 2:
                        image = mri_volume[:,:,i,0]
                if self.image_size is None:
                    self.image_size = max(image.shape)
                start0 = time.time()
                image_tensor = process_image_array(image, self.image_size).unsqueeze(0).repeat(self.n_color, 1,
                                                                                               1).float()  # [C, H, W]
                start1 = time.time()
                if (start1 - start0) > 0.1:
                    print(file_path + " start1 - start0 ", start1 - start0)
                    print_var_detail(image_tensor)
                image_volume.append(image_tensor.numpy())
                start2 = time.time()
                if (start2 - start1) > 0.1:
                    print("start2 - start1 ", start2 - start1)
        else:
            raise ValueError("File extension must be .mgz or .nii")

        image_volume = np.array(image_volume, np.float32)

        # remove slices given drop faction
        if min_tensor_size < len(image_volume):
            n = int(self.drop_fraction * len(image_volume))
            if n / 2 >= len(image_volume) / 2:  # if drop faction is too large, leave the middle slices unmoved
                n = int(len(image_volume) / 2) - 1
            if n > 0:
                image_volume = image_volume[n:-n]
        return torch.from_numpy(image_volume).float()  # convert to torch

    def load_mri_dir_tensor(self, patient, dir_index):
        """
        load mri as tensor given dir index, only make sense if you are sure that same mri series
         are stored in separate files in same dir
        :param file_index: [0,0,0]
        :return mri volume tensor: [num_slice, n_color, H, W]
        """
        volume_dir = torch.Tensor()
        for file_index in patient.get_file_index_list_in_dir(dir_index):
            volume_temp = self.load_mri_file_tensor(patient, file_index, self.max_volume_size, self.min_volume_size)
            volume_dir = torch.cat((volume_dir, volume_temp), 0)
        return volume_dir

    def sequence_detection_by_slice(self, model, image_slice, detect_sequence_list=None,
                                    **kwargs):
        # image_slice: [C, H, W]
        if len(image_slice.shape) == 3:
            image_slice = image_slice.unsqueeze(0)
        if detect_sequence_list is None:
            detect_sequence_list = self.detect_sequence_list
        model.eval()
        if next(model.parameters()).device != self.device:
            model = model.to(self.device)
        with torch.no_grad():
            image_slice = image_slice.to(self.device)
            image_label_pred = model(image=image_slice, **kwargs)
            image_label_pred = image_label_pred.detach().cpu()
            confidence, predicted = torch.max(image_label_pred.data, 1)
            pred_seq_index, _ = torch.mode(predicted, 0)
            detect_sequence = detect_sequence_list[pred_seq_index]
        return detect_sequence, torch.mean(confidence)

    def mask_generation_by_slice(self, model, image_slice,
                                    **kwargs):
        # image_slice: [C, H, W]
        if len(image_slice.shape) == 3:
            image_slice = image_slice.unsqueeze(0)
        model.eval()
        if next(model.parameters()).device != self.device:
            model = model.to(self.device)
        with torch.no_grad():
            image_slice = image_slice.to(self.device)
            image_mask_pred = model(image=image_slice, **kwargs)
            image_mask_pred = image_mask_pred.detach().cpu()
        return image_mask_pred

    def sequence_detection_by_volume(self, model, image_volume, detect_sequence_list=None, if_exclude_empty=True,
                                     confidence_threshold=1.0, min_threshold_size=10, **kwargs):
        # image_volume: [B, C, H, W]
        detect_sequence = None
        detected_confidence = torch.tensor([0.0])

        if image_volume is None: # return None sequence if image volume is None
            return detect_sequence, detected_confidence

        if len(image_volume.shape) > 1:
            if detect_sequence_list is None:
                detect_sequence_list = self.detect_sequence_list
            model.eval()
            if next(model.parameters()).device != self.device:
                model = model.to(self.device)
            with torch.no_grad():
                # load slice by volume
                image_label_pred_volume = torch.Tensor()
                if if_exclude_empty:
                    valid_cols = [col_idx for col_idx, col in enumerate(torch.split(image_volume, 1, dim=0)) if
                                  not torch.all(col < 1e-16)]
                    image_volume = image_volume[valid_cols, :]

                if image_volume.shape[0] > 0:
                    # for batch in torch_tensor_loader(image_volume, self.batch_size_volume):
                    for i, batch in enumerate(torch_tensor_loader(image_volume, self.batch_size_volume)):
                        if i > 0 and i % 100 == 0:
                            print("detected {} mri slices".format(i * self.batch_size_volume))
                        batch = batch.to(self.device)  # [batch, C, H, W]
                        image_label_pred = model(image=batch, **kwargs)
                        image_label_pred = image_label_pred.detach().cpu()
                        image_label_pred_volume = torch.cat((image_label_pred_volume, image_label_pred), 0)
                    confidence, predicted = torch.max(image_label_pred_volume.data, 1)

                    # use the most appears label
                    # pred_seq_index, _ = torch.mode(predicted, 0)  # int cooresponding to detect_sequence
                    # detect_sequence = detect_sequence_list[pred_seq_index]  # string
                    # use the most accumulate label
                    confidences_sum = np.zeros(len(detect_sequence_list))
                    confidences_sum_count = np.zeros(len(detect_sequence_list))

                    # set confidence threshold to -1 if min_threshold_size not met
                    if min_threshold_size > confidence.shape[0]:
                        confidence_threshold = -1
                    elif torch.max(confidence) < confidence_threshold:
                        # if confidence lower than confidence_threshold, using upper min_threshold_size confidence as threshold
                        confidence_sorted, _ = torch.sort(confidence)
                        # confidence_threshold = confidence_sorted[int(len(confidence_sorted) / 2)]
                        confidence_threshold = confidence_sorted[-min_threshold_size]

                    for j in range(len(confidence)):
                        confidence_j = confidence[j]
                        info_idx = predicted[j]

                        if confidence_j > confidence_threshold:
                            if info_idx > len(confidences_sum) - 1:
                                print(info_idx)
                                print(j)
                                print(confidence)
                                print(predicted)
                            confidences_sum[info_idx] += confidence_j
                            confidences_sum_count[info_idx] += 1
                    # find the most confident sequence
                    np.seterr(invalid='raise')  # Converts warnings to exceptions
                    try:
                        max_idx = confidences_sum.argmax(axis=0)
                        detect_sequence = detect_sequence_list[max_idx]
                        detected_confidence = confidences_sum[max_idx] / confidences_sum_count[max_idx]
                    except FloatingPointError:
                        print("Caught an invalid operation (NaN or Inf in division)")
                        print(confidences_sum[max_idx])
                        print(confidences_sum_count[max_idx])
                        detect_sequence = None
                        detected_confidence = None

        return detect_sequence, detected_confidence

    def mask_generation_by_volume(self, model, image_volume, if_exclude_empty=True, **kwargs):
        # image_volume: [B, C, H, W]
        valid_cols = []
        image_mask_pred_volume= None
        if len(image_volume.shape) > 1:
            model.eval()
            if next(model.parameters()).device != self.device:
                model = model.to(self.device)
            with torch.no_grad():
                # load slice by volume
                image_mask_pred_volume = torch.Tensor()
                if if_exclude_empty:
                    valid_cols = [col_idx for col_idx, col in enumerate(torch.split(image_volume, 1, dim=0)) if
                                  not torch.all(col < 1e-16)]
                else:
                    valid_cols = [col_idx for col_idx in range(image_volume.shape[0])]
                image_volume = image_volume[valid_cols, :]

                if image_volume.shape[0] > 0:
                    # for batch in torch_tensor_loader(image_volume, self.batch_size_volume):
                    for i, batch in enumerate(torch_tensor_loader(image_volume, self.batch_size_volume)):
                        if i > 0 and i % 100 == 0:
                            print("detected {} mri slices".format(i * self.batch_size_volume))
                        batch = batch.to(self.device)  # [batch, C, H, W]
                        image_mask_pred = model(image=batch, **kwargs)
                        image_mask_pred = image_mask_pred.detach().cpu()
                        image_mask_pred_volume = torch.cat((image_mask_pred_volume, image_mask_pred), 0)

        return image_mask_pred_volume, valid_cols

    def mask_generation_by_slice_index(self, patient, model, slice_index, **kwargs):
        # load slice by file
        image_tensor = self.load_mri_slice_tensor(patient, slice_index)

        image_mask_pred = self.mask_generation_by_slice(model, image_tensor, **kwargs)  # tensor
        return image_mask_pred

    def sequence_detection_by_file_index(self, patient, model, file_index, detect_sequence_list=None,
                                         confidence_threshold=0.999, if_include_mask = False,mask_ratio_threshold=0.5,
                                         npy_cache_root=None, **kwargs):
        # load slice by file
        is_volume = False
        if patient.mode == 'nifti':
            is_volume = True

        ### get dimention and load_plane
        load_plane = 2
        nifti_data = patient.load_nifti_mri_by_file(file_index)
        dim = nifti_data.header["dim"]
        if dim[0] == 3:
            dim3 = dim[1:4]
            min_idx = np.argmin(dim3)  # Index of min value
            min_val = dim3[min_idx]
            if min_val < 64:
                load_plane = min_idx



        if if_include_mask:
            image_volume, image_stripped_volume = self.load_mri_given_strip_tensor(patient,
                                                                                              file_index,
                                                                                              max_tensor_size=self.max_volume_size,
                                                                                              load_plane=load_plane,mask_ratio_threshold=mask_ratio_threshold,
                                                                                   npy_cache_root = npy_cache_root,
                                                                                              **kwargs)
        else:
            image_volume = self.load_mri_file_tensor(patient, file_index, self.max_volume_size, self.min_volume_size, is_volume, load_plane = load_plane)
        if detect_sequence_list is None:
            detect_sequence_list = self.detect_sequence_list
        detect_sequence, confidence = self.sequence_detection_by_volume(model, image_volume, detect_sequence_list, True,
                                                                        confidence_threshold,
                                                                        self.min_volume_size, **kwargs)  # string
        return detect_sequence, confidence

    def sequence_detection_by_dir_index(self, patient, model, dir_index, detect_sequence_list=None,
                                        confidence_threshold=1.0, **kwargs):
        # load slice by file
        image_volume = self.load_mri_dir_tensor(patient, dir_index)
        if detect_sequence_list is None:
            detect_sequence_list = self.detect_sequence_list
        detect_sequence, confidence = self.sequence_detection_by_volume(model, image_volume, detect_sequence_list, True,
                                                                        confidence_threshold,
                                                                        self.min_volume_size, **kwargs)  # string
        return detect_sequence, confidence

    def sequence_detection_update(self, patient, model, detect_sequence_list=None, detect_by_file=True, update_skip_size = None,
                                  update_sequence_list = None, **kwargs):
        """
        detect and update sequence in patient using trained model
        """
        if detect_sequence_list is None:
            detect_sequence_list = self.detect_sequence_list # total sequence list that can be detected by model
        if update_sequence_list is None:
            update_sequence_list = self.detect_sequence_list # current sequence to be updated from
        for dir_index in patient.dir_index_list:
            if detect_by_file:
                file_index_list = patient.get_file_index_list_in_dir(dir_index)
                for file_index in file_index_list:
                    current_sequence =patient.get_sequence_given_file_index(file_index)
                    if current_sequence in update_sequence_list:
                        # load slice by volume
                        detect_sequence, _ = self.sequence_detection_by_file_index(patient, model, file_index,
                                                                                   detect_sequence_list, **kwargs)  # string

                        # update detect sequence
                        patient.set_sequence_to_dir(sequence_name=detect_sequence, dir_index=dir_index)
                        slice_index_list = patient.get_slice_index_list_in_file(file_index)
                        if update_skip_size is not None:
                            if len(slice_index_list) < update_skip_size:
                                # for i in tqdm(range(len(slice_index_list))):
                                for i in range(len(slice_index_list)):
                                    slice_index = slice_index_list[i]
                                    patient.set_sequence_to_slice(sequence_name=detect_sequence, slice_index=slice_index)
                        else:
                            # for i in tqdm(range(len(slice_index_list))):
                            for i in range(len(slice_index_list)):
                                slice_index = slice_index_list[i]
                                patient.set_sequence_to_slice(sequence_name=detect_sequence, slice_index=slice_index)
            else:  # detect by dir
                current_sequence = patient.get_sequence_given_dir_index(dir_index)
                if current_sequence in update_sequence_list:
                    detect_sequence, _ = self.sequence_detection_by_dir_index(patient, model, dir_index,
                                                                              detect_sequence_list, **kwargs)  # string

                    # update detect sequence
                    patient.set_sequence_to_dir(sequence_name=detect_sequence, dir_index=dir_index)
                    slice_index_list = patient.get_slice_index_list_in_dir(dir_index)
                    if update_skip_size is not None:
                        if len(slice_index_list) < update_skip_size:
                            # for i in tqdm(range(len(slice_index_list))):
                            for i in range(len(slice_index_list)):
                                slice_index = slice_index_list[i]
                                patient.set_sequence_to_slice(sequence_name=detect_sequence, slice_index=slice_index)
                    else:
                        for i in range(len(slice_index_list)):
                            slice_index = slice_index_list[i]
                            patient.set_sequence_to_slice(sequence_name=detect_sequence, slice_index=slice_index)

        return patient

    def addition_info_detection_update(self, patient, model, detect_info_tags_list=None, detect_by_file=True,
                                       max_info_size=None, skip_max_file_size=1e20, **kwargs):
        """
        detect and update additional info in patient using trained model
        """
        if detect_info_tags_list is None:
            print("no tags detected")
            return None
        for dir_index in patient.dir_index_list:
            if detect_by_file:
                file_index_list = patient.get_file_index_list_in_dir(dir_index)
                for file_index in file_index_list:
                    slice_index_list = patient.get_slice_index_list_in_file(file_index)
                    if len(slice_index_list) < skip_max_file_size:
                        # load slice by volume
                        detect_tag, _ = self.sequence_detection_by_file_index(patient, model, file_index,
                                                                              detect_info_tags_list, **kwargs)  # string
                        # update detect tags
                        # for i in tqdm(range(len(slice_index_list))):
                        for i in range(len(slice_index_list)):
                            slice_index = slice_index_list[i]
                            patient.add_slice_info_given_slice(slice_index=slice_index, add_slice_info=detect_tag,
                                                               max_info_size=max_info_size)
            else:  # detect by dir
                detect_tag, _ = self.sequence_detection_by_dir_index(patient, model, dir_index,
                                                                     detect_info_tags_list, **kwargs)  # string

                # update detect sequence
                slice_index_list = patient.get_slice_index_list_in_dir(dir_index)
                # for i in tqdm(range(len(slice_index_list))):
                for i in range(len(slice_index_list)):
                    slice_index = slice_index_list[i]
                    patient.add_slice_info_given_slice(slice_index=slice_index, add_slice_info=detect_tag,
                                                       max_info_size=max_info_size)

        return patient
