import numpy as np

import utils.ipt_util as utility

import torch

from tqdm import tqdm
from help_func import print_var_detail
import time
from utils.evaluation_utils import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor, binaryMaskIOU
import copy
from utils.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from modeling.loss_functions.dice_loss import DC_and_Focal_loss

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Flatten tensors to calculate Dice Loss across all voxels
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Compute Dice coefficient
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.epsilon) / (probs.sum() + targets.sum() + self.epsilon)
        return 1 - dice


class PerVolumeDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(PerVolumeDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        Args:
            logits: Model output logits of shape [N, C, H, W]
            targets: Ground truth binary masks of shape [N, C, H, W]
        Returns:
            Dice loss averaged over all samples
        """
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)  # Convert logits to probabilities

        # Flatten the dimensions C, H, W into a single volume for each sample
        probs = probs.view(probs.size(0), -1)  # Shape: [N, C*H*W]
        targets = targets.view(targets.size(0), -1)  # Shape: [N, C*H*W]

        # Compute intersection and cardinality for each sample
        intersection = (probs * targets).sum(dim=1)  # Sum over the volume for each sample
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)  # Sum over the volume for each sample

        # Compute Dice coefficient for each sample
        dice = (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)  # Shape: [N]

        # Compute Dice loss (1 - Dice coefficient) and average across the batch
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class PerChannelDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(PerChannelDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Initialize loss
        total_loss = 0.0
        num_channels = logits.shape[1]

        for c in range(num_channels):
            # Compute per-channel Dice Loss
            p = probs[:, c].view(-1)
            t = targets[:, c].view(-1)
            intersection = (p * t).sum()
            dice = (2. * intersection + self.epsilon) / (p.sum() + t.sum() + self.epsilon)
            total_loss += 1 - dice

        # Average loss over all channels
        return total_loss / num_channels


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class TrainerSegBinaryDiceLoss3D_MAEUNT:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, scheduler, grad_scaler, amp,
                 gradient_clipping, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, labels=None, loss_type='dice',
                 target_channel=40, if_use_scaler=True):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        self.labels = labels
        self.loss_type = loss_type
        self.n_classes = len(labels)
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.gradient_clipping = gradient_clipping
        self.if_use_scaler = if_use_scaler
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        if self.n_classes == 1:
            print("binary mask")
        self.target_channel = target_channel

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (
            images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
            if_zero) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(device=self.device, dtype=torch.float32,
                                             memory_format=torch.channels_last)  # [batch, C, H, W]
                    # images_masked = images_masked.to(self.device)  # [batch, num_class]
                    images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)  # [batch, C, H, W]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_mask_binary, "images_mask_binary")

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        # images_mask_binary_pred = self.model(images_input)  # [batch, C, H, W]
                        images_mask_binary_pred_conv, images_mask_binary_pred_tran = self.model(images_input)
                        if self.n_classes == 1:  # for binary class, use dice loss only
                            # if images_mask_binary_pred.shape[1] == 1:
                            if self.loss_type == 'dice':
                                for j in range(images_mask_binary_pred_conv.shape[1]):
                                    if j == 0:
                                        loss_conv = self.loss(images_mask_binary_pred_conv[:, j, :, :],
                                                              images_mask_binary[:, j, :, :].float())
                                        loss_conv += dice_loss(F.sigmoid(images_mask_binary_pred_conv[:, j, :, :]),
                                                               images_mask_binary[:, j, :, :].float(),
                                                               multiclass=False)
                                        loss_tran = self.loss(images_mask_binary_pred_tran[:, j, :, :],
                                                              images_mask_binary[:, j, :, :].float())
                                        loss_tran += dice_loss(F.sigmoid(images_mask_binary_pred_tran[:, j, :, :]),
                                                               images_mask_binary[:, j, :, :].float(),
                                                               multiclass=False)
                                    else:
                                        loss_conv += self.loss(images_mask_binary_pred_conv[:, j, :, :],
                                                               images_mask_binary[:, j, :, :].float())
                                        loss_conv += dice_loss(F.sigmoid(images_mask_binary_pred_conv[:, j, :, :]),
                                                               images_mask_binary[:, j, :, :].float(),
                                                               multiclass=False)
                                        loss_tran += self.loss(images_mask_binary_pred_tran[:, j, :, :],
                                                               images_mask_binary[:, j, :, :].float())
                                        loss_tran += dice_loss(F.sigmoid(images_mask_binary_pred_tran[:, j, :, :]),
                                                               images_mask_binary[:, j, :, :].float(),
                                                               multiclass=False)
                        loss = loss_conv + loss_tran
                        self.optimizer.zero_grad()
                        if torch.isnan(loss):
                            print("nan loss occur")
                            # print(filename)
                            print('images_mask_binary_pred_conv', images_mask_binary_pred_conv)
                            print('images_mask_binary_pred_tran', images_mask_binary_pred_tran)
                            for j in range(images_input.size(0)):
                                print_var_detail(images_input[j], 'images_input')
                            num_nan += 1
                        else:
                            if self.if_use_scaler:
                                self.grad_scaler.scale(loss).backward()
                                self.grad_scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                                self.grad_scaler.step(self.optimizer)
                                self.running_loss_train += loss.item()
                                self.grad_scaler.update()
                            else:
                                loss.backward()
                                self.optimizer.step()
                                self.running_loss_train += loss.item()
                        if batch % 100 == 0 or batch < 4:
                            end = time.time()
                            print('trained batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU, dice_score = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        dice_score = 0
        nmse = 0.0
        psnr = 0.0

        if self.target_channel is None:
            ssim = 0.0
            IoU = 0
            num_valid_ioU = 0
        else:
            # ssim = torch.zeros(len(self.labels))
            # IoU = np.zeros(len(self.labels))
            # num_valid_ioU = np.zeros(len(self.labels))
            ssim = torch.zeros(self.target_channel)
            IoU = np.zeros(self.target_channel)
            num_valid_ioU = np.zeros(self.target_channel)
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                for batch, (
                images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
                if_zero) in enumerate(self.loader_test):
                    if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                        timer_data, timer_model = utility.timer(), utility.timer()
                        images_input = images.to(device=self.device, dtype=torch.float32,
                                                 memory_format=torch.channels_last)  # [pair]
                        # images_masked = images_masked.to(self.device)  # [pair]
                        images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)
                        timer_data.hold()
                        timer_model.tic()
                        images_mask_binary_pred_conv, images_mask_binary_pred_tran = self.model(images_input)

                        if self.n_classes == 1:
                            if self.loss_type == 'dice':
                                for j in range(images_mask_binary_pred_conv.shape[1]):
                                    if j == 0:
                                        loss_conv = self.loss(images_mask_binary_pred_conv[:, j, :, :],
                                                              images_mask_binary[:, j, :, :].float())
                                        loss_conv += dice_loss(F.sigmoid(images_mask_binary_pred_conv[:, j, :, :]),
                                                               images_mask_binary[:, j, :, :].float(),
                                                               multiclass=False)
                                        loss_tran = self.loss(images_mask_binary_pred_tran[:, j, :, :],
                                                              images_mask_binary[:, j, :, :].float())
                                        loss_tran += dice_loss(F.sigmoid(images_mask_binary_pred_tran[:, j, :, :]),
                                                               images_mask_binary[:, j, :, :].float(),
                                                               multiclass=False)
                                    else:
                                        loss_conv += self.loss(images_mask_binary_pred_conv[:, j, :, :],
                                                               images_mask_binary[:, j, :, :].float())
                                        loss_conv += dice_loss(F.sigmoid(images_mask_binary_pred_conv[:, j, :, :]),
                                                               images_mask_binary[:, j, :, :].float(),
                                                               multiclass=False)
                                        loss_tran += self.loss(images_mask_binary_pred_tran[:, j, :, :],
                                                               images_mask_binary[:, j, :, :].float())
                                        loss_tran += dice_loss(F.sigmoid(images_mask_binary_pred_tran[:, j, :, :]),
                                                               images_mask_binary[:, j, :, :].float(),
                                                               multiclass=False)
                        loss = loss_conv + loss_tran
                        # choose average of conv and tran output as inference
                        images_mask_binary_pred = (images_mask_binary_pred_conv + images_mask_binary_pred_tran) / 2.0

                        if self.n_classes == 1:
                            # if images_mask_binary_pred.shape[1] == 1:
                            assert images_mask_binary.min() >= 0 and images_mask_binary.max() <= 1, 'True mask indices should be in [0, 1]'
                            for k in range(images_mask_binary_pred.shape[1]):
                                images_mask_binary_pred[:, k, :, :] = (
                                            F.sigmoid(images_mask_binary_pred[:, k, :, :]) > 0.5).float()
                                # compute the Dice score
                                dice_score += dice_coeff(images_mask_binary_pred[:, k, :, :],
                                                         images_mask_binary[:, k, :, :],
                                                         reduce_batch_first=False)

                        if torch.isnan(loss):
                            num_nan += 1
                            print('nan loss detected')
                            continue
                        else:
                            self.running_loss_test += loss.item()

                        if batch % 20 == 0 or batch < 4:
                            end = time.time()
                            print('tested batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()

                        # evaluation metrics

                        if self.n_classes == 1:
                            images_masked_pred = images_input * images_mask_binary_pred
                            pred = images_masked_pred.detach().cpu()
                            images_masked = images_input * images_mask_binary
                            tg = images_masked.detach().cpu()
                            images_mask_binary = images_mask_binary.detach().cpu()
                            images_mask_binary_pred = images_mask_binary_pred.detach().cpu()

                        i_nmse = calc_nmse_tensor(tg, pred)
                        i_psnr = calc_psnr_tensor(tg, pred)
                        i_ssim = torch.zeros(images_mask_binary_pred.shape[1])
                        i_IoU = np.zeros(images_mask_binary_pred.shape[1])
                        for j in range(images_mask_binary_pred.shape[1]):
                            i_num_none_ioU = 0
                            # i_ssim[j] = calc_ssim_tensor(tg[:,j,:,:].unsqueeze(1), pred[:,j,:,:].unsqueeze(1))
                            for i in range(images_mask_binary_pred.shape[0]):
                                i_IoU_i = binaryMaskIOU(images_mask_binary[i][j], images_mask_binary_pred[i][j])
                                i_ssim_i = calc_ssim_tensor(tg[i, j, :, :].unsqueeze(0).unsqueeze(0),
                                                            pred[i, j, :, :].unsqueeze(0).unsqueeze(0))
                                if i_IoU_i is not None:
                                    i_IoU[j] += i_IoU_i
                                    i_ssim[j] += i_ssim_i
                                else:
                                    i_num_none_ioU += 1
                            if i_num_none_ioU < images_mask_binary_pred.shape[0]:
                                num_valid_ioU[j] += (images_mask_binary_pred.shape[0] - i_num_none_ioU)
                            else:
                                num_valid_ioU[j] += 0

                        nmse += i_nmse
                        psnr += i_psnr

                        ssim += i_ssim
                        IoU += i_IoU
                    else:
                        break

            for j in range(self.target_channel):
                num_valid = num_valid_ioU[j]
                if num_valid == 0:
                    ssim[j] = 0
                    IoU[j] = 0
                else:
                    ssim[j] /= num_valid
                    IoU[j] /= num_valid
            ssim_ave_channel = sum(ssim) / len(ssim)
            IoU_ave_channel = sum(IoU) / len(IoU)

            if self.max_num_batch_test is None:
                nmse /= (len(self.loader_test) - num_nan)
                psnr /= (len(self.loader_test) - num_nan)
                self.running_loss_test /= (len(self.loader_test) - num_nan)
                dice_score /= (len(self.loader_test) - num_nan)
            else:
                nmse /= (self.max_num_batch_test - num_nan)
                psnr /= (self.max_num_batch_test - num_nan)
                self.running_loss_test /= (self.max_num_batch_test - num_nan)
                dice_score /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(
                  ssim) + '|| ssim_ave_channel: ' + str(ssim_ave_channel)
              + '|| IoU: ' + str(IoU) + '|| IoU_ave_channel: ' + str(IoU_ave_channel) + '|| Dice score: ' + str(
                  dice_score))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU, dice_score


class TrainerSegBinaryDiceLoss3D_MAE:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, scheduler, grad_scaler, amp,
                 gradient_clipping, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, labels=None, loss_type='dice',
                 target_channel=40, if_use_scaler=True):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        self.labels = labels
        self.loss_type = loss_type
        self.n_classes = len(labels)
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.gradient_clipping = gradient_clipping
        self.if_use_scaler = if_use_scaler
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        if self.n_classes == 1:
            print("binary mask")
        self.target_channel = target_channel
        if self.loss_type == 'dice':
            self.dice = PerVolumeDiceLoss(epsilon=1e-6)

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (
            images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
            if_zero) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(device=self.device, dtype=torch.float32,
                                             memory_format=torch.channels_last)  # [batch, C, H, W]
                    # images_masked = images_masked.to(self.device)  # [batch, num_class]
                    images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)  # [batch, C, H, W]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_mask_binary, "images_mask_binary")

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        # images_mask_binary_pred = self.model(images_input)  # [batch, C, H, W]
                        # images_mask_binary_pred_tran = self.model(images_input)
                        _, pred, _ = self.model(images_input, mask_ratio=0)
                        images_mask_binary_pred_tran = self.model.unpatchify(pred, self.target_channel)
                        if self.n_classes == 1:  # for binary class, use dice loss only
                            # if images_mask_binary_pred.shape[1] == 1:
                            if self.loss_type == 'dice':
                                loss_tran = self.loss(images_mask_binary_pred_tran,
                                                      images_mask_binary.float())
                                loss_tran += self.dice(images_mask_binary_pred_tran,
                                                       images_mask_binary.float())
                                # for j in range(images_mask_binary_pred_tran.shape[1]):
                                #     if j == 0:
                                #         loss_tran = self.loss(images_mask_binary_pred_tran[:, j, :, :],
                                #                               images_mask_binary[:, j, :, :].float())
                                #         loss_tran += dice_loss(F.sigmoid(images_mask_binary_pred_tran[:, j, :, :]),
                                #                                images_mask_binary[:, j, :, :].float(),
                                #                                multiclass=False)
                                #     else:
                                #         loss_tran += self.loss(images_mask_binary_pred_tran[:, j, :, :],
                                #                           images_mask_binary[:, j, :, :].float())
                                #         loss_tran += dice_loss(F.sigmoid(images_mask_binary_pred_tran[:, j, :, :]),
                                #                           images_mask_binary[:, j, :, :].float(),
                                #                           multiclass=False)
                        loss = loss_tran
                        self.optimizer.zero_grad()
                        if torch.isnan(loss):
                            print("nan loss occur")
                            # print(filename)
                            print('images_mask_binary_pred_tran', images_mask_binary_pred_tran)
                            for j in range(images_input.size(0)):
                                print_var_detail(images_input[j], 'images_input')
                            num_nan += 1
                        else:
                            if self.if_use_scaler:
                                self.grad_scaler.scale(loss).backward()
                                self.grad_scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                                self.grad_scaler.step(self.optimizer)
                                self.running_loss_train += loss.item()
                                self.grad_scaler.update()
                                # self.scheduler.step()
                            else:
                                loss.backward()
                                self.optimizer.step()
                                # self.scheduler.step()
                                self.running_loss_train += loss.item()
                        if batch % 100 == 0 or batch < 4:
                            end = time.time()
                            print('trained batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU, dice_score = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        dice_score = 0
        nmse = 0.0
        psnr = 0.0

        if self.target_channel is None:
            ssim = 0.0
            IoU = 0
            num_valid_ioU = 0
        else:
            # ssim = torch.zeros(len(self.labels))
            # IoU = np.zeros(len(self.labels))
            # num_valid_ioU = np.zeros(len(self.labels))
            ssim = torch.zeros(self.target_channel)
            IoU = np.zeros(self.target_channel)
            num_valid_ioU = np.zeros(self.target_channel)
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                for batch, (
                images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
                if_zero) in enumerate(self.loader_test):
                    if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                        timer_data, timer_model = utility.timer(), utility.timer()
                        images_input = images.to(device=self.device, dtype=torch.float32,
                                                 memory_format=torch.channels_last)  # [pair]
                        # images_masked = images_masked.to(self.device)  # [pair]
                        images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)
                        timer_data.hold()
                        timer_model.tic()
                        # images_mask_binary_pred_tran= self.model(images_input)
                        _, pred, _ = self.model(images_input, mask_ratio=0)
                        images_mask_binary_pred_tran = self.model.unpatchify(pred, self.target_channel)
                        if self.n_classes == 1:
                            if self.loss_type == 'dice':
                                loss_tran = self.loss(images_mask_binary_pred_tran,
                                                      images_mask_binary.float())
                                loss_tran += self.dice(images_mask_binary_pred_tran,
                                                       images_mask_binary.float())
                                # for j in range(images_mask_binary_pred_tran.shape[1]):
                                #     if j == 0:
                                #         loss_tran = self.loss(images_mask_binary_pred_tran[:, j, :, :],
                                #                               images_mask_binary[:, j, :, :].float())
                                #         loss_tran += dice_loss(F.sigmoid(images_mask_binary_pred_tran[:, j, :, :]),
                                #                                images_mask_binary[:, j, :, :].float(),
                                #                                multiclass=False)
                                #     else:
                                #         loss_tran += self.loss(images_mask_binary_pred_tran[:, j, :, :],
                                #                                images_mask_binary[:, j, :, :].float())
                                #         loss_tran += dice_loss(F.sigmoid(images_mask_binary_pred_tran[:, j, :, :]),
                                #                                images_mask_binary[:, j, :, :].float(),
                                #                                multiclass=False)
                        loss = loss_tran
                        # choose average of conv and tran output as inference
                        images_mask_binary_pred = images_mask_binary_pred_tran

                        if self.n_classes == 1:
                            # if images_mask_binary_pred.shape[1] == 1:
                            assert images_mask_binary.min() >= 0 and images_mask_binary.max() <= 1, 'True mask indices should be in [0, 1]'
                            for k in range(images_mask_binary_pred.shape[1]):
                                images_mask_binary_pred[:, k, :, :] = (
                                            F.sigmoid(images_mask_binary_pred[:, k, :, :]) > 0.5).float()
                                # compute the Dice score
                                dice_score += dice_coeff(images_mask_binary_pred[:, k, :, :],
                                                         images_mask_binary[:, k, :, :],
                                                         reduce_batch_first=False)

                        if torch.isnan(loss):
                            num_nan += 1
                            print('nan loss detected')
                            continue
                        else:
                            self.running_loss_test += loss.item()

                        if batch % 20 == 0 or batch < 4:
                            end = time.time()
                            print('tested batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()

                        # evaluation metrics

                        if self.n_classes == 1:
                            images_masked_pred = images_input * images_mask_binary_pred
                            pred = images_masked_pred.detach().cpu()
                            images_masked = images_input * images_mask_binary
                            tg = images_masked.detach().cpu()
                            images_mask_binary = images_mask_binary.detach().cpu()
                            images_mask_binary_pred = images_mask_binary_pred.detach().cpu()

                        i_nmse = calc_nmse_tensor(tg, pred)
                        i_psnr = calc_psnr_tensor(tg, pred)
                        i_ssim = torch.zeros(images_mask_binary_pred.shape[1])
                        i_IoU = np.zeros(images_mask_binary_pred.shape[1])
                        for j in range(images_mask_binary_pred.shape[1]):
                            i_num_none_ioU = 0
                            # i_ssim[j] = calc_ssim_tensor(tg[:,j,:,:].unsqueeze(1), pred[:,j,:,:].unsqueeze(1))
                            for i in range(images_mask_binary_pred.shape[0]):
                                i_IoU_i = binaryMaskIOU(images_mask_binary[i][j], images_mask_binary_pred[i][j])
                                i_ssim_i = calc_ssim_tensor(tg[i, j, :, :].unsqueeze(0).unsqueeze(0),
                                                            pred[i, j, :, :].unsqueeze(0).unsqueeze(0))
                                if i_IoU_i is not None:
                                    i_IoU[j] += i_IoU_i
                                    i_ssim[j] += i_ssim_i
                                else:
                                    i_num_none_ioU += 1
                            if i_num_none_ioU < images_mask_binary_pred.shape[0]:
                                num_valid_ioU[j] += (images_mask_binary_pred.shape[0] - i_num_none_ioU)
                            else:
                                num_valid_ioU[j] += 0

                        nmse += i_nmse
                        psnr += i_psnr

                        ssim += i_ssim
                        IoU += i_IoU
                    else:
                        break

            for j in range(self.target_channel):
                num_valid = num_valid_ioU[j]
                if num_valid == 0:
                    ssim[j] = 0
                    IoU[j] = 0
                else:
                    ssim[j] /= num_valid
                    IoU[j] /= num_valid
            ssim_ave_channel = sum(ssim) / len(ssim)
            IoU_ave_channel = sum(IoU) / len(IoU)

            if self.max_num_batch_test is None:
                nmse /= (len(self.loader_test) - num_nan)
                psnr /= (len(self.loader_test) - num_nan)
                self.running_loss_test /= (len(self.loader_test) - num_nan)
                dice_score /= (len(self.loader_test) - num_nan)
            else:
                nmse /= (self.max_num_batch_test - num_nan)
                psnr /= (self.max_num_batch_test - num_nan)
                self.running_loss_test /= (self.max_num_batch_test - num_nan)
                dice_score /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(
                  ssim) + '|| ssim_ave_channel: ' + str(ssim_ave_channel)
              + '|| IoU: ' + str(IoU) + '|| IoU_ave_channel: ' + str(IoU_ave_channel) + '|| Dice score: ' + str(
                  dice_score))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU, dice_score


class TrainerSegBinaryDiceLoss3D_Conformer:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, scheduler, grad_scaler, amp,
                 gradient_clipping, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, labels=None, loss_type='dice',
                 target_channel=40, if_use_scaler=True, target_output = 'conv'):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        self.labels = labels
        self.loss_type = loss_type
        self.n_classes = len(labels)
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.gradient_clipping = gradient_clipping
        self.if_use_scaler = if_use_scaler
        self.target_output = target_output
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        if self.n_classes == 1:
            print("binary mask")
        self.target_channel = target_channel
        if self.loss_type == 'dice':
            self.dice = PerVolumeDiceLoss(epsilon=1e-6)

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (
            images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
            if_zero) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(device=self.device, dtype=torch.float32,
                                             memory_format=torch.channels_last)  # [batch, C, H, W]
                    # images_masked = images_masked.to(self.device)  # [batch, num_class]
                    images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)  # [batch, C, H, W]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_mask_binary, "images_mask_binary")

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        # images_mask_binary_pred = self.model(images_input)  # [batch, C, H, W]
                        # images_mask_binary_pred_tran = self.model(images_input)
                        # _, pred, _ = self.model(images_input, mask_ratio=0)
                        images_mask_binary_pred_conv, images_mask_binary_pred_tran = self.model(images_input)
                        # images_mask_binary_pred_tran = self.model.unpatchify(pred, self.target_channel)

                        # if self.target_output == 'conv':
                        #     images_mask_binary_pred = images_mask_binary_pred_conv
                        # elif self.target_output == 'trans':
                        #     images_mask_binary_pred = images_mask_binary_pred_tran
                        # elif self.target_output == 'both':
                        #     images_mask_binary_pred = (images_mask_binary_pred_conv + images_mask_binary_pred_tran) // 2


                        if self.n_classes == 1:  # for binary class, use dice loss only
                            # if images_mask_binary_pred.shape[1] == 1:

                            if self.target_output == 'conv':
                                loss = self.loss(images_mask_binary_pred_conv,
                                                      images_mask_binary.float())
                                if self.loss_type == 'dice':
                                    loss += self.dice(images_mask_binary_pred_conv,
                                                           images_mask_binary.float())
                            elif self.target_output == 'trans':
                                loss = self.loss(images_mask_binary_pred_tran,
                                                      images_mask_binary.float())
                                if self.loss_type == 'dice':
                                    loss += self.dice(images_mask_binary_pred_tran,
                                                      images_mask_binary.float())

                            elif self.target_output == 'both':
                                loss = self.loss(images_mask_binary_pred_tran,
                                                 images_mask_binary.float())
                                loss += self.loss(images_mask_binary_pred_conv,
                                                 images_mask_binary.float())
                                if self.loss_type == 'dice':
                                    loss += self.dice(images_mask_binary_pred_conv,
                                                      images_mask_binary.float())
                                    loss += self.dice(images_mask_binary_pred_tran,
                                                      images_mask_binary.float())

                        # loss = loss_tran
                        self.optimizer.zero_grad()
                        if torch.isnan(loss):
                            print("nan loss occur")
                            # print(filename)
                            print('images_mask_binary_pred_conv', images_mask_binary_pred_conv)
                            print('images_mask_binary_pred_tran', images_mask_binary_pred_tran)
                            for j in range(images_input.size(0)):
                                print_var_detail(images_input[j], 'images_input')
                                print("is input has nun: " + str(torch.isnan(images_input).any()))
                            num_nan += 1
                        else:
                            if self.if_use_scaler:
                                self.grad_scaler.scale(loss).backward()
                                self.grad_scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                                self.grad_scaler.step(self.optimizer)
                                self.running_loss_train += loss.item()
                                self.grad_scaler.update()
                                # self.scheduler.step()
                            else:
                                loss.backward()
                                self.optimizer.step()
                                # self.scheduler.step()
                                self.running_loss_train += loss.item()
                        if batch % 100 == 0 or batch < 4:
                            end = time.time()
                            print('trained batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU, dice_score = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        dice_score = 0
        nmse = 0.0
        psnr = 0.0

        if self.target_channel is None:
            ssim = 0.0
            IoU = 0
            num_valid_ioU = 0
        else:
            # ssim = torch.zeros(len(self.labels))
            # IoU = np.zeros(len(self.labels))
            # num_valid_ioU = np.zeros(len(self.labels))
            ssim = torch.zeros(self.target_channel)
            IoU = np.zeros(self.target_channel)
            num_valid_ioU = np.zeros(self.target_channel)
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                for batch, (
                images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
                if_zero) in enumerate(self.loader_test):
                    if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                        timer_data, timer_model = utility.timer(), utility.timer()
                        images_input = images.to(device=self.device, dtype=torch.float32,
                                                 memory_format=torch.channels_last)  # [pair]
                        # images_masked = images_masked.to(self.device)  # [pair]
                        images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)
                        timer_data.hold()
                        timer_model.tic()
                        # images_mask_binary_pred_tran= self.model(images_input)
                        # _, pred, _ = self.model(images_input, mask_ratio=0)
                        # images_mask_binary_pred_tran = self.model.unpatchify(pred, self.target_channel)
                        images_mask_binary_pred_conv, images_mask_binary_pred_tran = self.model(images_input)

                        if self.n_classes == 1:
                            if self.target_output == 'conv':
                                loss = self.loss(images_mask_binary_pred_conv,
                                                 images_mask_binary.float())
                                if self.loss_type == 'dice':
                                    loss += self.dice(images_mask_binary_pred_conv,
                                                      images_mask_binary.float())
                            elif self.target_output == 'trans':
                                loss = self.loss(images_mask_binary_pred_tran,
                                                 images_mask_binary.float())
                                if self.loss_type == 'dice':
                                    loss += self.dice(images_mask_binary_pred_tran,
                                                      images_mask_binary.float())

                            elif self.target_output == 'both':
                                loss = self.loss(images_mask_binary_pred_tran,
                                                 images_mask_binary.float())
                                loss += self.loss(images_mask_binary_pred_conv,
                                                  images_mask_binary.float())
                                if self.loss_type == 'dice':
                                    loss += self.dice(images_mask_binary_pred_conv,
                                                      images_mask_binary.float())
                                    loss += self.dice(images_mask_binary_pred_tran,
                                                      images_mask_binary.float())

                        # loss = loss_tran
                        # choose average of conv and tran output as inference
                        # images_mask_binary_pred = images_mask_binary_pred_conv

                        if self.target_output == 'conv':
                            images_mask_binary_pred = images_mask_binary_pred_conv
                        elif self.target_output == 'trans':
                            images_mask_binary_pred = images_mask_binary_pred_tran
                        elif self.target_output == 'both':
                            images_mask_binary_pred = (images_mask_binary_pred_tran + images_mask_binary_pred_conv ) / 2.0


                        if self.n_classes == 1:
                            # if images_mask_binary_pred.shape[1] == 1:
                            assert images_mask_binary.min() >= 0 and images_mask_binary.max() <= 1, 'True mask indices should be in [0, 1]'
                            for k in range(images_mask_binary_pred.shape[1]):
                                images_mask_binary_pred[:, k, :, :] = (
                                            F.sigmoid(images_mask_binary_pred[:, k, :, :]) > 0.5).float()
                                # compute the Dice score
                                dice_score += dice_coeff(images_mask_binary_pred[:, k, :, :],
                                                         images_mask_binary[:, k, :, :],
                                                         reduce_batch_first=False)

                        if torch.isnan(loss):
                            num_nan += 1
                            print('nan loss detected')
                            continue
                        else:
                            self.running_loss_test += loss.item()

                        if batch % 20 == 0 or batch < 4:
                            end = time.time()
                            print('tested batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()

                        # evaluation metrics

                        if self.n_classes == 1:
                            images_masked_pred = images_input * images_mask_binary_pred
                            pred = images_masked_pred.detach().cpu()
                            images_masked = images_input * images_mask_binary
                            tg = images_masked.detach().cpu()
                            images_mask_binary = images_mask_binary.detach().cpu()
                            images_mask_binary_pred = images_mask_binary_pred.detach().cpu()

                        i_nmse = calc_nmse_tensor(tg, pred)
                        i_psnr = calc_psnr_tensor(tg, pred)
                        i_ssim = torch.zeros(images_mask_binary_pred.shape[1])
                        i_IoU = np.zeros(images_mask_binary_pred.shape[1])
                        for j in range(images_mask_binary_pred.shape[1]):
                            i_num_none_ioU = 0
                            # i_ssim[j] = calc_ssim_tensor(tg[:,j,:,:].unsqueeze(1), pred[:,j,:,:].unsqueeze(1))
                            for i in range(images_mask_binary_pred.shape[0]):
                                i_IoU_i = binaryMaskIOU(images_mask_binary[i][j], images_mask_binary_pred[i][j])
                                i_ssim_i = calc_ssim_tensor(tg[i, j, :, :].unsqueeze(0).unsqueeze(0),
                                                            pred[i, j, :, :].unsqueeze(0).unsqueeze(0))
                                if i_IoU_i is not None:
                                    i_IoU[j] += i_IoU_i
                                    i_ssim[j] += i_ssim_i
                                else:
                                    i_num_none_ioU += 1
                            if i_num_none_ioU < images_mask_binary_pred.shape[0]:
                                num_valid_ioU[j] += (images_mask_binary_pred.shape[0] - i_num_none_ioU)
                            else:
                                num_valid_ioU[j] += 0

                        nmse += i_nmse
                        psnr += i_psnr

                        ssim += i_ssim
                        IoU += i_IoU
                    else:
                        break

            for j in range(self.target_channel):
                num_valid = num_valid_ioU[j]
                if num_valid == 0:
                    ssim[j] = 0
                    IoU[j] = 0
                else:
                    ssim[j] /= num_valid
                    IoU[j] /= num_valid
            ssim_ave_channel = sum(ssim) / len(ssim)
            IoU_ave_channel = sum(IoU) / len(IoU)

            if self.max_num_batch_test is None:
                nmse /= (len(self.loader_test) - num_nan)
                psnr /= (len(self.loader_test) - num_nan)
                self.running_loss_test /= (len(self.loader_test) - num_nan)
                dice_score /= (len(self.loader_test) - num_nan)
            else:
                nmse /= (self.max_num_batch_test - num_nan)
                psnr /= (self.max_num_batch_test - num_nan)
                self.running_loss_test /= (self.max_num_batch_test - num_nan)
                dice_score /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(
                  ssim) + '|| ssim_ave_channel: ' + str(ssim_ave_channel)
              + '|| IoU: ' + str(IoU) + '|| IoU_ave_channel: ' + str(IoU_ave_channel) + '|| Dice score: ' + str(
                  dice_score))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU, dice_score


class TrainerSegBinaryDiceLoss3D_Unet:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, scheduler, grad_scaler, amp,
                 gradient_clipping, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, labels=None, loss_type='dice',
                 target_channel=40, if_use_scaler=True):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        self.labels = labels
        self.loss_type = loss_type
        self.n_classes = len(labels)
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.gradient_clipping = gradient_clipping
        self.if_use_scaler = if_use_scaler
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        if self.n_classes == 1:
            print("binary mask")
        self.target_channel = target_channel
        if self.loss_type == 'dice':
            self.dice = PerVolumeDiceLoss(epsilon=1e-6)

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (
            images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
            if_zero) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(device=self.device, dtype=torch.float32,
                                             memory_format=torch.channels_last)  # [batch, C, H, W]
                    # images_masked = images_masked.to(self.device)  # [batch, num_class]
                    images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)  # [batch, C, H, W]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_mask_binary, "images_mask_binary")

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        # images_mask_binary_pred = self.model(images_input)  # [batch, C, H, W]
                        # images_mask_binary_pred_tran = self.model(images_input)
                        # _, pred, _ = self.model(images_input, mask_ratio=0)
                        images_mask_binary_pred_conv = self.model(images_input)
                        # images_mask_binary_pred_tran = self.model.unpatchify(pred, self.target_channel)
                        if self.n_classes == 1:  # for binary class, use dice loss only
                            # if images_mask_binary_pred.shape[1] == 1:
                            if self.loss_type == 'dice':
                                loss_tran = self.loss(images_mask_binary_pred_conv,
                                                      images_mask_binary.float())
                                loss_tran += self.dice(images_mask_binary_pred_conv,
                                                       images_mask_binary.float())
                            else:
                                loss_tran = self.loss(images_mask_binary_pred_conv,
                                                      images_mask_binary.float())
                        loss = loss_tran
                        self.optimizer.zero_grad()
                        if torch.isnan(loss):
                            print("nan loss occur")
                            # print(filename)
                            print('images_mask_binary_pred_conv', images_mask_binary_pred_conv)
                            for j in range(images_input.size(0)):
                                print_var_detail(images_input[j], 'images_input')
                            num_nan += 1
                        else:
                            if self.if_use_scaler:
                                self.grad_scaler.scale(loss).backward()
                                self.grad_scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                                self.grad_scaler.step(self.optimizer)
                                self.running_loss_train += loss.item()
                                self.grad_scaler.update()
                                # self.scheduler.step()
                            else:
                                loss.backward()
                                self.optimizer.step()
                                # self.scheduler.step()
                                self.running_loss_train += loss.item()
                        if batch % 100 == 0 or batch < 4:
                            end = time.time()
                            print('trained batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, nmse, psnr, ssim, IoU, dice_score = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        dice_score = 0
        nmse = 0.0
        psnr = 0.0

        if self.target_channel is None:
            ssim = 0.0
            IoU = 0
            num_valid_ioU = 0
        else:
            # ssim = torch.zeros(len(self.labels))
            # IoU = np.zeros(len(self.labels))
            # num_valid_ioU = np.zeros(len(self.labels))
            ssim = torch.zeros(self.target_channel)
            IoU = np.zeros(self.target_channel)
            num_valid_ioU = np.zeros(self.target_channel)
        num_nan = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                for batch, (
                images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
                if_zero) in enumerate(self.loader_test):
                    if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                        timer_data, timer_model = utility.timer(), utility.timer()
                        images_input = images.to(device=self.device, dtype=torch.float32,
                                                 memory_format=torch.channels_last)  # [pair]
                        # images_masked = images_masked.to(self.device)  # [pair]
                        images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)
                        timer_data.hold()
                        timer_model.tic()
                        # images_mask_binary_pred_tran= self.model(images_input)
                        # _, pred, _ = self.model(images_input, mask_ratio=0)
                        # images_mask_binary_pred_tran = self.model.unpatchify(pred, self.target_channel)
                        images_mask_binary_pred_conv = self.model(images_input)

                        if self.n_classes == 1:
                            if self.loss_type == 'dice':
                                loss_tran = self.loss(images_mask_binary_pred_conv,
                                                      images_mask_binary.float())
                                loss_tran += self.dice(images_mask_binary_pred_conv,
                                                       images_mask_binary.float())

                        loss = loss_tran
                        # choose average of conv and tran output as inference
                        images_mask_binary_pred = images_mask_binary_pred_conv

                        if self.n_classes == 1:
                            # if images_mask_binary_pred.shape[1] == 1:
                            assert images_mask_binary.min() >= 0 and images_mask_binary.max() <= 1, 'True mask indices should be in [0, 1]'
                            for k in range(images_mask_binary_pred.shape[1]):
                                images_mask_binary_pred[:, k, :, :] = (
                                            F.sigmoid(images_mask_binary_pred[:, k, :, :]) > 0.5).float()
                                # compute the Dice score
                                dice_score += dice_coeff(images_mask_binary_pred[:, k, :, :],
                                                         images_mask_binary[:, k, :, :],
                                                         reduce_batch_first=False)

                        if torch.isnan(loss):
                            num_nan += 1
                            print('nan loss detected')
                            continue
                        else:
                            self.running_loss_test += loss.item()

                        if batch % 20 == 0 or batch < 4:
                            end = time.time()
                            print('tested batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()

                        # evaluation metrics

                        if self.n_classes == 1:
                            images_masked_pred = images_input * images_mask_binary_pred
                            pred = images_masked_pred.detach().cpu()
                            images_masked = images_input * images_mask_binary
                            tg = images_masked.detach().cpu()
                            images_mask_binary = images_mask_binary.detach().cpu()
                            images_mask_binary_pred = images_mask_binary_pred.detach().cpu()

                        i_nmse = calc_nmse_tensor(tg, pred)
                        i_psnr = calc_psnr_tensor(tg, pred)
                        i_ssim = torch.zeros(images_mask_binary_pred.shape[1])
                        i_IoU = np.zeros(images_mask_binary_pred.shape[1])
                        for j in range(images_mask_binary_pred.shape[1]):
                            i_num_none_ioU = 0
                            # i_ssim[j] = calc_ssim_tensor(tg[:,j,:,:].unsqueeze(1), pred[:,j,:,:].unsqueeze(1))
                            for i in range(images_mask_binary_pred.shape[0]):
                                i_IoU_i = binaryMaskIOU(images_mask_binary[i][j], images_mask_binary_pred[i][j])
                                i_ssim_i = calc_ssim_tensor(tg[i, j, :, :].unsqueeze(0).unsqueeze(0),
                                                            pred[i, j, :, :].unsqueeze(0).unsqueeze(0))
                                if i_IoU_i is not None:
                                    i_IoU[j] += i_IoU_i
                                    i_ssim[j] += i_ssim_i
                                else:
                                    i_num_none_ioU += 1
                            if i_num_none_ioU < images_mask_binary_pred.shape[0]:
                                num_valid_ioU[j] += (images_mask_binary_pred.shape[0] - i_num_none_ioU)
                            else:
                                num_valid_ioU[j] += 0

                        nmse += i_nmse
                        psnr += i_psnr

                        ssim += i_ssim
                        IoU += i_IoU
                    else:
                        break

            for j in range(self.target_channel):
                num_valid = num_valid_ioU[j]
                if num_valid == 0:
                    ssim[j] = 0
                    IoU[j] = 0
                else:
                    ssim[j] /= num_valid
                    IoU[j] /= num_valid
            ssim_ave_channel = sum(ssim) / len(ssim)
            IoU_ave_channel = sum(IoU) / len(IoU)

            if self.max_num_batch_test is None:
                nmse /= (len(self.loader_test) - num_nan)
                psnr /= (len(self.loader_test) - num_nan)
                self.running_loss_test /= (len(self.loader_test) - num_nan)
                dice_score /= (len(self.loader_test) - num_nan)
            else:
                nmse /= (self.max_num_batch_test - num_nan)
                psnr /= (self.max_num_batch_test - num_nan)
                self.running_loss_test /= (self.max_num_batch_test - num_nan)
                dice_score /= (self.max_num_batch_test - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(
                  ssim) + '|| ssim_ave_channel: ' + str(ssim_ave_channel)
              + '|| IoU: ' + str(IoU) + '|| IoU_ave_channel: ' + str(IoU_ave_channel) + '|| Dice score: ' + str(
                  dice_score))
        print('----------------------------------------------------------------------')
        return self.running_loss_test, nmse, psnr, ssim, IoU, dice_score



class TrainerClassgBinary3D_Conformer:
    """
            trainer for ipt model, includes train() and test() function

            Args:
            ----------
            args : argparse
                args saved in option.py
            loader_train: dataloader
                dataloader for training, expected tuple consists of pairs
            loader_test: dataloader
                dataloader for testing, expected tuple consists of pairs
            my_model: model
                ipt model with multi-heads/tails
            my_loss: nn.modules.loss._Loss
                loss modules defined in loss._init_
            ckp: checkpoint
                checkpoint class to load pretrained model if exits
            """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, scheduler, grad_scaler, amp,
                 gradient_clipping, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, labels=None, loss_type='dice',
                 target_channel=40, if_use_scaler=True, target_output = 'conv'):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
        self.labels = labels
        self.loss_type = loss_type
        self.n_classes = len(labels)
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.gradient_clipping = gradient_clipping
        self.if_use_scaler = if_use_scaler
        self.target_output = target_output
        # if load_ckp:
        #     checkpoint = torch.load(self.PATH_CKPOINT)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if RESUME_EPOCH > 0:
            self.model.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])
        self.device = device
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        if self.n_classes == 1:
            print("binary mask")
        self.target_channel = target_channel
        if self.loss_type == 'dice':
            self.dice = PerVolumeDiceLoss(epsilon=1e-6)

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            for batch, (
            images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
            if_zero) in enumerate(self.loader_train):
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = images.to(device=self.device, dtype=torch.float32,
                                             memory_format=torch.channels_last)  # [batch, C, H, W]
                    # images_masked = images_masked.to(self.device)  # [batch, num_class]
                    images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)  # [batch, C, H, W]
                    image_label = image_label.to(device=self.device, dtype=torch.long)  # [batch, C, H, W]
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(images_mask_binary, "images_mask_binary")

                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        # images_mask_binary_pred = self.model(images_input)  # [batch, C, H, W]
                        # images_mask_binary_pred_tran = self.model(images_input)
                        # _, pred, _ = self.model(images_input, mask_ratio=0)
                        images_mask_binary_pred_conv, images_label_pred_tran = self.model(images_input)
                        # images_mask_binary_pred_tran = self.model.unpatchify(pred, self.target_channel)

                        # if self.target_output == 'conv':
                        #     images_mask_binary_pred = images_mask_binary_pred_conv
                        # elif self.target_output == 'trans':
                        #     images_mask_binary_pred = images_mask_binary_pred_tran
                        # elif self.target_output == 'both':
                        #     images_mask_binary_pred = (images_mask_binary_pred_conv + images_mask_binary_pred_tran) // 2


                        if self.n_classes == 1:  # for binary class, use dice loss only
                            loss = self.loss(images_label_pred_tran, image_label.float())

                        # loss = loss_tran
                        self.optimizer.zero_grad()
                        if torch.isnan(loss):
                            print("nan loss occur")
                            # print(filename)
                            print('images_mask_binary_pred_conv', images_mask_binary_pred_conv)
                            print('images_label_pred_tran', images_label_pred_tran)
                            for j in range(images_input.size(0)):
                                print_var_detail(images_input[j], 'images_input')
                                print("is input has nan: " + str(torch.isnan(images_input).any()))
                            num_nan += 1
                        else:
                            if self.if_use_scaler:
                                self.grad_scaler.scale(loss).backward()
                                self.grad_scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                                self.grad_scaler.step(self.optimizer)
                                self.running_loss_train += loss.item()
                                self.grad_scaler.update()
                                # self.scheduler.step()
                            else:
                                loss.backward()
                                self.optimizer.step()
                                # self.scheduler.step()
                                self.running_loss_train += loss.item()
                        if batch % 100 == 0 or batch < 4:
                            end = time.time()
                            print('trained batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()
                else:
                    break
            # self.optimizer.schedule()
            if self.max_num_batch_train is None:
                self.running_loss_train /= (len(self.loader_train) - num_nan)
            else:
                self.running_loss_train /= (self.max_num_batch_train - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if i == 0 or (i + 1) % 1 == 0:
                print('*** EPOCH [%d / %d] || AVG LOSS: %f' % (i + 1, epochs, self.running_loss_train))
                # save model ckpt
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL CKPT SAVED.')
        # test model
        if show_test:
            running_loss_test, correct, total, accuracy = self.test()

        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance. WNet.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        dice_score = 0
        nmse = 0.0
        psnr = 0.0

        if self.target_channel is None:
            ssim = 0.0
            IoU = 0
            num_valid_ioU = 0
        else:
            # ssim = torch.zeros(len(self.labels))
            # IoU = np.zeros(len(self.labels))
            # num_valid_ioU = np.zeros(len(self.labels))
            ssim = torch.zeros(self.target_channel)
            IoU = np.zeros(self.target_channel)
            num_valid_ioU = np.zeros(self.target_channel)
        num_nan = 0
        correct = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                for batch, (
                images, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment,
                if_zero) in enumerate(self.loader_test):
                    if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                        timer_data, timer_model = utility.timer(), utility.timer()
                        images_input = images.to(device=self.device, dtype=torch.float32,
                                                 memory_format=torch.channels_last)  # [pair]
                        # images_masked = images_masked.to(self.device)  # [pair]
                        images_mask_binary = images_mask_binary.to(device=self.device, dtype=torch.long)
                        image_label = image_label.to(device=self.device, dtype=torch.long)
                        timer_data.hold()
                        timer_model.tic()
                        # images_mask_binary_pred_tran= self.model(images_input)
                        # _, pred, _ = self.model(images_input, mask_ratio=0)
                        # images_mask_binary_pred_tran = self.model.unpatchify(pred, self.target_channel)
                        images_mask_binary_pred_conv, images_label_pred_tran = self.model(images_input)

                        if self.n_classes == 1:
                            loss = self.loss(images_label_pred_tran,
                                                 image_label.float())

                        if torch.isnan(loss):
                            num_nan += 1
                            print('nan loss detected')
                            continue
                        else:
                            self.running_loss_test += loss.item()

                        if batch % 20 == 0 or batch < 4:
                            end = time.time()
                            print('tested batches:', batch)
                            print('time: ' + str(end - start) + ' sec')
                            start = time.time()

                        # evaluation metrics
                        tg = image_label.detach()
                        pred = images_label_pred_tran.detach()

                        if tg.shape[1] > 1:  # multi class
                            _, predicted = torch.max(pred.data, 1)
                            _, label_target = torch.max(tg.data, 1)
                        else:  # binary class
                            predicted = F.sigmoid(pred) > 0.5
                            predicted[predicted >= 0.5] = 1
                            predicted[predicted < 0.5] = 0
                            label_target = tg
                            label_target[label_target >= 0.5] = 1
                            label_target[label_target < 0.5] = 0

                        total += label_target.size(0)
                        correct += (predicted == label_target).sum().item()
                    else:
                        break
                if self.max_num_batch_test is None:
                    self.running_loss_test /= (len(self.loader_test) - num_nan)
                else:
                    self.running_loss_test /= (self.max_num_batch_test - num_nan)
                accuracy = 100. * correct / total

            print('### TEST LOSS: ',
                  str(self.running_loss_test) + '|| correct: ' + str(correct) + '|| total: ' + str(
                      total) + '|| accuracy: ' + str(accuracy))
            print('----------------------------------------------------------------------')
            print('----------------------------------------------------------------------')
            return self.running_loss_test, correct, total, accuracy