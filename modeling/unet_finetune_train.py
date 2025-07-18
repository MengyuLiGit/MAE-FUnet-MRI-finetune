import utils.ipt_util as utility

import torch

from tqdm import tqdm
from help_func import print_var_detail
import time
from utils.evaluation_utils import calc_nmse_tensor, calc_psnr_tensor, calc_ssim_tensor, binaryMaskIOU
import copy
import numpy as np
from utils.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
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
class Trainer:
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

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, if_fix_encoder=False,
                 task='classify'):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
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
        self.if_fix_encoder = if_fix_encoder
        self.task = task

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        if self.if_fix_encoder:  # freeze all param in encoder
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            # for batch, (data_i, image_label, patient_label) in enumerate(self.loader_train):
            for batch, data_i in enumerate(self.loader_train):
                image_tensor, image_label, patient_ID, direction = data_i
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = image_tensor.to(self.device)  # [batch, chn, H, W]
                    image_label = image_label
                    image_label = image_label.to(self.device)  # [batch, num_class]
                    patient_label = patient_ID
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(image_label, "image_label")

                    self.optimizer.zero_grad()
                    image_label_pred = self.model(images_input)
                    loss = self.loss(image_label_pred, image_label)
                    if torch.isnan(loss):
                        print("nan loss occur")
                        # print(filename)
                        print('image_label_pred', image_label_pred)
                        print('image_label', image_label)
                        for j in range(images_input.size(0)):
                            print_var_detail(images_input[j], 'images_input')
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.running_loss_train += loss.item()
                    if batch % 2000 == 0 or batch < 4:
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
            # torch.save({
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            # }, self.PATH_CKPOINT)
            # print('MODEL SAVED at epoch: ' + str(i + 1))
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
            loss_test, correct, total, accuracy = self.test()

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
        num_nan = 0
        correct = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            # for batch, (data_i, image_label, _) in enumerate(self.loader_test):
            for batch, data_i in enumerate(self.loader_test):
                if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                    timer_data, timer_model = utility.timer(), utility.timer()
                    image_tensor, image_label, patient_ID, direction = data_i
                    images_input = image_tensor.to(self.device)  # [pair]
                    image_label = image_label
                    image_label = image_label.to(self.device)  # [pair]
                    timer_data.hold()
                    timer_model.tic()
                    image_label_pred = self.model(image=images_input)
                    loss = self.loss(image_label_pred, image_label)
                    if torch.isnan(loss):
                        num_nan += 1
                    else:
                        self.running_loss_test += loss.item()

                    if batch % 100 == 0 or batch < 4:
                        end = time.time()
                        print('tested batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()

                    # evaluation metrics
                    tg = image_label.detach()
                    pred = image_label_pred.detach()

                    if tg.shape[1] > 1: # multi class
                        _, predicted = torch.max(pred.data, 1)
                        _, label_target = torch.max(tg.data, 1)
                    else: # binary class
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

class Trainer3D:
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

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, if_fix_encoder=False,
                 task='classify'):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.cpu = False
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
        self.if_fix_encoder = if_fix_encoder
        self.task = task

    def train(self, show_step=-1, epochs=5, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        if self.if_fix_encoder:  # freeze all param in encoder
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False

        # training iteration
        # pbar = tqdm(range(epochs), desc='LOSS')
        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0

            start = time.time()
            # for batch, (data_i, image_label, patient_label) in enumerate(self.loader_train):
            for batch, data_i in enumerate(self.loader_train):
                image_tensor, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment, if_zero = data_i
                if self.max_num_batch_train is None or batch < self.max_num_batch_train:
                    images_input = image_tensor.to(self.device)  # [batch, chn, H, W]
                    image_label = image_label
                    image_label = image_label.to(self.device)  # [batch, num_class]
                    patient_label = patient_ID
                    if i == 0 and batch == 0:
                        print_var_detail(images_input, "images_input")
                        print_var_detail(image_label, "image_label")

                    self.optimizer.zero_grad()
                    image_label_pred = self.model(images_input)
                    loss = self.loss(image_label_pred, image_label)
                    if torch.isnan(loss):
                        print("nan loss occur")
                        # print(filename)
                        print('image_label_pred', image_label_pred)
                        print('image_label', image_label)
                        for j in range(images_input.size(0)):
                            print_var_detail(images_input[j], 'images_input')
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.running_loss_train += loss.item()
                    if batch % 2000 == 0 or batch < 4:
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
            # torch.save({
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            # }, self.PATH_CKPOINT)
            # print('MODEL SAVED at epoch: ' + str(i + 1))
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
            loss_test, correct, total, accuracy = self.test()

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
        num_nan = 0
        correct = 0

        total = 0
        with torch.no_grad():
            start = time.time()
            # for batch, (data_i, image_label, _) in enumerate(self.loader_test):
            for batch, data_i in enumerate(self.loader_test):
                if self.max_num_batch_test is None or batch < self.max_num_batch_test:
                    timer_data, timer_model = utility.timer(), utility.timer()
                    image_tensor, image_label, patient_ID, if_full, image_tensor_mask_total, images_mask_binary, use_argment, if_zero = data_i
                    images_input = image_tensor.to(self.device)  # [pair]
                    image_label = image_label
                    image_label = image_label.to(self.device)  # [pair]
                    timer_data.hold()
                    timer_model.tic()
                    image_label_pred = self.model(image=images_input)
                    loss = self.loss(image_label_pred, image_label)
                    if torch.isnan(loss):
                        num_nan += 1
                    else:
                        self.running_loss_test += loss.item()

                    if batch % 100 == 0 or batch < 4:
                        end = time.time()
                        print('tested batches:', batch)
                        print('time: ' + str(end - start) + ' sec')
                        start = time.time()

                    # evaluation metrics
                    tg = image_label.detach()
                    pred = image_label_pred.detach()

                    if tg.shape[1] > 1: # multi class
                        _, predicted = torch.max(pred.data, 1)
                        _, label_target = torch.max(tg.data, 1)
                    else: # binary class
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