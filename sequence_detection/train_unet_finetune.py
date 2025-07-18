import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.help_func import print_var_detail
import time
import torch
from packaging import version  # for safe version comparison
from torch.amp import autocast
def optimizer_to(optim, device):
    for param in optim.state.values():
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
    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None, force_float32=False
                 ):
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.model = my_model
        self.loss_fn = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.device = device
        self.cpu = False
        self.error_last = 1e8
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        self.force_float32 = force_float32
        if version.parse(torch.__version__) >= version.parse("2.2.0"):
            from torch.amp import GradScaler
            self.scaler = GradScaler(device=device)
        else:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        if RESUME_EPOCH > 0:
            checkpoint = torch.load(PATH_MODEL + f'model_E{RESUME_EPOCH}.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, epochs=5, show_step=1, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='LOSS')
        for epoch in pbar:
            self.running_loss_train = 0.0
            num_nan = 0

            if hasattr(self.loader_train.sampler, 'set_epoch'):
                self.loader_train.sampler.set_epoch(epoch)
            start = time.time()
            for batch_idx, (images, labels) in enumerate(self.loader_train):
                if self.max_num_batch_train is not None and batch_idx >= self.max_num_batch_train:
                    break
                images = images.to(self.device)
                if self.force_float32:
                    images = images.float()
                labels = labels.to(self.device).long()
                labels = labels.argmax(dim=1)  # one-hot → index

                if epoch == 0 and batch_idx == 0:
                    print_var_detail(images, "images_input")
                    print_var_detail(labels, "image_label (as indices)")

                self.optimizer.zero_grad()

                if images.dtype == torch.float16:
                    with autocast(device_type=self.device.type):
                        logits = self.model(images)

                    with autocast(device_type=self.device.type, enabled=False):  # Disable autocast for loss computation
                        logits = logits.float()
                        loss = self.loss_fn(logits, labels)

                    if torch.isnan(loss):
                        print("⚠️ NaN loss at batch", batch_idx)
                        num_nan += 1
                    else:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.running_loss_train += loss.item()
                else:
                    logits = self.model(images)  # Unet forwards directly
                    logits = logits.float()
                    loss = self.loss_fn(logits, labels)

                    if torch.isnan(loss):
                        print("⚠️ NaN loss at batch", batch_idx)
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        self.running_loss_train += loss.item()

                if (batch_idx % 1000 == 0 or batch_idx < 20) and epoch == 0:
                    end = time.time()
                    print('trained batches:', batch_idx)
                    print('time: ' + str(end - start) + ' sec')
                    start = time.time()

            denom = len(self.loader_train) if self.max_num_batch_train is None else self.max_num_batch_train
            avg_loss = self.running_loss_train / (denom - num_nan)
            pbar.set_description(f"EPOCH [{epoch + 1}/{epochs}] || AVG LOSS: {avg_loss:.6f}")

            if (epoch + 1) % show_step == 0 or epoch == 0:
                print(f'*** EPOCH {epoch + 1} || AVG LOSS: {avg_loss:.6f}')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + f'model_E{epoch + 1}.pt')
                print('MODEL CKPT SAVED.')

        if show_test:
            self.test()

        return self.model

    def test(self):
        self.model.eval()
        self.model.to(self.device)

        correct = 0
        total = 0
        running_loss = 0.0
        num_nan = 0

        # Try to infer number of classes from the final layer or default to 8
        num_classes = self.model.class_dim if hasattr(self.model, 'class_dim') else 8
        correct_per_class = [0] * num_classes
        total_per_class = [0] * num_classes

        test_loader = tqdm(enumerate(self.loader_test), total=len(self.loader_test), desc='Testing')

        with torch.no_grad():
            for batch_idx, (images, labels) in test_loader:
                if self.max_num_batch_test is not None and batch_idx >= self.max_num_batch_test:
                    break

                images = images.to(self.device).float()
                labels = labels.to(self.device).long()
                labels_idx = labels.argmax(dim=1)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels_idx)

                if torch.isnan(loss):
                    num_nan += 1
                    continue

                running_loss += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += (preds == labels_idx).sum().item()
                total += labels.size(0)

                # Per-class accuracy stats
                for i in range(labels.size(0)):
                    label = labels_idx[i].item()
                    pred = preds[i].item()
                    total_per_class[label] += 1
                    if pred == label:
                        correct_per_class[label] += 1

        avg_loss = running_loss / (len(self.loader_test) - num_nan)
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # === Build log string ===
        log_lines = []
        log_lines.append(f"### TEST LOSS: {avg_loss:.6f} || Accuracy: {accuracy:.2f}% || Correct: {correct} / {total}")
        log_lines.append("----------------------------------------------------------------------")
        log_lines.append("Per-class accuracy:")

        for cls in range(num_classes):
            total_cls = total_per_class[cls]
            correct_cls = correct_per_class[cls]
            acc_cls = 100.0 * correct_cls / total_cls if total_cls > 0 else 0.0
            log_lines.append(f"Class {cls}: {acc_cls:.2f}% ({correct_cls} / {total_cls})")

        log_str = "\n".join(log_lines)
        print(log_str)

        # === Logging to file ===
        base_log_path = os.path.join(self.PATH_MODEL, "log.txt")
        log_path = base_log_path
        log_index = 1
        while os.path.exists(log_path):
            log_path = os.path.join(self.PATH_MODEL, f"log{log_index}.txt")
            log_index += 1

        with open(log_path, "w") as log_file:
            log_file.write(log_str + "\n")

        return avg_loss, correct, total, accuracy, correct_per_class, total_per_class

