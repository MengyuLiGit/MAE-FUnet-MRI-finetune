import torch
import os
import time
from tqdm.autonotebook import tqdm
from utils.help_func import print_var_detail
from torch.cuda.amp import autocast, GradScaler
def freeze_mae_encoder(model):
    if hasattr(model, 'mae'):  # wrapped in fusion
        mae = model.mae
    else:
        mae = model

    for param in mae.patch_embed.parameters():
        param.requires_grad = False
    for param in mae.blocks.parameters():
        param.requires_grad = False
    for param in mae.norm.parameters():
        param.requires_grad = False
    mae.cls_token.requires_grad = False
    mae.pos_embed.requires_grad = False

def freeze_mae_encoder_and_decoder(model):
    if hasattr(model, 'mae'):
        mae = model.mae
    else:
        mae = model

    for name, param in mae.named_parameters():
        if name.startswith("patch_embed") or name.startswith("blocks") or \
           name.startswith("norm") or name in ["cls_token", "pos_embed"] or \
           name.startswith("decoder") or name == "mask_token":
            param.requires_grad = False


def unfreeze_mae(model):
    if hasattr(model, 'mae'):
        for param in model.mae.parameters():
            param.requires_grad = True


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


def remap_logits_old_to_new(logits_old, mapping, num_new_classes):
    B, _, H, W = logits_old.shape
    logits_new = torch.zeros((B, num_new_classes, H, W), device=logits_old.device)
    for old_cls, new_cls in mapping.items():
        logits_new[:, new_cls] += logits_old[:, old_cls]
    return logits_new

def remap_labels_old_to_new(labels_old, mapping, default_class=0):
    labels_new = torch.full_like(labels_old, default_class)
    for old_cls, new_cls in mapping.items():
        labels_new[labels_old == old_cls] = new_cls
    return labels_new

class Trainer:
    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer, RESUME_EPOCH,
                 PATH_MODEL, device, max_num_batch_train=None, max_num_batch_test=None,
                 task='classify', cls_strategy='cls_token', freeze_mae_encoder=False, freeze_mae_encoder_decoder=False,
                 force_float32 = False, num_classes = None, unfreeze_at_epoch=None,weight_decay=1e-4, remap_classes=False, num_remap_classes = 0,
                 old_to_new_mapping ={}):
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.model = my_model
        self.loss_fn = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.RESUME_EPOCH = RESUME_EPOCH
        self.device = device
        self.max_num_batch_train = max_num_batch_train
        self.max_num_batch_test = max_num_batch_test
        self.task = task  # classify or segment
        self.cls_strategy = cls_strategy
        self.freeze_mae_encoder = freeze_mae_encoder
        self.freeze_mae_encoder_decoder = freeze_mae_encoder_decoder
        self.force_float32 = force_float32
        self.scaler = GradScaler()
        self.num_classes = num_classes

        # Enable remapping mode
        self.remap_classes = remap_classes
        self.num_remap_classes = num_remap_classes
        self.old_to_new_mapping = old_to_new_mapping


        if RESUME_EPOCH > 0:
            checkpoint = torch.load(PATH_MODEL + f'model_E{RESUME_EPOCH}.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.unfrozen = False
        self.weight_decay = weight_decay

    def train(self, epochs=5, show_step=1, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)

        if self.freeze_mae_encoder:
            freeze_mae_encoder(self.model)
        if self.freeze_mae_encoder_decoder:
            freeze_mae_encoder_and_decoder(self.model)
        if self.freeze_mae_encoder or self.freeze_mae_encoder_decoder:
            if hasattr(self.model, 'if_freeze_mae'):
                self.model.if_freeze_mae = True
                print("MAE freeze mode is enabled:", self.model.if_freeze_mae)

        print(f"Total params: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        pbar = tqdm(range(self.RESUME_EPOCH, epochs), desc='Training')
        for epoch in pbar:
            if self.unfreeze_at_epoch is not None and epoch >= self.unfreeze_at_epoch and not self.unfrozen:
                print(f"🔓 Unfreezing model at epoch {epoch}")
                for param in self.model.parameters():
                    param.requires_grad = True
                # unfreeze_mae(self.model)
                self.unfrozen = True
                if hasattr(self.model, 'if_freeze_mae'):
                    self.model.if_freeze_mae = False
                    print("MAE freeze mode is disabled:", self.model.if_freeze_mae)

                # 🔁 Reinitialize optimizer with weight decay after unfreezing, only activate when hit unfreeze epoch
                # no need to reinitialize if resume epoch larger than the unfreeze epoch
                if epoch == self.unfreeze_at_epoch:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=lr,
                        weight_decay=self.weight_decay
                    )
                    optimizer_to(self.optimizer, self.device)
                    print("✅ Reinitialized optimizer with weight decay after unfreezing.")

            self.model.train()
            running_loss = 0.0
            num_nan = 0

            if hasattr(self.loader_train.sampler, 'set_epoch'):
                self.loader_train.sampler.set_epoch(epoch)
            start_time = time.time()
            for batch_idx, (images, labels) in enumerate(self.loader_train):
                if self.max_num_batch_train is not None and batch_idx >= self.max_num_batch_train:
                    break
                if self.force_float32:
                    images = images.to(self.device).float() # if want to use float32 as input
                else:
                    images = images.to(self.device)
                if images.dtype not in [torch.float32, torch.float16]:
                    print(f"images input is not in float type: {images.dtype}")
                    images = images.float()


                labels = labels.to(self.device)

                if labels.ndim == 4 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)  # 🛠️ Add this to fix label shape

                if self.task == 'classify':
                    labels = labels.argmax(dim=1).long()  # for classification only
                elif self.task == 'segment':
                    labels = labels.long()
                self.optimizer.zero_grad()

                if images.dtype == torch.float16:
                    # Mixed precision branch
                    with autocast():
                        if self.task == 'classify':
                            logits = self.model.forward_cls(images, cls_strategy=self.cls_strategy)
                        elif self.task == 'segment':
                            logits = self.model(images)
                    logits = logits.float()
                    # logitstype = logits.dtype
                    # labelstype = labels.dtype
                    # print(logitstype)
                    # print(labelstype)
                    # print(f"Logits max: {logits.max().item()}, min: {logits.min().item()}")
                    # print(f"Images mean: {images.mean().item()}, std: {images.std().item()}")
                    # print(f"Labels unique: {labels.unique()}")
                    # Now disable autocast for loss calculation
                    with torch.cuda.amp.autocast(enabled=False):
                        loss = self.loss_fn(logits, labels)


                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                        print(f"❌ Invalid loss: {loss.item():.6f}")
                    if torch.isnan(loss):
                        print("⚠️ NaN loss at batch", batch_idx)
                        num_nan += 1
                    else:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        running_loss += loss.item()
                else:
                    # # Normal float32 branch
                    if self.task == 'classify':
                        logits = self.model.forward_cls(images, cls_strategy=self.cls_strategy)
                    elif self.task == 'segment':
                        logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                        print(f"❌ Invalid loss: {loss.item():.6f}")
                        # continue
                    if torch.isnan(loss):
                        print("⚠️ NaN loss at batch", batch_idx)
                        num_nan += 1
                    else:
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()

                if(epoch == 0 or epoch == self.RESUME_EPOCH) and batch_idx == 0:
                    print_var_detail(images, "images_input")
                    print(f"images type: {images.dtype}")
                    print_var_detail(labels, "labels_input")
                    print(f"labels type: {labels.dtype}")
                    print_var_detail(logits, "logits")
                    print(f"logits type: {logits.dtype}")



                batch_time = time.time() - start_time
                if batch_idx < 20 and (epoch == 0 or epoch == self.RESUME_EPOCH):
                    print(f"⏱️ Batch {batch_idx} time: {batch_time:.4f} seconds")
                start_time = time.time()
            denom = len(self.loader_train) if self.max_num_batch_train is None else self.max_num_batch_train
            avg_loss = running_loss / (denom - num_nan)
            pbar.set_description(f"EPOCH [{epoch + 1}/{epochs}] || AVG LOSS: {avg_loss:.6f}")

            if (epoch + 1) % show_step == 0 or epoch == 0:
                self.save_checkpoint(epoch + 1)

        if show_test:
            self.test()

        return self.model

    def save_checkpoint(self, epoch):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.PATH_MODEL + f'model_E{epoch}.pt')
        print(f'MODEL SAVED at epoch {epoch}')

    def test(self, exclude_background=True):
        self.model.eval()
        self.model.to(self.device)
        correct, total, running_loss, num_nan = 0, 0, 0.0, 0
        eps = 1e-6
        num_classes = self.num_remap_classes if self.remap_classes else (self.num_classes or 8)
        correct_per_class = [0] * num_classes
        total_per_class = [0] * num_classes
        dice_scores = [0.0] * num_classes
        iou_scores = [0.0] * num_classes

        test_loader = tqdm(enumerate(self.loader_test), total=len(self.loader_test), desc='Testing')

        with torch.no_grad():
            for batch_idx, (images, labels) in test_loader:
                if self.max_num_batch_test is not None and batch_idx >= self.max_num_batch_test:
                    break

                images = images.to(self.device).float()
                labels = labels.to(self.device)
                if labels.ndim == 4 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)

                if self.task == 'classify':
                    labels_idx = labels.argmax(dim=1)
                    logits = self.model.forward_cls(images, cls_strategy=self.cls_strategy)
                    loss = self.loss_fn(logits, labels_idx)
                    _, preds = torch.max(logits, dim=1)

                    if exclude_background:
                        mask = labels_idx != 0
                        correct += (preds[mask] == labels_idx[mask]).sum().item()
                        total += mask.sum().item()
                    else:
                        correct += (preds == labels_idx).sum().item()
                        total += labels_idx.numel()

                    for i in range(labels_idx.size(0)):
                        label = labels_idx[i].item()
                        if exclude_background and label == 0:
                            continue
                        pred = preds[i].item()
                        total_per_class[label] += 1
                        if pred == label:
                            correct_per_class[label] += 1

                elif self.task == 'segment':
                    logits = self.model(images)
                    # if self.remap_classes:
                    #     logits = remap_logits_old_to_new(logits, self.old_to_new_mapping, self.num_remap_classes)
                    #     labels = remap_labels_old_to_new(labels, self.old_to_new_mapping)

                    loss = self.loss_fn(logits, labels)
                    running_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    if self.remap_classes:
                        preds = remap_labels_old_to_new(preds, self.old_to_new_mapping)

                    mask = (labels != 0) if exclude_background else torch.ones_like(labels).bool()
                    correct += (preds[mask] == labels[mask]).sum().item()
                    total += mask.sum().item()

                    for cls in range(num_classes):
                        if exclude_background and cls == 0:
                            continue
                        pred_cls = (preds == cls).float()
                        label_cls = (labels == cls).float()
                        intersection = (pred_cls * label_cls).sum()
                        union = pred_cls.sum() + label_cls.sum() - intersection
                        iou = (intersection + eps) / (union + eps) if union > 0 else torch.tensor(0.0)
                        dice = (2 * intersection + eps) / (pred_cls.sum() + label_cls.sum() + eps)
                        dice_scores[cls] += dice.item()
                        iou_scores[cls] += iou.item()

        denom = max(1, len(self.loader_test) - num_nan)
        avg_loss = running_loss / denom
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        log_lines = []
        bg_info = " (excluding background)" if exclude_background else ""
        log_lines.append(
            f"### TEST LOSS: {avg_loss:.6f} || Accuracy{bg_info}: {accuracy:.2f}% || Correct: {correct} / {total}")
        log_lines.append("----------------------------------------------------------------------")

        if self.task == 'segment':
            log_lines.append(f"Per-class Dice and IoU{bg_info}:")
            if exclude_background:
                class_range = range(1, num_classes)  # Skip background
            else:
                class_range = range(num_classes)

            for cls in class_range:
                dice_avg = dice_scores[cls] / len(self.loader_test)
                iou_avg = iou_scores[cls] / len(self.loader_test)
                log_lines.append(f"Class {cls}: Dice {dice_avg:.4f} || IoU {iou_avg:.4f}")

            mean_dice = sum([dice_scores[cls] for cls in class_range]) / (len(self.loader_test) * len(class_range))
            mean_iou = sum([iou_scores[cls] for cls in class_range]) / (len(self.loader_test) * len(class_range))
            log_lines.append(f"Mean Dice{bg_info}: {mean_dice:.4f}")
            log_lines.append(f"Mean IoU{bg_info}: {mean_iou:.4f}")

        log_str = "\n".join(log_lines)
        print(log_str)

        log_path = self.get_log_path()
        with open(log_path, "w") as log_file:
            log_file.write(log_str + "\n")

        return avg_loss, correct, total, accuracy

    def get_log_path(self):
        base_log_path = os.path.join(self.PATH_MODEL, "log.txt")
        log_path = base_log_path
        log_index = 1
        while os.path.exists(log_path):
            log_path = os.path.join(self.PATH_MODEL, f"log{log_index}.txt")
            log_index += 1
        return log_path




