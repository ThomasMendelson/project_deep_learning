import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from monai.losses import DiceCELoss, FocalLoss, DiceFocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import albumentations as A
from model3D import UNET3D
from model3D_2 import UNet3D as UNet3D_2
from dataset3D import Dataset3D


from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loader,
    check_accuracy,
    check_accuracy_multy_models,
    # save_predictions_as_imgs,
    save_instance_by_colors,
    apply_color_map,
)

# Hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
L1_LAMBDA = 1e-5
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 8
CROP_SIZE = (32, 256, 256)
CLASS_WEIGHTS = [0.2, 0.6, 0.2]  # [0.15, 0.6, 0.25]  # [0.1, 0.6, 0.3] [0.1, 0.7, 0.2]
PIN_MEMORY = False
LOAD_MODEL = False
WANDB_TRACKING = True
THREE_D = True
# 3D
TRAIN_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02"
TRAIN_SEG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02_GT/SEG"
TRAIN_TRA_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02_GT/TRA2"
VAL_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01"
VAL_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG"
VAL_TRA_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/TRA2"


# TRAIN_IMG_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/02"
# TRAIN_SEG_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/SEG"
# TRAIN_TRA_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/TRA2"
# VAL_IMG_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/01"
# VAL_MASK_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/SEG"
# VAL_TRA_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/TRA2"

# TRAIN_IMG_DIR = r"C:\Users\beaviv\PycharmProjects\ImageProcessing\Datasets\Fluo-N3DH-SIM+\02"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02"
# TRAIN_MASK_DIR = r"C:\Users\beaviv\PycharmProjects\ImageProcessing\Datasets\Fluo-N3DH-SIM+\02_GT\SEG"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02_ERR_SEG"
# VAL_IMG_DIR = r"C:\Users\beaviv\PycharmProjects\ImageProcessing\Datasets\Fluo-N3DH-SIM+\01"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01"
# VAL_MASK_DIR = r"C:\Users\beaviv\PycharmProjects\ImageProcessing\Datasets\Fluo-N3DH-SIM+\01_GT\SEG"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01_ERR_SEG"

def calculate_l1_loss(model):
    l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
    return L1_LAMBDA * l1_loss


def train_fn(loader, model, optimizer, loss_functions, scaler):
    loop = tqdm(loader)
    model.train()
    total_loss = 0.0

    for batch_idx, (data, class_targets, marker_targets) in enumerate(loop):
        loss = 0
        data = data.to(device=DEVICE)
        class_targets = Dataset3D.split_mask(class_targets.to(device=DEVICE)).long()
        marker_targets[marker_targets > 0] = 1
        marker_targets = marker_targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            class_predictions, marker_predictions = model(data)
            for loss_fn in loss_functions:
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    # CrossEntropyLoss expects targets as class indices (without one-hot encoding)
                    loss += loss_fn(class_predictions, class_targets)
                elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
                    # BCEWithLogitsLoss expects targets to be of the same shape as predictions
                    loss += loss_fn(marker_predictions.squeeze(1), marker_targets)
                elif isinstance(loss_fn, DiceCELoss):
                    class_targets_one_hot = F.one_hot(class_targets, num_classes=3).permute(0, 4, 1, 2, 3).float()
                    loss += loss_fn(class_predictions, class_targets_one_hot)
            # Add L1 regularization
            loss += calculate_l1_loss(model)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def evaluate_fn(loader, model, loss_functions):
    loop = tqdm(loader)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, class_targets, marker_targets in loop:
            loss = 0
            data = data.to(device=DEVICE)

            class_targets = Dataset3D.split_mask(class_targets.to(device=DEVICE)).long()
            marker_targets[marker_targets > 0] = 1
            marker_targets = marker_targets.to(device=DEVICE)

            class_predictions, marker_predictions = model(data)

            for loss_fn in loss_functions:
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss += loss_fn(class_predictions, class_targets)

                elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
                    loss += loss_fn(marker_predictions.squeeze(1), marker_targets)

                elif isinstance(loss_fn, DiceCELoss):
                    class_targets_one_hot = F.one_hot(class_targets, num_classes=3).permute(0, 4, 1, 2, 3).float()
                    loss += loss_fn(class_predictions, class_targets_one_hot)

            # Add L1 regularization
            loss += calculate_l1_loss(model)

            total_loss += loss.item()
            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def main():
    if WANDB_TRACKING:
        wandb.login(key="12b9b358323faf2af56dc288334e6247c1e8bc63")
        wandb.init(project="seg_unet_3D",
                   config={
                       "epochs": NUM_EPOCHS,
                       "batch_size": BATCH_SIZE,
                       "lr": LEARNING_RATE,
                       "L1_lambda": L1_LAMBDA,
                       "L2_lambda": WEIGHT_DECAY,
                       "CE_weight": CLASS_WEIGHTS,
                       "img_size": CROP_SIZE,
                   })
        wandb_step = 0

    model = UNET3D(in_channels=1, out_channels=3).to(DEVICE)
    # model = UNet3D_2(in_channels=1, num_classes=3).to(DEVICE)

    class_weights = torch.FloatTensor(CLASS_WEIGHTS).to(DEVICE)
    #CrossEntropyLoss
    # criterion = [nn.CrossEntropyLoss(weight=class_weights)]#, nn.BCEWithLogitsLoss()]
    #DiceCELoss
    criterion = [DiceCELoss(softmax=True, squared_pred=True, weight=class_weights, batch=True),
                 # in reference was , batch=True)
                 nn.BCEWithLogitsLoss()]
    #FocalLoss
    # criterion = [FocalLoss(gamma=2, alpha=class_weights),
    #              nn.BCEWithLogitsLoss()]
    #DiceFocalLoss
    # criterion = [DiceFocalLoss(softmax=True, squared_pred=True, weight=class_weights, gamma=2),
    #              nn.BCEWithLogitsLoss()]


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

    train_loader = get_loader(dir=TRAIN_IMG_DIR, seg_dir=TRAIN_SEG_DIR, tra_dir=TRAIN_TRA_DIR, train_aug=True,
                              shuffle=True,
                              batch_size=BATCH_SIZE, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)
    val_loader = get_loader(dir=VAL_IMG_DIR, seg_dir=VAL_MASK_DIR, tra_dir=VAL_TRA_DIR, train_aug=False, shuffle=False,
                            batch_size=BATCH_SIZE, crop_size=CROP_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            three_d=THREE_D, device=DEVICE)

    test_check_accuracy_loader = get_loader(dir=VAL_IMG_DIR, seg_dir=VAL_MASK_DIR, tra_dir=VAL_TRA_DIR, train_aug=False,
                                            shuffle=False,
                                            batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)

    if LOAD_MODEL:
        load_checkpoint(torch.load(f"checkpoint/unet3D.pth.tar", map_location=torch.device(DEVICE)), model)
        model.to(DEVICE)
        check_accuracy(val_loader, model, device=DEVICE, three_d=THREE_D)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"epoch: [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss = train_fn(train_loader, model, optimizer, criterion, scaler)
        val_loss = evaluate_fn(val_loader, model, criterion)

        scheduler.step(val_loss)

        if WANDB_TRACKING:
            wandb.log(
                {"Train Loss": train_loss, "Val Loss": val_loss, "Learning Rate": optimizer.param_groups[0]['lr']},
                step=wandb_step)

        if (epoch + 1) % 10 == 0:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="checkpoint/unet3D.pth.tar")

            # save_instance_by_colors(loader=val_loader, model=model, folder="checkpoint/saved_images", device=DEVICE,
            #                         three_d=THREE_D, wandb_tracking=WANDB_TRACKING, wandb_step=wandb_step)
            if (epoch + 1) != NUM_EPOCHS:
                # check_accuracy(test_check_accuracy_loader, model, num_image=50, device=DEVICE, three_d=THREE_D)
                check_accuracy_multy_models(test_check_accuracy_loader, [model], num_image=50, device=DEVICE, three_d=THREE_D)

        if WANDB_TRACKING:
            wandb_step += 1

    # check_accuracy(test_check_accuracy_loader, model, device=DEVICE, three_d=THREE_D)
    check_accuracy_multy_models(test_check_accuracy_loader, [model], device=DEVICE, three_d=THREE_D)
    # save instance image
    save_instance_by_colors(loader=test_check_accuracy_loader, model=model, folder="checkpoint", three_d=THREE_D,
                            device=DEVICE, wandb_tracking=WANDB_TRACKING, wandb_step=wandb_step)

    if WANDB_TRACKING:
        # torch.onnx.export(model, torch.randn(1, 1, CROP_SIZE, device=DEVICE), "model.onnx")
        # wandb.save("model.onnx")
        # print("=> saved model.onnx to wandb")
        wandb.finish()


def t_acc():
    model = UNET3D(in_channels=1, out_channels=3).to(DEVICE)
    load_checkpoint(torch.load("checkpoint/vit_checkpoint.pth.tar", map_location=torch.device(DEVICE)), model)

    test_check_accuracy_loader = get_loader(dir=VAL_IMG_DIR, seg_dir=VAL_MASK_DIR, tra_dir=VAL_TRA_DIR, train_aug=False,
                                            shuffle=False, batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)
    check_accuracy(test_check_accuracy_loader, model, device=DEVICE, num_image=None, three_d=THREE_D)


# def t_acc_mul_models():
#     model1 = UNET3D(in_channels=1, out_channels=3).to(DEVICE)
#     load_checkpoint(torch.load("checkpoint/my_checkpoint_59.pth.tar", map_location=torch.device(DEVICE)), model1)
#
#     model2 = UNET3D(in_channels=1, out_channels=3).to(DEVICE)
#     load_checkpoint(torch.load("checkpoint/my_checkpoint_49.pth.tar", map_location=torch.device(DEVICE)), model2)
#
#     test_check_accuracy_loader = get_loader(dir=VAL_IMG_DIR, maskdir=VAL_MASK_DIR, train_aug=False, shuffle=True,
#                                             batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
#                                             pin_memory=PIN_MEMORY, three_d=True, device=DEVICE)
#     check_accuracy_multy_models(test_check_accuracy_loader, [model1], device=DEVICE, one_image=False, three_d=True)
#     check_accuracy_multy_models(test_check_accuracy_loader, [model2], device=DEVICE, one_image=False, three_d=True)
#     check_accuracy_multy_models(test_check_accuracy_loader, [model1, model2], device=DEVICE, one_image=False,
#                                 three_d=True)
#
#
# def t_save_instance_by_colors():
#     model = UNET3D(in_channels=1, out_channels=3).to(DEVICE)
#     load_checkpoint(torch.load("checkpoint/my_checkpoint.pth.tar", map_location=torch.device(DEVICE)), model)
#
#     loader = get_loader(dir=VAL_IMG_DIR, maskdir=VAL_MASK_DIR, train_aug=False, shuffle=True,
#                         batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
#                         pin_memory=PIN_MEMORY, three_d=True, device=DEVICE)
#     save_instance_by_colors(loader, model, folder="checkpoint", device=DEVICE, three_d=True)


if __name__ == "__main__":
    main()
    # t_acc()
    # t_acc_mul_models()
    # t_save_instance_by_colors()
