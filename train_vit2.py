import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from tqdm import tqdm
from dataset import Dataset
from dataset3D import Dataset3D
from models.transformer.vit import ViT_UNet
from monai.losses import DiceCELoss, FocalLoss, DiceFocalLoss
import torch.nn.functional as F

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loader,
    check_accuracy,
    check_accuracy_without_markers,
    save_instance_by_colors,
    save_instance_by_colors_without_markers,
    three_d_to_two_d_represantation,
)

# Hyperparameters
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-3
L1_LAMBDA = 1e-5

BATCH_SIZE = 8
NUM_EPOCHS = 150
NUM_WORKERS = 8
CLASS_WEIGHTS = [0.2, 0.6, 0.2]  # [0.15, 0.6, 0.25] [0.2, 0.6, 0.2] [0.1, 0.6, 0.3] [0.1, 0.7, 0.2]
MARKERS_WEIGHTS = [2]

PATCH_SIZE = 16
HIDDEN_SIZE = 512
MLP_DIM = 2048

NUM_LAYERS = 4
NUM_HEADS = 8
PROJ_TYPE = "conv"
DROPOUT_RATE = 0.1
SPARSE_DIM = 2
QKV_BIAS = True

PIN_MEMORY = False
LOAD_MODEL = False
WANDB_TRACKING = False

CROP_SIZE = (32, 128, 128)  # (32, 256, 256)
# CROP_SIZE = (256, 256)
THREE_D = True
THREE_D_BY_TWO_D = False
WITH_MARKERS = True
RUNAI = False

if THREE_D:
    # 3D
    if RUNAI:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        TRAIN_IMG_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/02"
        TRAIN_SEG_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/02_GT/SEG"
        TRAIN_TRA_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/02_GT/TRA2"
        VAL_IMG_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/01"
        VAL_MASK_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/01_GT/SEG"
        VAL_TRA_DIR = "/gpfs0/tamyr/projects/data/CellTrackingChallenge/Fluo-N3DH-SIM+/01_GT/TRA2"
    else:
        DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
        TRAIN_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02"
        TRAIN_SEG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02_GT/SEG"
        TRAIN_TRA_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02_GT/TRA2"
        VAL_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01"
        VAL_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG"
        VAL_TRA_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/TRA2"
else:
    # 2D
    TRAIN_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02"
    TRAIN_SEG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_GT/SEG"
    TRAIN_TRA_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_GT/TRA2"
    VAL_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01"
    VAL_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01_GT/SEG"
    VAL_TRA_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01_GT/TRA2"

if RUNAI:
    SAVE_PATH = "/gpfs0/tamyr/users/thomasm/ckpts/"
else:
    SAVE_PATH = "/raid/data/users/thomasm/ckpts/"


def calculate_l1_loss(model):
    l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
    return L1_LAMBDA * l1_loss


def train_fn(loader, model, optimizer, loss_functions, scaler):
    loop = tqdm(loader)
    model.train()
    total_loss = 0.0

    for batch_idx, (data, class_targets, marker_targets) in enumerate(loop):
        loss = 0

        if THREE_D:
            class_targets = Dataset3D.split_mask(class_targets.to(device=DEVICE)).long()
            if THREE_D_BY_TWO_D:
                data = three_d_to_two_d_represantation(images=data.to(device=DEVICE))
        else:
            class_targets = Dataset.split_mask(class_targets.to(device=DEVICE)).long()
        data = data.to(device=DEVICE)
        marker_targets[marker_targets > 0] = 1

        if THREE_D_BY_TWO_D:
            marker_targets = marker_targets.view(BATCH_SIZE * CROP_SIZE[0], CROP_SIZE[1], CROP_SIZE[2])
            class_targets = class_targets.view(BATCH_SIZE * CROP_SIZE[0], CROP_SIZE[1], CROP_SIZE[2])
        marker_targets = marker_targets.to(device=DEVICE)
        class_targets = class_targets.to(device=DEVICE)
        # forward
        with (torch.cuda.amp.autocast()):
            class_predictions, marker_predictions = model(data)
            for loss_fn in loss_functions:
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss += loss_fn(class_predictions, class_targets)
                elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
                    loss += loss_fn(marker_predictions.squeeze(1), marker_targets)
                elif isinstance(loss_fn, DiceCELoss) or isinstance(loss_fn, FocalLoss) or isinstance(loss_fn,
                                                                                                     DiceFocalLoss):
                    if THREE_D and (not THREE_D_BY_TWO_D):
                        class_targets_one_hot = F.one_hot(class_targets, num_classes=3).permute(0, 4, 1, 2, 3).float()
                    else:
                        class_targets_one_hot = F.one_hot(class_targets, num_classes=3).permute(0, 3, 1, 2).float()
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
            if THREE_D:
                class_targets = Dataset3D.split_mask(class_targets.to(device=DEVICE)).long()
                if THREE_D_BY_TWO_D:
                    data = three_d_to_two_d_represantation(images=data.to(device=DEVICE))
            else:
                class_targets = Dataset.split_mask(class_targets.to(device=DEVICE)).long()
            data = data.to(device=DEVICE)
            marker_targets[marker_targets > 0] = 1

            if THREE_D_BY_TWO_D:
                marker_targets = marker_targets.view(BATCH_SIZE * CROP_SIZE[0], CROP_SIZE[1], CROP_SIZE[2])
                class_targets = class_targets.view(BATCH_SIZE * CROP_SIZE[0], CROP_SIZE[1], CROP_SIZE[2])
            marker_targets = marker_targets.to(device=DEVICE)
            class_targets = class_targets.to(device=DEVICE)

            class_predictions, marker_predictions = model(data)

            for loss_fn in loss_functions:
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    loss += loss_fn(class_predictions, class_targets)

                elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
                    loss += loss_fn(marker_predictions.squeeze(1), marker_targets)


                elif isinstance(loss_fn, DiceCELoss) or isinstance(loss_fn, FocalLoss) or isinstance(loss_fn,
                                                                                                     DiceFocalLoss):
                    if THREE_D and (not THREE_D_BY_TWO_D):
                        class_targets_one_hot = F.one_hot(class_targets, num_classes=3).permute(0, 4, 1, 2, 3).float()
                    else:
                        class_targets_one_hot = F.one_hot(class_targets, num_classes=3).permute(0, 3, 1, 2).float()
                    loss += loss_fn(class_predictions, class_targets_one_hot)

            # Add L1 regularization
            loss += calculate_l1_loss(model)

            total_loss += loss.item()
            # update tqdm loop
            loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def main():
    if WANDB_TRACKING:
        wandb.login(key="")
        wandb.init(project="vit_unet",
                   config={
                       "THREE_D": THREE_D,
                       "THREE_D_BY_TWO_D": THREE_D_BY_TWO_D,
                       "epochs": NUM_EPOCHS,
                       "batch_size": BATCH_SIZE,
                       "lr": LEARNING_RATE,
                       "L1_lambda": L1_LAMBDA,
                       "L2_lambda": WEIGHT_DECAY,
                       "CE_weight": CLASS_WEIGHTS,
                       "img_size": CROP_SIZE,
                       "with_markers": WITH_MARKERS,
                       "patch_size": PATCH_SIZE,
                       "hidden_size": HIDDEN_SIZE,
                       "mlp_dim": MLP_DIM,
                       "num_layers": NUM_LAYERS,
                       "num_heads": NUM_HEADS,
                       "proj_type": PROJ_TYPE,
                       "dropout_rate": DROPOUT_RATE,
                   })
        wandb_step = 0
    if THREE_D:
        if THREE_D_BY_TWO_D:
            # 3D by 2D slices
            model = ViT_UNet(in_channels=3, out_channels=3, img_size=CROP_SIZE[-2:], qkv_bias=QKV_BIAS,
                             patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, mlp_dim=MLP_DIM, num_layers=NUM_LAYERS,
                             num_heads=NUM_HEADS, proj_type=PROJ_TYPE, dropout_rate=DROPOUT_RATE,
                             spatial_dims=SPARSE_DIM, classification=False, with_markers=WITH_MARKERS,
                             three_d=(THREE_D and (not THREE_D_BY_TWO_D)), device=DEVICE).to(DEVICE)
        else:
            # 3D
            SPARSE_DIM = 3
            model = ViT_UNet(in_channels=1, out_channels=3, img_size=CROP_SIZE, qkv_bias=QKV_BIAS,
                             patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, mlp_dim=MLP_DIM, num_layers=NUM_LAYERS,
                             num_heads=NUM_HEADS, proj_type=PROJ_TYPE, dropout_rate=DROPOUT_RATE,
                             spatial_dims=SPARSE_DIM, with_markers=WITH_MARKERS,
                             classification=False, three_d=THREE_D, device=DEVICE).to(DEVICE)

    else:
        # 2D
        model = ViT_UNet(in_channels=1, out_channels=3, img_size=CROP_SIZE, qkv_bias=QKV_BIAS,
                         patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, mlp_dim=MLP_DIM, num_layers=NUM_LAYERS,
                         num_heads=NUM_HEADS, proj_type=PROJ_TYPE, dropout_rate=DROPOUT_RATE, spatial_dims=SPARSE_DIM,
                         classification=False, three_d=THREE_D, with_markers=WITH_MARKERS, device=DEVICE).to(DEVICE)

    class_weights = torch.FloatTensor(CLASS_WEIGHTS).to(DEVICE)
    markers_weights = torch.FloatTensor(MARKERS_WEIGHTS).to(DEVICE)
    if WITH_MARKERS:
        # CrossEntropyLoss
        # criterion = [nn.CrossEntropyLoss(weight=class_weights), nn.BCEWithLogitsLoss(pos_weight=markers_weights)]
        # DiceCELoss
        criterion = [DiceCELoss(softmax=True, squared_pred=True, weight=class_weights, batch=True),
                     # in reference was , batch=True)
                     nn.BCEWithLogitsLoss(pos_weight=markers_weights)]
        # FocalLoss
        # criterion = [FocalLoss(softmax=True, squared_pred=True, gamma=2, alpha=class_weights),
        #              nn.BCEWithLogitsLoss(pos_weight=markers_weights)]
        # DiceFocalLoss
        # criterion = [DiceFocalLoss(softmax=True, squared_pred=True, weight=class_weights, gamma=2),
        #              nn.BCEWithLogitsLoss(pos_weight=markers_weights)]
    else:
        # CrossEntropyLoss
        criterion = [nn.CrossEntropyLoss(weight=class_weights)]
        # DiceCELoss
        # criterion = [DiceCELoss(softmax=True, squared_pred=True, weight=class_weights, batch=True)]
        # FocalLoss
        # criterion = [FocalLoss(softmax=True, squared_pred=True, gamma=2, alpha=class_weights)]
        # DiceFocalLoss
        # criterion = [DiceFocalLoss(softmax=True, squared_pred=True, weight=class_weights, gamma=2)]

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
        load_checkpoint(torch.load(f"{SAVE_PATH}vit_checkpoint.pth.tar", map_location=torch.device(DEVICE)), model)
        model.to(DEVICE)
        check_accuracy(val_loader, model, device=DEVICE, three_d=THREE_D, three_d_by_two_d=THREE_D_BY_TWO_D)

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
            save_checkpoint(checkpoint, filename=f"{SAVE_PATH}vit_checkpoint.pth.tar")


            if (epoch + 1) != NUM_EPOCHS:
                if WITH_MARKERS:
                    check_accuracy(test_check_accuracy_loader, model, num_image=50, device=DEVICE, three_d=THREE_D,
                                   three_d_by_two_d=THREE_D_BY_TWO_D, save_path=SAVE_PATH, name=f"vit_{epoch}")
                    save_instance_by_colors(loader=val_loader, model=model, folder=f"{SAVE_PATH}saved_images",
                                            device=DEVICE,
                                            three_d=THREE_D, wandb_tracking=WANDB_TRACKING, wandb_step=wandb_step,
                                            three_d_by_two_d=THREE_D_BY_TWO_D)
                else:
                    check_accuracy_without_markers(test_check_accuracy_loader, model, num_image=50, device=DEVICE,
                                                   three_d=THREE_D, three_d_by_two_d=THREE_D_BY_TWO_D,
                                                   save_path=SAVE_PATH, name=f"vit_{epoch}")
                    save_instance_by_colors_without_markers(loader=val_loader, model=model, three_d=THREE_D,
                                                            folder=f"{SAVE_PATH}saved_images",device=DEVICE,
                                                            wandb_tracking=WANDB_TRACKING, wandb_step=wandb_step,
                                                            three_d_by_two_d=THREE_D_BY_TWO_D)
        if WANDB_TRACKING:
            wandb_step += 1

    if WITH_MARKERS:
        check_accuracy(test_check_accuracy_loader, model, device=DEVICE, three_d=THREE_D, three_d_by_two_d=THREE_D_BY_TWO_D)
        save_instance_by_colors(loader=test_check_accuracy_loader, model=model, folder=f"{SAVE_PATH}", three_d=THREE_D,
                                device=DEVICE, wandb_tracking=WANDB_TRACKING, wandb_step=wandb_step,
                                three_d_by_two_d=THREE_D_BY_TWO_D)
    else:
        check_accuracy_without_markers(test_check_accuracy_loader, model, device=DEVICE,
                                       three_d=THREE_D, three_d_by_two_d=THREE_D_BY_TWO_D)
        save_instance_by_colors_without_markers(loader=test_check_accuracy_loader, model=model, folder=f"{SAVE_PATH}",
                                                three_d=THREE_D,device=DEVICE, wandb_tracking=WANDB_TRACKING,
                                                wandb_step=wandb_step, three_d_by_two_d=THREE_D_BY_TWO_D)

    if WANDB_TRACKING:
        wandb.finish()


def t_acc():
    SPARSE_DIM = 3
    model = ViT_UNet(in_channels=1, out_channels=3, img_size=CROP_SIZE, qkv_bias=QKV_BIAS,
                     patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, mlp_dim=MLP_DIM, num_layers=NUM_LAYERS,
                     num_heads=NUM_HEADS, proj_type=PROJ_TYPE, dropout_rate=DROPOUT_RATE, spatial_dims=SPARSE_DIM,
                     classification=False, three_d=THREE_D, device=DEVICE, with_markers=WITH_MARKERS).to(DEVICE)
    load_checkpoint(torch.load(f"{SAVE_PATH}/vit_79_0.7116.pth.tar", map_location=torch.device(DEVICE)), model)
    test_check_accuracy_loader = get_loader(dir=VAL_IMG_DIR, seg_dir=VAL_MASK_DIR, tra_dir=VAL_TRA_DIR, train_aug=False,
                                            shuffle=False, batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)

    train_accuracy_loader = get_loader(dir=TRAIN_IMG_DIR, seg_dir=TRAIN_SEG_DIR, tra_dir=TRAIN_TRA_DIR, train_aug=False,
                                       shuffle=False, batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                                       pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)

    # check_accuracy_multy_models(test_check_accuracy_loader, [model], device=DEVICE, three_d=THREE_D)
    # save_images_to_check_accuracy(test_check_accuracy_loader, model, save_path=f"{SAVE_PATH}/Fluo-N3DH-SIM+/",
    #                               device=DEVICE, three_d=THREE_D, three_d_by_two_d=THREE_D_BY_TWO_D)
    if WITH_MARKERS:
        print("check accuracy train with markers")
        check_accuracy(train_accuracy_loader, model, device=DEVICE, num_image=None, three_d=THREE_D)
        print("check accuracy test with markers")
        check_accuracy(test_check_accuracy_loader, model, device=DEVICE, num_image=None, three_d=THREE_D)
    print("check accuracy train dir without markers")
    check_accuracy_without_markers(train_accuracy_loader, model, device=DEVICE, num_image=None, three_d=THREE_D)
    print("check accuracy test dir without markers")
    check_accuracy_without_markers(test_check_accuracy_loader, model, device=DEVICE, num_image=None, three_d=THREE_D)

if __name__ == "__main__":
    # main()
    t_acc()
