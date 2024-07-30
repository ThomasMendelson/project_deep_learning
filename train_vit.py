import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from tqdm import tqdm

from model import UNET
from dataset import Dataset
from dataset3D import Dataset3D
from transformer.models.vit import ViT_UNet
from monai.losses import DiceCELoss
import torch.nn.functional as F





from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loader,
    check_accuracy,
    check_accuracy_multy_models,
    save_predictions_as_imgs,
    save_instance_by_colors,
)

# Hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
L1_LAMBDA = 1e-5
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 4
CROP_SIZE = (32, 128, 128)
CLASS_WEIGHTS = [0.15, 0.6, 0.25] #  [0.1, 0.6, 0.3]  # [0.1, 0.7, 0.2]
PATCH_SIZE = 16
HIDDEN_SIZE = 512
MLP_DIM = 2048
NUM_LAYERS = 4
NUM_HEADS = 8
PROJ_TYPE = "conv"
DROPOUT_RATE = 0.

THREE_D = True
PIN_MEMORY = False
LOAD_MODEL = False
WANDB_TRACKING = True

# 2D
# TRAIN_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02"
# TRAIN_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/02_GT/SEG"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02_ERR_SEG"
# VAL_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01"
# VAL_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N2DH-SIM+/01_GT/SEG"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01_ERR_SEG"

# 3D
TRAIN_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02"
TRAIN_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/02_GT/SEG"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/02_ERR_SEG"
VAL_IMG_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01"
VAL_MASK_DIR = "/mnt/tmp/data/users/thomasm/Fluo-N3DH-SIM+/01_GT/SEG"  # "Fluo-N2DH-SIM+_training-datasets/Fluo-N2DH-SIM+/01_ERR_SEG"


def calculate_l1_loss(model):
    l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
    return L1_LAMBDA * l1_loss


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = Dataset3D.split_mask(targets.to(device=DEVICE)).long()

        if isinstance(loss_fn, DiceCELoss):
            targets_one_hot = F.one_hot(targets, num_classes=3).permute(0, 4, 1, 2, 3).float()

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets_one_hot)
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


def evaluate_fn(loader, model, loss_fn):
    loop = tqdm(loader)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(device=DEVICE)
            targets = Dataset3D.split_mask(targets.to(device=DEVICE)).long()

            if isinstance(loss_fn, DiceCELoss):
                targets_one_hot = F.one_hot(targets, num_classes=3).permute(0, 4, 1, 2, 3).float()

            predictions = model(data)
            loss = loss_fn(predictions, targets_one_hot)
            # Add L1 regularization
            loss += calculate_l1_loss(model)

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    if WANDB_TRACKING:
        wandb.login(key="12b9b358323faf2af56dc288334e6247c1e8bc63")
        wandb.init(project="vit_unet",
                   config={
                       "epochs": NUM_EPOCHS,
                       "batch_size": BATCH_SIZE,
                       "lr": LEARNING_RATE,
                       "L1_lambda": L1_LAMBDA,
                       "L2_lambda": WEIGHT_DECAY,
                       "CE_weight": CLASS_WEIGHTS,
                       "img_size": CROP_SIZE,
                       "patch_size": PATCH_SIZE,
                       "hidden_size": HIDDEN_SIZE,
                       "mlp_dim": MLP_DIM,
                       "num_layers": NUM_LAYERS,
                       "num_heads": NUM_HEADS,
                       "proj_type": PROJ_TYPE,
                       "dropout_rate": DROPOUT_RATE,
                   })
    model = ViT_UNet(in_channels=1, out_channels=3, img_size=CROP_SIZE,
                     patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, mlp_dim=MLP_DIM, num_layers=NUM_LAYERS,
                     num_heads=NUM_HEADS, proj_type=PROJ_TYPE, dropout_rate=DROPOUT_RATE,
                     classification=False, three_d=THREE_D, device=DEVICE).to(DEVICE)


    class_weights = torch.FloatTensor(CLASS_WEIGHTS).to(DEVICE)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = DiceCELoss(to_onehot_y=False, softmax=True, squared_pred=True, weight=class_weights)  # in reference was , batch=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)

    train_loader = get_loader(dir=TRAIN_IMG_DIR, maskdir=TRAIN_MASK_DIR, train_aug=True, shuffle=True,
                              batch_size=BATCH_SIZE, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)
    val_loader = get_loader(dir=VAL_IMG_DIR, maskdir=VAL_MASK_DIR, train_aug=False, shuffle=False,
                            batch_size=BATCH_SIZE, crop_size=CROP_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            three_d=THREE_D, device=DEVICE)

    test_check_accuracy_loader = get_loader(dir=VAL_IMG_DIR, maskdir=VAL_MASK_DIR, train_aug=False, shuffle=False,
                                            batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint/vit_checkpoint.pth.tar", map_location=torch.device(DEVICE)), model)
        check_accuracy(val_loader, model, device=DEVICE, three_d=THREE_D)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"epoch: [{epoch + 1}/{NUM_EPOCHS}]")
        train_loss = train_fn(train_loader, model, optimizer, criterion, scaler)
        val_loss = evaluate_fn(val_loader, model, criterion)

        scheduler.step(val_loss)

        if WANDB_TRACKING:
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Learning Rate": optimizer.param_groups[0]['lr']})

        if (epoch + 1) % 10 == 0:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="checkpoint/vit_checkpoint.pth.tar")

            # print some examples to a folder
            save_predictions_as_imgs(loader=val_loader, model=model, folder="checkpoint/saved_images", device=DEVICE,
                                     three_d=THREE_D, wandb_tracking=WANDB_TRACKING)

    # check accuracy
    check_accuracy(test_check_accuracy_loader, model, device=DEVICE, three_d=THREE_D)

    # save instance image
    save_instance_by_colors(test_check_accuracy_loader, model, folder="checkpoint", three_d=THREE_D, device=DEVICE, wandb_tracking=WANDB_TRACKING)

    if WANDB_TRACKING:
        # torch.onnx.export(model, torch.randn(1, 1, CROP_SIZE, CROP_SIZE, device=DEVICE), "model.onnx")
        # wandb.save("model.onnx")
        # print("=> saved model.onnx to wandb")
        wandb.finish()


def t_acc():
    model = ViT_UNet(in_channels=1, out_channels=3, img_size=(32, 128, 128), patch_size=16, hidden_size=512,
                     mlp_dim=2048, num_layers=12, num_heads=8, proj_type="conv", dropout_rate=0.,
                     classification=False, three_d=THREE_D, device=DEVICE).to(DEVICE)
    load_checkpoint(torch.load("checkpoint/vit_checkpoint.pth.tar", map_location=torch.device(DEVICE)), model)

    test_check_accuracy_loader = get_loader(dir=VAL_IMG_DIR, maskdir=VAL_MASK_DIR, train_aug=False, shuffle=True,
                                            batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)
    check_accuracy(test_check_accuracy_loader, model, device=DEVICE, one_image=False, three_d=THREE_D)


# def t_acc_mul_models():
#     model = UNET(in_channels=1, out_channels=3).to(DEVICE)
#     load_checkpoint(torch.load("checkpoint/my_checkpoint_0.838.pth.tar", map_location=torch.device(DEVICE)), model)
#
#     model1 = UNET(in_channels=1, out_channels=3).to(DEVICE)
#     load_checkpoint(torch.load("checkpoint/my_checkpoint_0.82.pth.tar", map_location=torch.device(DEVICE)), model1)
#
#     model2 = UNET(in_channels=1, out_channels=3).to(DEVICE)
#     load_checkpoint(torch.load("checkpoint/my_checkpoint1.pth.tar", map_location=torch.device(DEVICE)), model2)
#
#     test_check_accuracy_loader = get_loader(dir=VAL_IMG_DIR, maskdir=VAL_MASK_DIR, train_aug=False, shuffle=True,
#                                             batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
#                                             pin_memory=PIN_MEMORY)
#     check_accuracy_multy_models(test_check_accuracy_loader, [model], device=DEVICE, one_image=False)
#     check_accuracy_multy_models(test_check_accuracy_loader, [model1], device=DEVICE, one_image=False)
#     check_accuracy_multy_models(test_check_accuracy_loader, [model2], device=DEVICE, one_image=False)
#     check_accuracy_multy_models(test_check_accuracy_loader, [model, model1], device=DEVICE, one_image=False)
#     check_accuracy_multy_models(test_check_accuracy_loader, [model, model1, model2], device=DEVICE, one_image=False)
#
#
#
def t_save_instance_by_colors():
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

    model = ViT_UNet(in_channels=1, out_channels=3, img_size=CROP_SIZE,
                     patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, mlp_dim=MLP_DIM, num_layers=NUM_LAYERS,
                     num_heads=NUM_HEADS, proj_type=PROJ_TYPE, dropout_rate=DROPOUT_RATE,
                     classification=False, three_d=THREE_D, device=DEVICE).to(DEVICE)

    load_checkpoint(torch.load("checkpoint/vit_checkpoint2.pth.tar", map_location=torch.device(DEVICE)), model)

    val_loader = get_loader(dir=VAL_IMG_DIR, maskdir=VAL_MASK_DIR, train_aug=False, shuffle=False,
                            batch_size=BATCH_SIZE, crop_size=CROP_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            three_d=THREE_D, device=DEVICE)

    test_check_accuracy_loader = get_loader(dir=VAL_IMG_DIR, maskdir=VAL_MASK_DIR, train_aug=False, shuffle=False,
                                            batch_size=1, crop_size=CROP_SIZE, num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY, three_d=THREE_D, device=DEVICE)

    save_predictions_as_imgs(loader=val_loader, model=model, folder="checkpoint/saved_images2", device=DEVICE,
                             three_d=THREE_D, wandb_tracking=WANDB_TRACKING)


    # save_instance_by_colors(test_check_accuracy_loader, model, folder="checkpoint", three_d=THREE_D, device=DEVICE, wandb_tracking=WANDB_TRACKING)


if __name__ == "__main__":
    main()
    # t_acc()
    # t_acc_mul_models()
    # t_save_instance_by_colors()
