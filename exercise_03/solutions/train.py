"""
Exercise 03 - Training Script (SOLUTION)
==========================================
Project structure inspired by: https://github.com/victoresque/pytorch-template
"""

import os
import sys
import argparse

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import PlantWildDataset
from transforms import get_train_transforms, get_val_transforms
from model import PlantCNN, count_parameters
from utils import set_seed, compute_accuracy, save_checkpoint


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, dataloader, criterion, optimizer, device, log_interval=10):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if (batch_idx + 1) % log_interval == 0:
            print(f"    Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def main(config_path):
    # --- Load config ---
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # --- Setup ---
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = os.path.join(config["logging"]["log_dir"], config["experiment_name"])
    writer = SummaryWriter(log_dir=log_dir)

    # --- Data ---
    image_size = config["data"]["image_size"]
    augmentation = config["data"]["augmentation"]

    train_dataset = PlantWildDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "train"),
        transform=get_train_transforms(image_size, augmentation),
    )
    val_dataset = PlantWildDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "val"),
        transform=get_val_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # --- Model ---
    model = PlantCNN(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()

    if config["training"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            momentum=0.9,
            weight_decay=config["training"]["weight_decay"],
        )

    # --- Log sample images ---
    sample_images, _ = next(iter(train_loader))
    writer.add_images("sample_training_images", sample_images, global_step=0)

    # --- Training Loop ---
    best_val_acc = 0.0
    num_epochs = config["training"]["epochs"]

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=config["logging"]["log_interval"],
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(
                config["checkpoint"]["save_dir"],
                config["experiment_name"] + "_best.pt",
            )
            save_checkpoint(model, optimizer, epoch, val_acc, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f}) → {ckpt_path}")

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlantCNN")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    main(args.config)
