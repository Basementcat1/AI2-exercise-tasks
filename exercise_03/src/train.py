"""
Exercise 03 - Training Script
================================
Main training script that ties together dataset, model, loss, optimizer,
TensorBoard logging, and checkpointing.

Usage:
    python src/train.py --config config.yaml

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
    """Load YAML config file and return as dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, dataloader, criterion, optimizer, device, log_interval=10):
    """
    Train the model for one epoch.

    Returns:
        tuple: (average_loss, average_accuracy) for the epoch.
    """
    # Set the model to training mode (model.train()).
    #       This enables dropout and batch normalization updates.
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
    # TODO: Complete the training step by filling in the individual TODOs.
    #
    #   Step 1: Move images and labels to the device using .to(device).
        images = images.to(device)
        labels = labels.to(device)
    
    #   Step 2: Clear old gradients from the optimizer (hint: .zero_grad()).
        # TODO: your code here - clear gradients
    
    #   Step 3: Forward pass — get model outputs from the images.
        outputs = model(images)

    #   Step 4: Compute the loss using the criterion (hint: criterion(outputs, labels)).
    #           Pass (outputs, labels) — not (labels, outputs).
        loss = criterion(outputs, labels)

    #   Step 5: Backward pass — compute gradients from the loss (hint: .backward()).
        # TODO: your code here - backward pass
        loss.backward()

    #   Step 6: Update weights with the optimizer (hint: .step()).
        # TODO: your code here - optimizer step
        optimizer.step()

    
        # --- Accumulate metrics (do not modify) ---
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
    """
    Evaluate the model on the validation set.

    Returns:
        tuple: (average_loss, average_accuracy) for the validation set.
    """
    # Set the model to evaluation mode (.eval).
    #       This disables dropout and freezes batch normalization.
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    # Disable gradient computation for the entire validation loop.
    #       Wrap the loop in a torch.no_grad() context manager.
    #       Inside, for each batch:

    with torch.no_grad():
        for images, labels in dataloader:
            # TODO: Complete the validation step by filling in the individual TODOs.
            
            # Step 1: Move images and labels to device.
            # TODO: your code here
            images = images.to(device)
            labels = labels.to(device)

            #   Step 2: Forward pass only — no backward, no optimizer step.
            # TODO: get model outputs from the images
            #outputs = None # TODO: replace "None" and add your solution 
            outputs = model(images)

            #   Step 3: Compute loss (for logging, not for training).
            #loss = None # TODO: replace "None" and add your solution
            loss = criterion(outputs, labels)
        
        
            # --- Accumulate metrics (do not modify) ---
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def main(config_path):
    """Main training function."""

    # --- Load config ---
    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

     # --- Setup ---
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the TensorBoard SummaryWriter. (see https://docs.pytorch.org/docs/2.14/tensorboard.html)
    #   The log directory should include the experiment name so each
    #   experiment gets its own subfolder. Combine config["logging"]["log_dir"]
    #   and config["experiment_name"] using os.path.join().
    log_dir = os.path.join(config["logging"]["log_dir"], config["experiment_name"])
    writer = SummaryWriter(log_dir=log_dir)

    image_size = config["data"]["image_size"]
    augmentation = config["data"]["augmentation"]

    # Create training and validation datasets.
    #   Use PlantWildDataset with the appropriate root_dir
    #   (os.path.join with "train" or "val") and transforms.
    # Check the config file for specific keywords related to root directory and datasets
    train_dataset = PlantWildDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "train"),
        transform=get_train_transforms(image_size, augmentation),
    )

    # TODO: get the validation dataset, similar to train dataset, use the correct transforms!
    #val_dataset = None # TODO: remove "None" and add your solution here
    val_dataset = PlantWildDataset(
        root_dir=os.path.join(config["data"]["root_dir"], "val"),
        transform=get_val_transforms(image_size),
    )

    # Create DataLoaders for both datasets.
    #   Use batch_size and num_workers from config["data"].
    #   Training: shuffle=True. Validation: shuffle=False.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    # TODO: create the validation dataloader, similar to train loader, use the val_dataset instead of train_dataset
    #       everything else stays the same
    #val_loader = None
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )


    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Instantiate the model.
    #   Use PlantCNN with num_classes and dropout from config.
    #   Move to device.
    model = PlantCNN(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)
    

    print(f"Model parameters: {count_parameters(model):,}")

    # Define the loss function — nn.CrossEntropyLoss().
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer based on config["training"]["optimizer"].
    #   If "adam", use torch.optim.Adam.
    #   If "sgd", use torch.optim.SGD with momentum=0.9.
    #   Pass learning_rate and weight_decay from config.
    if config["training"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        # TODO: Instantiate a torch.optim.SGD() optimizer with the same config values
        #       SGD additionally has a parameter "momentum", set it to momentum=0.9
        #pass # TODO: remove "pass" and fill in your solution here
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            momentum=0.9
        )

    # Log sample images to TensorBoard for visual verification.
    #   Fetch one batch from train_loader using next(iter(...)).
    #   Log images using writer.add_images("sample_training_images", images, 0).
    sample_images, _ = next(iter(train_loader))
    writer.add_images("sample_training_images", sample_images, global_step=0)


    # --- Training Loop ---
    best_val_acc = 0.0
    num_epochs = config["training"]["epochs"]

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        # Train the model for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=config["logging"]["log_interval"],
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # --- Logging ---
        # Log four scalar metrics to TensorBoard using writer.add_scalar():
        #   "Loss/train", "Loss/val", "Accuracy/train", "Accuracy/val"
        #   Use epoch as the global_step.
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # --- Checkpointing ---
        # Save checkpoint if validation accuracy improved.
        #   Compare val_acc with best_val_acc. If better, update best_val_acc
        #   and call save_checkpoint(). Save to the path built from
        #   config["checkpoint"]["save_dir"] and experiment_name + "_best.pt".
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(
                config["checkpoint"]["save_dir"],
                config["experiment_name"] + "_best.pt",
            )
            save_checkpoint(model, optimizer, epoch, val_acc, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f}) → {ckpt_path}")


    # Close the TensorBoard writer.
    writer.close()

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlantCNN")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    main(args.config)
