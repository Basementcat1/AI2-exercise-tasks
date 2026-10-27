"""
Exercise 02 - Tensors and Tensor Operations
============================================
Solve each task below. Run this script to verify your solutions:
    python tensors.py
"""

import torch


# ===========================================================================
# Task 1: Tensor Creation and Inspection
# ===========================================================================
# Create a random tensor of shape (2, 3, 224, 224) — simulating a batch
# of 2 RGB images of size 224x224.
# Print its shape, dtype, and device.
#
# Hint: torch.randn() creates a tensor with random values from a
# normal distribution. Pass the dimensions as arguments.
batch = torch.randn(2, 3, 224, 224)
print("Task 1:")
print(batch.shape)
print(batch.dtype)
print(batch.device)

# ===========================================================================
# Task 2: Device Transfer
# ===========================================================================
# Move the tensor from Task 1 to GPU (if available).
# Print the device before and after the transfer.
#
# Step 1: Check if CUDA is available using torch.cuda.is_available().
# Step 2: Move the tensor using .to('cuda').
# Step 3: Print the .device attribute to confirm.
print("\nTask 2:")
print(f"Is cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    batch_gpu = batch.to("cuda")
    print(batch_gpu.device)
else:
    print("Sorry, no cuda...")

# ===========================================================================
# Task 3: Indexing and Slicing (Homework)
# ===========================================================================
# Create a random tensor of shape (8, 3, 32, 32) — a batch of 8 images.
#
# Step 1: Extract the 5th image from the batch. Remember that Python
#         indexing starts at 0, so the 5th image is at index 4.
#         Print its shape — should be (3, 32, 32).
#
# Step 2: Extract only the green channel (index 1) of that image.
#         Print its shape — should be (32, 32).
print("\nTask 3:")
batch2 = torch.randn(8, 3, 32, 32)
fifth_image = batch2[4]
print(f"5th image shape: {fifth_image.shape}")
green_channel = batch2[4, 1]
print(f"green channel shape: {green_channel.shape}")

# ===========================================================================
# Task 4: Permute — HWC to CHW (Homework)
# ===========================================================================
# Create a random tensor of shape (64, 64, 3) representing an image
# in HWC format (Height, Width, Channels).
#
# Step 1: Convert to PyTorch's CHW format using .permute().
#         You need to reorder dimensions: dim 2 → 0, dim 0 → 1, dim 1 → 2.
# Step 2: Print both shapes to confirm (64, 64, 3) → (3, 64, 64).
print("\nTask 4:")
image_hwc = torch.randn(64, 64, 3)
image_chw = image_hwc.permute(2, 0, 1)
print(f"image_hwc shape: {image_hwc.shape}")
print(f"image_chw shape: {image_chw.shape}")

# ===========================================================================
# Task 5: Squeeze and Unsqueeze (Homework)
# ===========================================================================
# a) Create a tensor of shape (3, 128, 128) representing a single image.
#    Add a batch dimension at position 0 using .unsqueeze().
#    Print the shape — should be (1, 3, 128, 128).
#
# b) Remove the batch dimension using .squeeze() and verify you get
#    back to (3, 128, 128).
#
# This pattern is common: models expect batched input [B, C, H, W],
# so a single image needs unsqueeze before passing through.
print("\nTask 5:")
image2 = torch.randn(3, 128, 128)
print(f"image2 shape: {image2.shape}")
image2_batched = image2.unsqueeze(0)
print(f"image2_batched shape: {image2_batched.shape}")
image2_back = image2_batched.squeeze(0)
print(f"image2_back shape: {image2_back.shape}")

# ===========================================================================
# Task 6: Reshaping — Flatten (Homework)
# ===========================================================================
# Create a random tensor of shape (4, 16, 8, 8) — simulating a batch
# of 4 feature maps with 16 channels and 8x8 spatial dimensions.
#
# Flatten each sample so the result has shape (4, 1024).
# Note: 16 * 8 * 8 = 1024.
#
# Use .view() or .reshape(). The batch dimension (4) stays, everything
# else gets flattened into one dimension. You can pass -1 to let
# PyTorch infer the flattened size automatically.
print("\nTask 6:")
batch3 = torch.randn(4, 16, 8, 8)
batch3_flattened = batch3.view(4, 1024)
print(f"batch3 shape: {batch3.shape}")
print(f"batch3_flattened shape: {batch3_flattened.shape}")

# ===========================================================================
# Task 7: Basic Math (Homework)
# ===========================================================================
# a) Create two random tensors of shape (3, 3).
#    Compute their element-wise product (use * operator) and their
#    matrix product (use @ operator). Print both results.
#
# b) Create a random tensor of shape (4, 3, 32, 32).
#    Compute the mean pixel value across the batch dimension (dim=0).
#    Print the resulting shape — should be (3, 32, 32).
#    This is how dataset mean is computed for normalization.
print("\nTask 7a:")
tensor1 = torch.randn(3, 3)
tensor2 = torch.randn(3, 3)
element_wise_product = tensor1 * tensor2
matrix_product = tensor1 @ tensor2
print(f"element_wise_product: {element_wise_product}")
print(f"matrix_product: {matrix_product}")

print("\nTask 7b")
batch4 = torch.randn(4, 3, 32, 32)
mean_batch4 = batch4.mean(dim = 0)
print(f"batch4 shape {batch4.shape}")
print(f"mean_batch4 shape: {mean_batch4.shape}")