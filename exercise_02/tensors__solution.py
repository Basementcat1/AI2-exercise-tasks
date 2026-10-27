"""
Exercise 02 - Tensors and Tensor Operations (SOLUTION)
=======================================================
"""

import torch


# ===========================================================================
# Task 1: Tensor Creation and Inspection
# ===========================================================================
batch = torch.randn(2, 3, 224, 224)
print("Task 1:")
print(f"  Shape: {batch.shape}")       # torch.Size([2, 3, 224, 224])
print(f"  Dtype: {batch.dtype}")       # torch.float32
print(f"  Device: {batch.device}")     # cpu


# ===========================================================================
# Task 2: Device Transfer
# ===========================================================================
print("\nTask 2:")
print(f"  Before: {batch.device}")     # cpu
if torch.cuda.is_available():
    batch_gpu = batch.to('cuda')
    print(f"  After: {batch_gpu.device}")  # cuda:0
else:
    print("  CUDA not available, skipping GPU transfer.")


# ===========================================================================
# Task 3: Indexing and Slicing
# ===========================================================================
x = torch.randn(8, 3, 32, 32)
fifth_image = x[4]
green_channel = x[4, 1]

print("\nTask 3:")
print(f"  5th image shape: {fifth_image.shape}")       # torch.Size([3, 32, 32])
print(f"  Green channel shape: {green_channel.shape}")  # torch.Size([32, 32])


# ===========================================================================
# Task 4: Permute — HWC to CHW
# ===========================================================================
img_hwc = torch.randn(64, 64, 3)
img_chw = img_hwc.permute(2, 0, 1)

print("\nTask 4:")
print(f"  HWC shape: {img_hwc.shape}")  # torch.Size([64, 64, 3])
print(f"  CHW shape: {img_chw.shape}")  # torch.Size([3, 64, 64])


# ===========================================================================
# Task 5: Squeeze and Unsqueeze
# ===========================================================================
img = torch.randn(3, 128, 128)
img_batched = img.unsqueeze(0)
img_back = img_batched.squeeze(0)

print("\nTask 5:")
print(f"  Original: {img.shape}")           # torch.Size([3, 128, 128])
print(f"  Unsqueezed: {img_batched.shape}")  # torch.Size([1, 3, 128, 128])
print(f"  Squeezed: {img_back.shape}")       # torch.Size([3, 128, 128])


# ===========================================================================
# Task 6: Reshaping — Flatten
# ===========================================================================
features = torch.randn(4, 16, 8, 8)
flattened = features.view(4, -1)

print("\nTask 6:")
print(f"  Before: {features.shape}")    # torch.Size([4, 16, 8, 8])
print(f"  After: {flattened.shape}")    # torch.Size([4, 1024])


# ===========================================================================
# Task 7: Basic Math
# ===========================================================================
a = torch.randn(3, 3)
b = torch.randn(3, 3)

elementwise = a * b
matmul = a @ b

print("\nTask 7a:")
print(f"  Element-wise product:\n{elementwise}")
print(f"  Matrix product:\n{matmul}")

batch_images = torch.randn(4, 3, 32, 32)
mean_across_batch = batch_images.mean(dim=0)

print(f"\nTask 7b:")
print(f"  Batch shape: {batch_images.shape}")             # torch.Size([4, 3, 32, 32])
print(f"  Mean across batch: {mean_across_batch.shape}")   # torch.Size([3, 32, 32])