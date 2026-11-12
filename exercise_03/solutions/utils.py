"""
Exercise 03 - Utility Functions (SOLUTION)
============================================
"""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def save_checkpoint(model, optimizer, epoch, val_accuracy, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ===========================================================================
# Verification
# ===========================================================================
if __name__ == "__main__":
    import tempfile
    import torch.nn as nn

    print("=== VERIFICATION ===")

    def _assert_true(condition, message):
        assert condition, message


    def check(label, assertion_fn):
        try:
            assertion_fn()
        except AssertionError as e:
            print(f"[FAIL] {label} - {e}")
            return False
        except NameError as e:
            print(f"[FAIL] {label} (variable not defined: {e})")
            return False
        except Exception as e:
            print(f"[FAIL] {label} (error while checking: {e})")
            return False
        print(f"[PASS] {label}")
        return True


    def _check_accuracy(outputs, targets, expected):
        acc = compute_accuracy(outputs, targets)
        _assert_true(acc is not None, "compute_accuracy returned None - did you return the accuracy?")
        _assert_true(isinstance(acc, float),
                     f"expected a Python float, got {type(acc).__name__} - use .item() to convert the count")
        _assert_true(abs(acc - expected) < 1e-6, f"expected {expected:.4f}, got {acc:.4f}")


    def _check_set_seed():
        set_seed(42)
        a = torch.randn(3)
        set_seed(42)
        b = torch.randn(3)
        _assert_true(torch.equal(a, b), "the same seed produced different random numbers")


    def _check_checkpoint_roundtrip():
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        saved_weight = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "checkpoints", "test_ckpt.pt")
            save_checkpoint(model, optimizer, epoch=5, val_accuracy=0.85, filepath=path)
            with torch.no_grad():
                model.weight.zero_()
            ckpt = load_checkpoint(path, model, optimizer)
        _assert_true(ckpt["epoch"] == 5 and ckpt["val_accuracy"] == 0.85,
                     f"expected epoch=5, val_accuracy=0.85, got epoch={ckpt['epoch']}, "
                     f"val_accuracy={ckpt['val_accuracy']}")
        _assert_true(torch.equal(model.weight, saved_weight), "loading the checkpoint did not restore the weights")


    # 4 samples x 3 classes: predictions per row are 0, 1, 2, 0
    outputs = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5], [0.9, 0.3, 0.1]])

    results = []

    results.append(check("compute_accuracy - all predictions correct gives 1.0",
                         lambda: _check_accuracy(outputs, torch.tensor([0, 1, 2, 0]), 1.0)))
    results.append(check("compute_accuracy - 3 of 4 correct gives 0.75",
                         lambda: _check_accuracy(outputs, torch.tensor([0, 1, 2, 2]), 0.75)))
    results.append(check("compute_accuracy - no prediction correct gives 0.0",
                         lambda: _check_accuracy(outputs, torch.tensor([1, 0, 0, 1]), 0.0)))
    results.append(check("set_seed - the same seed reproduces the same random numbers", _check_set_seed))
    results.append(check("save_checkpoint / load_checkpoint - round trip restores epoch, accuracy and weights",
                         _check_checkpoint_roundtrip))

    print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
