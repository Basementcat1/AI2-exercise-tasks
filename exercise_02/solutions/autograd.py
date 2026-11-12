"""
Exercise 02 - Autograd (SOLUTION)
==================================
"""

import torch


# ===========================================================================
# Task 1: Basic Autograd
# ===========================================================================
x = torch.tensor(4.0, requires_grad=True)
y = 3 * x**3 - 2 * x**2 + x
y.backward()

print("Task 1:")
print(f"  x = {x.item()}")
print(f"  y = {y.item()}")
print(f"  x.grad (dy/dx) = {x.grad.item()}")


# ===========================================================================
# Task 2: Manual Gradient Verification
# ===========================================================================
# Given: y = 3x³ - 2x² + x
#
# Step 1: dy/dx = 9x² - 4x + 1
#
# Step 2: dy/dx at x = 4.0 = 9(16) - 4(4) + 1 = 144 - 16 + 1 = 129.0
#
# Step 3: x.grad from Task 1 = 129.0 → They match.

print("\nTask 2:")
manual_grad = 9 * (4.0**2) - 4 * 4.0 + 1
print(f"  Manual gradient: {manual_grad}")
print(f"  Autograd result: {x.grad.item()}")
print(f"  Match: {manual_grad == x.grad.item()}")


# ===========================================================================
# Task 3: Second Function — Full Verification
# ===========================================================================
# f(x) = 5x² - 4x + 7
# df/dx = 10x - 4
# df/dx at x = 2.0 = 10(2) - 4 = 16.0

x2 = torch.tensor(2.0, requires_grad=True)
f = 5 * x2**2 - 4 * x2 + 7
f.backward()

manual_grad_2 = 10 * 2.0 - 4

print("\nTask 3:")
print(f"  x = {x2.item()}")
print(f"  f(x) = {f.item()}")
print(f"  Manual df/dx = {manual_grad_2}")
print(f"  Autograd df/dx = {x2.grad.item()}")
print(f"  Match: {manual_grad_2 == x2.grad.item()}")


# ===========================================================================
# Task 4: torch.no_grad()
# ===========================================================================
x3 = torch.tensor(5.0, requires_grad=True)

with torch.no_grad():
    y_no_grad = x3 * 2

z_with_grad = x3 * 2

print("\nTask 4:")
print(f"  Inside no_grad, y.requires_grad = {y_no_grad.requires_grad}")   # False
print(f"  Outside no_grad, z.requires_grad = {z_with_grad.requires_grad}") # True


# ===========================================================================
# Verification
# ===========================================================================
print("\n=== VERIFICATION ===")

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


def _assert_leaf(tensor, name, value):
    _assert_true(isinstance(tensor, torch.Tensor), f"expected {name} to be a tensor, got {type(tensor).__name__}")
    _assert_true(tensor.item() == value,
                 f"expected {name} = {value}, got {tensor.item()} - "
                 f"use a separate tensor per task (x, x2, x3) instead of overwriting one")
    _assert_true(tensor.requires_grad, f"expected {name} to be created with requires_grad=True")


def _assert_grad(tensor, name, expected):
    _assert_true(tensor.grad is not None, f"{name}.grad is None - did you call .backward()?")
    torch.testing.assert_close(tensor.grad, torch.tensor(expected),
                               msg=f"expected {name}.grad = {expected}, got {tensor.grad.item()}")


results = []

results.append(check("Task 1 - x is a tensor with value 4.0 and requires_grad=True",
                     lambda: _assert_leaf(x, "x", 4.0)))
results.append(check("Task 1 - x.grad matches dy/dx = 9x^2 - 4x + 1 at x=4.0 (129.0)",
                     lambda: _assert_grad(x, "x", 129.0)))

results.append(check("Task 3 - x2 is a tensor with value 2.0 and requires_grad=True",
                     lambda: _assert_leaf(x2, "x2", 2.0)))
results.append(check("Task 3 - x2.grad matches df/dx = 10x - 4 at x=2.0",
                     lambda: _assert_grad(x2, "x2", 16.0)))

results.append(check("Task 4 - x3 is a tensor with value 5.0 and requires_grad=True",
                     lambda: _assert_leaf(x3, "x3", 5.0)))
results.append(check("Task 4 - y_no_grad.requires_grad is False (computed inside torch.no_grad())",
                     lambda: _assert_true(y_no_grad.requires_grad is False,
                                          "expected y_no_grad.requires_grad to be False")))
results.append(check("Task 4 - z_with_grad.requires_grad is True (computed outside torch.no_grad())",
                     lambda: _assert_true(z_with_grad.requires_grad is True,
                                          "expected z_with_grad.requires_grad to be True")))

print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
