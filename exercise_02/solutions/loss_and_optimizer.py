"""
Exercise 02 - Loss Functions and Optimizers (SOLUTION)
=======================================================
"""

import torch
import torch.nn as nn

from model import TinyNet


# ===========================================================================
# Task 1: Compute Loss
# ===========================================================================
model = TinyNet(num_classes=5)
dummy_input = torch.randn(4, 3, 32, 32)
targets = torch.tensor([0, 2, 4, 1])

criterion = nn.CrossEntropyLoss()
outputs = model(dummy_input)
loss = criterion(outputs, targets)

print("Task 1:")
print(f"  Loss: {loss.item()}")


# ===========================================================================
# Task 2: One Optimizer Step
# ===========================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTask 2:")
print(f"  Loss before update: {loss.item()}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

outputs_after = model(dummy_input)
loss_after = criterion(outputs_after, targets)

print(f"  Loss after update: {loss_after.item()}")


# ===========================================================================
# Task 3: Understanding zero_grad
# ===========================================================================
model2 = TinyNet(num_classes=5)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

# First forward + backward (no zero_grad needed on first pass — grads start at zero)
out1 = model2(dummy_input)
loss1 = criterion(out1, targets)
loss1.backward()

# Pick the first conv layer's weight gradient
first_param = list(model2.parameters())[0]
grad_after_first = first_param.grad.clone()

# Second forward + backward WITHOUT zero_grad
out2 = model2(dummy_input)
loss2 = criterion(out2, targets)
loss2.backward()

grad_after_second = first_param.grad.clone()

print("\nTask 3:")
print(f"  Grad sum after 1st backward: {grad_after_first.sum().item():.6f}")
print(f"  Grad sum after 2nd backward: {grad_after_second.sum().item():.6f}")
print(f"  (Second is ~2x the first because gradients accumulated)")


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


results = []

results.append(check("Task 1 - outputs shape is (4, 5)",
                      lambda: _assert_true(outputs.shape == torch.Size([4, 5]),
                                            f"expected shape (4, 5), got {tuple(outputs.shape)}")))
results.append(check("Task 1 - loss is a scalar that requires grad",
                      lambda: _assert_true(loss.dim() == 0 and loss.requires_grad,
                                            "expected loss to be a 0-dim tensor with requires_grad=True")))

results.append(check(
    "Task 2 - optimizer is Adam with lr=0.001",
    lambda: _assert_true(isinstance(optimizer, torch.optim.Adam) and optimizer.defaults["lr"] == 0.001,
                          "expected an Adam optimizer with lr=0.001")))
results.append(check(
    "Task 2 - loss_after differs from the pre-update loss",
    lambda: _assert_true(loss_after.item() != loss.item(),
                          "expected the loss to change after optimizer.step()")))

results.append(check(
    "Task 3 - grad_after_second accumulated on top of grad_after_first",
    lambda: torch.testing.assert_close(grad_after_second, grad_after_first * 2)))

print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
