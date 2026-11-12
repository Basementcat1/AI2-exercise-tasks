"""
Exercise 02 - nn.Module: Building a Network (SOLUTION)
=======================================================
"""

import torch
import torch.nn as nn


# ===========================================================================
# Task 1: Define TinyNet
# ===========================================================================
class TinyNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # ===========================================================================
    # Task 2: Forward Pass with Dummy Input
    # ===========================================================================
    model = TinyNet(num_classes=5)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)

    print("Task 2:")
    print(f"  Output shape: {output.shape}")   # torch.Size([1, 5])
    print(f"  Output values: {output}")


    # ===========================================================================
    # Task 3: Inspect Model Parameters
    # ===========================================================================
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nTask 3:")
    print(f"  Model architecture:\n{model}")
    print(f"\n  Total trainable parameters: {total_params}")


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


    EXPECTED_PARAM_COUNT = 6517  # fixed by the architecture spec in Task 1

    results = []

    results.append(check(
        "Task 1 - a fresh TinyNet maps (1, 3, 32, 32) to (1, 5)",
        lambda: _assert_true(
            TinyNet(num_classes=5)(torch.zeros(1, 3, 32, 32)).shape == torch.Size([1, 5]),
            "expected TinyNet(num_classes=5) to map a (1, 3, 32, 32) input to shape (1, 5)")))
    results.append(check(
        "Task 1 - TinyNet has 6517 trainable parameters",
        lambda: _assert_true(
            sum(p.numel() for p in TinyNet(num_classes=5).parameters() if p.requires_grad) == EXPECTED_PARAM_COUNT,
            f"expected {EXPECTED_PARAM_COUNT} trainable parameters for the specified architecture")))

    results.append(check("Task 2 - dummy_input shape is (1, 3, 32, 32)",
                          lambda: _assert_true(dummy_input.shape == torch.Size([1, 3, 32, 32]),
                                                f"expected shape (1, 3, 32, 32), got {tuple(dummy_input.shape)}")))
    results.append(check("Task 2 - output shape is (1, 5)",
                          lambda: _assert_true(output.shape == torch.Size([1, 5]),
                                                f"expected shape (1, 5), got {tuple(output.shape)}")))

    results.append(check("Task 3 - total_params equals 6517",
                          lambda: _assert_true(total_params == EXPECTED_PARAM_COUNT,
                                                f"expected {EXPECTED_PARAM_COUNT}, got {total_params}")))

    print(f"\n--- {sum(results)}/{len(results)} checks passed. ---")
