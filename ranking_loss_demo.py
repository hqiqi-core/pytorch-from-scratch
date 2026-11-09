import torch
import torch.nn.functional as F
from torch import nn

initial_scores = torch.tensor([
    [-1.0,  1.0],  # 负例排在正例前  正例，分数 -1，该高却低   负例，分数 1，该低却高
    [ 0.0,  0.0],  # 同分
    [ 1.0, -1.0],  # 正例排在负例前
])

targets = torch.tensor([
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
])

# ---- BCE ----  BCE 逐个格子算损失，再平均。它不管“row0 的正例和负例谁高”，只管“这个格子该高还是该低，现在够不够”。
bce_scores = nn.Parameter(initial_scores.clone())
bce_loss = F.binary_cross_entropy_with_logits(bce_scores, targets)
bce_loss.backward()
print("=== BCE ===")
print("initial loss:", bce_loss.item())
print("gradient:")
print(bce_scores.grad)

# ---- Pairwise ----  Pairwise 不看单个格子，看每行两个格子的差：
pair_scores = nn.Parameter(initial_scores.clone())
gap = pair_scores[:, 0] - pair_scores[:, 1]
pair_loss = F.softplus(-gap).mean()
pair_loss.backward()
print("\n=== Pairwise ===")
print("initial loss:", pair_loss.item())
print("gradient:")
print(pair_scores.grad)

print("\n=== SGD 20 steps ===")

def run_sgd(loss_fn, name):
    scores = nn.Parameter(initial_scores.clone())
    optimizer = torch.optim.SGD([scores], lr=0.5)

    initial_loss = loss_fn(scores).item()

    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(scores)
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(scores).item()
    gaps = (scores[:, 0] - scores[:, 1]).detach()
    correct = (gaps > 0).sum().item()

    print(f"\n{name}:")
    print(f"  initial loss: {initial_loss:.4f}")
    print(f"  final loss:   {final_loss:.4f}")
    print(f"  gaps: {[round(g, 3) for g in gaps.tolist()]}")
    print(f"  rows with s_pos > s_neg: {correct}/3")

run_sgd(
    lambda s: F.binary_cross_entropy_with_logits(s, targets),
    "BCE",
)
run_sgd(
    lambda s: F.softplus(-(s[:, 0] - s[:, 1])).mean(),
    "Pairwise",
)

print("\n=== Shift invariance ===")

shifted = initial_scores + 5.0

# Pairwise：gap 不变，loss 应不变
gap_orig = initial_scores[:, 0] - initial_scores[:, 1]
gap_shift = shifted[:, 0] - shifted[:, 1]
pair_orig = F.softplus(-gap_orig).mean().item()
pair_shift = F.softplus(-gap_shift).mean().item()
print(f"Pairwise  orig={pair_orig:.6f}  shifted={pair_shift:.6f}")
assert abs(pair_orig - pair_shift) < 1e-6

# BCE：绝对值变了，loss 应变
bce_orig = F.binary_cross_entropy_with_logits(initial_scores, targets).item()
bce_shift = F.binary_cross_entropy_with_logits(shifted, targets).item()
print(f"BCE       orig={bce_orig:.6f}  shifted={bce_shift:.6f}")
assert abs(bce_orig - bce_shift) > 0.1

print("\nShift invariance check passed.")
