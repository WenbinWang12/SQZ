# -*- coding: utf-8 -*-
"""
对比 Forward Learning（黑盒扰动梯度） vs Backprop（交叉熵）—— CIFAR-10 / CIFAR-100（修正版）
- 关键修正：
  1) Forward 目标改为最小化 J = CE + local（local 为正则项，越大越差），避免原版“符号反了”。
  2) SPSA 采用相对扰动：theta + eps * sign * (|theta| + 1e-3)，提升信噪比。
  3) 调整超参：eps 网格放大；K 增大、inner_steps 减少；步长裁剪上限增大。
- 结构保持与原版一致：统一随机性、参数向量化、local loss、网格搜索、最优对比、可选 landscape。
- 可切换模型：小型 ConvNet（默认）或 torchvision ResNet18（--model resnet18）。
- 可选择对子采样训练集以加速调参（SUBSET_RATIO / MAX_TRAIN_SAMPLES）。
- 输出：最优曲线图、（可选）loss landscape、模型权重与 CSV。
"""
import os
import math
import csv
import copy
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# 随机性与设备
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] 使用设备: {device}")

# =========================
# 全局对齐设置 & 超参数（硬对齐）
# =========================
DATASET = "cifar10"  # 可选: "cifar10" / "cifar100"
NUM_WORKERS = 2

GAMMA = 0.99  # 仅在 RL 有用，这里保留以保持结构一致
MAX_UPDATES = 500              # 两侧一致（每次 update 由若干小步组成）
EVAL_EVERY = 2                  # 打印频率
HIDDEN_SIZES = (256,)           # 仅在 MLP 用；ConvNet 不用
MOVING_AVG_W = 0.95
STEADY_K = 5                    # 评分时取尾部 K 次验证均值
REWARD_CLIP = None              # 图像分类不使用

# 数据子采样（加速调参；设为 1.0 表示全量；或者指定 MAX_TRAIN_SAMPLES 更直接）
SUBSET_RATIO = 1.0
MAX_TRAIN_SAMPLES = None  # 如 20000，可显著加速
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256

# =========================
# “软对齐”搜索空间（Forward & Backprop）
# =========================
# —— Forward Learning（黑盒扰动梯度：eps / lr）
# 放大 eps 网格以提升有限差分信噪比；K 增大以降方差；inner_steps 降低以提高单次估计质量
FL_LR_GRID = [0.00030, 0.00040, 0.00050]
FL_EPS_GRID = [1e-2, 2e-2, 5e-2]
K_FL = 4
INNER_STEPS_FL = 10  # 每个 update 进行多少个 mini-batch 更新（Forward）

# —— Backprop（标准交叉熵）
BP_LR_GRID = [0.1, 0.05, 0.02]
BP_WEIGHT_DECAY_GRID = [5e-4, 1e-4]
EPOCHS_PER_UPDATE_BP = 1   # 每个 update 训练多少个 epoch
INNER_STEPS_BP = None      # 自动按 epoch 走

# =========================
# Local Loss 配置（可自由开关/加权）
# =========================
# 在监督学习里将 local 视为“正则项”（越大越差），因此目标是最小化 CE + local_weighted_sum。
LOCAL_LOSS_CFG = {
    "act_l2":     {"weight": 0.00},               # 默认为0，可按需开到 1e-4 ~ 1e-3
    "decor":      {"weight": 0.00},
    "slow":       {"weight": 0.00},
    "ent_target": {"weight": 0.002, "target": 1.8}  # CIFAR-10: 最大熵 log(10)≈2.303
}
GLOBAL_SUPERVISED_WEIGHT = 1.0  # w_S（分类交叉熵的权重）

# Forward 步长裁剪上限适当放宽，避免更新被过早钳死
MAX_STEP_NORM = 5.0
RELATIVE_PERTURB_EPS_FLOOR = 1e-3  # 相对扰动中的下限，避免权重过小而无效扰动

# =========================
# 工具
# =========================

def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(xs, w=MOVING_AVG_W):
    out, ma = [], 0.0
    for i, x in enumerate(xs):
        ma = x if i == 0 else (w * ma + (1 - w) * x)
        out.append(ma)
    return out

# =========================
# 数据集与模型
# =========================

def get_dataloaders(dataset=DATASET, batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE):
    if dataset.lower() == "cifar10":
        num_classes = 10
        ds_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True,
            transform=T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616))
            ])
        )
        ds_test = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616))
            ])
        )
    elif dataset.lower() == "cifar100":
        num_classes = 100
        ds_train = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True,
            transform=T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761))
            ])
        )
        ds_test = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761))
            ])
        )
    else:
        raise ValueError("dataset 必须是 cifar10 或 cifar100")

    # 子采样（可加速网格搜索）
    if MAX_TRAIN_SAMPLES is not None:
        indices = list(range(len(ds_train)))
        random.shuffle(indices)
        indices = indices[:MAX_TRAIN_SAMPLES]
        ds_train = torch.utils.data.Subset(ds_train, indices)
    elif SUBSET_RATIO < 1.0:
        n = int(len(ds_train) * SUBSET_RATIO)
        indices = list(range(len(ds_train)))
        random.shuffle(indices)
        ds_train = torch.utils.data.Subset(ds_train, indices[:n])

    pin = True if device.type == "cuda" else False
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=test_batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin
    )
    return train_loader, test_loader, num_classes


class SmallConvNet(nn.Module):
    """比 ResNet18 更轻量的小型 CNN，适合黑盒方法调参。"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x, need_acts=False):
        acts = []
        out = x
        for layer in self.features:
            out = layer(out)
            if isinstance(layer, nn.ReLU) and need_acts:
                acts.append(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        if need_acts:
            return logits, acts
        return logits


def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "convnet":
        return SmallConvNet(num_classes)
    elif name == "resnet18":
        net = torchvision.models.resnet18(num_classes=num_classes)
        return net
    else:
        raise ValueError("model 仅支持 convnet / resnet18")

# =========================
# 参数向量化（Forward Learning）
# =========================

def get_param_vector(model: nn.Module):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_param_vector(model: nn.Module, theta_vec: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(theta_vec[idx:idx+n].view_as(p))
        idx += n

# =========================
# Local Loss & 指标
# =========================
@torch.no_grad()

def eval_acc(model: nn.Module, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def _loss_act_l2(acts_list):
    if not acts_list:
        return torch.tensor(0.0, device=device)
    vals = []
    for A in acts_list:
        vals.append((A**2).mean())
    return torch.stack(vals).mean()


def _loss_decorrelation(acts_list, eps=1e-6):
    if not acts_list:
        return torch.tensor(0.0, device=device)
    vals = []
    for A in acts_list:
        # A: (B,C,H,W) 或 (B,H)；拉平空间维度
        B = A.shape[0]
        feat = A.view(B, -1)
        if feat.shape[1] < 2 or B < 2:
            continue
        X = feat - feat.mean(dim=0, keepdim=True)
        C = (X.t() @ X) / (B + eps)          # (D,D)
        off = C - torch.diag(torch.diag(C))
        vals.append((off**2).mean())
    return torch.stack(vals).mean() if vals else torch.tensor(0.0, device=device)


def _loss_entropy_target_from_logits(logits, target):
    # logits -> softmax 熵
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(dim=-1)  # (B,)
    return (H.mean() - float(target))**2


def compute_local_losses_from_forward(logits, acts_list, cfg: dict):
    """返回 (dict, weighted_sum)。local 是正则（越大越差），将被直接相加到 CE。"""
    if cfg is None:
        return {}, torch.tensor(0.0, device=device)

    losses = {}
    total = torch.tensor(0.0, device=device)

    w = cfg.get("act_l2", {}).get("weight", 0.0)
    if w != 0.0:
        val = _loss_act_l2(acts_list)
        losses["act_l2"] = val.detach().item()
        total = total + w * val

    w = cfg.get("decor", {}).get("weight", 0.0)
    if w != 0.0:
        val = _loss_decorrelation(acts_list)
        losses["decor"] = val.detach().item()
        total = total + w * val

    ent_cfg = cfg.get("ent_target", {})
    w = ent_cfg.get("weight", 0.0)
    if w != 0.0:
        tgt = ent_cfg.get("target", 0.0)
        val = _loss_entropy_target_from_logits(logits, tgt)
        losses["ent_target"] = val.detach().item()
        total = total + w * val

    # slowness 在监督学习中通常无意义，这里不计
    return losses, total

# 小工具：兼容不同模型的 forward（是否支持 need_acts，返回值可能是 Tensor 或 Tuple）
def forward_logits_and_acts(model, x, need_acts: bool):
    try:
        out = model(x, need_acts=need_acts)
    except TypeError:
        out = model(x)
    # 统一为 (logits, acts_list)
    if isinstance(out, tuple):
        logits = out[0]
        acts = out[1] if len(out) > 1 else []
        if isinstance(acts, (list, tuple)):
            acts_list = list(acts)
        else:
            acts_list = [acts]
    else:
        logits = out
        acts_list = []
    return logits, acts_list

# =========================
# 训练：Forward Learning（黑盒扰动梯度）
# 目标：最小化  J_total(θ) = w_S * CE + sum_i weight_i * local_i
# SPSA 梯度估计：g ≈ [(J(θ+εΔ) - J(θ-εΔ)) / (2ε)] · Δ
# 使用“相对扰动”：θ ± (eps * (|θ| + floor)) ⊙ Δ
# =========================

def train_forward_local(
    model, train_loader, val_loader,
    updates=MAX_UPDATES, eps=2e-2, lr=5e-3, K=4, inner_steps=10,
    local_cfg=LOCAL_LOSS_CFG, w_supervised=GLOBAL_SUPERVISED_WEIGHT,
    track_thetas: bool=False
):
    theta = get_param_vector(model).to(device)
    best_theta = theta.clone()
    best_val = -1.0
    theta_traj = [theta.detach().cpu().clone()] if track_thetas else None

    train_obj_curve = []
    val_acc_curve = []

    it = iter(train_loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(train_loader)
            return next(it)

    for up in range(1, updates + 1):
        g_est = torch.zeros_like(theta)

        # ===== 累计 inner_steps 次小批次的黑盒梯度估计 =====
        model.train()
        for inner in range(inner_steps):
            x, y = next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # 相对尺度（逐元素），增强在不同量级参数上的可见性
            rel_scale = (theta.abs() + RELATIVE_PERTURB_EPS_FLOOR)

            # 每次用 K 组 Rademacher 向量
            for k in range(K):
                delta = torch.randint_like(theta, low=0, high=2, device=device, dtype=torch.long)
                delta = delta.float().mul_(2.0).sub_(1.0)  # {-1, +1}

                # θ + eps * rel_scale * Δ
                set_param_vector(model, theta + (eps * rel_scale) * delta)
                logits_p, acts_p = forward_logits_and_acts(model, x, need_acts=(local_cfg is not None))
                ce_p = F.cross_entropy(logits_p, y)
                _, local_p = compute_local_losses_from_forward(logits_p, acts_p if local_cfg else [], local_cfg)
                Jp = w_supervised * ce_p + local_p

                # θ - eps * rel_scale * Δ
                set_param_vector(model, theta - (eps * rel_scale) * delta)
                logits_m, acts_m = forward_logits_and_acts(model, x, need_acts=(local_cfg is not None))
                ce_m = F.cross_entropy(logits_m, y)
                _, local_m = compute_local_losses_from_forward(logits_m, acts_m if local_cfg else [], local_cfg)
                Jm = w_supervised * ce_m + local_m

                diff = (Jp - Jm).detach()
                g_est += (diff / (2.0 * eps)) * (delta / (rel_scale + 1e-12))  # SPSA: 对相对扰动，应当按元素除以 rel_scale

            # 恢复到 θ，准备下一个 inner batch
            set_param_vector(model, theta)

        g_est /= max(1, inner_steps*K)

        # 步长限制（防爆）
        step = lr * g_est
        step_norm = torch.norm(step)
        if torch.isfinite(step_norm) and step_norm > MAX_STEP_NORM:
            step = step * (MAX_STEP_NORM / (step_norm + 1e-12))

        prev_theta = theta.clone()
        theta = theta - step  # 最小化 J_total
        set_param_vector(model, theta)

        # 非有限回退
        if not torch.isfinite(theta).all():
            print("[FWD][Guard] non-finite params; revert & reduce lr.")
            theta = prev_theta
            set_param_vector(model, theta)
            lr *= 0.5
            continue

        # 记录训练目标值（再走一小批次估计）
        model.train()
        xb, yb = next_batch()
        xb, yb = xb.to(device), yb.to(device)
        logits_b, acts_b = forward_logits_and_acts(model, xb, need_acts=(local_cfg is not None))
        ce_b = F.cross_entropy(logits_b, yb)
        _, local_b = compute_local_losses_from_forward(logits_b, acts_b if local_cfg else [], local_cfg)
        J_new = (w_supervised * ce_b + local_b).detach().item()
        train_obj_curve.append(J_new)

        # 验证精度
        val_acc = eval_acc(model, val_loader)
        val_acc_curve.append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_theta = theta.clone()

        if track_thetas:
            theta_traj.append(theta.detach().cpu().clone())

        if up % EVAL_EVERY == 0 or up == 1:
            tag = "LocalON" if local_cfg is not None else "LocalOFF"
            print(f"[FWD-{tag}] Update {up:3d} | train(obj): {J_new:7.4f} | VAL(acc): {val_acc*100:6.2f}%")

    # 恢复最佳验证点
    set_param_vector(model, best_theta)
    return (train_obj_curve, val_acc_curve, best_theta) if not track_thetas else (train_obj_curve, val_acc_curve, best_theta, theta_traj)

# =========================
# 训练：Backprop（交叉熵）
# =========================

def train_backprop(
    model, train_loader, val_loader,
    updates=MAX_UPDATES, epochs_per_update=1,
    lr=0.1, weight_decay=5e-4,
    track_thetas: bool=False
):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates*epochs_per_update)

    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0

    train_loss_curve = []
    val_acc_curve = []
    theta_traj = [get_param_vector(model).detach().cpu().clone()] if track_thetas else None

    for up in range(1, updates + 1):
        # 每个 update 训练 epochs_per_update 个 epoch
        for _ in range(epochs_per_update):
            model.train()
            losses = []
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(loss.detach().item())
            scheduler.step()
        train_loss_curve.append(float(np.mean(losses)) if losses else float("nan"))

        model.eval()
        val_acc = eval_acc(model, val_loader)
        val_acc_curve.append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if track_thetas:
            theta_traj.append(get_param_vector(model).detach().cpu().clone())

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[BP]   Update {up:3d} | train(loss): {train_loss_curve[-1]:7.4f} | VAL(acc): {val_acc*100:6.2f}%")

    model.load_state_dict(best_state)
    return (train_loss_curve, val_acc_curve) if not track_thetas else (train_loss_curve, val_acc_curve, theta_traj)

# =========================
# 实验/搜索封装
# =========================

def run_one_forward_trial(base_model, train_loader, val_loader, lr_fl, eps, local_cfg, track_thetas=True):
    model = copy.deepcopy(base_model).to(device)
    out = train_forward_local(
        model, train_loader, val_loader,
        updates=MAX_UPDATES, eps=eps, lr=lr_fl, K=K_FL, inner_steps=INNER_STEPS_FL,
        local_cfg=local_cfg, w_supervised=GLOBAL_SUPERVISED_WEIGHT,
        track_thetas=track_thetas
    )
    if track_thetas:
        train_obj, val_accs, best_theta, theta_traj = out
    else:
        train_obj, val_accs, best_theta = out
        theta_traj = None

    k = min(STEADY_K, len(val_accs))
    score = float(np.mean(val_accs[-k:])) if k > 0 else float(np.mean(val_accs))
    return score, train_obj, val_accs, best_theta, copy.deepcopy(model.state_dict()), theta_traj


def run_one_bp_trial(base_model, train_loader, val_loader, lr_bp, wd, track_thetas=True):
    model = copy.deepcopy(base_model).to(device)
    out = train_backprop(
        model, train_loader, val_loader,
        updates=MAX_UPDATES, epochs_per_update=EPOCHS_PER_UPDATE_BP,
        lr=lr_bp, weight_decay=wd, track_thetas=track_thetas
    )
    if track_thetas:
        train_loss, val_accs, theta_traj = out
    else:
        train_loss, val_accs = out
        theta_traj = None

    k = min(STEADY_K, len(val_accs))
    score = float(np.mean(val_accs[-k:])) if k > 0 else float(np.mean(val_accs))
    return score, train_loss, val_accs, copy.deepcopy(model.state_dict()), theta_traj

# =========================
# Loss landscape（可选）
# =========================

def sample_normalized_direction_like(model: nn.Module, rng: np.random.RandomState):
    vecs = []
    for p in model.parameters():
        noise = torch.from_numpy(rng.randn(*p.data.shape)).to(dtype=p.data.dtype, device=p.data.device)
        scale = p.data.norm().item()
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        noise = noise / (noise.norm() + 1e-12) * scale
        vecs.append(noise.reshape(-1))
    d = torch.cat(vecs).float()
    d = d / (d.norm() + 1e-12)
    return d


def gram_schmidt_2(d1: torch.Tensor, d2: torch.Tensor):
    u1 = d1 / (d1.norm() + 1e-12)
    proj = torch.dot(d2, u1) * u1
    v2 = d2 - proj
    n2 = v2.norm()
    if not torch.isfinite(n2) or n2 < 1e-12:
        v2 = torch.randn_like(d2)
        v2 -= torch.dot(v2, u1) * u1
        n2 = v2.norm()
    u2 = v2 / (n2 + 1e-12)
    return u1, u2


@torch.no_grad()

def coeff_from_theta(theta_ref: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor, theta: torch.Tensor):
    D = torch.stack([d1, d2], dim=1)   # (N,2)
    rhs = (theta - theta_ref)
    G = D.T @ D
    b = D.T @ rhs
    sol = torch.linalg.solve(G, b) if torch.det(G) > 1e-12 else torch.zeros(2, device=theta.device)
    return float(sol[0].item()), float(sol[1].item())


@torch.no_grad()

def make_landscape(model, theta_ref, d1, d2, loader, span=1.0, grid_n=31):
    alphas = np.linspace(-span, span, grid_n)
    betas  = np.linspace(-span, span, grid_n)
    Z = np.zeros((grid_n, grid_n), dtype=np.float32)

    theta_bak = get_param_vector(model).detach().clone()

    model.eval()
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            theta_ij = theta_ref + a*d1 + b*d2
            set_param_vector(model, theta_ij)
            # 使用验证交叉熵作为 landscape（更平滑）
            losses = []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                losses.append(loss.item())
            Z[j, i] = float(np.mean(losses))

    set_param_vector(model, theta_bak)
    A, B = np.meshgrid(alphas, betas)
    return A, B, Z


def plot_landscape_with_trajectories(A, B, Z, trajs_ab, labels, tag_png):
    plt.figure(figsize=(7.5, 6.5))
    cs = plt.contourf(A, B, Z, levels=30)
    plt.colorbar(cs, label=r"$\\mathcal{L}(\\alpha,\\beta)\\;=\\;\\text{Val CE}$")
    for (ab, lb) in zip(trajs_ab, labels):
        ab = np.asarray(ab)
        if ab.size == 0:
            continue
        plt.plot(ab[:, 0], ab[:, 1], marker='o', markersize=3, linewidth=1.5, label=lb, alpha=0.9)
    plt.xlabel(r"$\\alpha$ along $d_1$")
    plt.ylabel(r"$\\beta$ along $d_2$")
    plt.title("Validation Loss Landscape & Optimization Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(tag_png, dpi=150)
    plt.show()

# =========================
# 主流程
# =========================

def main():
    global MAX_UPDATES, INNER_STEPS_FL, EPOCHS_PER_UPDATE_BP

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET, choices=["cifar10", "cifar100"])
    parser.add_argument("--model", type=str, default="convnet", choices=["convnet", "resnet18"])
    parser.add_argument("--do_landscape", action="store_true", help="是否评估 loss landscape（较慢）")
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--inner_steps_fl", type=int, default=INNER_STEPS_FL)
    parser.add_argument("--epochs_per_update_bp", type=int, default=EPOCHS_PER_UPDATE_BP)
    args = parser.parse_args()
    
    MAX_UPDATES = args.updates
    INNER_STEPS_FL = args.inner_steps_fl
    EPOCHS_PER_UPDATE_BP = args.epochs_per_update_bp

    # 数据加载
    train_loader, val_loader, num_classes = get_dataloaders(args.dataset)

    # 统一初始化模型
    set_global_seed(SEED)
    base_model = build_model(args.model, num_classes).to(device)
    base_theta0 = get_param_vector(base_model).detach().cpu().clone()

    # ===== Forward(Local ON) Grid Search =====
    print("\n=== Grid Search: Forward Learning (Local Loss = ON) ===")
    fwd_on_results = []  # (lr, eps, score, final_val_acc)
    best_fwd_on = {"score": -1e9}

    for lr_fl in FL_LR_GRID:
        for eps in FL_EPS_GRID:
            print(f"\n[FWD-ON TRY] lr={lr_fl:.5g}, eps={eps:.3g}")
            score, tr_curve, val_curve, best_theta, state_dict, theta_traj = run_one_forward_trial(
                base_model, train_loader, val_loader, lr_fl, eps, LOCAL_LOSS_CFG, track_thetas=True
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            fwd_on_results.append((lr_fl, eps, score, final_val))
            if score > best_fwd_on["score"]:
                best_fwd_on.update({
                    "score": score, "lr": lr_fl, "eps": eps,
                    "train_curve": tr_curve, "val_curve": val_curve,
                    "best_theta": best_theta.detach().cpu(),
                    "state_dict": state_dict, "theta_traj": theta_traj
                })
            print(f"[FWD-ON RESULT] score(steady-avg)={score*100:.2f}%, final_val_acc={final_val*100:.2f}%")

    print("\n[FWD-ON BEST] lr={:.5g}, eps={:.3g}, score={:.2f}%".format(
        best_fwd_on['lr'], best_fwd_on['eps'], best_fwd_on['score']*100))

    # ===== Forward(Local OFF) Grid Search =====
    print("\n=== Grid Search: Forward Learning (Local Loss = OFF) ===")
    fwd_off_results = []
    best_fwd_off = {"score": -1e9}

    for lr_fl in FL_LR_GRID:
        for eps in FL_EPS_GRID:
            print(f"\n[FWD-OFF TRY] lr={lr_fl:.5g}, eps={eps:.3g}")
            score, tr_curve, val_curve, best_theta, state_dict, theta_traj = run_one_forward_trial(
                base_model, train_loader, val_loader, lr_fl, eps, None, track_thetas=True
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            fwd_off_results.append((lr_fl, eps, score, final_val))
            if score > best_fwd_off["score"]:
                best_fwd_off.update({
                    "score": score, "lr": lr_fl, "eps": eps,
                    "train_curve": tr_curve, "val_curve": val_curve,
                    "best_theta": best_theta.detach().cpu(),
                    "state_dict": state_dict, "theta_traj": theta_traj
                })
            print(f"[FWD-OFF RESULT] score(steady-avg)={score*100:.2f}%, final_val_acc={final_val*100:.2f}%")

    print("\n[FWD-OFF BEST] lr={:.5g}, eps={:.3g}, score={:.2f}%".format(
        best_fwd_off['lr'], best_fwd_off['eps'], best_fwd_off['score']*100))

    # ===== Backprop Grid Search =====
    print("\n=== Grid Search: Backprop (CE) ===")
    bp_results = []  # (lr, wd, score, final_val_acc)
    best_bp = {"score": -1e9}

    for lr_bp in BP_LR_GRID:
        for wd in BP_WEIGHT_DECAY_GRID:
            print(f"\n[BP-TRY] lr={lr_bp:.5g}, weight_decay={wd:.1e}")
            score, tr_curve, val_curve, state_dict, theta_traj = run_one_bp_trial(
                base_model, train_loader, val_loader, lr_bp, wd, track_thetas=True
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            bp_results.append((lr_bp, wd, score, final_val))
            if score > best_bp["score"]:
                best_bp.update({
                    "score": score, "lr": lr_bp, "wd": wd,
                    "train_curve": tr_curve, "val_curve": val_curve,
                    "state_dict": state_dict, "theta_traj": theta_traj
                })
            print(f"[BP-RESULT] score(steady-avg)={score*100:.2f}%, final_val_acc={final_val*100:.2f}%")

    print("\n[BP-BEST] lr={:.5g}, weight_decay={:.1e}, score={:.2f}%".format(
        best_bp['lr'], best_bp['wd'], best_bp['score']*100))

    # ===== 最优模型最终评估 =====
    # 统一重建模型载入最优权重
    fwd_on_model = build_model(args.model, num_classes).to(device)
    fwd_on_model.load_state_dict(best_fwd_on["state_dict"])
    fwd_off_model = build_model(args.model, num_classes).to(device)
    fwd_off_model.load_state_dict(best_fwd_off["state_dict"])
    bp_model = build_model(args.model, num_classes).to(device)
    bp_model.load_state_dict(best_bp["state_dict"])

    final_fwd_on_acc = eval_acc(fwd_on_model, val_loader)
    final_fwd_off_acc = eval_acc(fwd_off_model, val_loader)
    final_bp_acc = eval_acc(bp_model, val_loader)

    print("\n===== 最终评估（各自最优超参；验证集精度） =====")
    print(f"  Forward(Local ON ): {final_fwd_on_acc*100:.2f}% "
          f"(lr={best_fwd_on['lr']:.5g},  eps={best_fwd_on['eps']:.3g})")
    print(f"  Forward(Local OFF): {final_fwd_off_acc*100:.2f}% "
          f"(lr={best_fwd_off['lr']:.5g}, eps={best_fwd_off['eps']:.3g})")
    print(f"  Backprop (CE)     : {final_bp_acc*100:.2f}% "
          f"(lr={best_bp['lr']:.5g}, wd={best_bp['wd']:.1e})")

    # ===== 绘图（最优曲线） =====
    plt.figure(figsize=(14, 5))
    # 左：训练趋势（Forward 画 train obj；Backprop 画 train loss）
    plt.subplot(1, 2, 1)
    xs_on = np.arange(1, len(best_fwd_on["train_curve"]) + 1)
    xs_off = np.arange(1, len(best_fwd_off["train_curve"]) + 1)
    xb = np.arange(1, len(best_bp["train_curve"]) + 1)
    plt.plot(xs_on,  best_fwd_on["train_curve"],  alpha=0.25, label="Forward(ON) train(obj)")
    plt.plot(xs_on,  moving_average(best_fwd_on["train_curve"]),  label="Forward(ON) EMA")
    plt.plot(xs_off, best_fwd_off["train_curve"], alpha=0.25, label="Forward(OFF) train(obj)")
    plt.plot(xs_off, moving_average(best_fwd_off["train_curve"]), label="Forward(OFF) EMA")
    plt.plot(xb,     best_bp["train_curve"],      alpha=0.25, label="BP train (loss)")
    plt.plot(xb,     moving_average(best_bp["train_curve"]),     label="BP train EMA")
    plt.xlabel("Update #"); plt.ylabel("Objective / Loss")
    plt.title("Training metrics (trend)")
    plt.legend()

    # 右：验证精度
    plt.subplot(1, 2, 2)
    x1 = np.arange(1, len(best_fwd_on["val_curve"]) + 1)
    x2 = np.arange(1, len(best_fwd_off["val_curve"]) + 1)
    x3 = np.arange(1, len(best_bp["val_curve"]) + 1)
    plt.plot(x1, np.array(best_fwd_on["val_curve"]) * 100.0,  label="Forward(ON) VAL acc")
    plt.plot(x2, np.array(best_fwd_off["val_curve"]) * 100.0, label="Forward(OFF) VAL acc")
    plt.plot(x3, np.array(best_bp["val_curve"]) * 100.0,      label="BP VAL acc")
    plt.xlabel("Update #"); plt.ylabel("Accuracy (%)")
    plt.title("Validation accuracy (best hparams)")
    plt.legend()

    plt.tight_layout()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = f"cifar_best_curves_{tag}.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()

    # ===== 可选：Loss landscape & trajectories =====
    if args.do_landscape:
        print("[Landscape] 开始评估（较慢）...")
        theta_ref = base_theta0.clone().to(device)
        rng = np.random.RandomState(SEED + 777)
        d1_raw = sample_normalized_direction_like(base_model, rng)
        d2_raw = sample_normalized_direction_like(base_model, rng)
        d1, d2 = gram_schmidt_2(d1_raw, d2_raw)

        # 使用验证集计算 landscape（交叉熵）
        tmp_model = build_model(args.model, num_classes).to(device)
        set_param_vector(tmp_model, theta_ref)

        A, B, Z = make_landscape(tmp_model, theta_ref, d1, d2, val_loader, span=1.0, grid_n=31)

        def project_traj(thetas):
            ab = []
            for th in thetas:
                th = th.to(device)
                a, b = coeff_from_theta(theta_ref, d1, d2, th)
                ab.append((a, b))
            return np.array(ab, dtype=np.float32)

        traj_on  = project_traj([t for t in best_fwd_on.get("theta_traj", [])])   if "theta_traj" in best_fwd_on  else np.zeros((0,2))
        traj_off = project_traj([t for t in best_fwd_off.get("theta_traj", [])])  if "theta_traj" in best_fwd_off else np.zeros((0,2))
        traj_bp  = project_traj([t for t in best_bp.get("theta_traj", [])])       if "theta_traj" in best_bp      else np.zeros((0,2))

        land_png = f"cifar_loss_landscape_{tag}.png"
        plot_landscape_with_trajectories(A, B, Z,
            trajs_ab=[traj_on, traj_off, traj_bp],
            labels=["Forward (Local ON)", "Forward (Local OFF)", "Backprop (CE)"],
            tag_png=land_png
        )
        print(f"[Landscape] Saved: {land_png}")

    # ===== 保存成果 =====
    torch.save(best_fwd_on["best_theta"],  f"forward_on_best_theta_{tag}.pt")
    torch.save(best_fwd_on["state_dict"],  f"model_forward_on_best_{tag}.pt")
    torch.save(best_fwd_off["best_theta"], f"forward_off_best_theta_{tag}.pt")
    torch.save(best_fwd_off["state_dict"], f"model_forward_off_best_{tag}.pt")
    torch.save(best_bp["state_dict"],      f"model_bp_best_{tag}.pt")

    with open(f"forward_on_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_forward", "eps", "score_steady_avg_acc", "final_val_acc"])
        for row in fwd_on_results:
            w.writerow(row)
    with open(f"forward_off_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_forward", "eps", "score_steady_avg_acc", "final_val_acc"])
        for row in fwd_off_results:
            w.writerow(row)
    with open(f"bp_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_bp", "weight_decay", "score_steady_avg_acc", "final_val_acc"])
        for row in bp_results:
            w.writerow(row)

    print("\n===== 已保存文件 =====")
    print(f"  图像: {fig_path}")
    if args.do_landscape:
        print(f"  Landscape: cifar_loss_landscape_{tag}.png")
    print(f"  Forward(Local ON ): model_forward_on_best_{tag}.pt, forward_on_best_theta_{tag}.pt, forward_on_grid_{tag}.csv")
    print(f"  Forward(Local OFF): model_forward_off_best_{tag}.pt, forward_off_best_theta_{tag}.pt, forward_off_grid_{tag}.csv")
    print(f"  Backprop (CE)     : model_bp_best_{tag}.pt, bp_grid_{tag}.csv")


if __name__ == "__main__":
    main()