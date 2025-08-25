import math
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE


# =============== 核心：FF层定义 ===============
class FFLayer(nn.Module):
    """
    单个 Forward-Forward 线性层：h = relu(x W + b)
    goodness = mean(h^2)  (对每个样本逐层计算，再取 batch 平均)
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        # Hinton 原文建议 ReLU + squared activations；你也可尝试 SiLU/Tanh 等
        self.act = nn.ReLU()

        # 参数初始化（可按需微调）
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        if bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))

    @staticmethod
    def goodness(h: torch.Tensor) -> torch.Tensor:
        # 对每个样本：goodness_i = mean_j h_{ij}^2
        return (h ** 2).mean(dim=1)


# =============== 网络封装（若干层串联） ===============
class FFNet(nn.Module):
    """
    若干 FFLayer 串联。训练/推断均“无反传”，每层局部判别。
    监督方式：把 label one-hot 拼到输入上（class-conditional 输入），
    正样本：真实标签，负样本：随机错误标签。
    """
    def __init__(self, input_dim: int, layers: List[int], num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        dims = [input_dim + num_classes] + layers  # 第一层输入拼上 one-hot 标签
        self.layers = nn.ModuleList([FFLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward_through_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        返回各层的激活结果，便于逐层计算 goodness。
        """
        hs = []
        h = x
        for layer in self.layers:
            h = layer(h)
            hs.append(h)
        return hs

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        推断：对每个类 c，构造 [x, one_hot(c)]，前向通过所有层，累加 goodness，取 argmax。
        x shape: (B, D_in)
        return: 预测标签 (B,)
        """
        B = x.size(0)
        device = x.device
        scores = torch.zeros(B, self.num_classes, device=device)

        for c in range(self.num_classes):
            onehot = F.one_hot(torch.full((B,), c, device=device), num_classes=self.num_classes).float()
            xc = torch.cat([x, onehot], dim=1)
            hs = self.forward_through_layers(xc)
            # 累加所有层的 goodness 作为该类的打分
            score_c = torch.stack([FFLayer.goodness(h) for h in hs], dim=1).sum(dim=1)
            scores[:, c] = score_c

        return scores.argmax(dim=1)


# =============== 训练例程（逐层 FF 训练） ===============
@dataclass
class FFTrainConfig:
    epochs_per_layer: int = 2
    lr: float = 1e-3
    margin: float = 2.0     # 正负 goodness 的间隔（b，越大越严格）
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_pos_neg_pairs(x_flat: torch.Tensor, y: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构造监督式的正/负样本对：
      - 正：拼接真实标签的 one-hot
      - 负：为每个样本随机采样一个错误标签，拼接其 one-hot
    """
    B, Din = x_flat.shape
    device = x_flat.device

    # 正样本
    y_pos = F.one_hot(y, num_classes=num_classes).float()
    x_pos = torch.cat([x_flat, y_pos], dim=1)

    # 负样本（随机错误标签）
    y_neg_idx = []
    for i in range(B):
        wrong = random.randrange(num_classes - 1)
        if wrong >= y[i].item():
            wrong += 1
        y_neg_idx.append(wrong)
    y_neg = F.one_hot(torch.tensor(y_neg_idx, device=device), num_classes=num_classes).float()
    x_neg = torch.cat([x_flat, y_neg], dim=1)

    return x_pos, x_neg


def ff_layer_train_step(layer: FFLayer,
                        x_pos: torch.Tensor,
                        x_neg: torch.Tensor,
                        optimizer: torch.optim.Optimizer,
                        margin: float) -> float:
    """
    单层的 FF 损失：
      L = - [ log σ(good_pos - b) + log (1 - σ(good_neg - b)) ]
      这里 b=margin。把 goodness 当二分类打分（正应大、负应小）。
    """
    layer.train()
    optimizer.zero_grad()

    h_pos = layer(x_pos)
    h_neg = layer(x_neg)

    g_pos = FFLayer.goodness(h_pos)  # (B,)
    g_neg = FFLayer.goodness(h_neg)  # (B,)

    # logistic 判别目标（可替换为 hinge/margin loss 等）
    loss = - (torch.log(torch.sigmoid(g_pos - margin)) + torch.log(1 - torch.sigmoid(g_neg - margin))).mean()
    loss.backward()           # 这里对该层做一次反传（但**不**跨层传播），体现“局部可学习”
    optimizer.step()

    return loss.item()


def ff_train_layerwise(model: FFNet,
                       train_loader: DataLoader,
                       cfg: FFTrainConfig) -> None:
    device = cfg.device
    model.to(device)

    for li, layer in enumerate(model.layers):
        # 只训练当前层
        for p in model.layers.parameters():
            p.requires_grad_(False)
        for p in layer.parameters():
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(layer.parameters(), lr=cfg.lr)

        for epoch in range(cfg.epochs_per_layer):
            running = 0.0
            n = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                x_flat = x.view(x.size(0), -1)

                if li == 0:
                    # 第0层：这里才需要拼 one-hot
                    x_pos, x_neg = make_pos_neg_pairs(x_flat, y, model.num_classes)
                else:
                    # 更深层：先在冻结的前 li 层上前向，得到第 li 层的正/负输入
                    with torch.no_grad():
                        x_pos0, x_neg0 = make_pos_neg_pairs(x_flat, y, model.num_classes)
                        h_pos, h_neg = x_pos0, x_neg0
                        for j in range(li):
                            h_pos = model.layers[j](h_pos)
                            h_neg = model.layers[j](h_neg)
                    # 不再拼标签，直接作为当前层输入
                    x_pos, x_neg = h_pos, h_neg

                loss = ff_layer_train_step(layer, x_pos, x_neg, optimizer, cfg.margin)
                running += loss * x.size(0)
                n += x.size(0)

            print(f"[Layer {li+1}/{len(model.layers)}] Epoch {epoch+1}/{cfg.epochs_per_layer}  "
                  f"loss={running / n:.4f}")



@torch.no_grad()
def evaluate(model: FFNet, data_loader: DataLoader) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        x_flat = x.view(x.size(0), -1)
        yhat = model.classify(x_flat)
        correct += (yhat == y).sum().item()
        total += y.size(0)
    return correct / total


# =============== 数据 & 训练脚本 ===============
def get_dataloaders(batch_size=256, num_workers=0, pin_memory=None):
    """
    Windows 上先用 num_workers=0，确认跑通后再尝试 >0。
    Normalize((0.5,), (0.5,)) 等价于 t*2-1，避免了 Lambda 的 pickle 问题。
    """
    if pin_memory is None:
        # 只有在用 GPU 时再启用 pin_memory 比较有意义
        pin_memory = torch.cuda.is_available()

    tfm = transforms.Compose([
        transforms.ToTensor(),                     # [0,1]
        transforms.Normalize((0.5,), (0.5,)),      # 映射到 [-1,1]，无 lambda
    ])

    train = datasets.MNIST(root="./data", train=True,  download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=False  # Windows 建议关掉
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=False
    )
    return train_loader, test_loader



def main():
    torch.manual_seed(0)
    train_loader, test_loader = get_dataloaders(batch_size=256, num_workers=0)

    # MNIST: 28*28 输入；分类数 10；给一个小的多层网络
    input_dim = 28 * 28
    num_classes = 10
    layers = [1024, 512]  # 你可以加深/变宽

    model = FFNet(input_dim=input_dim, layers=layers, num_classes=num_classes)

    cfg = FFTrainConfig(
        epochs_per_layer=2,   # 演示用，实际可调大
        lr=1e-3,
        margin=2.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    ff_train_layerwise(model, train_loader, cfg)

    acc = evaluate(model, test_loader)
    print(f"Test accuracy: {acc*100:.2f}%")

    # # === 可视化 1：混淆矩阵 ===
    # plot_confusion_matrix(model, test_loader, class_names=[str(i) for i in range(10)])

    # # === 可视化 2：错分样本网格 ===
    # show_misclassified_grid(model, test_loader, num_samples=25, denormalize=True)

    # # === 可视化 3：goodness 直方图（第1层；可改成 1 表示第2层） ===
    # plot_goodness_histogram(model, test_loader, layer_index=0, num_batches=2, bins=40)

    # === 可视化 4：t-SNE（最后一层；可改 layer_index=0 看第1层）===
    plot_tsne_activations(model, test_loader, layer_index=0, sample_size=2000, perplexity=30.0)

    # 为 10 个类别各生成 1 张（10x1 的网格）
    _ = visualize_generation(
        model,
        classes_to_generate=list(range(10)),  # 0..9
        per_class=1,                          # 每类生成几张
        steps=200,                            # 梯度上升步数，可调 100~500
        lr=0.1,                               # 学习率；较大时容易糊，较小时收敛慢
        tv_weight=0.002,                      # TV 正则；越大越平滑，但也可能失真
        l2_weight=0.0,                        # L2 正则；一般可 0，如不稳定可加 1e-4~1e-3
        img_h=28, img_w=28,                   # MNIST
        save_path="./ff_generations/ff_gen_10x1.png",
        show=True
    )

@torch.no_grad()
def _gather_preds(model, data_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        x_flat = x.view(x.size(0), -1)
        preds = model.classify(x_flat)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred


# 1) 混淆矩阵
def plot_confusion_matrix(model, data_loader, class_names: Optional[List[str]] = None):
    device = next(model.parameters()).device
    y_true, y_pred = _gather_preds(model, data_loader, device)

    if class_names is None:
        num_classes = int(np.max(y_true)) + 1
        class_names = list(map(str, range(num_classes)))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# 2) 错分样本网格（显示原图、真实/预测标签）
@torch.no_grad()
def show_misclassified_grid(model,
                            data_loader,
                            num_samples: int = 25,
                            denormalize: bool = True):
    """
    num_samples: 展示的错分个数（尽量凑够）
    denormalize: 如果你在 ToTensor 后做了 Normalize((0.5,), (0.5,)),
                 这里做一次反变换，把 [-1,1] 显示回 [0,1]
    """
    device = next(model.parameters()).device
    model.eval()
    mis_imgs, mis_true, mis_pred = [], [], []

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        B = x.size(0)
        preds = model.classify(x.view(B, -1))

        mask = (preds != y)
        if mask.any():
            idxs = torch.where(mask)[0]
            for i in idxs:
                mis_imgs.append(x[i].detach().cpu())   # (1, H, W)
                mis_true.append(int(y[i].item()))
                mis_pred.append(int(preds[i].item()))
                if len(mis_imgs) >= num_samples:
                    break
        if len(mis_imgs) >= num_samples:
            break

    if len(mis_imgs) == 0:
        print("No misclassified samples found on this split. 🎉")
        return

    # 反归一化回 [0,1] 方便显示
    if denormalize:
        mis_imgs = [(img * 0.5 + 0.5).clamp(0, 1) for img in mis_imgs]

    cols = int(math.ceil(math.sqrt(len(mis_imgs))))
    rows = int(math.ceil(len(mis_imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows))
    axes = np.array(axes).reshape(rows, cols)

    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        ax.axis("off")
        if k < len(mis_imgs):
            img = mis_imgs[k].squeeze(0).numpy()  # (H, W)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"T:{mis_true[k]} / P:{mis_pred[k]}", fontsize=9)
    fig.suptitle("Misclassified Samples", y=0.98)
    plt.tight_layout()
    plt.show()


# 3) 指定层的 goodness 直方图（正/负样本）
@torch.no_grad()
def plot_goodness_histogram(model,
                            data_loader,
                            layer_index: int = 0,
                            num_batches: int = 1,
                            bins: int = 40):
    """
    layer_index: 选择第几层的 goodness（0-based）
    num_batches: 从 data_loader 取多少个 batch 聚合绘图
    """
    device = next(model.parameters()).device
    model.eval()

    gpos_all, gneg_all = [], []
    it = iter(data_loader)
    taken = 0
    while taken < num_batches:
        try:
            x, y = next(it)
        except StopIteration:
            break
        taken += 1

        x = x.to(device)
        y = y.to(device)
        x_flat = x.view(x.size(0), -1)

        # 构造正/负输入（与训练一致）
        x_pos0, x_neg0 = make_pos_neg_pairs(x_flat, y, model.num_classes)

        # 通过到指定层的输入：若层索引 > 0，需要先过前几层
        h_pos, h_neg = x_pos0, x_neg0
        for j in range(layer_index):
            h_pos = model.layers[j](h_pos)
            h_neg = model.layers[j](h_neg)

        # 在该层上前向一次，算 goodness
        h_pos = model.layers[layer_index](h_pos)
        h_neg = model.layers[layer_index](h_neg)

        g_pos = (h_pos ** 2).mean(dim=1).cpu().numpy()
        g_neg = (h_neg ** 2).mean(dim=1).cpu().numpy()

        gpos_all.append(g_pos)
        gneg_all.append(g_neg)

    if not gpos_all:
        print("No batches available.")
        return

    gpos = np.concatenate(gpos_all, axis=0)
    gneg = np.concatenate(gneg_all, axis=0)

    plt.figure(figsize=(6, 4))
    plt.hist(gpos, bins=bins, alpha=0.6, label="Positive")
    plt.hist(gneg, bins=bins, alpha=0.6, label="Negative")
    plt.xlabel("Goodness")
    plt.ylabel("Count")
    plt.title(f"Goodness Distribution (Layer {layer_index+1})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 4) 指定层激活的 t-SNE 可视化
@torch.no_grad()
def plot_tsne_activations(model,
                          data_loader,
                          layer_index: int = -1,
                          sample_size: int = 2000,
                          perplexity: float = 30.0,
                          random_state: int = 0):
    """
    layer_index: -1 表示最后一层；否则选定 0-based 层索引
    sample_size: 从测试集中抽样的样本数（太大会很慢）
    """
    device = next(model.parameters()).device
    model.eval()

    xs, ys = [], []
    cnt = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        xs.append(x)
        ys.append(y)
        cnt += x.size(0)
        if cnt >= sample_size:
            break

    x = torch.cat(xs, dim=0)[:sample_size]
    y = torch.cat(ys, dim=0)[:sample_size]
    x_flat = x.view(x.size(0), -1)

    # 仅用“正样本构造”的输入，以匹配推断使用
    x_pos, _ = make_pos_neg_pairs(x_flat, y, model.num_classes)

    # 前向到指定层
    if layer_index < 0:
        layer_index = len(model.layers) - 1

    h = x_pos
    for j in range(layer_index + 1):
        h = model.layers[j](h)

    H = h.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    H2 = tsne.fit_transform(H)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(H2[:, 0], H2[:, 1], c=y_np, cmap="tab10", s=10)
    cbar = plt.colorbar(sc, ticks=list(range(int(y_np.max())+1)))
    cbar.ax.set_ylabel("Class")
    plt.title(f"t-SNE of Activations (Layer {layer_index+1})")
    plt.tight_layout()
    plt.show()


# -------------------- 显示/保存用的小工具 --------------------
def _to_display_range(x: torch.Tensor) -> torch.Tensor:
    """
    训练时我们把像素放在 [-1,1]，显示时转回 [0,1]
    x: (B, 1, H, W) 或 (B, H, W)
    """
    y = (x + 1.0) / 2.0
    return y.clamp(0, 1)

def save_image_grid(tensor: torch.Tensor, nrow: int, save_path: str):
    """
    tensor: (B, 1, H, W) in [-1, 1]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid = make_grid(_to_display_range(tensor), nrow=nrow, padding=2)
    plt.figure(figsize=(nrow * 1.6, math.ceil(tensor.size(0) / nrow) * 1.6))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def show_image_grid(tensor: torch.Tensor, nrow: int, title: str = None):
    grid = make_grid(_to_display_range(tensor), nrow=nrow, padding=2)
    plt.figure(figsize=(nrow * 1.6, math.ceil(tensor.size(0) / nrow) * 1.6))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
    plt.show()


# -------------------- 正则：Total Variation (TV) --------------------
def total_variation(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    x: (B, D) in [-1,1]，把它 reshape 成 (B,1,H,W) 做 TV
    """
    B, D = x.shape
    img = x.view(B, 1, height, width)
    tv_h = (img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2).mean()
    tv_w = (img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2).mean()
    return tv_h + tv_w


# -------------------- 计算 FF 得分（各层 goodness 之和） --------------------
@torch.no_grad()
def ff_class_score(model, x_flat: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """
    仅用于评估：对每个样本 c，拼 one-hot，经所有层，返回 goodness 总和。
    x_flat: (B, D)
    label_ids: (B,) int64 in [0, num_classes)
    """
    B = x_flat.size(0)
    onehot = F.one_hot(label_ids, num_classes=model.num_classes).float().to(x_flat.device)
    xc = torch.cat([x_flat, onehot], dim=1)
    hs = model.forward_through_layers(xc)
    score = torch.stack([FFLayer.goodness(h) for h in hs], dim=1).sum(dim=1)
    return score


# -------------------- 生成：对输入做梯度上升，最大化 goodness --------------------
def generate_ff_images(model,
                       target_labels,
                       steps: int = 200,
                       lr: float = 0.1,
                       tv_weight: float = 0.001,
                       l2_weight: float = 0.0,
                       init: str = "noise",
                       img_h: int = 28,
                       img_w: int = 28,
                       verbose: bool = True):
    """
    参数：
      - model: 训练好的 FFNet
      - target_labels: 长度为 B 的 list/ndarray/LongTensor，表示要生成的类别
      - steps: 梯度上升步数
      - lr: 学习率（建议 0.05~0.2 之间尝试）
      - tv_weight: TV 正则强度（去噪/平滑）
      - l2_weight: L2 正则强度（防发散，可设 0）
      - init: 初始化方式 {"noise", "zeros", "gaussian"}
      - img_h, img_w: 图像尺寸（MNIST 为 28×28）

    返回：
      - imgs: (B, 1, H, W) in [-1, 1]
      - history (可选): 记录每步的平均 score（便于画收敛曲线）
    """
    device = next(model.parameters()).device
    model.eval()

    if isinstance(target_labels, torch.Tensor):
        labels = target_labels.to(device).long()
    else:
        labels = torch.tensor(target_labels, device=device, dtype=torch.long)

    B = labels.size(0)
    D = img_h * img_w

    # 初始化输入
    if init == "noise":
        x = torch.empty(B, D, device=device).uniform_(-1.0, 1.0)
    elif init == "gaussian":
        x = torch.randn(B, D, device=device).clamp_(-2.0, 2.0) / 2.0
        x = x.tanh()  # 大致落在 [-1,1]
    elif init == "zeros":
        x = torch.zeros(B, D, device=device)
    else:
        raise ValueError("init must be one of {'noise','gaussian','zeros'}")

    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    score_trace = []

    for t in range(steps):
        optimizer.zero_grad()

        # 拼 one-hot，前向通过所有层，累计 goodness
        onehot = F.one_hot(labels, num_classes=model.num_classes).float()
        inp = torch.cat([x, onehot], dim=1)
        hs = model.forward_through_layers(inp)
        score = torch.stack([FFLayer.goodness(h) for h in hs], dim=1).sum(dim=1)   # (B,)

        # 正则
        tv = total_variation(x, img_h, img_w) if tv_weight > 0 else x.new_tensor(0.0)
        l2 = x.pow(2).mean() if l2_weight > 0 else x.new_tensor(0.0)

        # 我们做“最大化”，所以用负号当作损失
        loss = -(score.mean() - tv_weight * tv - l2_weight * l2)
        loss.backward()
        optimizer.step()

        # 将像素保持在训练域 [-1,1]
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)

        score_trace.append(score.mean().item())
        if verbose and (t % max(1, steps // 10) == 0 or t == steps - 1):
            print(f"[Gen] step {t+1:4d}/{steps}, avg_score={score_trace[-1]:.4f}, "
                  f"tv={tv.item():.5f}, l2={l2.item():.5f}")

    imgs = x.view(B, 1, img_h, img_w).detach()
    return imgs, score_trace


# -------------------- 一键可视化（显示+保存） --------------------
def visualize_generation(model,
                         classes_to_generate=None,
                         per_class: int = 1,
                         steps: int = 200,
                         lr: float = 0.1,
                         tv_weight: float = 0.001,
                         l2_weight: float = 0.0,
                         img_h: int = 28,
                         img_w: int = 28,
                         save_path: str = "./ff_generations/ff_gen.png",
                         show: bool = True):
    """
    为若干类别各生成 per_class 张，网格显示并保存。
    """
    if classes_to_generate is None:
        classes_to_generate = list(range(model.num_classes))

    labels = []
    for c in classes_to_generate:
        labels.extend([c] * per_class)

    imgs, trace = generate_ff_images(
        model,
        target_labels=labels,
        steps=steps,
        lr=lr,
        tv_weight=tv_weight,
        l2_weight=l2_weight,
        init="noise",
        img_h=img_h,
        img_w=img_w,
        verbose=True
    )

    nrow = per_class
    title = f"FF generations | steps={steps}, lr={lr}, tv={tv_weight}, l2={l2_weight}"
    if show:
        show_image_grid(imgs, nrow=nrow, title=title)
    save_image_grid(imgs, nrow=nrow, save_path=save_path)
    print(f"[Saved] {save_path}")

    return imgs, trace


if __name__ == "__main__":
    main()