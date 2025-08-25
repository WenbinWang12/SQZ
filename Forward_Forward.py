# pip install torch torchvision  (如本地未安装)
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
    """
    逐层训练：第 l 层训练时，把前面 l-1 层固定为“特征提取器”并只前向，
    只对第 l 层做局部判别训练。
    """
    device = cfg.device
    model.to(device)

    for li, layer in enumerate(model.layers):
        # 仅训练当前层参数
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

                # 展平图像到向量
                x_flat = x.view(x.size(0), -1)

                # 前置层（若 li>0 ）仅做前向得到“中间表示”
                if li > 0:
                    with torch.no_grad():
                        # 注意：前置层的输入也要拼 one-hot（因为我们网络第一层的输入定义如此）
                        x_pos, _ = make_pos_neg_pairs(x_flat, y, model.num_classes)
                        h = x_pos
                        for j in range(li):
                            h = model.layers[j](h)
                        # 此时 h 作为“当前层”的输入
                        x_flat = h

                # 针对当前层构造正负样本（拼接 one-hot）
                x_pos, x_neg = make_pos_neg_pairs(x_flat, y, model.num_classes)

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
def get_dataloaders(batch_size=256) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),                # [0,1]
        transforms.Lambda(lambda t: t*2-1),   # 映射到 [-1,1]，更稳定
    ])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def main():
    torch.manual_seed(0)
    train_loader, test_loader = get_dataloaders(batch_size=256)

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

if __name__ == "__main__":
    main()
