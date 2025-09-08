import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 随机性与设备
# ======================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================
# 超参数（可按需调整）
# ======================
batch_size    = 64
epochs        = 50          # ResNet 更强，先用 50 确认稳定；OK 后再加大
learning_rate = 3e-5
perturb_coef  = 1e-3        # SPSA 扰动幅度 ε
grad_clip     = 1.0         # 全局范数裁剪（L2）
weight_decay  = 1e-4        # 解耦式 weight decay；置 0 可关闭

# ======================
# 数据
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)

# ======================
# ResNet 组件
# ======================
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNetMNIST(nn.Module):
    """
    轻量可加深的 ResNet：
      stem: 1x28x28 -> 32 通道
      layer1: 32 -> 64  (stride=2)  # 14x14
      layer2: 64 -> 128 (stride=2)  # 7x7
      layer3: 128 -> 128 (stride=1) # 7x7
      GAP + Linear(128 -> 10)
    通过 blocks_per_layer 控制深度，如 (2,2,2) / (3,3,3) 等。
    """
    def __init__(self, blocks_per_layer=(2, 2, 2), num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32,  64,  blocks_per_layer[0], stride=2)
        self.layer2 = self._make_layer(64,  128, blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(128, 128, blocks_per_layer[2], stride=1)
        self.head   = nn.Linear(128, num_classes)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)                 # (N, 32, 28, 28)
        x = self.layer1(x)               # (N, 64, 14, 14)
        x = self.layer2(x)               # (N, 128, 7, 7)
        x = self.layer3(x)               # (N, 128, 7, 7)
        x = F.adaptive_avg_pool2d(x, 1)  # (N, 128, 1, 1)
        x = torch.flatten(x, 1)          # (N, 128) —— 安全展开（避免 squeeze 把 batch 维挤没）
        # 可调试时启用：确保与 head 的 in_features 匹配
        # assert x.dim() == 2 and x.size(1) == self.head.in_features, f"got {tuple(x.shape)}"
        logits = self.head(x)            # (N, 10)
        return logits

model = ResNetMNIST(blocks_per_layer=(2, 2, 2)).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# ======================
# 损失
# ======================
criterion = nn.CrossEntropyLoss()

# ======================
# 工具函数：收集/还原参数、扰动、裁剪
# ======================
def clone_params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}

@torch.no_grad()
def apply_params(model, param_dict):
    for name, p in model.named_parameters():
        p.copy_(param_dict[name])

def rademacher_like(param):
    # ±1 分布；用 sign(randn) 稳定
    return torch.sign(torch.randn_like(param))

def global_clip_(grads_dict, max_norm):
    # 计算全局 L2 范数并按比例缩放
    total_sq = 0.0
    for g in grads_dict.values():
        total_sq += g.pow(2).sum().item()
    total_norm = np.sqrt(total_sq)
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / (total_norm + 1e-12)
        for k in grads_dict.keys():
            grads_dict[k].mul_(scale)

# ======================
# 训练（SPSA 两点前向梯度）
# ======================
def train_forward_gradient(model, device, train_loader, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 原始参数 θ
        theta = clone_params(model)

        # 为每个参数张量生成 Rademacher 扰动 Δ
        deltas = {name: rademacher_like(p) for name, p in model.named_parameters()}

        # θ + εΔ
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.add_(perturb_coef * deltas[name])
        output_plus = model(data)
        loss_plus = criterion(output_plus, target)

        # θ - εΔ
        with torch.no_grad():
            apply_params(model, theta)  # 先回到 θ
            for name, p in model.named_parameters():
                p.add_(-perturb_coef * deltas[name])
        output_minus = model(data)
        loss_minus = criterion(output_minus, target)

        # 恢复 θ
        with torch.no_grad():
            apply_params(model, theta)

        # SPSA 前向梯度估计：g_hat = ((L+ - L-) / (2ε)) * Δ
        coef = (loss_plus.item() - loss_minus.item()) / (2.0 * perturb_coef)
        grads = {name: deltas[name] * coef for name in deltas.keys()}

        # 全局范数裁剪
        global_clip_(grads, grad_clip)

        # 解耦式 weight decay（可选）
        with torch.no_grad():
            if weight_decay > 0:
                for name, p in model.named_parameters():
                    p.mul_(1.0 - learning_rate * weight_decay)

        # 参数更新：θ <- θ - η g_hat
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.add_(-learning_rate * grads[name])

        # 统计指标（使用更新后的参数）
        with torch.no_grad():
            logits = model(data)
            loss = criterion(logits, target)
            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total
    print(f"Epoch {epoch} Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)")
    return avg_loss, acc

# ======================
# 测试
# ======================
@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        test_loss += criterion(logits, target).item()
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    test_loss /= len(test_loader)
    acc = 100.0 * correct / total
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)")
    return test_loss, acc

# ======================
# 训练循环
# ======================
train_losses, train_accuracies = [], []
test_losses,  test_accuracies  = [], []

start_time = time.time()
for epoch in range(1, epochs + 1):
    tr_l, tr_a = train_forward_gradient(model, device, train_loader, epoch)
    te_l, te_a = test(model, device, test_loader)

    train_losses.append(tr_l)
    train_accuracies.append(tr_a)
    test_losses.append(te_l)
    test_accuracies.append(te_a)

end_time = time.time()
print(f"总训练时间: {end_time - start_time:.2f}秒")

# ======================
# 可视化
# ======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses,  label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies,  label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# ======================
# 保存模型
# ======================
torch.save(model.state_dict(), "resnet_spsa_mnist.pth")
print("模型已保存到 resnet_spsa_mnist.pth")
