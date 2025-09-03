import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数 - 调整这些参数以解决NaN问题
batch_size = 64  # 减小批量大小
epochs = 2000       # 减少epoch数以便调试
learning_rate = 0.00003  # 降低学习率
perturb_coef = 1e-3    # 增大扰动系数
alpha = 0.1            # 增加全局损失权重
grad_clip = 1.0        # 添加梯度裁剪

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# 定义带有辅助分类器的网络
class FGLLNet(nn.Module):
    def __init__(self):
        super(FGLLNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # 辅助分类器 - 使用更简单的结构
        self.aux1 = nn.Linear(256, 10)
        self.aux2 = nn.Linear(128, 10)
        
        # 初始化权重 - 添加适当的初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = x.view(-1, 784)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        
        # 辅助输出
        aux1_out = self.aux1(h1.detach())  # 分离辅助梯度
        aux2_out = self.aux2(h2.detach())  # 分离辅助梯度
        
        return out, aux1_out, aux2_out

model = FGLLNet().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练函数 - 改进版本
def train_forward_gradient(model, device, train_loader, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 保存原始参数
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # 为每一层参数生成随机扰动向量
        perturbations = {}
        for name, param in model.named_parameters():
            perturbations[name] = torch.randn_like(param)
        
        # 原始前向传播
        output, aux1_out, aux2_out = model(data)
        loss_global_original = criterion(output, target)
        loss_aux1_original = criterion(aux1_out, target)
        loss_aux2_original = criterion(aux2_out, target)
        
        # 计算每一层的前向梯度
        gradients = {}
        
        # 第一层和辅助分类器1
        with torch.no_grad():
            for name in ['fc1.weight', 'fc1.bias', 'aux1.weight', 'aux1.bias']:
                model.state_dict()[name].copy_(original_params[name] + perturb_coef * perturbations[name])
        
        output_perturbed, aux1_out_perturbed, _ = model(data)
        loss_global_perturbed = criterion(output_perturbed, target)
        loss_aux1_perturbed = criterion(aux1_out_perturbed, target)
        
        # 计算梯度
        delta_global = (loss_global_perturbed - loss_global_original) / perturb_coef
        delta_aux1 = (loss_aux1_perturbed - loss_aux1_original) / perturb_coef
        
        # 保存梯度
        gradients['fc1.weight'] = alpha * delta_global * perturbations['fc1.weight'] + (1 - alpha) * delta_aux1 * perturbations['fc1.weight']
        gradients['fc1.bias'] = alpha * delta_global * perturbations['fc1.bias'] + (1 - alpha) * delta_aux1 * perturbations['fc1.bias']
        gradients['aux1.weight'] = delta_aux1 * perturbations['aux1.weight']
        gradients['aux1.bias'] = delta_aux1 * perturbations['aux1.bias']
        
        # 恢复原始参数
        for name in ['fc1.weight', 'fc1.bias', 'aux1.weight', 'aux1.bias']:
            model.state_dict()[name].copy_(original_params[name])
        
        # 第二层和辅助分类器2
        with torch.no_grad():
            for name in ['fc2.weight', 'fc2.bias', 'aux2.weight', 'aux2.bias']:
                model.state_dict()[name].copy_(original_params[name] + perturb_coef * perturbations[name])
        
        output_perturbed, _, aux2_out_perturbed = model(data)
        loss_global_perturbed = criterion(output_perturbed, target)
        loss_aux2_perturbed = criterion(aux2_out_perturbed, target)
        
        # 计算梯度
        delta_global = (loss_global_perturbed - loss_global_original) / perturb_coef
        delta_aux2 = (loss_aux2_perturbed - loss_aux2_original) / perturb_coef
        
        # 保存梯度
        gradients['fc2.weight'] = alpha * delta_global * perturbations['fc2.weight'] + (1 - alpha) * delta_aux2 * perturbations['fc2.weight']
        gradients['fc2.bias'] = alpha * delta_global * perturbations['fc2.bias'] + (1 - alpha) * delta_aux2 * perturbations['fc2.bias']
        gradients['aux2.weight'] = delta_aux2 * perturbations['aux2.weight']
        gradients['aux2.bias'] = delta_aux2 * perturbations['aux2.bias']
        
        # 恢复原始参数
        for name in ['fc2.weight', 'fc2.bias', 'aux2.weight', 'aux2.bias']:
            model.state_dict()[name].copy_(original_params[name])
        
        # 第三层
        with torch.no_grad():
            for name in ['fc3.weight', 'fc3.bias']:
                model.state_dict()[name].copy_(original_params[name] + perturb_coef * perturbations[name])
        
        output_perturbed, _, _ = model(data)
        loss_global_perturbed = criterion(output_perturbed, target)
        
        # 计算梯度
        delta_global = (loss_global_perturbed - loss_global_original) / perturb_coef
        
        # 保存梯度
        gradients['fc3.weight'] = alpha * delta_global * perturbations['fc3.weight']
        gradients['fc3.bias'] = alpha * delta_global * perturbations['fc3.bias']
        
        # 恢复原始参数
        for name in ['fc3.weight', 'fc3.bias']:
            model.state_dict()[name].copy_(original_params[name])
        
        # 应用梯度裁剪
        for name in gradients:
            torch.nn.utils.clip_grad_norm_([gradients[name]], grad_clip)
        
        # 使用前向梯度更新权重
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in gradients:
                    param -= learning_rate * gradients[name]
        
        # 计算训练指标
        output, _, _ = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_loss /= len(train_loader)
    train_accuracy = 100. * correct / total
    
    print(f'Epoch: {epoch} Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({train_accuracy:.2f}%)')
    
    return train_loss, train_accuracy

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_accuracy:.2f}%)')
    
    return test_loss, test_accuracy

# 训练模型
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

start_time = time.time()

for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train_forward_gradient(model, device, train_loader, epoch)
    test_loss, test_accuracy = test(model, device, test_loader)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

end_time = time.time()
print(f"总训练时间: {end_time - start_time:.2f}秒")

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# 保存模型
torch.save(model.state_dict(), "fgll_mnist_model.pth")
print("模型已保存到 fgll_mnist_model.pth")