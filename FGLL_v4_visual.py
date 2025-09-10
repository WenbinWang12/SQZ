import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import copy

# 设置随机种子以确保可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 创建目录保存可视化结果
os.makedirs('layer_visualizations', exist_ok=True)
os.makedirs('feature_visualizations', exist_ok=True)

# 超参数
batch_size = 256
epochs = 100  # 减少epoch数以便快速测试
learning_rate = 0.00003
perturb_coef = 1e-3
alpha = 0.1
grad_clip = 1.0
visualization_interval = 5  # 每隔几个epoch可视化一次

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("开始下载数据集...")
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
print("训练数据集下载完成")
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
print("测试数据集加载完成")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
print("训练DataLoader创建完成")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
print("测试DataLoader创建完成")

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# 定义ResNet模型
class ResNetFGLLNet(nn.Module):
    def __init__(self):
        super(ResNetFGLLNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.aux1 = nn.Linear(512, 10)
        self.aux2 = nn.Linear(128, 10)
        
        # 注册钩子来捕获中间层输出
        self.activations = {}
        
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def register_hooks(self):
        # 为每一层注册前向钩子
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        self.conv1.register_forward_hook(get_activation('conv1'))
        self.layer1.register_forward_hook(get_activation('layer1'))
        self.layer2.register_forward_hook(get_activation('layer2'))
        self.layer3.register_forward_hook(get_activation('layer3'))
        self.fc1.register_forward_hook(get_activation('fc1'))
        self.fc2.register_forward_hook(get_activation('fc2'))

    def forward(self, x):
        # 清除之前的激活
        self.activations = {}
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        
        aux1_out = self.aux1(h1.detach())
        aux2_out = self.aux2(h2.detach())
        
        return out, aux1_out, aux2_out

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 特征可视化函数
def visualize_features(model, epoch, test_loader):
    model.eval()
    
    # 获取一些测试样本
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[:10].to(device)  # 只取前10个样本
    
    # 注册钩子捕获激活
    model.register_hooks()
    
    # 前向传播
    with torch.no_grad():
        outputs, _, _ = model(images)
    
    # 可视化每一层的特征
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'fc1', 'fc2']
    
    for layer_name in layer_names:
        if layer_name not in model.activations:
            continue
            
        activations = model.activations[layer_name]
        
        # 对于卷积层，可视化特征图
        if len(activations.shape) == 4:  # 卷积层输出 [B, C, H, W]
            # 选择第一个样本的特征图
            sample_activations = activations[0]
            
            # 只显示前16个通道
            num_channels = min(16, sample_activations.shape[0])
            sample_activations = sample_activations[:num_channels]
            
            # 创建网格
            grid = make_grid(sample_activations.unsqueeze(1), nrow=4, normalize=True, pad_value=1)
            
            # 绘制特征图
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='viridis')
            plt.title(f'{layer_name} Feature Maps at Epoch {epoch}')
            plt.axis('off')
            plt.savefig(f'feature_visualizations/{layer_name}_epoch_{epoch}.png')
            plt.close()
            
            # 绘制特征图统计信息
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.hist(activations.flatten().cpu().numpy(), bins=50)
            plt.title(f'{layer_name} Activation Distribution')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            channel_means = activations.mean(dim=(0, 2, 3)).cpu().numpy()
            plt.bar(range(len(channel_means)), channel_means)
            plt.title(f'{layer_name} Channel Means')
            plt.xlabel('Channel')
            plt.ylabel('Mean Activation')
            
            plt.tight_layout()
            plt.savefig(f'feature_visualizations/{layer_name}_stats_epoch_{epoch}.png')
            plt.close()
        
        # 对于全连接层，可视化激活分布
        elif len(activations.shape) == 2:  # 全连接层输出 [B, D]
            # 绘制激活分布
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.hist(activations.flatten().cpu().numpy(), bins=50)
            plt.title(f'{layer_name} Activation Distribution')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            neuron_means = activations.mean(dim=0).cpu().numpy()
            plt.bar(range(min(50, len(neuron_means))), neuron_means[:50])
            plt.title(f'{layer_name} Neuron Means (first 50)')
            plt.xlabel('Neuron')
            plt.ylabel('Mean Activation')
            
            plt.tight_layout()
            plt.savefig(f'feature_visualizations/{layer_name}_stats_epoch_{epoch}.png')
            plt.close()
    
    model.train()

# 激活最大化函数 - 可视化每一层学习到的特征
def visualize_learned_features(model, epoch, layer_names=['conv1', 'layer1', 'layer2', 'layer3']):
    model.eval()
    
    for layer_name in layer_names:
        # 获取指定层的引用
        if layer_name == 'conv1':
            layer = model.conv1
        elif layer_name == 'layer1':
            layer = model.layer1[0].conv1  # 取第一个残差块的第一个卷积层
        elif layer_name == 'layer2':
            layer = model.layer2[0].conv1
        elif layer_name == 'layer3':
            layer = model.layer3[0].conv1
        else:
            continue
        
        # 为每个滤波器创建输入图像
        num_filters = min(16, layer.out_channels)  # 只可视化前16个滤波器
        synthesized_images = []
        
        for filter_idx in range(num_filters):
            # 创建一个随机初始化的输入图像
            input_img = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
            
            # 优化输入图像以最大化特定滤波器的激活
            optimizer = optim.Adam([input_img], lr=0.1)
            
            for i in range(100):  # 100次迭代
                optimizer.zero_grad()
                
                # 前向传播到目标层
                if layer_name == 'conv1':
                    output = layer(input_img)
                else:
                    # 对于更深的层，需要先通过前面的层
                    x = model.relu(model.bn1(model.conv1(input_img)))
                    if layer_name != 'layer1':
                        x = model.layer1(x)
                    if layer_name != 'layer2' and layer_name != 'layer1':
                        x = model.layer2(x)
                    if layer_name != 'layer3' and layer_name != 'layer2' and layer_name != 'layer1':
                        x = model.layer3(x)
                    output = layer(x)
                
                # 最大化特定滤波器的激活
                loss = -output[0, filter_idx].mean()
                loss.backward()
                optimizer.step()
            
            # 保存合成的图像
            synthesized_images.append(input_img.detach().cpu())
        
        # 创建网格显示所有合成的图像
        grid = make_grid(torch.cat(synthesized_images), nrow=4, normalize=True)
        
        # 绘制合成的图像
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='viridis')
        plt.title(f'{layer_name} Learned Features at Epoch {epoch}')
        plt.axis('off')
        plt.savefig(f'feature_visualizations/{layer_name}_learned_features_epoch_{epoch}.png')
        plt.close()
    
    model.train()

# 可视化函数
def visualize_layers(model, epoch, test_loader):
    # 可视化特征
    visualize_features(model, epoch, test_loader)
    
    # 可视化学习到的特征
    visualize_learned_features(model, epoch)
    
    # 可视化权重
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Layer Weights at Epoch {epoch}')
    
    # 可视化conv1权重
    conv1_weights = model.conv1.weight.detach().cpu()
    # 只显示前16个滤波器
    conv1_weights = conv1_weights[:16]
    # 重新排列维度为 (B, C, H, W)
    conv1_weights = conv1_weights.view(-1, 1, 3, 3)
    grid = make_grid(conv1_weights, nrow=4, normalize=True)
    axes[0, 0].imshow(grid.permute(1, 2, 0), cmap='viridis')
    axes[0, 0].set_title('Conv1 Weights (first 16 filters)')
    axes[0, 0].axis('off')
    
    # 可视化layer1第一个卷积层的权重
    layer1_weights = model.layer1[0].conv1.weight.detach().cpu()
    # 只显示前16个滤波器的第一个输入通道
    layer1_weights = layer1_weights[:16, :1]
    # 重新排列维度为 (B, C, H, W)
    layer1_weights = layer1_weights.view(-1, 1, 3, 3)
    grid = make_grid(layer1_weights, nrow=4, normalize=True)
    axes[0, 1].imshow(grid.permute(1, 2, 0), cmap='viridis')
    axes[0, 1].set_title('Layer1 Conv1 Weights (first 16 filters)')
    axes[0, 1].axis('off')
    
    # 可视化fc1权重
    fc1_weights = model.fc1.weight.detach().cpu()
    axes[0, 2].hist(fc1_weights.flatten().numpy(), bins=50)
    axes[0, 2].set_title('FC1 Weight Distribution')
    axes[0, 2].set_xlabel('Weight Value')
    axes[0, 2].set_ylabel('Frequency')
    
    # 可视化fc2权重
    fc2_weights = model.fc2.weight.detach().cpu()
    axes[1, 0].hist(fc2_weights.flatten().numpy(), bins=50)
    axes[1, 0].set_title('FC2 Weight Distribution')
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # 可视化fc3权重
    fc3_weights = model.fc3.weight.detach().cpu()
    axes[1, 1].hist(fc3_weights.flatten().numpy(), bins=50)
    axes[1, 1].set_title('FC3 Weight Distribution')
    axes[1, 1].set_xlabel('Weight Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # 可视化权重范数
    weight_norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_norms[name] = param.norm().item()
    
    axes[1, 2].bar(range(len(weight_norms)), list(weight_norms.values()))
    axes[1, 2].set_title('Weight Norms by Layer')
    axes[1, 2].set_ylabel('L2 Norm')
    axes[1, 2].set_xticks(range(len(weight_norms)))
    axes[1, 2].set_xticklabels(list(weight_norms.keys()), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'layer_visualizations/weights_epoch_{epoch}.png')
    plt.close()

# 训练函数
def train_forward_gradient(model, device, train_loader, epoch, test_loader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
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
        
        # 简化梯度计算过程，只计算关键层的梯度
        layer_names = [
            'conv1.weight', 'conv1.bias',
            'layer1.0.conv1.weight', 'layer1.0.conv1.bias',
            'layer2.0.conv1.weight', 'layer2.0.conv1.bias',
            'layer3.0.conv1.weight', 'layer3.0.conv1.bias',
            'fc1.weight', 'fc1.bias',
            'fc2.weight', 'fc2.bias',
            'fc3.weight', 'fc3.bias'
        ]
        
        # 只对部分层进行扰动和梯度计算
        for name in layer_names:
            if name not in dict(model.named_parameters()):
                continue
                
            # 应用扰动
            with torch.no_grad():
                param = dict(model.named_parameters())[name]
                original_value = original_params[name]
                param.data.copy_(original_value + perturb_coef * perturbations[name])
            
            # 前向传播计算损失
            output_perturbed, aux1_out_perturbed, aux2_out_perturbed = model(data)
            loss_global_perturbed = criterion(output_perturbed, target)
            loss_aux1_perturbed = criterion(aux1_out_perturbed, target)
            loss_aux2_perturbed = criterion(aux2_out_perturbed, target)
            
            # 计算梯度
            delta_global = (loss_global_perturbed - loss_global_original) / perturb_coef
            delta_aux1 = (loss_aux1_perturbed - loss_aux1_original) / perturb_coef
            delta_aux2 = (loss_aux2_perturbed - loss_aux2_original) / perturb_coef
            
            # 根据层级确定使用哪个梯度
            if 'conv1' in name or 'layer1' in name:
                gradients[name] = alpha * delta_global * perturbations[name] + (1 - alpha) * delta_aux1 * perturbations[name]
            elif 'layer2' in name:
                gradients[name] = alpha * delta_global * perturbations[name] + (1 - alpha) * delta_aux2 * perturbations[name]
            elif 'layer3' in name or 'fc' in name:
                gradients[name] = delta_global * perturbations[name]
            
            # 恢复原始参数
            with torch.no_grad():
                param.data.copy_(original_value)
        
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
    
    # 每隔一定epoch可视化层
    if epoch % visualization_interval == 0:
        visualize_layers(model, epoch, test_loader)
    
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

# 逐步构建模型
print("开始创建模型...")
model = ResNetFGLLNet()
print("模型创建完成")

print("尝试将模型移动到CPU...")
model = model.to('cpu')
print("模型已移动到CPU")

print("检查模型参数设备...")
print(f"模型参数设备: {next(model.parameters()).device}")

if torch.cuda.is_available():
    print("尝试将模型移动到GPU...")
    try:
        model = model.cuda()
        print("整个模型已移动到GPU")
    except Exception as e:
        print(f"模型移动失败: {e}")

# 检查模型是否真的在GPU上
print(f"模型已创建，参数设备: {next(model.parameters()).device}")

# 训练模型
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

start_time = time.time()

for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train_forward_gradient(model, device, train_loader, epoch, test_loader)
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
torch.save(model.state_dict(), "resnet_fgll_mnist_model.pth")
print("模型已保存到 resnet_fgll_mnist_model.pth")