import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np

# 设置随机种子以确保可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数
batch_size = 128
epochs = 10
learning_rate = 0.01
perturb_coef = 1e-5  # 前向梯度扰动系数
alpha = 0.5  # 全局损失权重

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
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        # 辅助分类器
        self.aux1 = nn.Linear(512, 10)
        self.aux2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        
        # 辅助输出
        aux1_out = self.aux1(h1)
        aux2_out = self.aux2(h2)
        
        return out, aux1_out, aux2_out

model = FGLLNet().to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练函数
def train_forward_gradient(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 为每一层参数生成随机扰动向量
        v1 = torch.randn_like(model.fc1.weight)
        v2 = torch.randn_like(model.fc2.weight)
        v3 = torch.randn_like(model.fc3.weight)
        v_aux1 = torch.randn_like(model.aux1.weight)
        v_aux2 = torch.randn_like(model.aux2.weight)
        
        # 原始前向传播
        output, aux1_out, aux2_out = model(data)
        loss_global_original = criterion(output, target)
        loss_aux1_original = criterion(aux1_out, target)
        loss_aux2_original = criterion(aux2_out, target)
        
        # 保存原始参数
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
        
        # 应用扰动并计算扰动损失
        with torch.no_grad():
            # 扰动第一层
            model.fc1.weight.data = original_params['fc1.weight'] + perturb_coef * v1
            model.fc1.bias.data = original_params['fc1.bias'] + perturb_coef * torch.randn_like(model.fc1.bias)
            
            # 扰动辅助分类器1
            model.aux1.weight.data = original_params['aux1.weight'] + perturb_coef * v_aux1
            model.aux1.bias.data = original_params['aux1.bias'] + perturb_coef * torch.randn_like(model.aux1.bias)
            
        # 前向传播计算扰动后的损失
        output_perturbed, aux1_out_perturbed, _ = model(data)
        loss_global_perturbed_fc1 = criterion(output_perturbed, target)
        loss_aux1_perturbed = criterion(aux1_out_perturbed, target)
        
        # 计算第一层和辅助分类器1的前向梯度
        delta_loss_global_fc1 = (loss_global_perturbed_fc1 - loss_global_original) / perturb_coef
        delta_loss_aux1 = (loss_aux1_perturbed - loss_aux1_original) / perturb_coef
        
        # 恢复原始参数
        for name, param in model.named_parameters():
            param.data = original_params[name]
        
        # 应用扰动到第二层
        with torch.no_grad():
            model.fc2.weight.data = original_params['fc2.weight'] + perturb_coef * v2
            model.fc2.bias.data = original_params['fc2.bias'] + perturb_coef * torch.randn_like(model.fc2.bias)
            
            # 扰动辅助分类器2
            model.aux2.weight.data = original_params['aux2.weight'] + perturb_coef * v_aux2
            model.aux2.bias.data = original_params['aux2.bias'] + perturb_coef * torch.randn_like(model.aux2.bias)
        
        # 前向传播计算扰动后的损失
        output_perturbed, _, aux2_out_perturbed = model(data)
        loss_global_perturbed_fc2 = criterion(output_perturbed, target)
        loss_aux2_perturbed = criterion(aux2_out_perturbed, target)
        
        # 计算第二层和辅助分类器2的前向梯度
        delta_loss_global_fc2 = (loss_global_perturbed_fc2 - loss_global_original) / perturb_coef
        delta_loss_aux2 = (loss_aux2_perturbed - loss_aux2_original) / perturb_coef
        
        # 恢复原始参数
        for name, param in model.named_parameters():
            param.data = original_params[name]
        
        # 应用扰动到第三层
        with torch.no_grad():
            model.fc3.weight.data = original_params['fc3.weight'] + perturb_coef * v3
            model.fc3.bias.data = original_params['fc3.bias'] + perturb_coef * torch.randn_like(model.fc3.bias)
        
        # 前向传播计算扰动后的损失
        output_perturbed, _, _ = model(data)
        loss_global_perturbed_fc3 = criterion(output_perturbed, target)
        
        # 计算第三层的前向梯度
        delta_loss_global_fc3 = (loss_global_perturbed_fc3 - loss_global_original) / perturb_coef
        
        # 恢复原始参数
        for name, param in model.named_parameters():
            param.data = original_params[name]
        
        # 使用前向梯度更新权重
        with torch.no_grad():
            # 更新第一层
            model.fc1.weight -= learning_rate * (alpha * delta_loss_global_fc1 * v1 + (1 - alpha) * delta_loss_aux1 * v1)
            model.fc1.bias -= learning_rate * (alpha * delta_loss_global_fc1 * torch.randn_like(model.fc1.bias) + 
                                              (1 - alpha) * delta_loss_aux1 * torch.randn_like(model.fc1.bias))
            
            # 更新辅助分类器1
            model.aux1.weight -= learning_rate * delta_loss_aux1 * v_aux1
            model.aux1.bias -= learning_rate * delta_loss_aux1 * torch.randn_like(model.aux1.bias)
            
            # 更新第二层
            model.fc2.weight -= learning_rate * (alpha * delta_loss_global_fc2 * v2 + (1 - alpha) * delta_loss_aux2 * v2)
            model.fc2.bias -= learning_rate * (alpha * delta_loss_global_fc2 * torch.randn_like(model.fc2.bias) + 
                                              (1 - alpha) * delta_loss_aux2 * torch.randn_like(model.fc2.bias))
            
            # 更新辅助分类器2
            model.aux2.weight -= learning_rate * delta_loss_aux2 * v_aux2
            model.aux2.bias -= learning_rate * delta_loss_aux2 * torch.randn_like(model.aux2.bias)
            
            # 更新第三层
            model.fc3.weight -= learning_rate * alpha * delta_loss_global_fc3 * v3
            model.fc3.bias -= learning_rate * alpha * delta_loss_global_fc3 * torch.randn_like(model.fc3.bias)
        
        # 计算训练指标
        output, _, _ = model(data)
        train_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {train_loss / (batch_idx + 1):.6f}')
    
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
    train_loss, train_accuracy = train_forward_gradient(model, device, train_loader, None, epoch)
    test_loss, test_accuracy = test(model, device, test_loader)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

end_time = time.time()
print(f"总训练时间: {end_time - start_time:.2f}秒")

# 保存模型
torch.save(model.state_dict(), "fgll_mnist_model.pth")
print("模型已保存到 fgll_mnist_model.pth")