import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
from test_model import UNet  # 导入模型
from Read_H5 import SSNetHDF5Dataset  # 导入您的数据集类

# 复数MSE损失函数
class ComplexMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        loss_real = self.mse(input.real, target.real)
        loss_imag = self.mse(input.imag, target.imag)
        return loss_real + loss_imag


# ============================== 配置参数 =============================
os.makedirs('output', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据路径与维度配置（根据实际数据调整）
XKAI_H5_PATH = "./output/hdf5_data/xkai_all.h5"
ETOT_H5_PATH = "./output/hdf5_data/etot_all.h5"
nfreq = 1  # 与数据集匹配的频率数

# 训练超参数
batch_size = 10
num_epochs = 80
lr = 0.001
bilinear = True  # U-Net上采样方式

# ============================== 数据准备 =============================
print("加载数据集...")
# 初始化自定义数据集
dataset = SSNetHDF5Dataset(
    xkai_h5_path=XKAI_H5_PATH,
    etot_h5_path=ETOT_H5_PATH,
    nfreq=nfreq
)

# 批量加载样本并整理形状
X = torch.stack([dataset[i][0] for i in range(len(dataset))])
Y = torch.stack([dataset[i][1] for i in range(len(dataset))])

# 适配U-Net输入格式 [B, C, H, W]
# 如需筛选传输方向，可取消注释：
# Y = Y[:, :, 0, :, :]  # 取第一个传输方向（与原代码逻辑一致）

print(f"数据加载完成：")
print(f"  - 总样本数：{len(dataset)}")
print(f"  - Xkai维度：{X.shape}")
print(f"  - Etot维度：{Y.shape}")

# 构建DataLoader
tensor_dataset = TensorDataset(X, Y)
dataloader = DataLoader(
    tensor_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True if torch.cuda.is_available() else False
)

# ============================== 模型初始化 =============================
# 获取通道数和空间尺寸（从数据集自动适配）
in_channels = X.shape[1]  # 输入通道数
out_channels = Y.shape[1]  # 输出通道数
mz, mx = X.shape[-2], X.shape[-1]  # 空间尺寸

print(f"\n初始化模型：")
print(f"  - 输入通道数：{in_channels}")
print(f"  - 输出通道数：{out_channels}")
print(f"  - 空间尺寸：{mz}×{mx}")

model = UNet(
    in_channels=in_channels,
    out_channels=out_channels,
    bilinear=bilinear
).to(device)

# ============================== 优化器与损失函数 =============================
criterion = ComplexMSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 学习率调度

# ============================== 训练循环 =============================
print("\n开始训练...")
epoch_losses = []
step_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 损失计算与反向传播
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        step_losses.append(loss.item())

        # 打印中间结果
        if (i + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # 学习率更新
    scheduler.step()

    # 记录轮次平均损失
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], 平均损失: {avg_loss:.4f}\n')

# ============================== 模型保存 =============================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'avg_loss': avg_loss,
    'in_channels': in_channels,
    'out_channels': out_channels
}, 'output/unet_complex_model.pth')
print("模型已保存至 output/unet_complex_model.pth")

# ============================== 损失曲线绘制 =============================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 5))

# 每步损失
plt.subplot(1, 2, 1)
plt.plot(step_losses, label='步损失', color='blue', alpha=0.7)
plt.xlabel('训练步数')
plt.ylabel('损失值')
plt.title('训练步损失曲线')
plt.legend()
plt.grid(alpha=0.3)

# 每轮平均损失
plt.subplot(1, 2, 2)
plt.plot(epoch_losses, label='轮次平均损失', color='orange', linewidth=2)
plt.xlabel('训练轮次')
plt.ylabel('平均损失值')
plt.title('轮次平均损失曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('output/unet_loss_curve.png', dpi=300, bbox_inches='tight')
print("损失曲线已保存至 output/unet_loss_curve.png")
plt.show()

# ============================== 训练日志保存 =============================
with open('output/training_log.txt', 'w', encoding='utf-8') as f:
    f.write(f"训练配置：\n")
    f.write(f"  - 设备：{device}\n")
    f.write(f"  - 批大小：{batch_size}\n")
    f.write(f"  - 训练轮次：{num_epochs}\n")
    f.write(f"  - 初始学习率：{lr}\n")
    f.write(f"  - 最终学习率：{optimizer.param_groups[0]['lr']:.6f}\n")
    f.write(f"  - 输入维度：{X.shape}\n")
    f.write(f"  - 输出维度：{Y.shape}\n")
    f.write(f"  - 模型参数总数：{sum(p.numel() for p in model.parameters()):,}\n")
    f.write(f"\n训练结果：\n")
    f.write(f"  - 最终轮次平均损失：{avg_loss:.6f}\n")

print("训练日志已保存至 output/training_log.txt")