import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import matplotlib.pyplot as plt
from model import SSNet
from Read_H5 import SSNetHDF5Dataset


class ComplexMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        loss_real = self.mse(input.real, target.real)
        loss_imag = self.mse(input.imag, target.imag)
        return loss_real + loss_imag


os.makedirs('output', exist_ok=True)

# ============================== 数据准备 =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temp_model = SSNet()
nfreq = temp_model.integral_kernel.nfreq
mz, mx = temp_model.integral_kernel.mz, temp_model.integral_kernel.mx
print(f"从模型获取到的维度信息: nfreq={nfreq}, mz={mz}, mx={mx}")

batch_size = 10
num_epochs = 80
lr = 0.001

print("准备训练数据...")

real_dataset = SSNetHDF5Dataset(
    xkai_h5_path="./output/hdf5_data/xkai_all.h5",
    etot_h5_path="./output/hdf5_data/etot_all.h5",
    nfreq=nfreq
)

X = torch.stack([real_dataset[i][0] for i in range(len(real_dataset))])
Y = torch.stack([real_dataset[i][1][:, 0, :, :] for i in range(len(real_dataset))])

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"训练数据准备完成，共 {len(dataset)} 个样本，批大小 {batch_size}")
print(f"Xkai维度: {X.shape}")
print(f"Etot维度: {Y.shape}")

# ============================== 模型初始化 =============================
print("初始化模型...")
model = temp_model.to(device)
print(f"模型已初始化，使用设备: {device}")

# ============================== 损失函数和优化器 =============================
criterion = ComplexMSELoss()
optimizer = optim.Adam(model.unet.parameters(), lr=lr)

# ============================== 训练循环 =============================
print("开始训练...")
epoch_losses = []
step_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for i, (inputs, targets) in enumerate(dataloader):
        # 转移数据到GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        pre_para, fin_para, Etot = model(inputs)

        # 损失计算
        loss = criterion(Etot, targets)

        # 反向传播+参数更新
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        step_losses.append(loss.item())

        # 打印中间结果
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # 记录每轮平均损失
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], 平均损失: {avg_loss:.4f}')

# ============================== 保存模型 =============================

torch.save(model.state_dict(), 'output/ssnet_k2etot_model.pth')
print("k→Etot模型已保存至 output/ssnet_k2etot_model.pth")

# ============================== 绘制损失曲线 =============================
plt.figure(figsize=(12, 5))

# 每步损失曲线
plt.subplot(1, 2, 1)
plt.plot(step_losses, label='Step Loss (k→Etot)')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Loss per Training Step (k→Etot)')
plt.legend()

# 每轮平均损失曲线
plt.subplot(1, 2, 2)
plt.plot(epoch_losses, label='Epoch Avg Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Loss per Epoch (k→Etot)')
plt.legend()

plt.tight_layout()
# 重命名损失曲线，避免覆盖
plt.savefig('output/loss_curve_k2etot.png')
print("k→Etot损失曲线已保存至 output/loss_curve_k2etot.png")
plt.show()
