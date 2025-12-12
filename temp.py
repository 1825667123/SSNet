import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import matplotlib.pyplot as plt
import pickle  # 用于保存/加载损失记录
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

# ============================== 续训配置 =============================
MODEL_PATH = 'output/ssnet_model.pth'
LOSS_RECORD_PATH = 'output/loss_records.pkl'  # 保存损失记录的文件
RESUME_TRAINING = True  # 是否开启断点续训

# ============================== 数据准备 =============================
print("CUDA可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("GPU数量:", torch.cuda.device_count())
print("初始化模型以获取维度信息...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temp_model = SSNet()
nfreq = temp_model.integral_kernel.nfreq
mz, mx = temp_model.integral_kernel.mz, temp_model.integral_kernel.mx
print(f"从模型获取到的维度信息: nfreq={nfreq}, mz={mz}, mx={mx}")

batch_size = 10
num_epochs = 3
lr = 0.001

print("准备训练数据...")

real_dataset = SSNetHDF5Dataset(
    xkai_h5_path="./output/hdf5_data/xkai_all.h5",
    etot_h5_path="./output/hdf5_data/etot_all.h5",
    nfreq=nfreq
)

X = torch.stack([real_dataset[i][0] for i in range(len(real_dataset))])
Y = torch.stack([real_dataset[i][1][:, 0, :, :] for i in range(len(real_dataset))])  # Y.shape=(N, 1, 20, 20)

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"训练数据准备完成，共 {len(dataset)} 个样本，批大小 {batch_size}")

# ============================== 模型初始化 =============================
print("初始化模型...")
model = temp_model.to(device)
start_epoch = 0  # 起始epoch
epoch_losses = []  # 历史epoch损失
step_losses = []  # 历史step损失

# 加载预训练模型
if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"成功加载预训练模型: {MODEL_PATH}")

        # 加载损失记录
        if os.path.exists(LOSS_RECORD_PATH):
            with open(LOSS_RECORD_PATH, 'rb') as f:
                loss_records = pickle.load(f)
                epoch_losses = loss_records['epoch_losses']
                step_losses = loss_records['step_losses']
                start_epoch = len(epoch_losses)  # 从上次结束的epoch+1开始

print(f"模型已初始化，使用设备: {device}，起始epoch: {start_epoch}")

# ============================== 损失函数和优化器 =============================
criterion = ComplexMSELoss()
optimizer = optim.Adam(model.unet.parameters(), lr=lr)

# 续训调整学习率
if start_epoch > 0:
    # 训练超过100轮后降低学习率
    if start_epoch > 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.1
        print(f"续训学习率调整为: {optimizer.param_groups[0]['lr']}")

# ============================== 训练循环 =============================
print("开始训练...")
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0.0

    for i, (inputs, targets) in enumerate(dataloader):
        # 转移数据到GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        pre_para, fin_para, Etot = model(inputs)

        # 计算损失
        loss = criterion(fin_para, targets)

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


# ============================== 最终保存模型和损失 =============================
# 训练结束后保存最终模型
torch.save(model.state_dict(), MODEL_PATH)
# 保存最终损失记录
loss_records = {
    'epoch_losses': epoch_losses,
    'step_losses': step_losses
}
with open(LOSS_RECORD_PATH, 'wb') as f:
    pickle.dump(loss_records, f)

print("模型已保存至 output/ssnet_model.pth")
print("损失记录已保存至 output/loss_records.pkl")

# ============================== 绘制损失曲线 =============================
plt.figure(figsize=(12, 5))

# 每步损失曲线
plt.subplot(1, 2, 1)
plt.plot(step_losses, label='Step Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Loss per Training Step (Full History)')
plt.legend()

# 每轮平均损失曲线
plt.subplot(1, 2, 2)
plt.plot(epoch_losses, label='Epoch Avg Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Loss per Epoch (Full History)')
plt.legend()

plt.tight_layout()
plt.savefig('output/loss_curve.png')
print("损失曲线已保存至 output/loss_curve.png")
plt.show()