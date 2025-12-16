import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
from model import SSNet
from Read_H5 import SSNetHDF5Dataset


class ComplexMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        input = input.view(target.shape)
        loss_real = self.mse(input.real, target.real)
        loss_imag = self.mse(input.imag, target.imag)
        return loss_real + loss_imag


os.makedirs('output', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 10
num_epochs = 30
lr = 0.001

temp_model = SSNet()
nfreq = temp_model.integral_kernel.nfreq
mz, mx = temp_model.integral_kernel.mz, temp_model.integral_kernel.mx
print(f"从模型获取到的维度信息: nfreq={nfreq}, mz={mz}, mx={mx}")

print("准备训练数据...")
real_dataset = SSNetHDF5Dataset(
    xkai_h5_path="./output/hdf5_data/xkai_all.h5",
    etot_h5_path="./output/hdf5_data/etot_all.h5",
    nfreq=nfreq
)

X = torch.stack([real_dataset[i][0] for i in range(len(real_dataset))])
Y = torch.stack([real_dataset[i][1] for i in range(len(real_dataset))])

X_max = X.abs().max() if X.abs().max() > 1e-8 else 1e-8
Y_max = Y.abs().max() if Y.abs().max() > 1e-8 else 1e-8

X = X / X_max
Y = Y / Y_max

print(f"【归一化验证】")
print(f"X归一化前最大模值: {X_max.item():.4f}, 归一化后: {X.abs().max().item():.4f}")
print(f"Y归一化前最大模值: {Y_max.item():.4f}, 归一化后: {Y.abs().max().item():.4f}")
print(f"X维度: {X.shape}, Y维度: {Y.shape}")

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"训练数据准备完成，共 {len(dataset)} 个样本，批大小 {batch_size}")

print("初始化模型...")
model = temp_model.to(device)
optimizer = optim.Adam(model.unet.parameters(), lr=lr)
criterion = ComplexMSELoss()

print("开始训练...")
epoch_losses = []
step_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        pre_para, fin_para, Etot = model(inputs)

        loss = criterion(Etot, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        step_losses.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], 平均损失: {avg_loss:.4f}')

torch.save(model.state_dict(), 'output/ssnet_k2etot_model_test.pth')
torch.save({"X_max": X_max, "Y_max": Y_max}, 'output/normalize_params.pth')
print("✅ 模型和归一化参数已保存")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(step_losses, label='Step Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Loss per Training Step')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epoch_losses, label='Epoch Avg Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Loss per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('output/loss_curve_k2etot.png')
plt.show()

print("\n✅ 训练完成，验证预测效果：")
model.eval()
with torch.no_grad():
    test_k = X[0:1].to(device)
    _, _, test_etot_pred = model(test_k)
    test_etot_true = Y[0:1].to(device)

    test_etot_pred_denorm = test_etot_pred * Y_max
    test_etot_true_denorm = test_etot_true * Y_max

    loss_normed = criterion(test_etot_pred, test_etot_true).item()
    loss_denormed = criterion(test_etot_pred_denorm, test_etot_true_denorm).item()

    print(f"归一化后损失: {loss_normed:.6f}")
    print(f"真实量级损失: {loss_denormed:.6f}")
    print(f"验证样本Etot模值范围（真实）: [{test_etot_true_denorm.abs().min():.4f}, {test_etot_true_denorm.abs().max():.4f}]")
    print(f"验证样本Etot模值范围（预测）: [{test_etot_pred_denorm.abs().min():.4f}, {test_etot_pred_denorm.abs().max():.4f}]")