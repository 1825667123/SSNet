import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Compute_IncField import Compute_IncField
from Compute_Integral_Kernel import compute_integral_kernel

class IntegralKernelLayer(nn.Module):
    def __init__(self, load_from_file=False, file_path="output/integral_kernel_matrix.npy"):
        super(IntegralKernelLayer, self).__init__()

        # kernel_np = np.load(file_path)
        kernel_np = compute_integral_kernel()
        self.nfreq, self.mz, self.mx = kernel_np.shape
        self.kernel = nn.Parameter(
            torch.tensor(kernel_np, dtype=torch.complex128),
            requires_grad=False        # 不可训练
        )

    def forward(self, x):
        return x * self.kernel


class UNet(nn.Module):
    def __init__(self, in_channels, spatial_size):
        super(UNet, self).__init__()
        self.mz, self.mx = spatial_size
        self.in_channels = in_channels

        # ===================== 复刻TF的下采样模块 =====================
        # Down1: Conv(3→8) + BN + ReLU + MaxPool
        self.down1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),  # 复刻TF Conv2D
            nn.BatchNorm2d(8),  # 复刻TF BatchNormalization
            nn.ReLU(inplace=True)  # 复刻TF ReLU
        )
        self.down1_pool = nn.MaxPool2d(2, stride=2)  # 复刻TF MaxPool2D

        # Down2: Conv(8→16) + BN + ReLU + MaxPool
        self.down2_conv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.down2_pool = nn.MaxPool2d(2, stride=2)

        # Down3: Conv(16→32) + BN + ReLU + MaxPool
        self.down3_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.down3_pool = nn.MaxPool2d(2, stride=2)

        # ===================== 复刻TF的上采样模块 =====================
        # Up4: ConvTranspose(32→16) + 拼接 + Conv(32→16) + BN + ReLU
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)  # 复刻TF Conv2DTranspose
        self.up4_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),  # 拼接后通道=16+16=32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Up5: ConvTranspose(16→8) + 拼接 + Conv(16→8) + BN + ReLU
        self.up5 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0)
        self.up5_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),  # 拼接后通道=8+8=16
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # ===================== 复刻TF的输出层 =====================
        self.out_conv = nn.Conv2d(8, in_channels, kernel_size=1, padding=0)  # 复刻TF 1x1 Conv

    def _process_single_channel(self, input_tensor):
        """复刻TF中对实/虚部的处理逻辑（单通道处理函数）"""
        # 下采样（复刻TF的down_block）
        conv1 = self.down1_conv(input_tensor)  # TF: down_block的conv输出
        pool1 = self.down1_pool(conv1)         # TF: down_block的pool输出

        conv2 = self.down2_conv(pool1)
        pool2 = self.down2_pool(conv2)

        conv3 = self.down3_conv(pool2)
        pool3 = self.down3_pool(conv3)

        # 上采样（复刻TF的up_block）
        up4 = self.up4(pool3)
        # 复刻TF的tf.image.resize对齐尺寸
        conv2_resized = F.interpolate(conv2, size=up4.shape[2:], mode='bilinear', align_corners=True)
        merge4 = torch.cat([conv2_resized, up4], dim=1)  # 复刻TF的tf.concat
        conv4 = self.up4_conv(merge4)

        up5 = self.up5(conv4)
        conv1_resized = F.interpolate(conv1, size=up5.shape[2:], mode='bilinear', align_corners=True)
        merge5 = torch.cat([conv1_resized, up5], dim=1)
        conv5 = self.up5_conv(merge5)

        # 复刻TF的最终resize和输出层
        out = F.interpolate(conv5, size=(self.mz, self.mx), mode='bilinear', align_corners=True)
        out = self.out_conv(out)
        return out

    def forward(self, x):
        """完全复刻TF的前向逻辑：复数拆分→分别处理→合并"""
        # 复刻TF的tf.math.real/tf.math.imag
        x_real = x.real.to(torch.float32)
        x_imag = x.imag.to(torch.float32)

        # 复刻TF中对实/虚部的相同处理逻辑
        out_real = self._process_single_channel(x_real)
        out_imag = self._process_single_channel(x_imag)

        # 复刻TF的tf.complex
        return torch.complex(out_real, out_imag)


class SSNet(nn.Module):
    def __init__(self):
        super(SSNet, self).__init__()
        self.integral_kernel = IntegralKernelLayer(load_from_file=True)
        self.nfreq = self.integral_kernel.nfreq
        self.mz, self.mx = self.integral_kernel.mz, self.integral_kernel.mx
        self.unet = UNet(
            in_channels=self.nfreq,
            spatial_size=(self.mz, self.mx)
        )

        einc_path = "output/Einc/Eyinc_Fre0001.dat"
        einc_raw = np.loadtxt(einc_path)  # shape=(400, 2)：每行[实部, 虚部]
        einc_np = einc_raw[:, 0] + 1j * einc_raw[:, 1]  # 转为复数数组
        # 2. 恢复维度：(nfreq=1, mz=20, mx=20)
        einc_np = einc_np.reshape(1, self.mz, self.mx)
        assert einc_np.shape == (self.nfreq, self.mz, self.mx)
        self.Einc = nn.Parameter(torch.tensor(einc_np, dtype=torch.complex128), requires_grad=False)
        print(f"✅ 已加载入射场，维度：{self.Einc.shape}")

        # self.Einc = nn.Parameter(torch.tensor(Compute_IncField(), dtype=torch.complex128), requires_grad=False)
        # print(f"已计算Einc，维度：{self.Einc.shape}")


    def forward(self, x):
        pre_para = self.integral_kernel(x)
        fin_para = self.unet(pre_para)

        # 计算Etot：统一数据类型
        batch_size = x.shape[0]
        # 将Einc转置并扩展为批量，同时转换为与fin_para相同的类型
        Einc_mat = self.Einc.transpose(-1, -2).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(fin_para.dtype)
        # 矩阵乘法
        Etot = torch.matmul(fin_para, Einc_mat).squeeze(-1)

        return pre_para, fin_para, Etot

if __name__ == "__main__":
    model = SSNet()  # 初始化模型
    print(f"积分核维度：{model.integral_kernel.kernel.shape}")

    # 生成测试输入x（批量大小为2，维度与积分核匹配）
    batch_size = 2
    x = torch.randn(
        batch_size,
        int(model.nfreq),
        int(model.mz),
        int(model.mx),
        dtype=torch.complex128
    )

    # 调用模型，触发forward方法
    pre_para, fin_para, Etot = model(x)
    print(f"测试输出：pre_para.shape={pre_para.shape}, fin_para.shape={fin_para.shape}, Etot.shape={Etot.shape}")