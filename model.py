import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Compute_Integral_Kernel import compute_integral_kernel
import global_para_var as gv
from ReadInput import Read_Input

class IntegralKernelLayer(nn.Module):
    def __init__(self):
        super(IntegralKernelLayer, self).__init__()

        Read_Input()
        self.nfreq = gv.nfreq
        self.mz = gv.mz
        self.mx = gv.mx

    def forward(self, x):

        batch_size = x.shape[0]
        integral_kernel = torch.zeros((batch_size, self.nfreq, self.mz, self.mx), dtype=torch.complex128).to(x.device)
        # 逐样本计算积分核
        for b in range(batch_size):

            # 提取单个样本的k，转为numpy数组
            contrast_np = x[b].cpu().numpy()
            kernel_np = compute_integral_kernel(contrast_np)
            # 转回tensor并放入对应设备
            integral_kernel[b] = torch.tensor(kernel_np, dtype=torch.complex128).to(x.device)

        return integral_kernel


class UNet(nn.Module):
    def __init__(self, in_channels, spatial_size):
        super(UNet, self).__init__()
        self.mz, self.mx = spatial_size
        self.in_channels = in_channels

        # ===================== 下采样模块 =====================
        # Down1: Conv(3→8) + BN + ReLU + MaxPool
        self.down1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.down1_pool = nn.MaxPool2d(2, stride=2)

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

        # ===================== 上采样模块 =====================
        # Up4: ConvTranspose(32→16) + 拼接 + Conv(32→16) + BN + ReLU
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.up4_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Up5: ConvTranspose(16→8) + 拼接 + Conv(16→8) + BN + ReLU
        self.up5 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0)
        self.up5_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # ===================== 输出层 =====================
        self.out_conv = nn.Conv2d(8, in_channels, kernel_size=1, padding=0)

    def _process_single_channel(self, input_tensor):
        """对实/虚部的处理逻辑（单通道处理函数）"""
        # 下采样
        conv1 = self.down1_conv(input_tensor)
        pool1 = self.down1_pool(conv1)

        conv2 = self.down2_conv(pool1)
        pool2 = self.down2_pool(conv2)

        conv3 = self.down3_conv(pool2)
        pool3 = self.down3_pool(conv3)

        # 上采样
        up4 = self.up4(pool3)

        # 对齐尺寸
        conv2_resized = F.interpolate(conv2, size=up4.shape[2:], mode='bilinear', align_corners=True)
        merge4 = torch.cat([conv2_resized, up4], dim=1)
        conv4 = self.up4_conv(merge4)

        up5 = self.up5(conv4)
        conv1_resized = F.interpolate(conv1, size=up5.shape[2:], mode='bilinear', align_corners=True)
        merge5 = torch.cat([conv1_resized, up5], dim=1)
        conv5 = self.up5_conv(merge5)

        # 最终resize和输出层
        out = F.interpolate(conv5, size=(self.mz, self.mx), mode='bilinear', align_corners=True)
        out = self.out_conv(out)
        return out

    def forward(self, x):
        """前向逻辑：复数拆分→分别处理→合并"""
        x_real = x.real.to(torch.float32)
        x_imag = x.imag.to(torch.float32)

        # 对实/虚部处理
        out_real = self._process_single_channel(x_real)
        out_imag = self._process_single_channel(x_imag)

        return torch.complex(out_real, out_imag)


class SSNet(nn.Module):
    def __init__(self):
        super(SSNet, self).__init__()
        self.integral_kernel = IntegralKernelLayer()
        self.nfreq = self.integral_kernel.nfreq
        self.mz, self.mx = self.integral_kernel.mz, self.integral_kernel.mx
        self.unet = UNet(
            in_channels=self.nfreq,
            spatial_size=(self.mz, self.mx)
        )

        einc_path = "output/Einc/Eyinc_Fre0001.dat"
        einc_raw = np.loadtxt(einc_path)
        einc_np = einc_raw[:, 0] + 1j * einc_raw[:, 1]
        einc_np = einc_np.reshape(1, self.mz, self.mx)
        assert einc_np.shape == (self.nfreq, self.mz, self.mx)
        self.Einc = nn.Parameter(torch.tensor(einc_np, dtype=torch.complex128), requires_grad=False)
        print(f"✅ 已加载入射场，维度：{self.Einc.shape}")

    def forward(self, x):

        pre_para = self.integral_kernel(x)
        fin_para = self.unet(pre_para)

        batch_size = x.shape[0]
        Einc_mat = self.Einc.transpose(-1, -2).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(fin_para.dtype)
        Etot = torch.matmul(fin_para, Einc_mat).squeeze(-1)
        return pre_para, fin_para, Etot

# if __name__ == "__main__":
#     model = SSNet()
#     print(f"积分核维度：{model.integral_kernel.kernel.shape}")
#
#     batch_size = 2
#     x = torch.randn(
#         batch_size,
#         int(model.nfreq),
#         int(model.mz),
#         int(model.mx),
#         dtype=torch.complex128
#     )
#
#     # 调用模型，触发forward方法
#     pre_para, fin_para, Etot = model(x)
#     print(f"测试输出：pre_para.shape={pre_para.shape}, fin_para.shape={fin_para.shape}, Etot.shape={Etot.shape}")