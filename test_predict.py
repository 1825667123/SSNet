import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from test_model import UNet  # 导入你的UNet模型
from Read_H5 import SSNetHDF5Dataset  # 导入数据集读取类

# ============================== 配置参数 ==============================
# 预测目标配置
TARGET_SCT_NUM = 1234  # 要预测的散射体编号（对应Sct1234）
NFREQ = 1  # 频率数（与训练时一致）

# 文件路径配置
MODEL_PATH = "output/unet_complex_model.pth"  # 训练好的模型路径
XKAI_H5_PATH = "./output/hdf5_data/xkai_all.h5"
ETOT_H5_PATH = "./output/hdf5_data/etot_all.h5"

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可视化配置
PLOT_SAVE_PATH = "output/prediction_result.png"
ERROR_LOG_PATH = "output/prediction_error.txt"


# ============================== 核心工具函数 ==============================
def find_sct_idx(dataset, target_sct_num):
    """根据散射体编号查找样本索引"""
    target_sct_name = f"Sct{target_sct_num:04d}"  # 修复：正确的f-string格式化
    for idx, (sct_name, _) in enumerate(dataset.samples):
        if sct_name == target_sct_name:
            return idx
    # 修复：字符串格式化错误
    raise ValueError(f"数据集未找到散射体编号 {target_sct_num}（对应Sct{target_sct_num:04d}）")


def calc_complex_error(pred, true):
    """计算复数预测结果的误差指标"""
    # 转为numpy数组并压缩维度
    pred = pred.squeeze().cpu().numpy()
    true = true.squeeze().cpu().numpy()

    # 实部误差
    real_abs_err = np.abs(pred.real - true.real)
    real_rel_err = (real_abs_err / (np.abs(true.real) + 1e-8)).mean() * 100  # 平均相对误差(%)
    real_mse = np.mean((pred.real - true.real) ** 2)

    # 虚部误差
    imag_abs_err = np.abs(pred.imag - true.imag)
    imag_rel_err = (imag_abs_err / (np.abs(true.imag) + 1e-8)).mean() * 100
    imag_mse = np.mean((pred.imag - true.imag) ** 2)

    # 整体误差
    total_mse = real_mse + imag_mse
    abs_error = np.abs(pred - true)
    avg_abs_error = abs_error.mean()
    max_abs_error = abs_error.max()

    return {
        "real_rel_err": real_rel_err,
        "imag_rel_err": imag_rel_err,
        "real_mse": real_mse,
        "imag_mse": imag_mse,
        "total_mse": total_mse,
        "avg_abs_error": avg_abs_error,
        "max_abs_error": max_abs_error
    }


def plot_prediction_result(pred, true, save_path):
    """可视化预测结果与真实值对比"""
    # 压缩维度并转为numpy数组
    pred = pred.squeeze().cpu().numpy()
    true = true.squeeze().cpu().numpy()

    # 计算误差
    abs_error = np.abs(pred - true)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 真实值 - 实部
    im1 = axes[0, 0].imshow(true.real, cmap='jet', aspect='auto')
    axes[0, 0].set_title('真实值 - 实部', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0])
    axes[0, 0].axis('off')

    # 2. 预测值 - 实部
    im2 = axes[0, 1].imshow(pred.real, cmap='jet', aspect='auto')
    axes[0, 1].set_title('预测值 - 实部', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].axis('off')

    # 3. 实部误差
    im3 = axes[0, 2].imshow(np.abs(pred.real - true.real), cmap='Reds', aspect='auto')
    axes[0, 2].set_title('实部绝对误差', fontsize=12)
    plt.colorbar(im3, ax=axes[0, 2])
    axes[0, 2].axis('off')

    # 4. 真实值 - 虚部
    im4 = axes[1, 0].imshow(true.imag, cmap='jet', aspect='auto')
    axes[1, 0].set_title('真实值 - 虚部', fontsize=12)
    plt.colorbar(im4, ax=axes[1, 0])
    axes[1, 0].axis('off')

    # 5. 预测值 - 虚部
    im5 = axes[1, 1].imshow(pred.imag, cmap='jet', aspect='auto')
    axes[1, 1].set_title('预测值 - 虚部', fontsize=12)
    plt.colorbar(im5, ax=axes[1, 1])
    axes[1, 1].axis('off')

    # 6. 虚部误差
    im6 = axes[1, 2].imshow(np.abs(pred.imag - true.imag), cmap='Reds', aspect='auto')
    axes[1, 2].set_title('虚部绝对误差', fontsize=12)
    plt.colorbar(im6, ax=axes[1, 2])
    axes[1, 2].axis('off')

    # 整体标题
    fig.suptitle(f'散射体Sct{TARGET_SCT_NUM:04d} 预测结果对比', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 预测结果可视化已保存至：{save_path}")


# ============================== 主预测逻辑 ==============================
def main():
    print("=" * 60)
    print("🔍 UNet 预测脚本启动")
    print("=" * 60)
    print(f"📌 计算设备：{DEVICE}")
    print(f"📌 目标散射体：Sct{TARGET_SCT_NUM:04d}")
    print(f"📌 模型路径：{MODEL_PATH}")

    # 1. 加载模型
    print("\n📥 加载训练好的UNet模型...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

    # 先加载数据集获取维度信息
    dataset = SSNetHDF5Dataset(
        xkai_h5_path=XKAI_H5_PATH,
        etot_h5_path=ETOT_H5_PATH,
        nfreq=NFREQ
    )
    print(f"✅ 数据集加载完成，总样本数：{len(dataset)}")

    # 查找目标散射体索引
    sct_idx = find_sct_idx(dataset, TARGET_SCT_NUM)
    print(f"✅ 目标散射体Sct{TARGET_SCT_NUM:04d} 对应的样本索引：{sct_idx}")

    # 读取样本获取维度（用于初始化模型）
    sample_x, sample_y = dataset[0]
    IN_CHANNELS = sample_x.shape[0]
    OUT_CHANNELS = sample_y[:, 0, :, :].shape[0]  # 匹配训练时的维度

    # 初始化模型并加载权重
    model = UNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        bilinear=True  # 与训练时一致
    ).to(DEVICE)

    # 加载模型权重（兼容多GPU训练的情况）
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ 模型加载完成")

    # 2. 读取目标样本数据
    print("\n📥 读取目标样本数据...")
    k_input, etot_true_all = dataset[sct_idx]
    etot_true = etot_true_all[:, 0, :, :].to(DEVICE)  # 取第一个传输方向（与训练对齐）
    k_input = k_input.unsqueeze(0).to(DEVICE)  # 增加batch维度

    # 3. 模型预测
    print("\n🔮 执行模型预测...")
    with torch.no_grad():
        etot_pred = model(k_input)
    etot_pred = etot_pred.squeeze(0)  # 移除batch维度

    # 4. 误差分析
    print("\n📊 计算误差指标...")
    error_metrics = calc_complex_error(etot_pred, etot_true)

    # 5. 打印结果
    print("\n" + "=" * 60)
    print(f"散射体Sct{TARGET_SCT_NUM:04d} 预测结果分析")
    print("=" * 60)

    # 基础维度信息
    print(f"📌 维度信息：")
    print(f"   - 输入Xkai维度：{k_input.shape}")
    print(f"   - 预测Etot维度：{etot_pred.shape}")
    print(f"   - 真实Etot维度：{etot_true.shape}")

    # 误差指标
    print(f"\n📊 误差指标：")
    print(f"   - 实部平均相对误差：{error_metrics['real_rel_err']:.4f}%")
    print(f"   - 虚部平均相对误差：{error_metrics['imag_rel_err']:.4f}%")
    print(f"   - 实部MSE：{error_metrics['real_mse']:.8f}")
    print(f"   - 虚部MSE：{error_metrics['imag_mse']:.8f}")
    print(f"   - 总MSE（实+虚）：{error_metrics['total_mse']:.8f}")
    print(f"   - 平均绝对误差：{error_metrics['avg_abs_error']:.8f}")
    print(f"   - 最大绝对误差：{error_metrics['max_abs_error']:.8f}")

    # 关键位置数值对比
    print(f"\n📍 关键位置数值对比：")
    pred_np = etot_pred.squeeze().cpu().numpy()
    true_np = etot_true.squeeze().cpu().numpy()
    for (z, x) in [(0, 0),(2, 6),(7, 8), (14, 7),(10, 10), (19, 19)]:
        print(f"   位置({z},{x})：")
        print(f"      预测：{pred_np[z, x].real:.6f} + {pred_np[z, x].imag:.6f}j")
        print(f"      真实：{true_np[z, x].real:.6f} + {true_np[z, x].imag:.6f}j")
        print(f"      误差：{np.abs(pred_np[z, x] - true_np[z, x]):.6f}")

    # 6. 保存误差日志
    print("\n💾 保存误差分析日志...")
    with open(ERROR_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"UNet 预测误差分析 - Sct{TARGET_SCT_NUM:04d}\n")
        f.write("=" * 60 + "\n\n")

        f.write("📌 基础信息：\n")
        f.write(f"   - 预测时间：{np.datetime64('now')}\n")
        f.write(f"   - 模型路径：{MODEL_PATH}\n")
        f.write(f"   - 散射体编号：{TARGET_SCT_NUM}\n")
        f.write(f"   - 样本索引：{sct_idx}\n\n")

        f.write("📊 维度信息：\n")
        f.write(f"   - 输入Xkai维度：{k_input.shape}\n")
        f.write(f"   - 预测Etot维度：{etot_pred.shape}\n")
        f.write(f"   - 真实Etot维度：{etot_true.shape}\n\n")

        f.write("📈 误差指标：\n")
        f.write(f"   - 实部平均相对误差：{error_metrics['real_rel_err']:.4f}%\n")
        f.write(f"   - 虚部平均相对误差：{error_metrics['imag_rel_err']:.4f}%\n")
        f.write(f"   - 实部MSE：{error_metrics['real_mse']:.8f}\n")
        f.write(f"   - 虚部MSE：{error_metrics['imag_mse']:.8f}\n")
        f.write(f"   - 总MSE（实+虚）：{error_metrics['total_mse']:.8f}\n")
        f.write(f"   - 平均绝对误差：{error_metrics['avg_abs_error']:.8f}\n")
        f.write(f"   - 最大绝对误差：{error_metrics['max_abs_error']:.8f}\n\n")

        f.write("📍 关键位置数值：\n")
        for (z, x) in [(0, 0),(2, 6),(7, 8), (14, 7),(10, 10), (19, 19)]:
            f.write(f"   位置({z},{x})：\n")
            f.write(f"      预测：{pred_np[z, x].real:.6f} + {pred_np[z, x].imag:.6f}j\n")
            f.write(f"      真实：{true_np[z, x].real:.6f} + {true_np[z, x].imag:.6f}j\n")
            f.write(f"      误差：{np.abs(pred_np[z, x] - true_np[z, x]):.6f}\n")

    # 7. 可视化结果
    print("\n🎨 生成预测结果可视化...")
    plot_prediction_result(etot_pred, etot_true, PLOT_SAVE_PATH)

    print("\n" + "=" * 60)
    print("🎉 预测完成！")
    print(f"📄 误差日志：{ERROR_LOG_PATH}")
    print(f"📊 可视化结果：{PLOT_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()