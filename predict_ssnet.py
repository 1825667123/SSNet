import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import SSNet  # 导入自定义的SSNet模型
from Read_H5 import SSNetHDF5Dataset  # 改为从dataset.py导入

# ============================== 配置项 ==============================
# 模型权重路径
MODEL_PATH = "output/ssnet_model.pth"
# 预测数据路径（HDF5格式）
X_TEST_H5_PATH = "./output/hdf5_data/xkai_all.h5"
ETOT_TEST_H5_PATH = "./output/hdf5_data/etot_all.h5"
# 输出预测结果的目录
OUTPUT_DIR = "output/predictions"
# 设备配置（优先GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 是否可视化预测结果
VISUALIZE = True
# 要预测的样本索引
TEST_SAMPLE_INDICES = [0, 1, 2]
# 打印数值的位置
PRINT_POSITIONS = [(0, 0), (5, 5), (10, 10), (15, 15), (19, 19)]
# 是否打印统计信息
PRINT_STATISTICS = True
# Etot传输方向选择（对应ntrans维度，0或1）
ETOT_TRANS_DIM = 0  # 选择第0个传输方向的Etot进行对比


# ============================== 工具函数 ==============================
def create_output_dirs():
    """创建输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "npy_files"), exist_ok=True)


def save_complex_results(pred_data, true_data, sample_idx, prefix):
    """保存复数预测结果"""
    pred_data_np = pred_data.cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "npy_files", f"{prefix}_pred_real_idx{sample_idx}.npy"), pred_data_np.real)
    np.save(os.path.join(OUTPUT_DIR, "npy_files", f"{prefix}_pred_imag_idx{sample_idx}.npy"), pred_data_np.imag)
    if true_data is not None:
        true_data_np = true_data.cpu().numpy()
        np.save(os.path.join(OUTPUT_DIR, "npy_files", f"{prefix}_true_real_idx{sample_idx}.npy"), true_data_np.real)
        np.save(os.path.join(OUTPUT_DIR, "npy_files", f"{prefix}_true_imag_idx{sample_idx}.npy"), true_data_np.imag)


def plot_complex_field(pred, true, sample_idx, field_name):
    """可视化复数场对比"""
    pred_real = pred.real.cpu().numpy().squeeze()
    pred_imag = pred.imag.cpu().numpy().squeeze()
    true_real = true.real.cpu().numpy().squeeze() if true is not None else np.zeros_like(pred_real)
    true_imag = true.imag.cpu().numpy().squeeze() if true is not None else np.zeros_like(pred_imag)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{field_name} - Sample {sample_idx} (Trans Dim {ETOT_TRANS_DIM})", fontsize=14)

    im1 = axes[0, 0].imshow(pred_real, cmap='jet', aspect='auto')
    axes[0, 0].set_title(f"Pred {field_name} Real")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(true_real, cmap='jet', aspect='auto')
    axes[0, 1].set_title(f"True {field_name} Real")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(pred_imag, cmap='jet', aspect='auto')
    axes[1, 0].set_title(f"Pred {field_name} Imag")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(true_imag, cmap='jet', aspect='auto')
    axes[1, 1].set_title(f"True {field_name} Imag")
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"{field_name}_sample{sample_idx}.png"))
    plt.close()


def print_complex_values(pred, true, sample_idx, field_name):
    """打印指定位置的复数预测值与真实值"""
    print(f"\n📌 {field_name} - 样本 {sample_idx} 关键位置数值对比：")
    print(f"{'位置':<10} {'预测实部':<20} {'真实实部':<20} {'预测虚部':<20} {'真实虚部':<20}")
    print("-" * 90)

    pred_squeezed = pred.squeeze()
    true_squeezed = true.squeeze() if true is not None else torch.zeros_like(pred_squeezed)

    for (z, x) in PRINT_POSITIONS:
        if z >= pred_squeezed.shape[0] or x >= pred_squeezed.shape[1]:
            continue

        pred_real = pred_squeezed[z, x].real.item()
        pred_imag = pred_squeezed[z, x].imag.item()
        true_real = true_squeezed[z, x].real.item() if true is not None else "N/A"
        true_imag = true_squeezed[z, x].imag.item() if true is not None else "N/A"

        print(f"({z:2d},{x:2d})    {pred_real:<20.6f} {str(true_real):<20} {pred_imag:<20.6f} {str(true_imag):<20}")


def print_complex_statistics(pred, true, sample_idx, field_name):
    """打印复数统计信息"""
    if not PRINT_STATISTICS or true is None:
        return

    print(f"\n📊 {field_name} - 样本 {sample_idx} 统计信息：")
    pred_squeezed = pred.squeeze().cpu()
    true_squeezed = true.squeeze().cpu()

    # 实部统计
    pred_real_mean = pred_squeezed.real.mean().item()
    true_real_mean = true_squeezed.real.mean().item()
    real_abs_error = torch.abs(pred_squeezed.real - true_squeezed.real).mean().item()
    real_rel_error = (real_abs_error / (abs(true_real_mean) + 1e-8)) * 100

    # 虚部统计
    pred_imag_mean = pred_squeezed.imag.mean().item()
    true_imag_mean = true_squeezed.imag.mean().item()
    imag_abs_error = torch.abs(pred_squeezed.imag - true_squeezed.imag).mean().item()
    imag_rel_error = (imag_abs_error / (abs(true_imag_mean) + 1e-8)) * 100

    print(
        f"实部 - 预测均值：{pred_real_mean:.6f} | 真实均值：{true_real_mean:.6f} | 平均绝对误差：{real_abs_error:.6f} | 相对误差：{real_rel_error:.2f}%")
    print(
        f"虚部 - 预测均值：{pred_imag_mean:.6f} | 真实均值：{true_imag_mean:.6f} | 平均绝对误差：{imag_abs_error:.6f} | 相对误差：{imag_rel_error:.2f}%")


# ============================== 核心预测逻辑 ==============================
def load_model():
    """加载训练好的模型"""
    model = SSNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 成功加载模型权重：{MODEL_PATH}")
    print(f"🔧 模型设备：{DEVICE}")
    return model


def load_test_data():
    """加载测试数据"""
    test_dataset = SSNetHDF5Dataset(
        xkai_h5_path=X_TEST_H5_PATH,
        etot_h5_path=ETOT_TEST_H5_PATH,
        nfreq=1
    )
    return test_dataset


def predict_single_sample(model, test_dataset, sample_idx):
    """对单个样本进行预测"""
    # 获取样本（x输入 + Etot真实值）
    x, etot_true_all = test_dataset[sample_idx]
    # 提取指定传输方向的Etot真实值：(1,2,20,20) → (1,20,20)
    etot_true = etot_true_all[:, ETOT_TRANS_DIM, :, :].to(DEVICE)
    # 训练时的fin_para目标值是etot_true_all[:, 0, :, :]（和训练脚本一致）
    fin_para_true = etot_true_all[:, 0, :, :].to(DEVICE)

    # 添加batch维度
    x_input = x.unsqueeze(0).to(DEVICE)

    # 禁用梯度计算
    with torch.no_grad():
        pre_para, fin_para, Etot_pred = model(x_input)

    # 去除batch维度
    pre_para = pre_para.squeeze(0)
    fin_para = fin_para.squeeze(0)
    Etot_pred = Etot_pred.squeeze(0)
    fin_para_true = fin_para_true.squeeze(0)
    etot_true = etot_true.squeeze(0)

    print(f"\n=== 样本 {sample_idx} 预测结果 ===")
    print(f"pre_para 维度: {pre_para.shape}")
    print(f"fin_para 维度: {fin_para.shape}")
    print(f"Etot_pred 维度: {Etot_pred.shape}")
    print(f"fin_para真实值维度: {fin_para_true.shape}")
    print(f"Etot真实值维度: {etot_true.shape}")

    # 打印数值
    print_complex_values(fin_para, fin_para_true, sample_idx, "fin_para (核心预测值)")
    print_complex_values(Etot_pred, etot_true, sample_idx, f"Etot (总电场-传输方向{ETOT_TRANS_DIM})")

    # 打印统计信息
    print_complex_statistics(fin_para, fin_para_true, sample_idx, "fin_para (核心预测值)")
    print_complex_statistics(Etot_pred, etot_true, sample_idx, f"Etot (总电场-传输方向{ETOT_TRANS_DIM})")

    # 保存结果
    save_complex_results(pre_para, None, sample_idx, "pre_para")
    save_complex_results(fin_para, fin_para_true, sample_idx, "fin_para")
    save_complex_results(Etot_pred, etot_true, sample_idx, f"Etot_trans{ETOT_TRANS_DIM}")

    # 可视化
    if VISUALIZE:
        plot_complex_field(fin_para, fin_para_true, sample_idx, "fin_para")
        plot_complex_field(Etot_pred, etot_true, sample_idx, f"Etot_trans{ETOT_TRANS_DIM}")

    return {
        "sample_idx": sample_idx,
        "pre_para": pre_para,
        "fin_para": fin_para,
        "fin_para_true": fin_para_true,
        "Etot_pred": Etot_pred,
        "Etot_true": etot_true
    }


# ============================== 主函数 ==============================
def main():
    create_output_dirs()
    model = load_model()
    test_dataset = load_test_data()

    all_predictions = []
    for idx in TEST_SAMPLE_INDICES:
        if idx >= len(test_dataset):
            print(f"⚠️ 样本索引 {idx} 超出数据集范围，跳过")
            continue
        pred_result = predict_single_sample(model, test_dataset, idx)
        all_predictions.append(pred_result)

    print("\n🎉 预测完成！结果保存至：", OUTPUT_DIR)
    return all_predictions


if __name__ == "__main__":
    main()