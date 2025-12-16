import torch
import numpy as np
from model import SSNet
from Read_H5 import SSNetHDF5Dataset

# ============================== 配置参数 ==============================
TARGET_SCT_NUM = 1234  # 要预测的散射体编号（1234→Sct1234）
MODEL_PATH = "output/ssnet_k2etot_model_test.pth"  # 训练保存的模型路径
NORMALIZE_PARAMS_PATH = "output/normalize_params.pth"  # 归一化参数路径
XKAI_H5_PATH = "./output/hdf5_data/xkai_all.h5"
ETOT_H5_PATH = "./output/hdf5_data/etot_all.h5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================== 工具函数 ==============================
def find_sct_idx(dataset, target_sct_num):
    """根据散射体编号找样本索引"""
    target_sct_name = f"Sct{target_sct_num:04d}"
    for idx, (sct_name, _) in enumerate(dataset.samples):
        if sct_name == target_sct_name:
            return idx
    raise ValueError(f"未找到散射体编号 {target_sct_num}")


def calc_relative_error(pred, true):
    """计算复数的相对误差（实虚部分开）"""
    # 转为numpy数组（确保维度压缩）
    pred = pred.squeeze().cpu().numpy()
    true = true.squeeze().cpu().numpy()

    # 实部相对误差（添加保护避免除零）
    real_abs_err = np.abs(pred.real - true.real)
    real_rel_err = (real_abs_err / (np.abs(true.real) + 1e-8)).mean() * 100

    # 虚部相对误差
    imag_abs_err = np.abs(pred.imag - true.imag)
    imag_rel_err = (imag_abs_err / (np.abs(true.imag) + 1e-8)).mean() * 100

    # MSE计算
    mse_real = np.mean((pred.real - true.real) ** 2)
    mse_imag = np.mean((pred.imag - true.imag) ** 2)
    total_mse = mse_real + mse_imag

    return {
        "real_rel_err": real_rel_err,
        "imag_rel_err": imag_rel_err,
        "total_mse": total_mse,
        "real_mse": mse_real,
        "imag_mse": mse_imag
    }


# ============================== 核心预测逻辑 ==============================
def main():
    # 1. 加载归一化参数
    normalize_dict = torch.load(NORMALIZE_PARAMS_PATH, map_location=DEVICE)
    X_max = normalize_dict["X_max"]
    Y_max = normalize_dict["Y_max"]
    print(f"✅ 加载归一化参数完成：X_max={X_max:.4f}, Y_max={Y_max:.4f}")

    # 2. 加载模型
    model = SSNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ 加载模型完成：{MODEL_PATH}")

    # 3. 加载原始数据集（未归一化）
    temp_model = SSNet()
    dataset = SSNetHDF5Dataset(
        xkai_h5_path=XKAI_H5_PATH,
        etot_h5_path=ETOT_H5_PATH,
        nfreq=temp_model.integral_kernel.nfreq
    )
    print(f"✅ 加载数据集完成，总样本数：{len(dataset)}")

    # 4. 找到目标散射体的索引
    sct_idx = find_sct_idx(dataset, TARGET_SCT_NUM)
    print(f"✅ 散射体Sct{TARGET_SCT_NUM}对应的样本索引：{sct_idx}")

    # 5. 读取目标样本原始数据
    k_input_original, etot_true_original = dataset[sct_idx]
    # 适配训练时的维度（无[:,0,:,:]切片）
    etot_true_original = etot_true_original.to(DEVICE)
    k_input_original = k_input_original.to(DEVICE)

    # 6. 对输入进行归一化（和训练保持一致）
    k_input_norm = k_input_original.unsqueeze(0) / X_max  # 加batch维度 + 归一化

    # 7. 模型预测
    with torch.no_grad():
        _, _, etot_pred_norm = model(k_input_norm)

    # 8. 预测结果反归一化（恢复真实量级）
    etot_pred_original = etot_pred_norm.squeeze(0) * Y_max  # 去batch维度 + 反归一化

    # 9. 计算误差（基于真实量级）
    error_metrics = calc_relative_error(etot_pred_original, etot_true_original)

    # 10. 打印结果
    print("\n" + "=" * 60)
    print(f"散射体Sct{TARGET_SCT_NUM} 预测结果 & 误差分析（真实量级）")
    print("=" * 60)

    # 基础维度信息
    print(f"📌 基础维度：")
    print(f"   原始输入k(xkai)维度：{k_input_original.shape}")
    print(f"   归一化输入k维度：{k_input_norm.shape}")
    print(f"   预测Etot（反归一化）维度：{etot_pred_original.shape}")
    print(f"   真实Etot维度：{etot_true_original.shape}")

    # 误差指标
    print(f"\n📊 误差指标：")
    print(f"   实部平均相对误差：{error_metrics['real_rel_err']:.4f}%")
    print(f"   虚部平均相对误差：{error_metrics['imag_rel_err']:.4f}%")
    print(f"   实部MSE：{error_metrics['real_mse']:.8f}")
    print(f"   虚部MSE：{error_metrics['imag_mse']:.8f}")
    print(f"   总MSE（实+虚）：{error_metrics['total_mse']:.8f}")

    # 关键位置数值对比
    print(f"\n📍 关键位置数值对比（(0,0)/(10,10)/(19,19)）：")
    pred_np = etot_pred_original.squeeze().cpu().numpy()
    true_np = etot_true_original.squeeze().cpu().numpy()

    # 确保索引不越界（适配不同的mz/mx）
    mz, mx = pred_np.shape
    check_positions = [(0, 0),
                       (min(10, mz - 1), min(10, mx - 1)),
                       (min(19, mz - 1), min(19, mx - 1))]

    for (z, x) in check_positions:
        print(f"   位置({z},{x})：")
        print(f"      预测：{pred_np[z, x].real:.6f} + {pred_np[z, x].imag:.6f}j")
        print(f"      真实：{true_np[z, x].real:.6f} + {true_np[z, x].imag:.6f}j")

    # 额外：归一化后的损失验证
    print(f"\n🔍 归一化后损失验证：")
    etot_true_norm = etot_true_original / Y_max
    loss_normed = torch.mean((etot_pred_norm.squeeze(0).real - etot_true_norm.real) ** 2 +
                             (etot_pred_norm.squeeze(0).imag - etot_true_norm.imag) ** 2).item()
    print(f"   归一化后总MSE：{loss_normed:.8f}")


if __name__ == "__main__":
    main()