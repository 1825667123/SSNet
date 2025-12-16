import torch
import numpy as np
from model import SSNet
from Read_H5 import SSNetHDF5Dataset


TARGET_SCT_NUM = 1234  # 要预测的散射体编号（1234→Sct1234）
MODEL_PATH = "output/ssnet_k2etot_model.pth"
XKAI_H5_PATH = "./output/hdf5_data/xkai_all.h5"
ETOT_H5_PATH = "./output/hdf5_data/etot_all.h5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_sct_idx(dataset, target_sct_num):
    """根据散射体编号找样本索引"""
    target_sct_name = f"Sct{target_sct_num:04d}"
    for idx, (sct_name, _) in enumerate(dataset.samples):
        if sct_name == target_sct_name:
            return idx
    raise ValueError(f"未找到散射体编号 {target_sct_num}")


def calc_relative_error(pred, true):
    """计算复数的相对误差"""
    # 转为numpy数组
    pred = pred.squeeze().cpu().numpy()
    true = true.squeeze().cpu().numpy()

    # 实部相对误差
    real_abs_err = np.abs(pred.real - true.real)
    real_rel_err = (real_abs_err / (np.abs(true.real) + 1e-8)).mean() * 100

    # 虚部相对误差
    imag_abs_err = np.abs(pred.imag - true.imag)
    imag_rel_err = (imag_abs_err / (np.abs(true.imag) + 1e-8)).mean() * 100

    # 整体MSE
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
    # 1. 加载模型
    model = SSNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ 加载模型完成：{MODEL_PATH}")

    # 2. 加载数据集
    temp_model = SSNet()
    dataset = SSNetHDF5Dataset(
        xkai_h5_path=XKAI_H5_PATH,
        etot_h5_path=ETOT_H5_PATH,
        nfreq=temp_model.integral_kernel.nfreq
    )
    print(f"✅ 加载数据集完成，总样本数：{len(dataset)}")

    # 3. 找到目标散射体的索引
    sct_idx = find_sct_idx(dataset, TARGET_SCT_NUM)
    print(f"✅ 散射体Sct1234对应的样本索引：{sct_idx}")

    # 4. 读取目标样本数据
    k_input, etot_true_all = dataset[sct_idx]
    etot_true = etot_true_all[:, 0, :, :].to(DEVICE)
    k_input = k_input.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, _, etot_pred = model(k_input)
    etot_pred = etot_pred.squeeze(0)

    error_metrics = calc_relative_error(etot_pred, etot_true)

    print("\n" + "=" * 50)
    print(f"散射体Sct1234 预测结果 & 误差分析")
    print("=" * 50)
    print(f"📌 基础维度：")
    print(f"   输入k(xkai)维度：{k_input.shape}")
    print(f"   预测Etot维度：{etot_pred.shape}")
    print(f"   真实Etot维度：{etot_true.shape}")

    print(f"\n📊 误差指标：")
    print(f"   实部平均相对误差：{error_metrics['real_rel_err']:.4f}%")
    print(f"   虚部平均相对误差：{error_metrics['imag_rel_err']:.4f}%")
    print(f"   实部MSE：{error_metrics['real_mse']:.8f}")
    print(f"   虚部MSE：{error_metrics['imag_mse']:.8f}")
    print(f"   总MSE（实+虚）：{error_metrics['total_mse']:.8f}")

    print(f"\n📍 关键位置数值对比（(0,0)/(10,10)/(19,19)）：")
    pred_np = etot_pred.squeeze().cpu().numpy()
    true_np = etot_true.squeeze().cpu().numpy()
    for (z, x) in [(0, 0), (10, 10), (19, 19)]:
        print(f"   位置({z},{x})：")
        print(f"      预测：{pred_np[z, x].real:.6f} + {pred_np[z, x].imag:.6f}j")
        print(f"      真实：{true_np[z, x].real:.6f} + {true_np[z, x].imag:.6f}j")


if __name__ == "__main__":
    main()