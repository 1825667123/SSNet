import h5py
import numpy as np
import os
from pathlib import Path

# ============================== 配置参数 ==============================
XKAI_INPUT_DIR = "./output/xkai"  # Xkai.dat文件目录
ETOT_INPUT_DIR = "./output/Etot"  # Etot.dat文件目录
HDF5_OUTPUT_DIR = "./output/hdf5_data"  # HDF5输出目录
SCATTERER_NUMBERS = list(range(1, 3001))  # 散射体编号范围（根据实际文件调整）
NFREQ = 1  # 频率数（每个散射体）
MZ, MX = 20, 20  # 空间网格尺寸
NTRANS = 1  # 传输方向数
COMPRESSION = "gzip"  # HDF5压缩方式（可选：None/"gzip"/"lzf"）
VERIFY_SCT_NUM = 171  # 验证用散射体编号（确保该编号有对应.dat文件）
VERIFY_FREQ_NUM = 1  # 验证用频率编号


# ============================== 核心函数 ==============================
def read_xkai_dat(file_path: str, mz: int, mx: int) -> np.ndarray:
    """读取Xkai.dat，返回 (mz, mx) 复数数组"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Xkai文件不存在：{file_path}")

    # 读取数据（跳过首行注释）
    data = np.loadtxt(file_path, skiprows=1)
    if data.shape[0] != mz * mx:
        raise ValueError(f"Xkai数据行数异常：期望{mz * mx}行，实际{data.shape[0]}行")

    xkai = np.zeros((mz, mx), dtype=np.complex64)
    for z in range(mz):
        for x in range(mx):
            row_idx = z * mx + x
            real = data[row_idx, 2]
            imag = data[row_idx, 3]
            xkai[z, x] = real + 1j * imag
    return xkai


def read_etot_dat(file_path: str, ntrans: int, mz: int, mx: int) -> np.ndarray:
    """读取Etot.dat，返回 (ntrans, mz, mx) 复数数组"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Etot文件不存在：{file_path}")

    data = np.loadtxt(file_path)
    expected_rows = ntrans * mz * mx
    if data.shape[0] != expected_rows:
        raise ValueError(f"Etot数据行数异常：期望{expected_rows}行，实际{data.shape[0]}行")

    etot = np.zeros((ntrans, mz, mx), dtype=np.complex64)
    for itrans in range(ntrans):
        for z in range(mz):
            for x in range(mx):
                row_idx = itrans * mz * mx + z * mx + x
                real = data[row_idx, 0]
                imag = data[row_idx, 1]
                etot[itrans, z, x] = real + 1j * imag
    return etot


def generate_hdf5():
    """生成HDF5文件（核心逻辑）"""
    # 创建输出目录
    Path(HDF5_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"📁 HDF5文件输出目录：{HDF5_OUTPUT_DIR}")

    xkai_h5_path = Path(HDF5_OUTPUT_DIR) / "xkai_all.h5"
    etot_h5_path = Path(HDF5_OUTPUT_DIR) / "etot_all.h5"

    # 统计成功/失败的散射体
    success_scts = []
    failed_scts = []

    # 打开HDF5文件并写入数据
    with h5py.File(xkai_h5_path, "w") as xkai_h5, h5py.File(etot_h5_path, "w") as etot_h5:
        # 写入元数据
        meta = {
            "scatterer_numbers": SCATTERER_NUMBERS,
            "nfreq": NFREQ,
            "ntrans": NTRANS,
            "mz": MZ,
            "mx": MX,
            "xkai_dat_format": "z x real imag",
            "etot_dat_format": "real imag",
            "generated_time": str(np.datetime64('now')),
            "compression": COMPRESSION
        }
        xkai_h5.attrs.update(meta)
        etot_h5.attrs.update(meta)

        # 遍历所有散射体
        for idx, sct_num in enumerate(SCATTERER_NUMBERS):
            sct_name = f"Sct{sct_num:04d}"
            print(f"\n[{idx + 1}/{len(SCATTERER_NUMBERS)}] 处理 {sct_name}...")

            try:
                # 创建散射体分组
                xkai_sct_group = xkai_h5.create_group(sct_name)
                etot_sct_group = etot_h5.create_group(sct_name)

                # 遍历所有频率
                for ifre in range(NFREQ):
                    freq_num = ifre + 1
                    freq_name = f"Fre{freq_num:04d}"

                    # 构造.dat文件路径
                    xkai_dat_path = Path(XKAI_INPUT_DIR) / f"Xkai_Fre{freq_num:04d}_Sct{sct_num:04d}.dat"
                    etot_dat_path = Path(ETOT_INPUT_DIR) / f"Eytot_Fre{freq_num:04d}_Sct{sct_num:04d}.dat"

                    # 读取数据
                    xkai_data = read_xkai_dat(xkai_dat_path, MZ, MX)
                    etot_data = read_etot_dat(etot_dat_path, NTRANS, MZ, MX)

                    # 写入HDF5
                    xkai_sct_group.create_dataset(
                        freq_name,
                        data=xkai_data,
                        compression=COMPRESSION,
                        dtype=np.complex64
                    )
                    etot_sct_group.create_dataset(
                        freq_name,
                        data=etot_data,
                        compression=COMPRESSION,
                        dtype=np.complex64
                    )

                    print(f"  ✅ {freq_name} - Xkai/Etot写入成功")

                success_scts.append(sct_num)

            except Exception as e:
                print(f"  ❌ {sct_name} 处理失败：{str(e)}")
                failed_scts.append((sct_num, str(e)))
                continue

    # 打印统计结果
    print("\n" + "=" * 60)
    print(f"🎉 HDF5生成完成！")
    print(f"📊 统计：成功{len(success_scts)}个，失败{len(failed_scts)}个")
    print(f"📄 文件路径：")
    print(f"  - Xkai: {xkai_h5_path.absolute()}")
    print(f"  - Etot: {etot_h5_path.absolute()}")

    if failed_scts:
        print(f"\n❌ 失败列表（前10个）：")
        for sct_num, err in failed_scts[:10]:
            print(f"  Sct{sct_num:04d}: {err}")


def verify_hdf5():
    """验证HDF5文件与原始.dat文件一致性"""
    xkai_h5_path = Path(HDF5_OUTPUT_DIR) / "xkai_all.h5"
    etot_h5_path = Path(HDF5_OUTPUT_DIR) / "etot_all.h5"

    # 检查HDF5文件是否存在
    if not xkai_h5_path.exists() or not etot_h5_path.exists():
        raise FileNotFoundError(
            f"HDF5文件不存在！请先执行generate_hdf5()生成文件\n"
            f"缺失文件：\n  - {xkai_h5_path}\n  - {etot_h5_path}"
        )

    # 验证指定散射体和频率
    sct_verify = f"Sct{VERIFY_SCT_NUM:04d}"
    freq_verify = f"Fre{VERIFY_FREQ_NUM:04d}"
    print(f"\n🔍 验证 {sct_verify}/{freq_verify} 数据一致性：")

    # 1. 从HDF5读取数据
    with h5py.File(xkai_h5_path, "r") as f_xkai, h5py.File(etot_h5_path, "r") as f_etot:
        # 读取元数据
        meta = dict(f_xkai.attrs)
        print(f"\n📋 HDF5元数据：")
        for k, v in meta.items():
            print(f"  {k}: {v}")

        # 读取目标数据
        h5_xkai = f_xkai[f"{sct_verify}/{freq_verify}"][:]
        h5_etot = f_etot[f"{sct_verify}/{freq_verify}"][:]

    # 2. 从原始.dat文件读取数据
    xkai_dat_path = Path(XKAI_INPUT_DIR) / f"Xkai_Fre{VERIFY_FREQ_NUM:04d}_Sct{VERIFY_SCT_NUM:04d}.dat"
    etot_dat_path = Path(ETOT_INPUT_DIR) / f"Eytot_Fre{VERIFY_FREQ_NUM:04d}_Sct{VERIFY_SCT_NUM:04d}.dat"
    dat_xkai = read_xkai_dat(xkai_dat_path, MZ, MX)
    dat_etot = read_etot_dat(etot_dat_path, NTRANS, MZ, MX)

    # 3. 对比数据
    print(f"\n【Xkai对比】")
    print(f"  HDF5形状：{h5_xkai.shape} | DAT形状：{dat_xkai.shape}")
    print(f"  HDF5前5个值：{np.round(h5_xkai.flatten()[:5], 6)}")
    print(f"  DAT前5个值：{np.round(dat_xkai.flatten()[:5], 6)}")
    xkai_max_diff = np.max(np.abs(h5_xkai - dat_xkai))
    print(f"  最大差值：{xkai_max_diff:.6e} | 一致性：{'✅' if xkai_max_diff < 1e-6 else '❌'}")

    print(f"\n【Etot对比】")
    print(f"  HDF5形状：{h5_etot.shape} | DAT形状：{dat_etot.shape}")
    print(f"  HDF5前5个值：{np.round(h5_etot.flatten()[:5], 6)}")
    print(f"  DAT前5个值：{np.round(dat_etot.flatten()[:5], 6)}")
    etot_max_diff = np.max(np.abs(h5_etot - dat_etot))
    print(f"  最大差值：{etot_max_diff:.6e} | 一致性：{'✅' if etot_max_diff < 1e-6 else '❌'}")

    print(f"\n✅ 验证完成！")


if __name__ == "__main__":
    try:
        # 第一步：生成HDF5文件（必须先执行）
        generate_hdf5()

        # 第二步：验证HDF5文件（可选）
        verify_hdf5()

    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        raise  # 重新抛出异常，便于调试