import h5py
import numpy as np
import os

# ============================== 配置参数==============================
XKAI_INPUT_DIR = "./output/xkai"
ETOT_INPUT_DIR = "./output/Etot"
HDF5_OUTPUT_DIR = "./output/hdf5_data"
SCATTERER_NUMBERS = list(range(1, 3001))  # 已有的散射体编号（Sct0001~Sct0010填1~11）
NFREQ = 1  # 每个散射体的频率数
MZ, MX = 20, 20  # 空间网格数
NTRANS = 1  # 传输方向数
COMPRESSION = "gzip"  # 不压缩


# ============================== 核心逻辑==============================
def read_xkai_dat(file_path: str, mz: int, mx: int) -> np.ndarray:
    """读取Xkai，返回 (mz, mx) 复数数组"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Xkai文件不存在：{file_path}")
    data = np.loadtxt(file_path, skiprows=1)
    xkai = np.zeros((mz, mx), dtype=np.complex64)
    for z in range(mz):
        for x in range(mx):
            row_idx = z * mx + x
            real = data[row_idx, 2]
            imag = data[row_idx, 3]
            xkai[z, x] = real + 1j * imag
    return xkai


def read_etot_dat(file_path: str, ntrans: int, mz: int, mx: int) -> np.ndarray:
    """读取Etot，返回 (ntrans, mz, mx) 复数数组"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Etot文件不存在：{file_path}")
    data = np.loadtxt(file_path)
    etot = np.zeros((ntrans, mz, mx), dtype=np.complex64)
    for itrans in range(ntrans):
        for z in range(mz):
            for x in range(mx):
                row_idx = itrans * mz * mx + z * mx + x
                real = data[row_idx, 0]
                imag = data[row_idx, 1]
                etot[itrans, z, x] = real + 1j * imag
    return etot


def main():
    os.makedirs(HDF5_OUTPUT_DIR, exist_ok=True)
    print(f"📁 HDF5文件将保存到：{HDF5_OUTPUT_DIR}")

    xkai_h5_path = os.path.join(HDF5_OUTPUT_DIR, "xkai_all.h5")
    etot_h5_path = os.path.join(HDF5_OUTPUT_DIR, "etot_all.h5")

    with h5py.File(xkai_h5_path, "w") as xkai_h5, h5py.File(etot_h5_path, "w") as etot_h5:
        meta = {
            "scatterer_numbers": SCATTERER_NUMBERS,
            "nfreq": NFREQ,
            "ntrans": NTRANS,
            "mz": MZ,
            "mx": MX,
            "xkai_dat_format": "z x real imag ",
            "etot_dat_format": "real imag"
        }
        xkai_h5.attrs.update(meta)
        etot_h5.attrs.update(meta)

        for sct_num in SCATTERER_NUMBERS:
            sct_name = f"Sct{sct_num:04d}"
            print(f"\n🔄 处理 {sct_name}...")

            xkai_sct_group = xkai_h5.create_group(sct_name)
            etot_sct_group = etot_h5.create_group(sct_name)

            for ifre in range(NFREQ):
                freq_num = ifre + 1
                freq_name = f"Fre{freq_num:04d}"

                xkai_dat_path = os.path.join(XKAI_INPUT_DIR, f"Xkai_Fre{freq_num:04d}_Sct{sct_num:04d}.dat")
                xkai_data = read_xkai_dat(xkai_dat_path, MZ, MX)

                etot_dat_path = os.path.join(ETOT_INPUT_DIR, f"Eytot_Fre{freq_num:04d}_Sct{sct_num:04d}.dat")
                etot_data = read_etot_dat(etot_dat_path, NTRANS, MZ, MX)

                xkai_sct_group.create_dataset(freq_name, data=xkai_data, compression=COMPRESSION)
                etot_sct_group.create_dataset(freq_name, data=etot_data, compression=COMPRESSION)
                print(f"  ✅ 完成 {freq_name}（Xkai+Etot）")

    print("\n" + "=" * 50)
    print("🎉 所有存在的.dat文件已转换完成！")
    print(f"📄 生成HDF5文件：")
    print(f"  - Xkai: {xkai_h5_path}")
    print(f"  - Etot: {etot_h5_path}")
    print("=" * 50)


if __name__ == "__main__":

    # main()

    # ============================== 验证逻辑==============================

    xkai_h5_path = os.path.join(HDF5_OUTPUT_DIR, "xkai_all.h5")
    etot_h5_path = os.path.join(HDF5_OUTPUT_DIR, "etot_all.h5")
    sct_verify_num = 171  # 要验证的散射体编号
    freq_verify_num = 1  # 要验证的频率编号
    sct_verify = f"Sct{sct_verify_num:04d}"
    freq_verify = f"Fre{freq_verify_num:04d}"

    print(f"\n🔍 对比 {sct_verify}/{freq_verify} 具体数据：")

    # 1. 从HDF5读取指定散射体和频率的数据
    with h5py.File(xkai_h5_path, "r") as f_xkai, h5py.File(etot_h5_path, "r") as f_etot:

        h5_xkai = f_xkai[f"{sct_verify}/{freq_verify}"][:]
        h5_etot = f_etot[f"{sct_verify}/{freq_verify}"][:]

    # 2. 从原始.dat文件读取相同散射体和频率的数据
    xkai_dat_verify_path = os.path.join(XKAI_INPUT_DIR,
                                        f"Xkai_Fre{freq_verify_num:04d}_Sct{sct_verify_num:04d}.dat")
    etot_dat_verify_path = os.path.join(ETOT_INPUT_DIR,
                                        f"Eytot_Fre{freq_verify_num:04d}_Sct{sct_verify_num:04d}.dat")

    dat_xkai = read_xkai_dat(xkai_dat_verify_path, MZ, MX)
    dat_etot = read_etot_dat(etot_dat_verify_path, NTRANS, MZ, MX)

    # 3. 打印Xkai对比
    print("\n【Xkai数据对比】")
    print(f"HDF5前5个：{h5_xkai.flatten()[:5]}")
    print(f"原始.dat前5个：{dat_xkai.flatten()[:5]}")
    print(f"HDF5后5个：{h5_xkai.flatten()[-5:]}")
    print(f"原始.dat后5个：{dat_xkai.flatten()[-5:]}")

    # 4. 打印Etot对比
    print("\n【Etot数据对比】")
    print(f"HDF5前5个：{h5_etot.flatten()[:5]}")
    print(f"原始.dat前5个：{dat_etot.flatten()[:5]}")
    print(f"HDF5后5个：{h5_etot.flatten()[-5:]}")
    print(f"原始.dat后5个：{dat_etot.flatten()[-5:]}")

    # 5. 打印维度和数据一致性校验
    print(f"\n【维度对比】")
    print(f"Xkai - HDF5维度：{h5_xkai.shape} | 原始.dat维度：{dat_xkai.shape}")
    print(f"Etot - HDF5维度：{h5_etot.shape} | 原始.dat维度：{dat_etot.shape}")

    # 校验数据是否完全一致
    xkai_diff = np.max(np.abs(h5_xkai - dat_xkai))
    etot_diff = np.max(np.abs(h5_etot - dat_etot))
    print(f"\n【数据一致性校验】")
    print(f"Xkai数据是否一致：{'✅' if xkai_diff < 1e-6 else '❌'}")
    print(f"Etot数据是否一致：{'✅' if etot_diff < 1e-6 else '❌'}")