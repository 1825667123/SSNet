"""
dataset.py：单独的HDF5数据读取模块，专门给SSNet提供训练数据
输出：(xkai_tensor, etot_tensor)
- xkai_tensor: 输入，shape=(nfreq, mz, mx)，torch.complex64
- etot_tensor: 目标，shape=(nfreq, ntrans, mz, mx)，torch.complex64
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SSNetHDF5Dataset(Dataset):
    def __init__(self, xkai_h5_path: str, etot_h5_path: str, nfreq: int):
        """
        初始化数据集
        :param xkai_h5_path: xkai_all.h5 的绝对/相对路径
        :param etot_h5_path: etot_all.h5 的绝对/相对路径
        :param nfreq: 模型的频率数（从SSNet中获取）
        """
        # 打开HDF5文件（只读模式，不加载到内存）
        try:
            self.xkai_h5 = h5py.File(xkai_h5_path, "r")
            self.etot_h5 = h5py.File(etot_h5_path, "r")
        except Exception as e:
            raise FileNotFoundError(f"无法打开HDF5文件：{str(e)}")

        # 读取数据元信息（从HDF5的attrs中获取，无需手动填写）
        self.meta = self.xkai_h5.attrs
        self.scatterer_numbers = self.meta["scatterer_numbers"]  # 所有散射体编号
        self.mz, self.mx = self.meta["mz"], self.meta["mx"]  # 空间网格数（20×20）
        self.ntrans = self.meta["ntrans"]  # Etot传输方向数（2）
        self.nfreq = nfreq  # 模型要求的频率数（1）

        # 生成样本列表：每个散射体+频率对应1个样本
        self.samples = []
        for sct_num in self.scatterer_numbers:
            sct_name = f"Sct{sct_num:04d}"  # 散射体名称：Sct0001~Sct0010（或Sct3000）
            freq_name = f"Fre0001"  # 频率名称：你的数据只有Fre0001
            # 检查该散射体+频率的路径是否存在（避免缺失文件）
            if (sct_name in self.xkai_h5 and freq_name in self.xkai_h5[sct_name] and
                    sct_name in self.etot_h5 and freq_name in self.etot_h5[sct_name]):
                self.samples.append((sct_name, freq_name))

        # 打印数据集信息（方便调试）
        print(f"📊 SSNet数据集初始化完成：")
        print(f"   - 总样本数：{len(self.samples)}")
        print(f"   - 散射体数量：{len(self.scatterer_numbers)}")
        print(f"   - 频率数：{self.nfreq}（仅Fre0001）")
        print(f"   - 空间网格：{self.mz}×{self.mx}")
        print(f"   - Etot传输方向数：{self.ntrans}")

    def __len__(self) -> int:
        """返回样本总数（训练时DataLoader会调用）"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        按索引获取单个样本（训练时自动调用）
        :param idx: 样本索引
        :return: (xkai_tensor, etot_tensor)
        """
        sct_name, freq_name = self.samples[idx]

        # 1. 读取Xkai（输入数据）：(mz, mx) → (nfreq, mz, mx)
        xkai_complex = self.xkai_h5[sct_name][freq_name][:]  # 原始数据：(20,20) 复数
        xkai_complex = np.expand_dims(xkai_complex, axis=0)  # 扩展频率维度：(1,20,20)
        xkai_tensor = torch.tensor(xkai_complex, dtype=torch.complex64)  # 转为torch张量

        # 2. 读取Etot（目标数据）：(ntrans, mz, mx) → (nfreq, ntrans, mz, mx)
        etot_complex = self.etot_h5[sct_name][freq_name][:]  # 原始数据：(2,20,20) 复数
        etot_complex = np.expand_dims(etot_complex, axis=0)  # 扩展频率维度：(1,2,20,20)
        etot_tensor = torch.tensor(etot_complex, dtype=torch.complex64)  # 转为torch张量

        return xkai_tensor, etot_tensor

    def __del__(self):
        """对象销毁时关闭HDF5文件（避免资源泄露）"""
        if hasattr(self, "xkai_h5"):
            self.xkai_h5.close()
        if hasattr(self, "etot_h5"):
            self.etot_h5.close()


if __name__ == "__main__":

    TEST_XKAI_H5 = "./output/hdf5_data/xkai_all.h5"
    TEST_ETOT_H5 = "./output/hdf5_data/etot_all.h5"
    dataset = SSNetHDF5Dataset(TEST_XKAI_H5, TEST_ETOT_H5, nfreq=1)

    # 读取第一个样本，验证维度
    xkai, etot = dataset[0]
    print(f"\n✅ 样本维度验证：")
    print(f"   - 输入Xkai形状：{xkai.shape}（期望：(1,20,20)）")
    print(f"   - 目标Etot形状：{etot.shape}（期望：(1,2,20,20)）")
    print(f"   - 数据类型：{xkai.dtype}（期望：torch.complex64）")