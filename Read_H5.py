import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset


class SSNetHDF5Dataset(Dataset):
    def __init__(self, xkai_h5_path: str, etot_h5_path: str, nfreq: int):

        try:
            self.xkai_h5 = h5py.File(xkai_h5_path, "r")
            self.etot_h5 = h5py.File(etot_h5_path, "r")
        except Exception as e:
            raise FileNotFoundError(f"无法打开HDF5文件：{str(e)}")

        self.meta = self.xkai_h5.attrs
        self.scatterer_numbers = self.meta["scatterer_numbers"]
        self.mz, self.mx = self.meta["mz"], self.meta["mx"]
        self.ntrans = self._get_actual_ntrans()
        self.nfreq = nfreq

        self.samples = []
        for sct_num in self.scatterer_numbers:
            sct_name = f"Sct{sct_num:04d}"
            freq_name = f"Fre0001"
            if (sct_name in self.xkai_h5 and freq_name in self.xkai_h5[sct_name] and
                    sct_name in self.etot_h5 and freq_name in self.etot_h5[sct_name]):
                self.samples.append((sct_name, freq_name))

        print(f"📊 SSNet数据集初始化完成：")
        print(f"   - 总样本数：{len(self.samples)}")
        print(f"   - 散射体数量：{len(self.scatterer_numbers)}")
        print(f"   - 频率数：{self.nfreq}")
        print(f"   - 空间网格：{self.mz}×{self.mx}")
        print(f"   - 传输方向数：{self.ntrans}（实际数据维度）")

    def _get_actual_ntrans(self):
        for sct_num in self.scatterer_numbers:
            sct_name = f"Sct{sct_num:04d}"
            freq_name = f"Fre0001"
            if sct_name in self.etot_h5 and freq_name in self.etot_h5[sct_name]:
                etot_sample = self.etot_h5[sct_name][freq_name][:]
                return etot_sample.shape[0]
        return 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sct_name, freq_name = self.samples[idx]

        xkai_complex = self.xkai_h5[sct_name][freq_name][:]
        xkai_complex = np.expand_dims(xkai_complex, axis=0)
        xkai_tensor = torch.tensor(xkai_complex, dtype=torch.complex64)

        etot_complex = self.etot_h5[sct_name][freq_name][:]
        etot_complex = np.expand_dims(etot_complex, axis=0)
        etot_tensor = torch.tensor(etot_complex, dtype=torch.complex64)

        return xkai_tensor, etot_tensor

    def __del__(self):
        if hasattr(self, "xkai_h5"):
            self.xkai_h5.close()
        if hasattr(self, "etot_h5"):
            self.etot_h5.close()


if __name__ == "__main__":
    TEST_XKAI_H5 = "./output/hdf5_data/xkai_all.h5"
    TEST_ETOT_H5 = "./output/hdf5_data/etot_all.h5"
    dataset = SSNetHDF5Dataset(TEST_XKAI_H5, TEST_ETOT_H5, nfreq=1)

    random.seed(42)
    np.random.seed(42)

    print("\n" + "="*50)
    print("1. 随机读取单个样本验证")
    print("="*50)
    random_idx = random.randint(0, len(dataset)-1)
    xkai_rand, etot_rand = dataset[random_idx]
    print(f"📌 随机索引：{random_idx}")
    print(f"   - 输入Xkai形状：{xkai_rand.shape}（期望：(1,20,20)）")
    print(f"   - 目标Etot形状：{etot_rand.shape}（期望：(1,{dataset.ntrans},20,20)）")
    print(f"   - 数据类型：Xkai={xkai_rand.dtype}，Etot={etot_rand.dtype}")

    xkai_abs = torch.abs(xkai_rand)
    etot_abs = torch.abs(etot_rand)
    print(f"\n📊 数值量级验证（随机样本 {random_idx}）：")
    print(f"   - Xkai模值范围：[{xkai_abs.min():.4f}, {xkai_abs.max():.4f}]")
    for trans_idx in range(dataset.ntrans):
        etot_trans_abs = etot_abs[0, trans_idx]
        print(f"   - Etot模值范围（传输方向{trans_idx}）：[{etot_trans_abs.min():.4f}, {etot_trans_abs.max():.4f}]")
        print(f"   - Etot关键位置(10,10)值（方向{trans_idx}）：{etot_rand[0, trans_idx, 10, 10]}")

    print("\n" + "="*50)
    print("2. 批量随机读取5个样本验证")
    print("="*50)
    sample_count = min(5, len(dataset))
    random_idxs = random.sample(range(len(dataset)), sample_count)
    for i, idx in enumerate(random_idxs):
        xkai, etot = dataset[idx]
        etot_abs = torch.abs(etot)
        print(f"\n📌 随机样本 {i+1}（索引{idx}）：")
        print(f"   - Xkai最大模值：{torch.abs(xkai).max():.4f}")
        for trans_idx in range(dataset.ntrans):
            print(f"   - Etot最大模值（方向{trans_idx}）：{etot_abs[0, trans_idx].max():.4f}")

    print("\n" + "="*50)
    print("3. 验证前100个样本的Etot量级分布")
    print("="*50)
    max_etot_list = []
    check_count = min(100, len(dataset))
    for idx in range(check_count):
        _, etot = dataset[idx]
        max_etot = torch.abs(etot).max().item()
        max_etot_list.append(max_etot)
        if max_etot > 1000:
            print(f"⚠️  异常样本（索引{idx}）：Etot最大模值={max_etot:.4f}")

    max_etot_arr = np.array(max_etot_list)
    print(f"\n📈 前{check_count}个样本Etot量级统计：")
    print(f"   - 平均值：{max_etot_arr.mean():.4f}")
    print(f"   - 中位数：{np.median(max_etot_arr):.4f}")
    print(f"   - 最大值：{max_etot_arr.max():.4f}")
    print(f"   - 最小值：{max_etot_arr.min():.4f}")

    print("\n" + "="*50)
    print("4. 原始第一个样本验证")
    print("="*50)
    xkai, etot = dataset[0]
    print(f"\n✅ 样本维度验证：")
    print(f"   - 输入Xkai形状：{xkai.shape}")
    print(f"   - 目标Etot形状：{etot.shape}")
    for trans_idx in range(dataset.ntrans):
        print(f"   - 第一个样本Etot(10,10)值（方向{trans_idx}）：{etot[0, trans_idx, 10, 10]}")