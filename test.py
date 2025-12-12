import h5py
import numpy as np

# 目标：找到真正的Etot真实值（量级1e4~1e6）和fin_para真实值
XKAI_PATH = "./output/hdf5_data/xkai_all.h5"
ETOT_PATH = "./output/hdf5_data/etot_all.h5"

def print_hdf5_structure(h5_path, name):
    print(f"\n===== {name} 文件结构 =====")
    with h5py.File(h5_path, "r") as f:
        def visit_func(path, obj):
            if isinstance(obj, h5py.Dataset):
                # 打印路径、形状、数值范围（关键：判断是否是Etot/fin_para）
                data = obj[:]
                print(f"路径：{path} | 形状：{data.shape} | 数值范围：{np.min(data.real)} ~ {np.max(data.real)}")
            else:
                print(f"路径：{path} | 类型：Group")
        f.visititems(visit_func)

# 打印两个文件的完整结构
print_hdf5_structure(XKAI_PATH, "xkai_all")
print_hdf5_structure(ETOT_PATH, "etot_all")