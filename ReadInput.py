import numpy as np
import matplotlib.pyplot as plt
import global_para_var as gv
import os
import re

def get_latest_scatter_file(input_dir="output/Scatterer"):
    """获取最新的散射体网格文件（修复：按数字编号排序）"""
    pattern = re.compile(r'^sct_(\d{4})\.inp$')
    files = [f for f in os.listdir(input_dir) if pattern.match(f)]
    if not files:
        raise FileNotFoundError("未找到任何散射体网格文件")
    # 关键修复：按文件编号的数字大小排序（而不是字符串排序）
    files.sort(key=lambda x: int(pattern.match(x).group(1)))
    return os.path.join(input_dir, files[-1])


def get_all_scatter_files(input_dir="output/Scatterer"):
    """获取所有散射体网格文件"""
    pattern = re.compile(r'^sct_(\d{4})\.inp$')
    files = [f for f in os.listdir(input_dir) if pattern.match(f)]
    if not files:
        raise FileNotFoundError("未找到任何散射体网格文件")
    # 按文件编号升序排序
    files.sort(key=lambda x: int(pattern.match(x).group(1)))
    return [os.path.join(input_dir, f) for f in files]


def Read_Input(scatter_file=None):
    # 读取背景介质参数
    print("read background parameters...")
    with open("input/background.inp", 'r') as fid:
        next(fid)
        gv.ttcer, gv.xsig, gv.ttmiur = list(map(float, fid.readline().strip().split()))

    # 读取发射机配置
    print("read transmitter config...")
    with open("input/tloc.inp", 'r') as fid:
        next(fid)
        gv.ntrtot = int(fid.readline().strip())
        gv.xptr = np.empty(gv.ntrtot, dtype=float)
        gv.zptr = np.empty(gv.ntrtot, dtype=float)
        next(fid)
        for m in range(gv.ntrtot):
            gv.xptr[m], gv.zptr[m] = list(map(float, fid.readline().strip().split()))

    # 读取接收机配置
    print("read receiver config...")
    with open("input/rloc.inp", 'r') as fid:
        next(fid)
        gv.nrectot = int(fid.readline().strip())
        gv.xrr = np.empty(gv.nrectot, dtype=float)
        gv.zrr = np.empty(gv.nrectot, dtype=float)
        next(fid)
        for m in range(gv.nrectot):
            gv.xrr[m], gv.zrr[m] = list(map(float, fid.readline().strip().split()))

    # 读取网格配置
    print("read cell config...")
    with open("input/cell.inp", 'r') as fid:
        next(fid)
        gv.mx, gv.mz = list(map(int, fid.readline().strip().split()))
        next(fid)
        gv.xx1, gv.zz1 = list(map(float, fid.readline().strip().split()))
        next(fid)
        gv.dx, gv.dz = list(map(float, fid.readline().strip().split()))
    gv.dxz = gv.dx * gv.dz
    gv.xxf = np.tile(np.arange(gv.xx1, gv.xx1 + gv.dx * gv.mx, gv.dx), [gv.mz, 1])
    gv.xzf = np.tile(np.arange(gv.zz1, gv.zz1 + gv.dz * gv.mz, gv.dz).reshape(-1, 1), [1, gv.mx])

    # 读取频率配置
    print("read frequency number...")
    with open("input/freq.inp", 'r') as fid:
        next(fid)
        gv.nfreq = int(fid.readline().strip())
        gv.freq = np.empty(gv.nfreq, dtype=float)
        next(fid)
        for m in range(gv.nfreq):
            gv.freq[m] = float(fid.readline().strip())
    gv.xomega = 2 * gv.pai * gv.freq
    gv.cer = gv.ttcer + gv.xsig / (1j * gv.xomega * gv.epsilon0)
    gv.mur = complex(gv.ttmiur)
    gv.xk = gv.xomega * np.sqrt(gv.cer * gv.epsilon0 * gv.mur * gv.mu0)

    # 读取频率配置
    print("read frequency number...")
    with open("input/freq.inp", 'r') as fid:
        next(fid)
        gv.nfreq = int(fid.readline().strip())
        gv.freq = np.empty(gv.nfreq, dtype=float)
        next(fid)
        for m in range(gv.nfreq):
            gv.freq[m] = float(fid.readline().strip())
    gv.xomega = 2 * gv.pai * gv.freq
    gv.cer = gv.ttcer + gv.xsig / (1j * gv.xomega * gv.epsilon0)
    gv.mur = complex(gv.ttmiur)
    gv.xk = gv.xomega * np.sqrt(gv.cer * gv.epsilon0 * gv.mur * gv.mu0)

    # 读取散射体信息
    print("read scatterer config from grid file...")
    # 初始化参数矩阵（用背景值填充）
    gv.scatepsr = np.full([gv.mz, gv.mx], gv.ttcer, dtype=float)
    gv.scatsigma = np.full([gv.mz, gv.mx], gv.xsig, dtype=float)
    gv.scatmur = np.full([gv.mz, gv.mx], gv.ttmiur, dtype=float)

    # 获取最新的散射体网格文件
    if scatter_file is None:
        scatter_file = get_latest_scatter_file()
    print(f"正在读取散射体文件: {scatter_file}")

    # 逐个网格点读取参数
    with open(scatter_file, 'r') as fid:
        next(fid)
        for line in fid:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            z = int(parts[0])
            x = int(parts[1])
            epsr = float(parts[2])
            mur = float(parts[3])
            sigma = float(parts[4])

            gv.scatepsr[z, x] = epsr
            gv.scatsigma[z, x] = sigma
            gv.scatmur[z, x] = mur

    gv.xkai = np.empty([gv.nfreq, gv.mz, gv.mx], dtype=complex)
    for ifre in range(gv.nfreq):
        gv.xkai[ifre, :, :] = (gv.scatepsr + gv.scatsigma / (1j * gv.xomega[ifre] * gv.epsilon0)) / gv.cer[ifre] - 1

    # 读取前向散射配置
    print("read forward calculation configuration...")
    with open("input/forwardconfig.inp", 'r') as fid:
        next(fid)
        gv.maxistep = int(fid.readline().strip())
        next(fid)
        gv.maxresidualerror = float(fid.readline().strip())

    return


if __name__ == "__main__":

    Read_Input()

    # 格式：网格(z,x) - 频率索引0（第一个频率）的xkai实部/虚部
    print("\n监控特定网格xkai值（频率0）：")
    print(f"网格(0,0): 实部={gv.xkai[0,0,0].real:.7e}, 虚部={gv.xkai[0,0,0].imag:.7e}")
    print(f"网格(5,5): 实部={gv.xkai[0,5,5].real:.7e}, 虚部={gv.xkai[0,5,5].imag:.7e}")
    print(f"网格(10,10): 实部={gv.xkai[0,10,10].real:.7e}, 虚部={gv.xkai[0,10,10].imag:.7e}")
    # 可根据需要添加更多网格，格式：print(f"网格(z,x): 实部={gv.xkai[频率索引,z,x].real:.7e}, 虚部={gv.xkai[频率索引,z,x].imag:.7e}")

    os.makedirs("output", exist_ok=True)
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 绘制第一个频率的xkai实部
    plt.figure(figsize=(8, 6))
    plt.imshow(np.real(gv.xkai[0]), cmap='jet', origin='lower', extent=[0, gv.mx, 0, gv.mz])
    plt.colorbar(label='xkai 实部')
    plt.title(f'{gv.freq[0] / 1e6:.1f}MHz散射体对比度')
    plt.xlabel('x 网格数')
    plt.ylabel('z 网格数')
    plt.tight_layout()
    plt.savefig("output/scatter_readxkai.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("热力图已保存至 output/scatter_readxkai.png")