import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import font_manager
import os

FONT_FILE_PATH = "/home/guanxin/.fonts/SIMHEI.TTF"
font_prop = font_manager.FontProperties(fname=FONT_FILE_PATH)

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
h5_dir = "./output/hdf5_data"
xkai_dat_dir = "./output/xkai"
etot_dat_dir = "./output/Etot"


def load_xkai_from_dat(sct_idx, fre_idx):
    fname = f"{xkai_dat_dir}/Xkai_Fre{fre_idx:04d}_Sct{sct_idx:04d}.dat"
    data = np.loadtxt(fname, skiprows=1)
    z = data[:, 0].astype(int)
    x = data[:, 1].astype(int)
    mz = z.max() + 1 if len(z) > 0 else 1
    mx = x.max() + 1 if len(x) > 0 else 1
    xkai_dat = np.zeros((mz, mx), dtype=np.complex128)
    xkai_dat[z, x] = data[:, 2] + 1j * data[:, 3]
    return xkai_dat


def load_Etot_from_dat(sct_idx, fre_idx, mz, mx):
    fname = f"{etot_dat_dir}/Eytot_Fre{fre_idx:04d}_Sct{sct_idx:04d}.dat"
    data = np.loadtxt(fname)
    etot_dat = (data[:, 0] + 1j * data[:, 1]).reshape(-1, mz, mx)[0].squeeze()
    return etot_dat


with h5py.File(f"{h5_dir}/xkai_all.h5", "r") as fx, h5py.File(f"{h5_dir}/etot_all.h5", "r") as fe:
    sct = random.choice([k for k in fx.keys() if k.startswith('Sct')])
    freq = random.choice([k for k in fx[sct].keys() if k.startswith('Fre')])
    sct_idx, fre_idx = int(sct[3:]), int(freq[3:])

    xkai_h5 = fx[f'{sct}/{freq}'][:]
    etot_h5 = fe[f'{sct}/{freq}'][:].squeeze()
    mz, mx = xkai_h5.shape

    xkai_dat = load_xkai_from_dat(sct_idx, fre_idx)
    etot_dat = load_Etot_from_dat(sct_idx, fre_idx, mz, mx)

    fig, ax = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle(f'散射体 {sct} - {freq}', fontsize=20, fontproperties=font_prop)
    plots = [
        (np.real(xkai_h5), 'Xkai(.h5) 实部', ax[0, 0]), (np.imag(xkai_h5), 'Xkai(.h5) 虚部', ax[0, 1]),
        (np.abs(xkai_h5), 'Xkai(.h5) 幅值', ax[0, 2]), (np.angle(xkai_h5), 'Xkai(.h5) 相位', ax[0, 3]),
        (np.real(xkai_dat), 'Xkai(.dat) 实部', ax[1, 0]), (np.imag(xkai_dat), 'Xkai(.dat) 虚部', ax[1, 1]),
        (np.abs(xkai_dat), 'Xkai(.dat) 幅值', ax[1, 2]), (np.angle(xkai_dat), 'Xkai(.dat) 相位', ax[1, 3]),
        (np.real(etot_h5), 'Etot(.h5) 实部', ax[2, 0]), (np.imag(etot_h5), 'Etot(.h5) 虚部', ax[2, 1]),
        (np.abs(etot_h5), 'Etot(.h5) 幅值', ax[2, 2]), (np.angle(etot_h5), 'Etot(.h5) 相位', ax[2, 3]),
        (np.real(etot_dat), 'Etot(.dat) 实部', ax[3, 0]), (np.imag(etot_dat), 'Etot(.dat) 虚部', ax[3, 1]),
        (np.abs(etot_dat), 'Etot(.dat) 幅值', ax[3, 2]), (np.angle(etot_dat), 'Etot(.dat) 相位', ax[3, 3])
    ]
    for d, t, a in plots:
        im = a.imshow(d, cmap='viridis')
        a.set_title(t, fontproperties=font_prop)
        a.set_xlabel('X 网格', fontproperties=font_prop)
        a.set_ylabel('Z 网格', fontproperties=font_prop)
        plt.colorbar(im, ax=a, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"/home/guanxin/SSNet/sct_{sct_idx}_fre_{fre_idx}_vis.png")
    plt.show()