import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from SaveData import Save_Scatterer

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PAI = np.pi
EPSILON0 = 8.8541878128e-12  # 真空介电常数 (F/m)


def read_global_params():
    """读取背景、频率参数"""
    # 读取背景介质参数
    with open("input/background.inp", 'r') as fid:
        next(fid)
        ttcer, xsig, _ = list(map(float, fid.readline().strip().split()))

    # 读取频率配置
    with open("input/freq.inp", 'r') as fid:
        next(fid)
        nfreq = int(fid.readline().strip())
        freq = np.empty(nfreq, dtype=float)
        next(fid)
        for m in range(nfreq):
            freq[m] = float(fid.readline().strip())

    # 预计算背景复介电常数和角频率
    xomega = 2 * PAI * freq
    cer = ttcer + xsig / (1j * xomega * EPSILON0)
    return ttcer, xsig, nfreq, xomega, cer


# 全局参数初始化
GLOBAL_PARAMS = read_global_params()
TTcer, Xsig, NFREQ, Xomega, Cer = GLOBAL_PARAMS


def point_in_triangle(x, z, vertices):
    """判断点是否在三角形内"""
    (x1, z1), (x2, z2), (x3, z3) = vertices
    v0 = (x3 - x1, z3 - z1)
    v1 = (x2 - x1, z2 - z1)
    v2 = (x - x1, z - z1)
    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def generate_scatter(
        grid_size=(20, 20),
        num_scatterers=3,
        min_size=1,
        max_size=2,
        epsr_range=(1.0, 2.5),
        mur_range=(1.0, 1.0),
        sigma_range=(0.001, 0.05),
        background_epsr=1.0,
        background_mur=1.0,
        background_sigma=0.0
):
    mz, mx = grid_size
    grid = np.zeros((mz, mx), dtype=int)
    scatterers = []

    # 生成散射体
    for s_id in range(1, num_scatterers + 1):
        shape = random.choice(["triangle", "rectangle", "circle"])
        placed = False
        max_attempts = 1000
        attempts = 0
        while not placed and attempts < max_attempts:
            attempts += 1
            temp_grid = np.zeros((mz, mx), dtype=bool)
            if shape == "rectangle":
                w, h = random.randint(min_size, max_size), random.randint(min_size, max_size)
                x0, z0 = random.randint(0, mx - w), random.randint(0, mz - h)
                temp_grid[z0:z0 + h, x0:x0 + w] = True
                params = (x0, z0, w, h)
            elif shape == "circle":
                r = random.randint(min_size, max_size // 2)
                cx, cz = random.randint(r, mx - r - 1), random.randint(r, mz - r - 1)
                z, x = np.ogrid[:mz, :mx]
                temp_grid[np.sqrt((z - cz) ** 2 + (x - cx) ** 2) <= r] = True
                params = (cx, cz, r)
            elif shape == "triangle":
                side_len = random.randint(min_size, max_size)
                center = (random.randint(side_len, mx - side_len), random.randint(side_len, mz - side_len))
                angle = random.uniform(0, 2 * np.pi)
                vertices = [
                    (center[0] + side_len * np.cos(angle), center[1] + side_len * np.sin(angle)),
                    (center[0] + side_len * np.cos(angle + 2 * np.pi / 3),
                     center[1] + side_len * np.sin(angle + 2 * np.pi / 3)),
                    (center[0] + side_len * np.cos(angle + 4 * np.pi / 3),
                     center[1] + side_len * np.sin(angle + 4 * np.pi / 3))
                ]
                vertices = [(int(round(x)), int(round(z))) for x, z in vertices]
                x1, z1 = vertices[0]
                x2, z2 = vertices[1]
                x3, z3 = vertices[2]
                for x in range(min(x1, x2, x3), max(x1, x2, x3) + 1):
                    for z in range(min(z1, z2, z3), max(z1, z2, z3) + 1):
                        if 0 <= x < mx and 0 <= z < mz and point_in_triangle(x, z, vertices):
                            temp_grid[z, x] = True
                params = vertices
            if not np.any(np.logical_and(grid > 0, temp_grid)):
                grid[temp_grid] = s_id
                placed = True
                scatterers.append({
                    "id": s_id, "shape": shape, "params": params,
                    "epsr": round(random.uniform(*epsr_range), 2),
                    "mur": round(random.uniform(*mur_range), 2),
                    "sigma": round(random.uniform(*sigma_range), 4)
                })
        if attempts >= max_attempts:
            print(f"警告：散射体{s_id}放置失败，已跳过")

    # 生成参数矩阵
    epsr_matrix = np.full((mz, mx), background_epsr, dtype=float)
    mur_matrix = np.full((mz, mx), background_mur, dtype=float)
    sigma_matrix = np.full((mz, mx), background_sigma, dtype=float)
    for s in scatterers:
        mask = (grid == s["id"])
        epsr_matrix[mask] = s["epsr"]
        mur_matrix[mask] = s["mur"]
        sigma_matrix[mask] = s["sigma"]

    save_path, next_index = Save_Scatterer(epsr_matrix, mur_matrix, sigma_matrix)
    print(f"网格参数已保存至: {save_path}")

    # 2. 保存Xkai文件
    output_dir = 'output/Xkai'
    os.makedirs(output_dir, exist_ok=True)
    for ifre in range(NFREQ):
        # 计算当前频率的xkai
        scatter_cer = epsr_matrix + sigma_matrix / (1j * Xomega[ifre] * EPSILON0)
        xkai = scatter_cer / Cer[ifre] - 1

        # 生成Xkai文件名
        fname = f"{output_dir}/Xkai_Fre{ifre + 1:04d}_Sct{next_index:04d}.dat"
        with open(fname, 'w') as f:
            f.write("# z x xkai_real xkai_imag\n")
            for z in range(mz):
                for x in range(mx):
                    f.write(f"{z:4d} {x:4d} {xkai[z, x].real:11.7e} {xkai[z, x].imag:11.7e}\n")
    print(f"Xkai数据已保存至: {output_dir} (Sct{next_index:04d})")
    # --------------------------------------------------------------------------

    return grid, scatterers, next_index


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    num_runs = 2890
    last_grid = None
    last_scatterers = None
    last_index = 0

    for i in range(num_runs):
        print(f"\n第{i + 1}/{num_runs}次生成:")
        grid, scatterers, next_index = generate_scatter(
            grid_size=(20, 20),
            num_scatterers=random.randint(6, 8),
            min_size=2,
            max_size=4
        )
        last_grid = grid
        last_scatterers = scatterers
        last_index = next_index

    # 绘制最后一次的图像
    if last_grid is not None and last_scatterers is not None:
        mz, mx = last_grid.shape
        num_scatterers = len(last_scatterers)

        plt.figure(figsize=(8, 6))
        colors = ["white"] + [plt.cm.Set3(i) for i in range(num_scatterers)]
        plt.imshow(last_grid, cmap=ListedColormap(colors), origin="lower", extent=[0, mx, 0, mz])
        plt.grid(True, color="gray", linewidth=0.5, linestyle="--")
        plt.xlabel("x (网格数)")
        plt.ylabel("z (网格数)")
        plt.title(f"散射体分布 ({mx}×{mz}网格) - sct_{last_index:04d}")

        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]) for i in range(1, num_scatterers + 1)]
        labels = [f"散射体{s['id']} ({s['shape']}):\nεr={s['epsr']}, μr={s['mur']}, σ={s['sigma']}" for s in
                  last_scatterers]
        plt.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.6, 1), fontsize=8)

        plt.tight_layout()
        plt.savefig("output/scatter_generate.png", bbox_inches="tight", dpi=200)
        plt.close()
        print(f"\n最后一次散射体图像已保存至 output/scatter_generate.png")

    print(f"\n已完成{num_runs}次生成，文件在output/Scatterer/")