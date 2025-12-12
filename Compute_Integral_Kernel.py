import numpy as np
import global_para_var as gv
from Compute_Gii import Compute_Gii
from ReadInput import Read_Input

def compute_integral_kernel():

    Read_Input()
    gejyym = np.empty((gv.nfreq, 2 * gv.mz - 1, 2 * gv.mx - 1), dtype=complex)
    Compute_Gii(gejyym)
    integral_kernel = np.empty((gv.nfreq, gv.mz, gv.mx), dtype=complex)
    unit_operator = np.zeros_like(integral_kernel, dtype=complex)  # 单位算子矩阵

    for ifre in range(gv.nfreq):

        greens_local = gejyym[ifre, :gv.mz, :gv.mx]
        contrast = gv.xkai[ifre, :, :]
        integral_kernel[ifre, :, :] = greens_local * contrast
        unit_operator[ifre] = np.eye(gv.mz, gv.mx, dtype=complex)
        print(f"频率 {ifre + 1}/{gv.nfreq} ，维度: {integral_kernel[ifre].shape}")

    final_intefral_kernel = unit_operator - integral_kernel

    print("积分核矩阵计算完成")
    return final_intefral_kernel

if __name__ == "__main__":

    kernel = compute_integral_kernel()
    np.save('output/integral_kernel_matrix.npy', kernel)
    print("积分核矩阵已保存为 integral_kernel_matrix.npy")