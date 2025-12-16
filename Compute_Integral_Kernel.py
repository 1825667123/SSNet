import numpy as np
import global_para_var as gv
from Compute_Gii import Compute_Gii
from ReadInput import Read_Input

_gejyym = None

def _precompute_greens_function():
    """预计算格林函数（gejyym）"""
    global _gejyym
    if _gejyym is None:
        Read_Input()
        _gejyym = np.empty((gv.nfreq, 2 * gv.mz - 1, 2 * gv.mx - 1), dtype=complex)
        Compute_Gii(_gejyym)
    return _gejyym


def compute_integral_kernel(contrast):

    # 预计算格林函数
    gejyym = _precompute_greens_function()

    # 初始化矩阵
    integral_kernel = np.empty((gv.nfreq, gv.mz, gv.mx), dtype=complex)
    unit_operator = np.zeros_like(integral_kernel, dtype=complex)  # 单位算子矩阵

    for ifre in range(gv.nfreq):

        greens_local = gejyym[ifre, :gv.mz, :gv.mx]
        integral_kernel[ifre, :, :] = greens_local * contrast[ifre, :, :]
        unit_operator[ifre] = np.eye(gv.mz, gv.mx, dtype=complex)

    final_integral_kernel = unit_operator - integral_kernel
    return final_integral_kernel

