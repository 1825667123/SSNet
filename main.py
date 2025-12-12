from ReadInput import Read_Input, get_all_scatter_files
from Compute_IncField import Compute_IncField
from Compute_Gii import Compute_Gii
from FFT_2DGreen import Gej_FFT
from Compute_TotField import Compute_TotField
from SaveData import Save_TotField,Save_IncField

import numpy as np
import global_para_var as gv
import torch
import os
import re

print("CUDA 可用: ", torch.cuda.is_available())
print("CUDA 版本: ", torch.version.cuda)
print("cuDNN 可用: ", torch.backends.cudnn.enabled)
print("cuDNN 版本: ", torch.backends.cudnn.version())



def main():

    Read_Input(scatter_file=None)

    Eyinc = np.empty((gv.nfreq,gv.ntrtot,gv.mz,gv.mx),dtype=complex)
    Compute_IncField(Eyinc)
    Save_IncField(Eyinc)

    all_scatter_files = get_all_scatter_files()
    print(f"共找到 {len(all_scatter_files)} 个散射体文件，开始批量计算入射场...")

    for scatter_file in all_scatter_files:

        file_name = os.path.basename(scatter_file)
        file_number = re.match(r'^sct_(\d{4})\.inp$', file_name).group(1)
        print(f"\n===== 处理散射体 {file_number} =====")

        Read_Input(scatter_file=scatter_file)

        #计算自身到自身格林函数，然后实施FFT
        gejyym=np.empty((gv.nfreq,2*gv.mz-1,2*gv.mx-1),dtype=complex)
        Compute_Gii(gejyym)
        gejyymfre=np.empty(np.shape(gejyym),dtype=complex)
        Gej_FFT(gejyym, gejyymfre)

        #计算总场，通过BCGS-FFT
        Eytot=np.empty(np.shape(Eyinc),dtype=complex)
        Compute_TotField(Eyinc, Eytot, gejyymfre)
        Save_TotField(Eytot, file_number)

if __name__ == "__main__":
    main()