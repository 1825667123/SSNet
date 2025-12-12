import global_para_var as gv
import os
import re

def Save_Scatterer(epsr_matrix, mur_matrix, sigma_matrix, output_dir="output\Scatterer"):

    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r'^sct_(\d{4})\.inp$')
    max_index = 0
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            max_index = max(max_index, int(match.group(1)))
    next_index = max_index + 1

    filename = os.path.join(output_dir, f"sct_{next_index:04d}.inp")
    with open(filename, "w") as f:
        f.write("# z x epsr mur sigma\n")
        for z in range(epsr_matrix.shape[0]):
            for x in range(epsr_matrix.shape[1]):
                f.write(f"{z} {x} {epsr_matrix[z, x]} {mur_matrix[z, x]} {sigma_matrix[z, x]}\n")

    return filename, next_index


def Save_IncField(Eyinc):

    for ifre in range(gv.nfreq):
        tempstr = f'{ifre + 1:04d}'
        filename = f'output/Einc/Eyinc_Fre{tempstr}.dat'
        with open(filename, 'w') as fid:
            for itrans in range(gv.ntrtot):
                for k in range(gv.mz):
                    for i in range(gv.mx):
                        fid.write(f'{Eyinc[ifre,itrans,k,i].real:11.7e} {Eyinc[ifre,itrans,k,i].imag:11.7e}      \n')
    return


def Save_TotField(Eytot, scatter_file_number):

    for ifre in range(gv.nfreq):
        tempstr = f'{ifre + 1:04d}'
        filename = f'output/Etot/Eytot_Fre{tempstr}_Sct{scatter_file_number}.dat'
        with open(filename, 'w') as fid:
            for itrans in range(gv.ntrtot):
                for k in range(gv.mz):
                    for i in range(gv.mx):
                        fid.write(f'{Eytot[ifre,itrans,k,i].real:11.7e} {Eytot[ifre,itrans,k,i].imag:11.7e}      \n')
    return


