from scipy.fftpack import fft2
import global_para_var as gv

def Gej_FFT(gejyym, gejyymfre):

    for ifre in range(gv.nfreq):
        gejyymfre[ifre,:,:]=fft2(gejyym[ifre,:,:])
        
    return