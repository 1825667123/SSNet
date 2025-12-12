import numpy as np
from scipy.fftpack import fft2,ifft2
import global_para_var as gv

def Compute_TotField(Eyinc, Eytot, gejyymfre):
    
    print("compute total fields...")
    for ifre in range(gv.nfreq):
        for itrans in range(gv.ntrtot):
            print(f"perform the BCGSFFT iteration for frequency {ifre + 1} and transmitter {itrans + 1}, please wait...")
            BCGSFFT(Eyinc[ifre,itrans,:, :], Eytot[ifre,itrans,:, :], gejyymfre[ifre,:, :], ifre)
    return

def BCGSFFT(Eyinc, Eytot, gejyymfre, ifre):

    rho = complex(1.0, 0.0)
    beta = complex(1.0, 0.0)
    dabu = complex(1.0, 0.0)
    rho1 = complex(0.0, 0.0)
    c1, c2 = complex(0.0, 0.0), complex(0.0, 0.0)

    itabnormal = 500
    memm = 0
    rerror0 = 1.0e+09
    rerror = 0.0
    rnormb = 0.0

    # 初始化数组
    xr0 = np.zeros((gv.mz,gv.mx), dtype=complex)
    xrn = np.zeros((gv.mz,gv.mx), dtype=complex)
    xpn = np.zeros((gv.mz,gv.mx), dtype=complex)
    xvn = np.zeros((gv.mz,gv.mx), dtype=complex)
    xxn = np.zeros((gv.mz,gv.mx), dtype=complex)
    xtn = np.zeros((gv.mz,gv.mx), dtype=complex)

    # 设置初始值
    xr0 = Eyinc
    xrn = xr0

    rnormb = np.sum(np.conj(xr0) * xr0).real
    print(f"rnornmb= {rnormb:.3f}   {gv.maxresidualerror:.7E}")

    # 迭代求解
    for istep in range(1, gv.maxistep + 1):
        rho1 = np.sum(np.conj(xr0) * xrn)
        beta = (rho1 / rho) * (beta / dabu)
        xpn= xrn + beta * (xpn - dabu * xvn)
        ComputeLJ(xvn, xpn, gejyymfre, ifre)
        beta = np.sum(np.conj(xr0) * xvn)
        beta = rho1 / beta
        xrn = xrn - beta * xvn
        rnorms = np.sum(np.conj(xrn) * xrn).real
        ComputeLJ(xtn, xrn, gejyymfre, ifre)
        c1 = np.sum(np.conj(xtn) * xrn)
        c2 = np.sum(np.conj(xtn) * xtn)
        if c2 == complex(0.0, 0.0):
            dabu = complex(1.0, 0.0)
        else:
            dabu = c1 / c2
        xxn= xxn + beta * xpn + dabu * xrn
        xrn= xrn - dabu * xtn
        rerror = np.sqrt(rnorms / rnormb)
        if rerror < rerror0:
            memm += 1
            rerror0 = rerror
        print(f"{istep:5d} {memm:5d} {rerror:.7f}")
        if rerror <= gv.maxresidualerror or memm >= itabnormal:
            break

    Eytot[:,:] = xxn
    return


def ComputeLJ(Eyinc, Eytot, gejyymfre, ifre):

    Jymfre = np.zeros((2*gv.mz-1, 2*gv.mx-1), dtype=complex)
    JFFT(Eytot, Jymfre, ifre)
    Eysct=ifft2(Jymfre * gejyymfre)[:gv.mz, :gv.mx]*gv.dxz
    Eyinc[:,:]= Eytot - Eysct
    return

def JFFT(Ey, Jyfre, ifre):

    Jy=1j*gv.xomega[ifre]*gv.cer[ifre]*gv.epsilon0*gv.xkai[ifre,::]*Ey
    Jyfre[:,:]=fft2(np.pad(Jy, ((0, gv.mz-1), (0, gv.mx-1))))
    return

