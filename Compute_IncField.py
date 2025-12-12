import numpy as np
import scipy.special as sp
import global_para_var as gv


def Compute_IncField(Eyinc):

    print("compute incident fields...")

    rho=np.empty([gv.ntrtot,gv.mz,gv.mx],dtype=float)
    for itrans in range(gv.ntrtot):
        rho[itrans,:,:]=np.sqrt((gv.xptr[itrans]-gv.xxf)**2 + (gv.zptr[itrans]-gv.xzf)**2)
    for ifre in range(gv.nfreq):
        Eyinc[ifre, :, :, : ] = -0.25*gv.xomega[ifre]*gv.mu0*gv.mur*(sp.jv(0, gv.xk[ifre]*rho)-1j*sp.yv(0, gv.xk[ifre]*rho))
    return