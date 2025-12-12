import numpy as np
import scipy.special as sp
import global_para_var as gv

def Compute_Gii(gejyym):

    print("compute self2self Greens...")
    rho=np.sqrt((gv.xxf[0,0]-gv.xxf)**2 + (gv.xzf[0,0]-gv.xzf)**2)
    a=np.sqrt(gv.dxz/gv.pai)

    for ifre in range(gv.nfreq):
        GreenOutputLoggingEm= -0.25*gv.xomega[ifre]*gv.mu0*gv.mur*(sp.jv(0, gv.xk[ifre]*rho)-1j*sp.yv(0, gv.xk[ifre]*rho))
        GreenOutputLoggingEm[0,0]= -gv.xomega[ifre]*gv.mu0*gv.mur/(2.0*gv.xk[ifre]*a)*(sp.jv(1, gv.xk[ifre]*a)-1j*sp.yv(1, gv.xk[ifre]*a))  \
                                                                    +1j*gv.xomega[ifre]*gv.mu0*gv.mur/(gv.pai*gv.xk[ifre]**2*a**2) # we use H_1^(2) ka here to deal with singularity
        AssembleGreen(gejyym[ifre,:,:],GreenOutputLoggingEm)
    return

def AssembleGreen(geyy,GreenEm):

    geyy[:gv.mz,:gv.mx]=GreenEm
    geyy[:gv.mz,-gv.mx+1:]=GreenEm[:,-gv.mx+1:][:,::-1]
    geyy[-gv.mz+1:,:gv.mx]=GreenEm[-gv.mz+1:,:][::-1,:]
    geyy[-gv.mz+1:,-gv.mx+1:]=GreenEm[-gv.mz+1:,-gv.mx+1:][::-1,::-1]

    return