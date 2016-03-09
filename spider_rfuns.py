import numpy
import ctypes
import harmonics
import time

mylib=ctypes.cdll.LoadLibrary("libspider_rfuns.so")


interpolate_covariance_c=mylib.interpolate_covariance
interpolate_covariance_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]

fill_gamma_alpha_c=mylib.fill_gamma_alpha
fill_gamma_alpha_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]

fill_QU_corr_ee_old_c=mylib.fill_QU_corr_ee_old
fill_QU_corr_ee_old_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]

fill_QU_corr_eb_c=mylib.fill_QU_corr_eb
fill_QU_corr_eb_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]

#void interpolate_covariance(double *ra, double *dec, int n, double *costheta, double *corr, int npt, double *mat)
def fill_QUcorr_eb(corr1E,corr2E,corr1B,corr2B,alpha,gamma):


    nn=corr1E.shape[0]

    corrE=numpy.zeros([2*nn,2*nn])
    corrB=numpy.zeros([2*nn,2*nn])

    fill_QU_corr_eb_c(corr1E.ctypes.data,corr2E.ctypes.data,corr1B.ctypes.data,corr2B.ctypes.data,alpha.ctypes.data,gamma.ctypes.data,nn,corrE.ctypes.data,corrB.ctypes.data)
    return corrE,corrB
                      
def fill_QUcorr_ee_old(corr1,corr2,alpha,gamma):
    if (True):
        corrQQ=numpy.zeros(corr1.shape)
        corrQU=numpy.zeros(corr1.shape)
        corrUQ=numpy.zeros(corr1.shape)
        corrUU=numpy.zeros(corr1.shape)
        fill_QU_corr_ee_old_c(corr1.ctypes.data,corr2.ctypes.data,alpha.ctypes.data,gamma.ctypes.data,corr1.size,corrQQ.ctypes.data,corrQU.ctypes.data,corrUQ.ctypes.data,corrUU.ctypes.data)
    else:
        c1=corr1*numpy.cos(-2*(alpha+gamma))
        c2=corr2*numpy.cos(2*(gamma-alpha))
        corrQQ=0.5*(c1+c2)
        corrUU=0.5*(c1-c2)
        
        c1=corr1*numpy.sin(-2*(alpha+gamma))
        c2=corr2*numpy.sin(2*(gamma-alpha))
        
        corrQU=0.5*(c2-c1)
        corrUQ=0.5*(c2+c1)
    
    return corrQQ,corrUU,corrQU,corrUQ

def interpolate_covariance(ra,dec,costheta,corr):
    n=len(ra)
    covmat=numpy.zeros([n,n])
    interpolate_covariance_c(ra.ctypes.data,dec.ctypes.data,len(ra),costheta.ctypes.data,corr.ctypes.data,len(costheta),covmat.ctypes.data)

    return covmat


def fill_gamma_alpha(ra,dec):
    npt=len(ra)
    alpha=numpy.zeros([npt,npt])
    gamma=numpy.zeros([npt,npt])
    fill_gamma_alpha_c(ra.ctypes.data,dec.ctypes.data,npt,alpha.ctypes.data,gamma.ctypes.data)
    return alpha,gamma


def spec2corr(pspec,x):
    mycorr=0*x;
    pp=numpy.zeros(len(pspec)+1)
    pp[1:]=pspec
    mycorr=mycorr+numpy.polynomial.legendre.legval(x,pp)
    return mycorr/(4*numpy.pi)


def get_EB_corrs(ra,dec,psE,psB):
    aa_start=time.time()
    aa=time.time()
    alpha,gamma=fill_gamma_alpha(ra,dec);
    bb=time.time();
    #print 'elapsed time to fill gamma and alpha is ' + repr(bb-aa)
    ninterp=5000;
    costh=1.0*numpy.arange(ninterp)
    costh=costh-numpy.mean(costh)
    costh=costh/max(costh)
    costh[0]=numpy.round(costh[0])
    costh[-1]=numpy.round(costh[-1])
    
    corr1E_vec=0*costh
    corr2E_vec=0*costh

    corr1B_vec=0*costh
    corr2B_vec=0*costh

    aa=time.time()
    for i in range(len(psE)):
        l=i+2
        fac=numpy.sqrt((2*l+1)/(4*numpy.pi))
        tmp=fac*harmonics.sYlm(2,l,-2,numpy.arccos(costh),0*costh)
        corr1E_vec=corr1E_vec+tmp.real*psE[i]
        corr1B_vec=corr1B_vec+tmp.real*psB[i]

        tmp=fac*harmonics.sYlm(2,l,2,numpy.arccos(costh),0*costh)#*(psE[i]-psB[i])
        corr2E_vec=corr2E_vec+tmp.real*psE[i]
        corr2B_vec=corr2B_vec-tmp.real*psB[i]
        #if (psE[i]>0):
        #    print "ell is " + repr(l)

    bb=time.time()
    #print 'elapsed time is to calculate correlation is ' + repr(bb-aa)

    nn=len(ra)


    aa=time.time()
    corr1=numpy.zeros([nn,nn])
    corr2=numpy.zeros([nn,nn])
    corr1E=interpolate_covariance(ra,dec,(costh),corr1E_vec.real)
    corr2E=interpolate_covariance(ra,dec,(costh),corr2E_vec.real)

    corr1B=interpolate_covariance(ra,dec,(costh),corr1B_vec.real)
    corr2B=interpolate_covariance(ra,dec,(costh),corr2B_vec.real)
    bb=time.time()
    #print 'elapsed time to interpolate correlation is ' + repr(bb-aa)


    aa=time.time()

    corrE,corrB=fill_QUcorr_eb(corr1E,corr2E,corr1B,corr2B,alpha,gamma)

    bb=time.time()
    #print 'elapsed time to form Q/U is ' + repr(bb-aa)
    bb_stop=time.time()
    print 'total elapsed time to form E/B covariances is ' + repr(bb_stop-aa_start)
    return corrE,corrB



def get_EB_corrs_old(ra,dec,psE,psB):
    aa_start=time.time()
    aa=time.time()
    alpha,gamma=fill_gamma_alpha(ra,dec);
    bb=time.time();
    print 'elapsed time to fill gamma and alpha is ' + repr(bb-aa)
    ninterp=5000;
    costh=1.0*numpy.arange(ninterp)
    costh=costh-numpy.mean(costh)
    costh=costh/max(costh)
    costh[0]=numpy.round(costh[0])
    costh[-1]=numpy.round(costh[-1])
    
    corr1_vec=0*costh
    corr2_vec=0*costh
    aa=time.time()
    for i in range(len(psE)):
        l=i+2
        fac=numpy.sqrt((2*l+1)/(4*numpy.pi))
        tmp=fac*harmonics.sYlm(2,l,-2,numpy.arccos(costh),0*costh)*(psE[i]+psB[i])
        corr1_vec=corr1_vec+tmp.real

        tmp=fac*harmonics.sYlm(2,l,2,numpy.arccos(costh),0*costh)*(psE[i]-psB[i])
        corr2_vec=corr2_vec+tmp.real
        if (psE[i]>0):
            print "ell is " + repr(l)

    bb=time.time()
    print 'elapsed time is to calculate correlation is ' + repr(bb-aa)

    nn=len(ra)


    aa=time.time()
    corr1=numpy.zeros([nn,nn])
    corr2=numpy.zeros([nn,nn])
    corr1=interpolate_covariance(ra,dec,(costh),corr1_vec.real)
    corr2=interpolate_covariance(ra,dec,(costh),corr2_vec.real)
    bb=time.time()
    print 'elapsed time to interpolate correlation is ' + repr(bb-aa)

    I=numpy.complex(0,1.0)



    aa=time.time()
    if (True):
        corrQQ,corrUU,corrQU,corrUQ=fill_QUcorr_ee_old(corr1,corr2,alpha,gamma)
    else:
        corr1=corr1*numpy.exp(-2*I*(alpha+gamma))
        corr2=corr2*numpy.exp(2*I*(gamma-alpha))
        #corr1=numpy.exp(2*I*(gamma-alpha));corr2=numpy.exp(-2*I*(alpha+gamma));return corr1,corr2

        corrQQ=0.5*(corr1.real+corr2.real)
        corrUU=0.5*(corr1.real-corr2.real)
        #be careful about signs here
        corrQU=0.5*(-corr1.imag+corr2.imag)
        corrUQ=0.5*(corr1.imag+corr2.imag)
    bb=time.time()
    print 'elapsed time to form Q/U is ' + repr(bb-aa)
    bb_stop=time.time()
    print 'total elapsed time is ' + repr(bb_stop-aa_start)
    return corrQQ,corrUU,corrQU,corrUQ
