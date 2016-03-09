import healpy
import numpy
import spider_rfuns
from matplotlib import pyplot as plt

nside=64
fsky=0.02
ipix=range(int(nside*nside*12*fsky))

dec,ra=healpy.pix2ang(nside,ipix)

lmax=100
lvec=numpy.arange(1.0,100.0)
ps=0.10/lvec**2

xx=0.001*numpy.arange(-1000,1001)





uk_arcmin=12.
pix_area=40000.0/12/nside/nside
noise_per_pix=uk_arcmin/60/numpy.sqrt(pix_area)

npix=len(ipix)
noise_mat=numpy.eye(npix)*(noise_per_pix**2)
mycorr=spider_rfuns.spec2corr(ps,xx)
mysig=spider_rfuns.interpolate_covariance(ra,dec,xx,mycorr)


noise_mat=noise_mat+mysig

#take advantage of the fact that the noise is diagonal
mycov_inv=numpy.eye(npix)/(noise_per_pix**2)
csc=numpy.dot(mycov_inv,mysig)
mycurve=numpy.sum(csc.transpose()*csc)
myerr=1.0/numpy.sqrt(mycurve)
print myerr
alpha=numpy.arange(-0.5,10.0,0.5)*myerr
mylike=0*alpha
mychisq=0*alpha
mylogdet=0*alpha
for i in range(len(alpha)):
    print alpha[i]
    mycov=noise_mat+alpha[i]*mysig
    mychol=numpy.linalg.cholesky(mycov)
    myinv=numpy.linalg.inv(mycov)

    mychisq[i]=numpy.sum(noise_mat   *myinv)
    mylogdet[i]=2*numpy.sum(numpy.log(numpy.diag(mychol)))

#covmat=spider_rfuns.interpolate_covariance(ra,dec,xx,yy)

mylike=-0.5*mychisq-0.5*mylogdet
plt.ion()
plt.clf()
plt.plot(alpha,mylike-mylike[0])
