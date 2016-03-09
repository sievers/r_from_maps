import healpy
import numpy
import harmonics
from matplotlib import pyplot as plt
import spider_rfuns
import time
plt.ion()
nside=32;

lmax=20;
cl=numpy.arange(lmax)*1.0
cl[0:2]=0
cl=0*cl
#cl[9:12]=1.0
cl[9]=2.0
zz=0*cl
cls=[zz,zz,zz,cl]
mmT,mmQ,mmU=healpy.synfast(cls,nside)
nsim=4000

Qmaps=numpy.zeros([nsim,len(mmQ)])
Umaps=numpy.zeros([nsim,len(mmU)])
for i in range(nsim):
    mmT,mmQ,mmU=healpy.synfast(cls,nside)
    Qmaps[i,:]=mmQ
    Umaps[i,:]=mmU

npix=nside*nside*12
ipix=range(npix)
dec,ra=healpy.pix2ang(nside,ipix)

Epred,Bpred=spider_rfuns.get_EB_corrs(ra,dec,cl[2:],cl[2:])

sz=Qmaps.shape
allmaps=numpy.zeros([sz[0],2*sz[1]])
allmaps[:,0::2]=Qmaps
allmaps[:,1::2]=Umaps
bigcov=numpy.dot(allmaps.transpose(),allmaps)/nsim
print (numpy.mean(numpy.abs(bigcov-Bpred))),numpy.mean(numpy.abs(bigcov+Bpred))
