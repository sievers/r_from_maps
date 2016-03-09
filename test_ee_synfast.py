import healpy
import numpy
import harmonics
from matplotlib import pyplot as plt
import spider_rfuns
import time
plt.ion()


nside=16;


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

#QQ_pred,UU_pred,QU_pred,UQ_pred=spider_rfuns.get_EB_corrs(ra,dec,cl[2:],0*cl[2:])
QQ_pred,UU_pred,QU_pred,UQ_pred=spider_rfuns.get_EB_corrs(ra,dec,0*cl[2:],cl[2:])

QQ_obs=numpy.dot(Qmaps.transpose(),Qmaps)/nsim
UU_obs=numpy.dot(Umaps.transpose(),Umaps)/nsim
UQ_obs=numpy.dot(Qmaps.transpose(),Umaps)/nsim
QU_obs=numpy.dot(Umaps.transpose(),Qmaps)/nsim
print numpy.mean(numpy.abs(QQ_obs-QQ_pred)),numpy.mean(numpy.abs(QQ_obs+QQ_pred))
print numpy.mean(numpy.abs(UU_obs-UU_pred)),numpy.mean(numpy.abs(UU_obs+UU_pred))
print numpy.mean(numpy.abs(UQ_obs-UQ_pred)),numpy.mean(numpy.abs(UQ_obs+UQ_pred))
print numpy.mean(numpy.abs(QU_obs-QU_pred)),numpy.mean(numpy.abs(QU_obs+QU_pred))
