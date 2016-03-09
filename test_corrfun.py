import healpy
import numpy
from matplotlib import pyplot as plt
nside=16;
npix=12*nside*nside
ipix=range(npix)
dec,ra=healpy.pix2ang(nside,ipix)
map=numpy.cos(2*dec)
alm=healpy.map2alm(map)


nn=len(alm)
lmax=numpy.int(numpy.sqrt(nn*2))
print lmax
alm[:]=0

I=numpy.complex(0,1)

#dipoles
#alm[1]=1
#alm[lmax]=1

#quadrupoles
#alm[2]=1
#alm[lmax+lmax-1]=1
#alm[lmax+1]=I

#octopoles


#i0=3
#ii=2
#for i in range(ii):
#    i0=i0+(lmax-i-1)
#alm[i0]=1



#mm=healpy.alm2map(alm,nside)

#healpy.mollview(mm)
#plt.ion()
#plt.show()


l_ps=9
alm[:]=0
alm[l_ps]=1
mm_tot=0
mm=healpy.alm2map(alm,nside)
mm_tot=mm_tot+mm*mm[0]
i0=l_ps
for m in range(1,l_ps+1):
    print m
    i0=i0+(lmax-m)
    alm[:]=0
    alm[i0]=1/numpy.sqrt(2)
    mm=healpy.alm2map(alm,nside)
    mm_tot=mm_tot+mm*mm[0]
    alm[i0]=I/numpy.sqrt(2)
    mm=healpy.alm2map(alm,nside)
    mm_tot=mm_tot+mm*mm[0]

mm_tot=mm_tot/(2*l_ps+1)
x=numpy.sin(dec)*numpy.sin(ra)
y=numpy.sin(dec)*numpy.cos(ra)
z=numpy.cos(dec)
dist=x*x[0]+y*y[0]+z*z[0]
#numpy.sqrt((x-x[0])**2+(y-y[0])**2+(z-z[0])**2)

plt.clf()
plt.plot(dist,mm_tot,'.')
xx=numpy.arange(-100,101)*0.01
pp=numpy.zeros(l_ps+1)
pp[l_ps]=1;
yy=numpy.polynomial.legendre.legval(xx,pp)/(4*numpy.pi)
plt.plot(xx,yy)


import spider_rfuns
covmat=spider_rfuns.interpolate_covariance(ra,dec,xx,yy)
