import healpy
import numpy
import harmonics
from matplotlib import pyplot as plt
import spider_rfuns
import time

nside=16;
npix=12*nside*nside
ipix=range(npix)
dec,ra=healpy.pix2ang(nside,ipix)
map=numpy.cos(2*dec)
alm=healpy.map2alm(map)

alm[:]=0
alm2=alm.copy()
#alm2[2]=1
lmax=numpy.int(numpy.sqrt(2*len(alm)))
print lmax

plt.ion()

nn=len(alm)
lmax=numpy.int(numpy.sqrt(nn*2))
print lmax
alm[:]=0

I=numpy.complex(0,1)

l_ps=7
alm[:]=0
zeros=alm.copy()
alm[l_ps]=1

mm_tot_QQ=0
mm_tot_QU=0
mm_tot_UQ=0
mm_tot_UU=0

sum_uu=0
sum_qq=0

myind=782;
mmT,mmQ,mmU=healpy.alm2map([zeros,alm,zeros],nside)
mm_tot_QQ=mm_tot_QQ+mmQ*mmQ[myind]
mm_tot_QU=mm_tot_QU+mmQ*mmU[myind]
mm_tot_UQ=mm_tot_UQ+mmU*mmQ[myind]
mm_tot_UU=mm_tot_UU+mmU*mmU[myind]

sum_uu=sum_uu+mmU*mmU
sum_qq=sum_qq+mmQ*mmQ

#mm=mmQ+I*mmU;mm_tot=mm_tot+mm*numpy.conj(mm[myind])

i0=l_ps

for m in range(1,l_ps+1):
    print m
    i0=i0+(lmax-m)
    alm[:]=0
    alm[i0]=1/numpy.sqrt(2)
    mmT,mmQ,mmU=healpy.alm2map([zeros,alm,zeros],nside)
    mm_tot_QQ=mm_tot_QQ+mmQ*mmQ[myind]
    mm_tot_QU=mm_tot_QU+mmQ*mmU[myind]
    mm_tot_UQ=mm_tot_UQ+mmU*mmQ[myind]
    mm_tot_UU=mm_tot_UU+mmU*mmU[myind]
    #mm=mmQ+I*mmU;mm_tot=mm_tot+mm*numpy.conj(mm[myind])
    sum_uu=sum_uu+mmU*mmU
    sum_qq=sum_qq+mmQ*mmQ


    alm[i0]=I/numpy.sqrt(2)
    mmT,mmQ,mmU=healpy.alm2map([zeros,alm,zeros],nside)
    mm_tot_QQ=mm_tot_QQ+mmQ*mmQ[myind]
    mm_tot_QU=mm_tot_QU+mmQ*mmU[myind]
    mm_tot_UQ=mm_tot_UQ+mmU*mmQ[myind]
    mm_tot_UU=mm_tot_UU+mmU*mmU[myind]

    #mm=mmQ+I*mmU;mm_tot=mm_tot+mm*numpy.conj(mm[myind])

    sum_uu=sum_uu+mmU*mmU
    sum_qq=sum_qq+mmQ*mmQ

#mm_tot_QQ=mm_tot_QQ/(2*l_ps+1)
#mm_tot_UQ=mm_tot_UQ/(2*l_ps+1)
#mm_tot_QU=mm_tot_QU/(2*l_ps+1)
#mm_tot_UU=mm_tot_UU/(2*l_ps+1)

x=numpy.sin(dec)*numpy.sin(ra)
y=numpy.sin(dec)*numpy.cos(ra)
z=numpy.cos(dec)
dist=x*x[myind]+y*y[myind]+z*z[myind]
#numpy.sqrt((x-x[0])**2+(y-y[0])**2+(z-z[0])**2)


mycorr=mm_tot_QQ+mm_tot_UU+I*(mm_tot_QU-mm_tot_UQ);
mycorr2=mm_tot_QQ-mm_tot_UU+I*(mm_tot_QU+mm_tot_UQ);
#plt.clf();plt.plot(dist,numpy.abs(mm_tot_QQ+mm_tot_UU+I*(mm_tot_QU-mm_tot_UQ)),'.')
plt.clf();plt.plot(dist,numpy.abs(mycorr),'.')
plt.plot(dist,numpy.abs(mycorr2),'.')



xx=numpy.arange(0,1.001,0.001)*numpy.pi
yy=harmonics.sYlm(2,l_ps,2,numpy.arccos(dist),0*dist)
yy=yy*numpy.sqrt((2*l_ps+1)/(4*numpy.pi))
yy2=harmonics.sYlm(2,l_ps,-2,numpy.arccos(dist),0*dist);
yy2=yy2*numpy.sqrt((2*l_ps+1)/(4*numpy.pi))

plt.plot(dist,(numpy.abs(yy2)),'.')
plt.plot(dist,numpy.abs(yy),'.')

#plt.plot(dist,(numpy.abs(yy2))/numpy.sqrt( (2*l_ps+1)*(4*numpy.pi)),'.')#/(2*l_ps+1))

print numpy.mean(numpy.abs(yy2)-numpy.abs(mycorr))

##do spherical law of cosines to get angles alpha and gamma 
#as defined in fig one of http://arxiv.org/pdf/astro-ph/9710012v2.pdf

mydra=ra-ra[myind];
mydra[mydra>numpy.pi]=mydra[mydra>numpy.pi]-2*numpy.pi
mydra[mydra<-numpy.pi]=mydra[mydra<-numpy.pi]+2*numpy.pi


aa=time.time();mycosalpha,mycosgamma=spider_rfuns.fill_gamma_alpha(ra,dec);bb=time.time();bb-aa
mycosalpha=mycosalpha[myind,:]
mycosgamma=mycosgamma[myind,:]
myalpha=numpy.arccos(-mycosalpha)
mygamma=numpy.arccos(mycosgamma)


myphase=numpy.exp(2*I*(mygamma-myalpha))
myphase2=numpy.exp(2*I*(mygamma+myalpha))

yy=yy*myphase2
yy2=yy2*myphase

QQ_pred=0.5*(yy2.real+yy.real)
UU_pred=0.5*(yy2.real-yy.real)
QU_pred=0.5*(yy2.imag+yy.imag)
UQ_pred=0.5*(-yy2.imag+yy.imag)

psE=numpy.zeros(20)
psE[l_ps-2]=1
#dd,c1,c2=spider_rfuns.get_EB_corrs(ra,dec,psE,0*psE);plt.plot(dd,numpy.abs(c1),'.');plt.plot(dd,numpy.abs(c2),'.')
#assert(1==0)
#c1,c2=spider_rfuns.get_EB_corrs(ra,dec,psE,0*psE);
#c1=c1[:,myind];c2=c2[:,myind];plt.plot(dist,numpy.abs(c1),'.');plt.plot(dist,numpy.abs(c2),'.')
#assert(1==0)

QQ_pred,UU_pred,QU_pred,UQ_pred=spider_rfuns.get_EB_corrs(ra,dec,psE,0*psE)
QQ_pred=QQ_pred[myind,:]
UU_pred=UU_pred[myind,:]
QU_pred=QU_pred[myind,:]
UQ_pred=UQ_pred[myind,:]

d1=numpy.mean(numpy.abs(numpy.abs(QQ_pred)-numpy.abs(mm_tot_QQ)))
d2=numpy.mean(numpy.abs(QU_pred-mm_tot_QU))
d3=numpy.mean(numpy.abs(UQ_pred-mm_tot_UQ))
d4=numpy.mean(numpy.abs(UU_pred-mm_tot_UU))
print [d1,d2,d3,d4]




assert(1==0)





ii=numpy.isfinite(myphase)
myphase[~ii]=1

v1=(numpy.angle(myphase))
v1[v1>numpy.pi]-=numpy.pi
v1[v1<0]+=numpy.pi
v2=(numpy.angle((mm_tot_QQ+mm_tot_UU)+I*(mm_tot_QU-mm_tot_UQ)))
v2[v2>numpy.pi]-=numpy.pi
v2[v2<0]+=numpy.pi

plt.clf();plt.plot(v1,v2,'.')

pred=myphase*yy2

ii=numpy.isfinite(pred)
print numpy.mean(numpy.abs(pred[ii])-abs(mycorr[ii]))


assert(1==0)

plt.clf()
plt.plot(dist,mm_tot_QQ,'.')
plt.plot(dist,mm_tot_UU,'.')
plt.plot(dist,mm_tot_QU,'.')
plt.plot(dist,mm_tot_UQ,'.')
assert(1==0)


import spider_rfuns
covmat=spider_rfuns.interpolate_covariance(ra,dec,xx,yy)


