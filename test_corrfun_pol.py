import healpy
import numpy
import harmonics
from matplotlib import pyplot as plt
def choose(a,b):
    tot=1.0
    for i in range(1,b+1):
        tot=(a+1-i)*tot/i
    return numpy.int(numpy.round(tot))
def get_ell_fac(l):
    #get coefficient for s=2,m=2 spin spherical harmonic
    #fac=1.0
    #fac=fac*(l+0.0)*(l-1.0)/(l+1.0)/(l+2.0)
    #fac=numpy.sqrt(fac/4/numpy.pi)
    fac=numpy.sqrt( (2*l+1)/(4*numpy.pi))
    return fac

def spin2(theta,l):
    tot=0
    s=2
    m=2
    for r in range(0,l-s+1):
        fac=choose(l-s,r)*choose(l+s,r+s-m)*(-1**(l-r-s))
        tot+=fac*numpy.tan(0.5*theta)**(m-s-2*r)
    ans=tot*(-1**m)*get_ell_fac(l)*numpy.sin(0.5*theta)**(2*l)
    return tot


nside=32;
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
alm2[3*lmax+2]=numpy.complex(-1,2)
alm2[5]=1.3
mapE=healpy.alm2map([alm,alm2,alm],nside);mapEQ=mapE[1];mapEU=mapE[2];
mapB=healpy.alm2map([alm,alm,alm2],nside);mapBQ=mapB[1];mapBU=mapB[2];

print numpy.mean(numpy.abs(mapEQ-mapBU))
print numpy.mean(numpy.abs(mapEU+mapBQ))

plt.ion()
#assert(1==0)



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


l_ps=7
alm[:]=0
zeros=alm.copy()
alm[l_ps]=1

mm_tot_QQ=0
mm_tot_QU=0
mm_tot_UQ=0
mm_tot_UU=0
mm_tot_TT=0
sum_uu=0
sum_qq=0
mm_tot=0;
myind=0;
mmT,mmQ,mmU=healpy.alm2map([zeros,alm,zeros],nside)
#mmT,mmQ,mmU=healpy.alm2map([alm,zeros,zeros],nside)
mm_tot_TT=mm_tot_TT+mmT*mmT[myind]
mm_tot_QQ=mm_tot_QQ+mmQ*mmQ[myind]
mm_tot_QU=mm_tot_QU+mmQ*mmU[myind]
mm_tot_UQ=mm_tot_UQ+mmU*mmQ[myind]
mm_tot_UU=mm_tot_UU+mmU*mmU[myind]

sum_uu=sum_uu+mmU*mmU
sum_qq=sum_qq+mmQ*mmQ

mm=mmQ+I*mmU;mm_tot=mm_tot+mm*numpy.conj(mm[myind])

i0=l_ps

for m in range(1,l_ps+1):
    print m
    i0=i0+(lmax-m)
    alm[:]=0
    alm[i0]=1/numpy.sqrt(2)
    mmT,mmQ,mmU=healpy.alm2map([zeros,alm,zeros],nside)
    #mmT,mmQ,mmU=healpy.alm2map([alm,zeros,zeros],nside)
    mm_tot_TT=mm_tot_TT+mmT*mmT[myind]
    mm_tot_QQ=mm_tot_QQ+mmQ*mmQ[myind]
    mm_tot_QU=mm_tot_QU+mmQ*mmU[myind]
    mm_tot_UQ=mm_tot_UQ+mmU*mmQ[myind]
    mm_tot_UU=mm_tot_UU+mmU*mmU[myind]
    mm=mmQ+I*mmU;mm_tot=mm_tot+mm*numpy.conj(mm[myind])
    sum_uu=sum_uu+mmU*mmU
    sum_qq=sum_qq+mmQ*mmQ


    alm[i0]=I/numpy.sqrt(2)
    mmT,mmQ,mmU=healpy.alm2map([zeros,alm,zeros],nside)
    #mmT,mmQ,mmU=healpy.alm2map([alm,zeros,zeros],nside)
    mm_tot_TT=mm_tot_TT+mmT*mmT[myind]
    mm_tot_QQ=mm_tot_QQ+mmQ*mmQ[myind]
    mm_tot_QU=mm_tot_QU+mmQ*mmU[myind]
    mm_tot_UQ=mm_tot_UQ+mmU*mmQ[myind]
    mm_tot_UU=mm_tot_UU+mmU*mmU[myind]

    mm=mmQ+I*mmU;mm_tot=mm_tot+mm*numpy.conj(mm[myind])

    sum_uu=sum_uu+mmU*mmU
    sum_qq=sum_qq+mmQ*mmQ

mm_tot_QQ=mm_tot_QQ/(2*l_ps+1)
mm_tot_UQ=mm_tot_UQ/(2*l_ps+1)
mm_tot_QU=mm_tot_QU/(2*l_ps+1)
mm_tot_UU=mm_tot_UU/(2*l_ps+1)

x=numpy.sin(dec)*numpy.sin(ra)
y=numpy.sin(dec)*numpy.cos(ra)
z=numpy.cos(dec)
dist=x*x[myind]+y*y[myind]+z*z[myind]
#numpy.sqrt((x-x[0])**2+(y-y[0])**2+(z-z[0])**2)


plt.clf();plt.plot(dist,numpy.abs(mm_tot_QQ+mm_tot_UU+I*(mm_tot_QU-mm_tot_UQ)),'.')


xx=numpy.arange(0,1.001,0.001)*numpy.pi
yy=harmonics.sYlm(2,l_ps,2,xx,0*xx);
yy2=harmonics.sYlm(2,l_ps,-2,xx,0*xx);
plt.plot(numpy.cos(xx),(numpy.abs(yy2))/numpy.sqrt( (2*l_ps+1)*(4*numpy.pi)))#/(2*l_ps+1))

#4-->9, 6-->13
#3/numpy.pi)
assert(1==0)


asdf=spin2(dec,l_ps)
ii=numpy.abs(asdf)>10
asdf[ii]=0
plt.plot(numpy.cos(dec),asdf,'.')
assert(1==0)
alm[:]=0
alm[l_ps]=1.0
mmT,mmQ1,mmU1=healpy.alm2map([zeros,alm,zeros],nside)
mmT,mmQ2,mmU2=healpy.alm2map([zeros,zeros,alm],nside)

mmE=mmQ1+I*mmU1
mmB=mmQ2+I*mmU2
mmE2=-2*mmE
mmB2=-2*I*mmB
a_plus=mmE2+mmB2
a_minus=mmE2-mmB2




xx=numpy.arange(-100,101)*0.01
#xx=1.0+(xx-1)*(l_ps/(l_ps+1))
pp=numpy.zeros(l_ps+2)
pp[l_ps-0]=1;
yy=numpy.polynomial.legendre.legval(xx,pp)/(4*numpy.pi)
plt.plot(xx,numpy.abs(yy))

plt.plot(numpy.cos(dec),numpy.abs(mmE.real),'.')



##do spherical law of cosines to get angles alpha and gamma 
#as defined in fig one of http://arxiv.org/pdf/astro-ph/9710012v2.pdf

mydra=ra-ra[myind];
mydra[mydra>numpy.pi]=mydra[mydra>numpy.pi]-2*numpy.pi
mydra[mydra<-numpy.pi]=mydra[mydra<-numpy.pi]+2*numpy.pi


cosa=dist;
cosb=numpy.cos(dec);
cosc=numpy.cos(dec[myind])
cosgamma=(cosc-cosa*cosb)/numpy.sin(numpy.arccos(cosa))/numpy.sin(numpy.arccos(cosb))
gamma=numpy.arccos(cosgamma)

#gamma[mydra<0]=-1*gamma[mydra<0]


tmp=(cosb-cosa*cosc)/numpy.sin(numpy.arccos(cosa))/numpy.sin(numpy.arccos(cosc))
#alpha=numpy.pi-numpy.arccos(tmp)
alpha=numpy.arccos(-tmp)

print 'nbig is ' + repr(numpy.sum(alpha>numpy.pi))
#alpha[mydra<0]=-1*gamma[mydra<0]




#alpha is angle from great circle down
#cosa=numpy.cos(numpy.arccos(dec)-numpy.arccos(dec[myind]))
#cosb=dist
#cosc=numpy.cos(ra-ra[myind])
#cosalpha=(cosc-cosa*cosb)/numpy.sin(numpy.arccos(cosa))/numpy.sin(numpy.arccos(cosb))



#if mydra<0:
#    myphase=numpy.exp(-2*I*(gamma-alpha))
#else:


myphase=numpy.exp(2*I*(gamma-alpha))
myphase[mydra>0]=numpy.conj(myphase[mydra>0])

#ii=numpy.abs(mydra)>numpy.pi/2
#myphase[ii]=numpy.conj(myphase[ii])

v1=(numpy.angle(myphase))
v1[v1>numpy.pi]-=numpy.pi
v1[v1<0]+=numpy.pi
v2=(numpy.angle((mm_tot_QQ+mm_tot_UU)+I*(mm_tot_QU-mm_tot_UQ)))
v2[v2>numpy.pi]-=numpy.pi
v2[v2<0]+=numpy.pi

#v1=(numpy.angle(myphase))
#v2=(numpy.angle((mm_tot_QQ+mm_tot_UU)+I*(mm_tot_QU-mm_tot_UQ)))


#ii=(dec>numpy.pi/2)&(dec!=dec[myind])&(ra!=ra[myind])
#ii=numpy.abs(v1-v2)>0.01



#correct sign
#ii=(dec>numpy.pi/2)&(mydra<0)&(dec!=dec[myind])&(ra!=ra[myind])
#ii=(dec<numpy.pi/2)&(mydra<0)&(dec!=dec[myind])&(ra!=ra[myind])

#this combination has all the wrong sign entries
#ii=(dec>numpy.pi/2)&(mydra>0)&(dec!=dec[myind])&(ra!=ra[myind])
#ii=(dec<numpy.pi/2)&(mydra>0)&(dec!=dec[myind])&(ra!=ra[myind])

#plt.clf();plt.plot(dec[ii]/numpy.pi,ra[ii],'.')


#plt.clf();plt.quiver(ra,dec,numpy.cos(v2),numpy.sin(v2))

#plt.clf();plt.plot(v1[ii],v2[ii]-v1[ii],'.')
plt.clf();plt.plot(v1,v2,'.')
#plt.clf();plt.plot(numpy.sin(numpy.angle(myphase)),numpy.sin(numpy.angle((mm_tot_QQ+mm_tot_UU)+I*(mm_tot_QU-mm_tot_UQ))),'.')
#plt.clf();plt.plot(cosalpha,ra-ra[myind],'.')


assert(1==0)

plt.clf()
plt.plot(dist,mm_tot_QQ,'.')
plt.plot(dist,mm_tot_UU,'.')
plt.plot(dist,mm_tot_QU,'.')
plt.plot(dist,mm_tot_UQ,'.')
assert(1==0)


import spider_rfuns
covmat=spider_rfuns.interpolate_covariance(ra,dec,xx,yy)


