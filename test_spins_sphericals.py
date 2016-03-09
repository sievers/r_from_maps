import numpy
import matplotlib.pyplot as plt
x=numpy.arange(-1000,1001)*0.001

lmax=50;
pl=numpy.zeros([lmax+1,len(x)])
pl[0,:]=1.0
ell=0;
pl[ell+1,:]=(2*ell+1.0)*x*pl[ell,:]/(ell+1.0)
for ell in range(1,lmax):
    pl[ell+1,:]=((2*ell+1.0)*x*pl[ell,:]-ell*pl[ell-1])/(ell+1.0)

pl1=numpy.zeros([lmax+1,len(x)])
pl1[1,:]=-numpy.sqrt(1-x*x)
m=1
for ell in range(1,lmax):
    pl1[ell+1,:]=((2*ell+1.0)*x*pl1[ell,:]-(ell+m)*pl1[ell-1,:])/(ell-m+1.0)

pl2=numpy.zeros([lmax+1,len(x)])
pl2[2,:]=3*(1-x*x)
m=2.0
for ell in range(2,lmax):
    pl2[ell+1,:]=((2*ell+1.0)*x*pl2[ell,:]-(ell+m)*pl2[ell-1,:])/(ell-m+1.0)

plt.ion()
plt.clf()
#plt.plot(x,pl[1])
#plt.plot(x,x)
#plt.plot(x,pl[4])
#plt.plot(x,(35.0*x**4-30*x**2+3)/8.0)

#plt.plot(x,pl1[3])
#plt.plot(x,-1.5*(5*x*x-1)*numpy.sqrt(1-x*x))

#plt.plot(x,pl1[4])
#plt.plot(x,-2.5*(7*x*x*x-3*x)*numpy.sqrt(1-x*x))

#plt.plot(x,pl2[4])
#plt.plot(x,15.0/2*(7*x**2-1)*(1-x**2))

#plt.plot(x,pl2[3,:])
#plt.plot(x,15.0*x*(1-x**2))
