import numpy
import healpy
import spider_rfuns
import time

mapfile='/home/sievers/spider/noise_sim_maps/noise_spider1_hfi_f.fits'
qmap,umap=healpy.read_map(mapfile,field=[1,2])
nside_org=numpy.int32(numpy.sqrt(len(qmap)/12))
nside=128
qwt=1/qmap**2
uwt=1/umap**2
qwt[qmap<0]=0
uwt[umap<0]=0
fac=(nside_org**2)/(1.0*nside**2)
qwt=healpy.ud_grade(qwt,nside)*fac
uwt=healpy.ud_grade(uwt,nside)*fac
ii=(qwt>0.0)&(uwt>0.0)
qmed=numpy.median(qwt[ii])
umed=numpy.median(uwt[ii])


#cut out pixels with noise more than sqrt(thresh) times the median.  
#for thresh=0.25 (double the noise), results in a 1% loss in data
thresh=0.25;  
ii=(qwt>thresh*qmed)&(umed>thresh*umed)

print numpy.sum(qwt[ii]), numpy.sum(uwt[ii]), numpy.sum(ii)

qwt=qwt[ii].copy()
uwt=uwt[ii].copy()

ipix=numpy.arange(12*nside*nside)
ipix=ipix[ii].copy()  #add bonus copies because numpy does strange, strange things internally

dec,ra=healpy.pix2ang(nside,ipix)


twin,polwin=healpy.pixwin(nside,True)
fwhm=49.0/60*numpy.pi/180
beam_spec=healpy.gauss_beam(fwhm,lmax=len(polwin)-1)

#this is the r-only model
#this is in l(l+1)C_l/2pi, goes from l=2
spec1=numpy.loadtxt('wmap7_r0p03_totCls.dat')
for i in range(1,spec1.shape[1]):
    spec1[:,i]=spec1[:,i]*2*numpy.pi/spec1[:,0]/(1.0+spec1[:,0])


#this is the total cls
#this is in C_l, goes from l=0
spec2=numpy.loadtxt('wmap7_r0p03_lensed_uK.txt')
spec2=spec2[2:,:]


tot_win=polwin*beam_spec
tot_win=tot_win[2:].copy()

nn=tot_win.size
spec2=spec2[0:nn,:]
spec1=spec1[0:nn,:]

cmb_spec=0*spec2
r_spec=0*spec1
cmb_spec[:,0]=spec2[:,0]
r_spec[:,0]=spec1[:,0]
for i in range(1,spec2.shape[1]):
    cmb_spec[:,i]=spec2[:,i]*tot_win
    r_spec[:,i]=spec1[:,i]*tot_win


Ecmb,Bcmb=spider_rfuns.get_EB_corrs(ra,dec,cmb_spec[:,2],cmb_spec[:,3])
Ecmb+=Bcmb
del Bcmb
noise=numpy.zeros(2*len(qwt))
noise[0::2]=1/numpy.sqrt(qwt)
noise[1::2]=1/numpy.sqrt(uwt)
data_cov=Ecmb+numpy.diag(noise**2)
del Ecmb


Er,Bpred_r=spider_rfuns.get_EB_corrs(ra,dec,r_spec[:,2],r_spec[:,3])
del Er


print numpy.sum(numpy.abs(data_cov-numpy.diag(numpy.diag(data_cov))))
rr=numpy.linalg.cholesky(data_cov)
#tmp=numpy.dot(rr,rr.transpose())

rr_inv=numpy.linalg.inv(rr)
del rr

data_cov2=numpy.dot(rr_inv,numpy.dot(data_cov,rr_inv.transpose()))
del data_cov
Bpred2=numpy.dot(rr_inv,numpy.dot(Bpred_r,rr_inv.transpose()))
del Bpred_r

del rr_inv


aa=time.time();ee,vv=numpy.linalg.eig(Bpred2);bb=time.time();
print 'took ' + repr(bb-aa) + ' seconds to do eigenvalues.'
vv=vv.real
ee=ee.real
Bpred3=numpy.dot(vv.transpose(),numpy.dot(Bpred2,vv))
del Bpred2
numpy.sum(numpy.abs(Bpred3-numpy.diag(numpy.diag(Bpred3))))

Bpred3=numpy.diag(Bpred3)

#>>> Bpred3.min() -8.8447301788987798e-11;>>> Bpred3.max() 0.0025283576171073905


fac=0.25

rvec=numpy.arange(-2,5,0.01)
f=open('spider_spec_' + repr(nside) + '.likes','w')
for i in range(len(rvec)):
    aa=time.time()
    #cov=data_cov2+rvec[i]*Bpred2
    #logdet[i]=spider_rfuns.invert_posdef_mat_double(cov)
    #chisq[i]=numpy.sum(cov*data_cov2)
    cov2=rvec[i]*Bpred3+1.0
    logdet2=numpy.sum(numpy.log(cov2))
    chi2=numpy.sum(1.0/cov2)
    cov3=rvec[i]*Bpred3+fac
    logdet3=numpy.sum(numpy.log(cov3))
    chi3=numpy.sum(0.25/cov3)
    bb=time.time()
    ###print rvec[i],-0.5*(logdet[i]+chisq[i]),-0.5*(chi2+logdet2),bb-aa
    #print rvec[i],-0.5*(chi2+logdet2),-0.5*(chi3+logdet3),chi2,chi3,bb-aa
    f.write(repr(rvec[i]) + ' ' + repr(-0.5*(chi2+logdet2))+ '\n')

f.close()
