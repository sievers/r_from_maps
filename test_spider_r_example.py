import numpy
import healpy
import spider_rfuns
import time


#this is the file with the relevant noise maps.  Below I assume it has T,Q,U, but
#one could easily modify as desired.
mapfile='/home/sievers/spider/noise_sim_maps/noise_spider1_hfi_f.fits'


nside=64

#cut out pixels with noise more than sqrt(thresh) times the median of non-zero weight pixels.  
#for thresh=0.25 (double the noise), results in a 1% loss in data for the above map
thresh=0.25;  

#set the beam FWHM, in radians
fwhm=49.0/60*numpy.pi/180


#load some spectra (from Aurelien).  Code looks rather strange
#since one file is in C_ell, one in l(l+1)C_ell/2pi.
#Final product should be spectra in C_ell, starting at ell=2.
#spec1 is the signal (e.g. spectrum just from r), spec2 is the (CMB) noise

#this is the r-only model
#this is in l(l+1)C_l/2pi, goes from l=2
spec1=numpy.loadtxt('wmap7_r0p03_totCls.dat')
for i in range(1,spec1.shape[1]):
    spec1[:,i]=spec1[:,i]*2*numpy.pi/spec1[:,0]/(1.0+spec1[:,0])

#this is the total cls
#this is in C_l, goes from l=0
spec2=numpy.loadtxt('wmap7_r0p03_lensed_uK.txt')
spec2=spec2[2:,:]

#amplitudes of the to-be-fit spectrum at which we'll evaluate the likelihood
rvec=numpy.arange(-2,5,0.01)

#file name at which we'll write the output
outfile_name='spider_spec_test_' + repr(nside) + '.likes'

#if you like, you can probably ignore everything below here

#--------------------------------------------------------------------------------



#get the noise maps set up, set to the desired nside
qmap,umap=healpy.read_map(mapfile,field=[1,2])
nside_org=numpy.int32(numpy.sqrt(len(qmap)/12))

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

#find all pixels where weight is within thresh of the median weight of non-empty pixels
ii=(qwt>thresh*qmed)&(umed>thresh*umed)

print numpy.sum(qwt[ii]), numpy.sum(uwt[ii]), numpy.sum(ii)

qwt=qwt[ii].copy()
uwt=uwt[ii].copy()

ipix=numpy.arange(12*nside*nside)
ipix=ipix[ii].copy()  #add bonus copies because numpy does strange, strange things internally

dec,ra=healpy.pix2ang(nside,ipix)

#get the pixel and beam window functions
twin,polwin=healpy.pixwin(nside,True)
beam_spec=healpy.gauss_beam(fwhm,lmax=len(polwin)-1)
tot_win=polwin*beam_spec
tot_win=tot_win[2:].copy()

#make spectra go to end of the window functions
nn=tot_win.size
spec2=spec2[0:nn,:]
spec1=spec1[0:nn,:]

#apply pixel/beam window functions to spectra
cmb_spec=0*spec2
r_spec=0*spec1
cmb_spec[:,0]=spec2[:,0]
r_spec[:,0]=spec1[:,0]
for i in range(1,spec2.shape[1]):
    cmb_spec[:,i]=spec2[:,i]*tot_win
    r_spec[:,i]=spec1[:,i]*tot_win

#get covariance from the CMB
Ecmb,Bcmb=spider_rfuns.get_EB_corrs(ra,dec,cmb_spec[:,2],cmb_spec[:,3])
Ecmb+=Bcmb
del Bcmb
#get the noise, and add it into the CMB
#one could easily put in a full noise covariance matrix here instead of the 
#diagonal instrument noise assumed
noise=numpy.zeros(2*len(qwt))
noise[0::2]=1/numpy.sqrt(qwt)
noise[1::2]=1/numpy.sqrt(uwt)
data_cov=Ecmb+numpy.diag(noise**2)
del Ecmb



#get the signal matrix
Er,Bpred_r=spider_rfuns.get_EB_corrs(ra,dec,r_spec[:,2],r_spec[:,3])
del Er

print numpy.sum(numpy.abs(data_cov-numpy.diag(numpy.diag(data_cov))))

#now we're set up.  One could of course directly evaluate likelihoods, but
#for problems of the form Cov=alpha*A+B one can do a transformation
#into a space where both the noise and signal are diagonal, so 
#at the price of an eigen decomposition, a cholesky, and a few matrix multiplies
#one can get the likelihood for arbitrarily many values of alpha for free.

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
#numpy returns not strictly real eigenvalues/eigenvectors for symmetric matrices
#it shouldn't do this, but well, this is just one of many reasons numpy annoys me...
vv=vv.real
ee=ee.real
Bpred3=numpy.dot(vv.transpose(),numpy.dot(Bpred2,vv))
del Bpred2
numpy.sum(numpy.abs(Bpred3-numpy.diag(numpy.diag(Bpred3))))

Bpred3=numpy.diag(Bpred3)

fac=0.25


#we now have rotated into a space where the signal matrix (r) is 
#diagonal, and the noise+CMB matrix is the identity matrix.  
#loop over the requested amplitudes and calculate the likelihoods, writing to a file.
f=open(outfile_name,'w')
for i in range(len(rvec)):
    aa=time.time()
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
