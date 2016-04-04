#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#include <gsl/gsl_interp.h>


//gcc-4.9 -I/Users/sievers/local/include -fopenmp -O3 -shared -fPIC -std=c99 -o libspider_rfuns.so spider_rfuns.c -L/Users/sievers/local/lib -lgsl -lm -lgomp
//gcc -I${HIPPO_GSL_DIR}/include -fopenmp -O3 -shared -fPIC -std=c99 -o libspider_rfuns.so spider_rfuns.c -L${HIPPO_GSL_DIR}/lib -lgsl -L${HIPPO_CBLAS_DIR}/lib -lcblas -L${HIPPO_OPENBLAS_DIR}/lib -lopenblas -lm -lgomp



/*--------------------------------------------------------------------------------*/

void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
void cdpotrf(char uplo, int n, double *a, int lda, int *info)
{
  dpotrf_(&uplo,&n,a,&lda,info);
}

void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
void cdpotri(char uplo, int n, double *a, int lda, int *info)
{
  dpotri_(&uplo,&n,a,&lda,info);
}

/*--------------------------------------------------------------------------------*/


int  invert_posdef_mat_double(double *mat, int n, double *logdet)
{


  *logdet=0;
  int info;
  cdpotrf('u', n, mat, n, &info);
  if (info) {
    printf("Cholesky failed in invert_posdef_mat_double with info %d\n",info);
    return info;
  }
  for (int i=0;i<n;i++)
    *logdet+=log(mat[i*n+i]);
  *logdet *=2;
  cdpotri('u', n,mat, n, &info);
  if (info)
    return info;
  double *mm=mat;
  for (int i = 0; i < n; i++)
    for (int j = i+1; j < n; j++) {
      mm[i*n+j]=mm[j*n+i];
    }
  return info;
}
/*--------------------------------------------------------------------------------*/

//void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *info);
void dsyevd_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *iwork, int *liwork, int *info, int jobzlen, int uplolen);


void cdsyevd(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork, int *iwork, int liwork, int *info)
{
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info,1,1);
}
/*--------------------------------------------------------------------------------*/
void eig_wrapper(double *mat, int n, double *eigvals)
{
  int info,lwork=-1,liwork=-1,iwork,*iworkvec;
  double *work,w;
  cdsyevd('v','u',n,mat,n,eigvals,&w,-1,&iwork,-1,&info);
  printf("workspace sizes are %f %d\n",w,iwork);
  work=(double *)malloc(sizeof(double)*w);
  iworkvec=(int *)malloc(sizeof(int)*iwork);
  
  cdsyevd('v','u',n,mat,n,eigvals,work,(int)w,iworkvec,iwork,&info);
  
  free(work);
  free(iworkvec);
}
/*--------------------------------------------------------------------------------*/



void gsl_spline_wrapper(double *x, double *y, int nx, double *xx, double *yy, int nxx)
{
  gsl_interp_accel *a = gsl_interp_accel_alloc ();
  //gsl_interp *interp = gsl_interp_alloc (gsl_interp_cspline_periodic, nx);
  gsl_interp *interp = gsl_interp_alloc (gsl_interp_cspline, nx);
  //printf("first x values are %12.6f %12.6f %12.6f\n",x[0],x[1],x[2]);                                                                                                          
  gsl_interp_init(interp,x,y,nx);

#pragma omp parallel for  
  for (int i=0;i<nxx;i++) {
    int status = gsl_interp_eval_e (interp, x, y,xx[i],a,&yy[i]);
    if (status)
      printf("status was %d %d %12.5f\n",status,i,xx[i]);
  }
  gsl_interp_accel_free (a);
  gsl_interp_free (interp);
  
}

/*--------------------------------------------------------------------------------*/

void interpolate_covariance(double *ra, double *dec, int n, double *costheta, double *corr, int npt, double *mat)
{
  //printf("first element was %12.5g %12.5g %12.5g\n",mat[0],corr[0],corr[npt-1]);
  //printf("limits are %12.5g %12.5g\n",costheta[0],costheta[npt-1]);



  double *x=(double *)malloc(sizeof(double)*n);
  double *y=(double *)malloc(sizeof(double)*n);
  double *z=(double *)malloc(sizeof(double)*n);

#pragma omp parallel for
  for (int i=0;i<n;i++) {
    x[i]=sin(dec[i])*sin(ra[i]);
    y[i]=sin(dec[i])*cos(ra[i]);
    z[i]=cos(dec[i]);
  }
  
#pragma omp parallel 
  {

    gsl_interp_accel *a;
    gsl_interp *interp;
#pragma omp critical 
    {
      a = gsl_interp_accel_alloc ();
      //interp = gsl_interp_alloc (gsl_interp_cspline_periodic, npt);
      interp = gsl_interp_alloc (gsl_interp_cspline, npt);
      gsl_interp_init(interp,costheta,corr,npt);
    }
#pragma omp for
    for (int i=0;i<n;i++)
      for (int j=i;j<n;j++) {
	
	double tmp=x[i]*x[j]+y[i]*y[j]+z[i]*z[j];
	if (tmp>1)
	  tmp=1;
	if (tmp<-1)
	  tmp=-1;
#if 0
	if (tmp==costheta[0])
	  mat[i*n+j]=corr[0];
	if (tmp==costheta[npt-1])
	  mat[i*n+j]=corr[npt-1];
	int ii=0;
	while (tmp>costheta[ii])
	  ii++;
	mat[i*n+j]=corr[ii];
	mat[j*n+i]=corr[ii];

#else	
	int status=gsl_interp_eval_e(interp,costheta,corr,tmp,a,&mat[i*n+j]);
	mat[j*n+i]=mat[i*n+j];

	if (status)
	  printf("status was %d %d %12.5f\n",status,i,tmp);
#endif
      }
	gsl_interp_accel_free (a);
	gsl_interp_free (interp);        
  }
   

  //free(mydot);
  free(x);
  free(y);
  free(z);

}

/*--------------------------------------------------------------------------------*/

void interpolate_covariance_old(double *ra, double *dec, int n, double *costheta, double *corr, int npt, double *mat)
{
  double *x=(double *)malloc(sizeof(double)*n);
  double *y=(double *)malloc(sizeof(double)*n);
  double *z=(double *)malloc(sizeof(double)*n);

#pragma omp parallel for
  for (int i=0;i<n;i++) {
    x[i]=sin(dec[i])*sin(ra[i]);
    y[i]=sin(dec[i])*cos(ra[i]);
    z[i]=cos(dec[i]);
  }

  double *mydot=(double *)malloc(sizeof(double)*n*n);
  for (int i=0;i<n;i++)
    for (int j=0;j<n;j++) {

      double tmp=x[i]*x[j]+y[i]*y[j]+z[i]*z[j];
      if (tmp>1)
	tmp=1;
      if (tmp<-1)
	tmp=-1;
      mydot[i*n+j]=tmp;
    }
  
  
  gsl_spline_wrapper(costheta,corr,npt,mydot,mat,n*n);

  free(mydot);
  free(x);
  free(y);
  free(z);

}

/*--------------------------------------------------------------------------------*/
void fill_gamma_alpha(double *ra, double *dec, int n, double *cosalpha, double *cosgamma)
{
  double *x=(double *)malloc(n*sizeof(double));
  double *y=(double *)malloc(n*sizeof(double));
  double *z=(double *)malloc(n*sizeof(double));
  double *sindec=(double *)malloc(n*sizeof(double));
  for (int i=0;i<n;i++) {
    x[i]=sin(dec[i])*sin(ra[i]);
    y[i]=sin(dec[i])*cos(ra[i]);
    z[i]=cos(dec[i]);
    sindec[i]=sin(dec[i]);
  }
  //  double pi=3.1415926535897;
#ifndef M_PI
#define M_PI 3.14159265358979
#endif
  double pi=M_PI;
  double twopi=2*pi;
#pragma omp parallel for
  for (int i=0;i<n;i++)
    for (int j=0;j<n;j++) {
      double mydra=ra[i]-ra[j];
      if (mydra>pi)
	mydra-=twopi;
      if (mydra<-pi)
	mydra+=twopi;
      double mysign=1.0;
      if (mydra<0)
	mysign *= -1;
	
      double cosa=x[i]*x[j]+y[i]*y[j]+z[i]*z[j];
      double cosb=z[i];
      double cosc=z[j];
      double numerator=cosc-cosa*cosb;
      double mysin=sin(acos(cosa));
      double denom=mysin*sindec[i];

      if (fabs(denom)>1e-14)
	cosgamma[i*n+j]=numerator/denom*mysign;
      else
	cosgamma[i*n+j]=1;

      if (cosgamma[i*n+j]>1)
	(cosgamma[i*n+j])=1;
      if (cosgamma[i*n+j]<-1)
	(cosgamma[i*n+j])=-1;
      cosgamma[i*n+j]=acos(cosgamma[i*n+j]);

      numerator=cosb-cosa*cosc;
      denom=mysin*sindec[j];
      if (fabs(denom)>1e-14)
	cosalpha[i*n+j]=numerator/denom*mysign;
      else
	cosalpha[i*n+j]=1;

      if (cosalpha[i*n+j]>1)
	(cosalpha[i*n+j])=1;
      if (cosalpha[i*n+j]<-1)
	(cosalpha[i*n+j])=-1;
      cosalpha[i*n+j]=acos(cosalpha[i*n+j]);
    }
  free(x);
  free(y);
  free(z);
  free(sindec);
}

/*--------------------------------------------------------------------------------*/
void fill_QU_corr_eb(double *corr1E, double *corr2E, double *corr1B, double *corr2B, double *alpha, double *gamma, int npt, double *corrE, double *corrB)
{
#pragma omp parallel for
  for (int i=0;i<npt;i++) 
    for (int j=0;j<npt;j++) {
      double c1=cos(-2*(alpha[i*npt+j]+gamma[i*npt+j]));
      double c2=cos(2*(gamma[i*npt+j]-alpha[i*npt+j]));
      
      corrE[4*i*npt+2*j]=0.5*(c1*corr1E[i*npt+j]+c2*corr2E[i*npt+j]);
      corrE[(4*i+2)*npt+2*j+1]=0.5*(c1*corr1E[i*npt+j]-c2*corr2E[i*npt+j]);

      corrB[4*i*npt+2*j]=0.5*(c1*corr1B[i*npt+j]+c2*corr2B[i*npt+j]);
      corrB[(4*i+2)*npt+2*j+1]=0.5*(c1*corr1B[i*npt+j]-c2*corr2B[i*npt+j]);

      c1=sin(-2*(alpha[i*npt+j]+gamma[i*npt+j]));
      c2=sin(2*(gamma[i*npt+j]-alpha[i*npt+j]));
      
      corrE[4*i*npt+2*j+1]=0.5*(c1*corr1E[i*npt+j]+c2*corr2E[i*npt+j]);
      corrE[(4*i+2)*npt+2*j]=0.5*(-c1*corr1E[i*npt+j]+c2*corr2E[i*npt+j]);
      
      corrB[4*i*npt+2*j+1]=0.5*(c1*corr1B[i*npt+j]+c2*corr2B[i*npt+j]);
      corrB[(4*i+2)*npt+2*j]=0.5*(-c1*corr1B[i*npt+j]+c2*corr2B[i*npt+j]);
      
  }
}

/*--------------------------------------------------------------------------------*/
void fill_QU_corr_ee_old(double *corr1, double *corr2, double *alpha, double *gamma, int npt, double *corrQQ, double *corrQU, double *corrUQ, double *corrUU)
{
#pragma omp parallel for
  for (int i=0;i<npt;i++) {
    double c1=corr1[i]*cos(-2*(alpha[i]+gamma[i]));
    double c2=corr2[i]*cos(2*(gamma[i]-alpha[i]));
    corrQQ[i]=0.5*(c1+c2);
    corrUU[i]=0.5*(c1-c2);

    c1=corr1[i]*sin(-2*(alpha[i]+gamma[i]));
    c2=corr2[i]*sin(2*(gamma[i]-alpha[i]));
    corrQU[i]=0.5*(c2-c1);
    corrUQ[i]=0.5*(c2+c1);
    
    
  }
}


/*--------------------------------------------------------------------------------*/
static inline double Cslm(int s, int l, int m)
{
  double tmp=l*l*(4.0*l*l-1.0);
  tmp /=((l*l-m*m+0.0)*(l*l-s*s+0.0));
  tmp=sqrt(tmp);
  //printf("Cslm in C is %12.5f\n",tmp);
  return tmp;
}

/*--------------------------------------------------------------------------------*/
double myfactorial(int n)
{
  double ans=1;
  for (int i=2;i<n+1;i++)
    ans*=i;
  
  return ans;
}


/*--------------------------------------------------------------------------------*/
void get_EB_corrs(double *x, int nx, double *psE, double *psB, int nell, double *Ecorr, double *Bcorr)
{
  memset(Ecorr,0,sizeof(double)*nx);
  memset(Bcorr,0,sizeof(double)*nx);
  //double *Pm=(double *)malloc(sizeof(double)*nx);
  double tmp=0.25;  //this is the tmp value from s_lambda_lm for |s|=|m|=2
  
  
  //do 2,2 one first
  int s=2;
  int m=2;
  double fac=myfactorial(2*m+1)/(4.0*M_PI*myfactorial(m+s)*myfactorial(m-s));
  fac=sqrt(fac);
  //for (int i=0;i<55;i++)
  //printf("ps %3d is %12.4g\n",i,psE[i]);
#pragma omp parallel for
  for (int i=0;i<nx;i++) {

    //double Pm=0.25;
    double Pm=pow(-0.5,m);
    if (m!=s) 
      Pm*=pow(1.0+x[i],(m-s)*(0.5));
    if (m!= -s)    
      Pm *= pow(1.0-x[i],(m+s)*0.5);
    Pm *=fac;
    


    //do quadrupole
    double tmp2=Pm*sqrt((2*2+1)/(4*M_PI));
    Ecorr[i]+=tmp2*psE[2];
    Bcorr[i]-=tmp2*psB[2];

    //do ell=3
    double mycslm_old=Cslm(s,m+1,m);
    double Pm1=(x[i]+s/(m+1.0))*mycslm_old*Pm;

    double tmp3=Pm1*sqrt((2*3+1)/(4*M_PI));
    Ecorr[i]+=tmp3*psE[3];
    Bcorr[i]-=tmp3*psB[3];

    if (i==0)
      printf("Pm and Pm1 in get_EB_corrs are %12.5g %12.5g\n",Pm,Pm1);
    
    for (int n=m+2;n<nell+1;n++) {
      double mycslm=Cslm(s,n,m);
      double Pn=mycslm*((x[i]+s*m/(n*(n-1.0)))*Pm1-Pm/mycslm_old);
      mycslm_old=mycslm;
      double tmp=Pn*sqrt((2*n+1)/(4*M_PI));
      Pm=Pm1;
      Pm1=Pn;
      //if (n==50) 
      //Ecorr[i]=tmp;
      Ecorr[i]+=tmp*psE[n];
      Bcorr[i]-=tmp*psB[n];

    }
  }
  printf("Ecorr[0]=%12.4g\n",Ecorr[0]);

  //do 2,-2 now
  s=-2;
  m=2;
  fac=myfactorial(2*m+1)/(4.0*M_PI*myfactorial(m+s)*myfactorial(m-s));
  fac=sqrt(fac);


#pragma omp parallel for
  for (int i=0;i<nx;i++) {
    double Pm=pow(-0.5,m);
    if (m!=s) 
      Pm*=pow(1.0+x[i],(m-s)*(0.5));
    if (m!= -s)    
      Pm *= pow(1.0-x[i],(m+s)*0.5);
    Pm *=fac;
    //do quadrupole
    double tmp2=Pm*sqrt((2*2+1)/(4*M_PI));
    Ecorr[i]+=tmp2*psE[2];
    Bcorr[i]+=tmp2*psB[2];


    //to ell=3
    double mycslm_old=Cslm(s,m+1,m);
    double Pm1=(x[i]+s/(m+1.0))*mycslm_old*Pm;

    if (i==0) {
      printf("in second half, Pm0 and Pm1 are %12.5g %12.5g\n",Pm,Pm1);
    }

    double tmp3=Pm1*sqrt((2*3+1)/(4*M_PI));
    Ecorr[i]+=tmp3*psE[3];
    Bcorr[i]+=tmp3*psB[3];

    for (int n=m+2;n<nell+1;n++) {
      double mycslm=Cslm(s,n,m);
      double Pn=mycslm*((x[i]+s*m/(n*(n-1.0)))*Pm1-Pm/mycslm_old);
      mycslm_old=mycslm;
      double tmp=Pn*sqrt((2*n+1)/(4*M_PI));
      Ecorr[i]+=tmp*psE[n];
      Bcorr[i]+=tmp*psB[n];
      Pm=Pm1;
      Pm1=Pn;
      //if (i==50)
      //Ecorr[i]=tmp;
    }
  }

}


/*--------------------------------------------------------------------------------*/
void get_EB_corrs_cached(double *x, int nx, double *psE, double *psB, int nell, double *Ecorr, double *Bcorr, double *Ecorr2, double *Bcorr2)
{
  memset(Ecorr,0,sizeof(double)*nx);
  memset(Bcorr,0,sizeof(double)*nx);
  memset(Ecorr2,0,sizeof(double)*nx);
  memset(Bcorr2,0,sizeof(double)*nx);

  double *Pm=(double *)malloc(sizeof(double)*nx);
  double *Pm1=(double *)malloc(sizeof(double)*nx);


  double tmp=0.25;  //this is the tmp value from s_lambda_lm for |s|=|m|=2
  
  
  //do 2,2 one first
  int s=2;
  int m=2;
  double fac=myfactorial(2*m+1)/(4.0*M_PI*myfactorial(m+s)*myfactorial(m-s));
  fac=sqrt(fac);

  
#pragma omp parallel 
  {
    #pragma omp for
    for (int i=0;i<nx;i++) {
      Pm[i]=pow(-0.5,m);
      if (m!=s) 
	Pm[i]*=pow(1.0+x[i],(m-s)*(0.5));
      if (m!= -s)    
	Pm[i] *= pow(1.0-x[i],(m+s)*0.5);
      Pm[i] *=fac;
      double tmp2=Pm[i]*sqrt((2*2+1)/(4*M_PI));
      Ecorr[i]+=tmp2*psE[2];
      Bcorr[i]-=tmp2*psB[2];
      double mycslm_old=Cslm(s,m+1,m);
      Pm1[i]=(x[i]+s/(m+1.0))*mycslm_old*Pm[i];
      
      double tmp3=Pm1[i]*sqrt((2*3+1)/(4*M_PI));
      Ecorr[i]+=tmp3*psE[3];
      Bcorr[i]-=tmp3*psB[3];    
    }
    
    double mycslm_old=Cslm(s,m+1,m);
    for (int n=m+2;n<nell+1;n++) {
      double mycslm=Cslm(s,n,m);
      double fac=sqrt((2*n+1)/(4*M_PI));
      double facE=fac*psE[n];
      double facB=fac*psB[n];
#pragma omp for
      for (int i=0;i<nx;i++) {
	double Pn=mycslm*((x[i]+s*m/(n*(n-1.0)))*Pm1[i]-Pm[i]/mycslm_old);
	Pm[i]=Pm1[i];
	Pm1[i]=Pn;
	Ecorr[i]+=Pn*facE;
	Bcorr[i]-=Pn*facB;
      }
      mycslm_old=mycslm;
      
    }
    
  }
#if 1
  //do 2,-2 now
  s=-2;
  m=2;
  fac=myfactorial(2*m+1)/(4.0*M_PI*myfactorial(m+s)*myfactorial(m-s));
  fac=sqrt(fac);


#pragma omp parallel 
  {
    double mycslm_old=Cslm(s,m+1,m);
#pragma omp for
    for (int i=0;i<nx;i++) {
      Pm[i]=pow(-0.5,m);
      if (m!=s) 
	Pm[i]*=pow(1.0+x[i],(m-s)*(0.5));
      if (m!= -s)    
	Pm[i] *= pow(1.0-x[i],(m+s)*0.5);
      Pm[i] *=fac;
      //do quadrupole
      double tmp2=Pm[i]*sqrt((2*2+1)/(4*M_PI));
      Ecorr2[i]+=tmp2*psE[2];
      Bcorr2[i]+=tmp2*psB[2];    
      
      Pm1[i]=(x[i]+s/(m+1.0))*mycslm_old*Pm[i];
      
      double tmp3=Pm1[i]*sqrt((2*3+1)/(4*M_PI));
      Ecorr2[i]+=tmp3*psE[3];
      Bcorr2[i]+=tmp3*psB[3];
      
    }

    mycslm_old=Cslm(s,m+1,m);
    for (int n=m+2;n<nell+1;n++) {
      double mycslm=Cslm(s,n,m);
      double fac=sqrt((2*n+1)/(4*M_PI));
      double facE=psE[n]*fac;
      double facB=psB[n]*fac;
#pragma omp for
      for (int i=0;i<nx;i++) {
	double Pn=mycslm*((x[i]+s*m/(n*(n-1.0)))*Pm1[i]-Pm[i]/mycslm_old);
	Ecorr2[i]+=Pn*facE;
	Bcorr2[i]+=Pn*facB;
	Pm[i]=Pm1[i];
	Pm1[i]=Pn;
      }
      mycslm_old=mycslm;
    }
  }
#endif
  free(Pm);
  free(Pm1);

}

/*--------------------------------------------------------------------------------*/
void s_lambda_lm (int s, int l, int m, double *x, int nx, double *Pm, double *Pm1)
{


  double tmp;
  if (m==2)
    tmp=-0.5*-0.5;
  else
    tmp=pow(-0.5,m);

  //printf("tmp is %12.5f\n",tmp);
  for (int i=0;i<nx;i++)
    Pm[i]=tmp;
  if (m!=s) 
    #pragma omp parallel for
    for (int i=0;i<nx;i++) 
      Pm[i]*=pow(1.0+x[i],(m-s)*(0.5));
  if (m!= -s)
    #pragma omp parallel for
    for (int i=0;i<nx;i++)
      Pm[i] *= pow(1.0-x[i],(m+s)*0.5);



  double fac=myfactorial(2*m+1)/(4.0*M_PI*myfactorial(m+s)*myfactorial(m-s));
  fac=sqrt(fac);
  

  for (int i=0;i<nx;i++)
    Pm[i]*=fac;
  //printf("Pm[0,mid,end] is now %12.5g %12.5g %12.5g\n",Pm[0],Pm[nx/2],Pm[nx-1]);

  //printf("fac is %12.4g and Pm[0] is %12.4g\n",fac,Pm[0]);

  if (l==m)
    return;


  double mycslm=Cslm(s,m+1,m);
#pragma omp parallel for
  for (int i=0;i<nx;i++) {
    Pm1[i]=(x[i]+s/(m+1.0))*mycslm*Pm[i];
  }
  //printf("Pm1[0,mid,end] is now %12.5g %12.5g %12.5g\n",Pm1[0],Pm1[nx/2],Pm1[nx-1]);  
  if (l==m+1) {
    for (int i=0;i<nx;i++)
      Pm[i]=Pm1[i];
    return;
  }



  double *mycslm_vec=(double *)malloc((l+1)*sizeof(double));
  for (int n=m;n<l+1;n++)
    mycslm_vec[n]=Cslm(s,n,m);


#if 0
#pragma omp parallel for
  for (int i=0;i<nx;i++) {
    double myPm=Pm[i];
    double myPm1=Pm1[i];
    double myx=x[i];
    for (int n=m+2;n<l+1;n++) {
      //double Pn=(myx+s*m/(n*(n-1.0)))*mycslm_vec[n]*myPm1-mycslm_vec[n]/mycslm_vec[n-1]*myPm;
      double Pn=mycslm_vec[n]*((myx+s*m/(n*(n-1.0)))*myPm1-myPm/mycslm_vec[n-1]);
      myPm=myPm1;
      myPm1=Pn;
    }
    Pm[i]=myPm1;
  }
#endif


#if 0
  int bs=16;
  //#pragma omp parallel for
  for (int i=0;i<nx;i+=bs) {    
    double myPm[bs];
    double myPm1[bs];
    double myx[bs];
    for (int j=0;j<bs;j++) {
      myPm[j]=Pm[i+j];
      myPm1[j]=Pm1[i+j];
      myx[j]=x[i+j];
    }
    for (int n=m+2;n<l;n++)
      for (int j=0;j<bs;j++) {
	double Pn=mycslm_vec[n]*((myx[j]+s*m/(n*(n-1.0)))*myPm1[j]-myPm[j]/mycslm_vec[n-1]);
	myPm[j]=myPm1[j];
	myPm1[j]=Pn;
      }
    for (int j=0;j<bs;j++) 
      Pm[i+j]=myPm1[j];
  }
#endif
  

#if 0
  int bs=1;
  for (int i=0;i<nx;i+=bs) {
    for (int n=m+2;n<l+1;n++) {
      //double Pn=(x[i]+s*m/(n*(n-1.0))*mycslm_vec[n]*Pm1[i]-mycslm_vec[n]/mycslm_vec[n-1]*Pm[i]);
      double Pn=(x[i]+s*m/(n*(n-1.0)))*mycslm_vec[n]*Pm1[i]-mycslm_vec[n]/mycslm_vec[n-1]*Pm[i];
      Pm[i]=Pm1[i];
      Pm1[i]=Pn;
    }
  }
  for (int i=0;i<nx;i++)
    Pm[i]=Pm1[i];
  
#endif
#if 0
  //this is the fastest, but do not yet have edge conditions appropriately handled
#define BS 64
#pragma omp parallel for
  for (int i=0;i<nx;i+=BS) {
    double Pn[BS];
    double myPm[BS];
    double myPm1[BS];
    for (int j=0;j<BS;j++) {
      myPm[j]=Pm[i+j];
      myPm1[j]=Pm1[i+j];
    }
    
    double mycslm_old=Cslm(s,m+1,m);
    for (int n=m+2;n<l+1;n++) {
      double mycslm_new=Cslm(s,n,m);
      
      for (int j=0;j<BS;j++) {	
	//double Pn=(x[i]+s*m/(n*(n-1.0))*mycslm_vec[n]*Pm1[i]-mycslm_vec[n]/mycslm_vec[n-1]*Pm[i]);	
	//Pn[j]=(x[i+j]+s*m/(n*(n-1.0)))*mycslm_new*Pm1[i+j]-mycslm_new/mycslm_old*Pm[i+j];
	//Pm[i+j]=Pm1[i+j];
	//Pm1[i+j]=Pn[j];	
	Pn[j]=(x[i+j]+s*m/(n*(n-1.0)))*mycslm_new*myPm1[j]-mycslm_new/mycslm_old*myPm[j];
	myPm[j]=myPm1[j];
	myPm1[j]=Pn[j];
	
      }
      mycslm_old=mycslm_new;
    }
    for (int j=0;j<BS;j++) 
      Pm[i+j]=myPm1[j];
  }
  
  //for (int i=0;i<nx;i++)
  //Pm[i]=Pm1[i];

#endif
  
  
  
  
#if 1
#pragma omp parallel 
  {
    double mycslm_old=Cslm(s,m+1,m);
    for (int n=m+2;n<l+1;n++) {
      double mycslm_new=Cslm(s,n,m);
#pragma omp for
      for (int i=0;i<nx;i++) {
	double Pn=(x[i]+s*m/(n*(n-1.0)))*mycslm_new*Pm1[i]-mycslm_new/mycslm_old*Pm[i];
	//if (x[i]==0)
	//	printf("n=%4d, Pn=%12.4g, Pm1=%12.4g, Pm=%12.4g, cslms are %12.4g %12.4g\n",i,Pn,Pm1[i],Pm[i],mycslm_new,mycslm_old);
	Pm[i]=Pm1[i];
	Pm1[i]=Pn;
      }
      
      mycslm_old=mycslm_new;
    }
  }
#pragma omp parallel for
  for (int i=0;i<nx;i++) {
    //printf("Pm[%d]=%12.4g\n",i,Pm[i]);
    Pm[i]=Pm1[i];
  }
#endif
  free(mycslm_vec);
}

/*--------------------------------------------------------------------------------*/
void s_lambda_lm_old (int s, int l, int m, double *x, int nx, double *Pm, double *Pm1)
{
  double tmp;
  if (m==2)
    tmp=-0.5*-0.5;
  else
    tmp=pow(-0.5,m);

  for (int i=0;i<nx;i++)
    Pm[i]=tmp;
  if (m!=s) 
    #pragma omp parallel for
    for (int i=0;i<nx;i++) 
      Pm[i]*=pow(1.0+x[i],(m-s)*(0.5));
  if (m!= -s)
    #pragma omp parallel for
    for (int i=0;i<nx;i++)
      Pm[i] *= pow(1.0-x[i],(m+s)*0.5);

  double fac=myfactorial(2*m+1)/(4.0*M_PI*myfactorial(m+s)*myfactorial(m-s));
  fac=sqrt(fac);

  for (int i=0;i<nx;i++)
    Pm[i]*=fac;
  //printf("fac is %12.4g and Pm[0] is %12.4g\n",fac,Pm[0]);

  if (l==m)
    return;


  double mycslm=Cslm(s,m+1,m);
#pragma omp parallel for
  for (int i=0;i<nx;i++) {
    Pm1[i]=(x[i]+s/(m+1.0))*mycslm*Pm[i];
  }
  
  if (l==m+1) {
    for (int i=0;i<nx;i++)
      Pm[i]=Pm1[i];
    return;
  }


#pragma omp parallel 
  {
    double mycslm_old=Cslm(s,m+1,m);
    for (int n=m+2;n<l+1;n++) {
      double mycslm_new=Cslm(s,n,m);
#pragma omp for
      for (int i=0;i<nx;i++) {
	double Pn=(x[i]+s*m/(n*(n-1.0)))*mycslm_new*Pm1[i]-mycslm_new/mycslm_old*Pm[i];
	//if (x[i]==0)
	//	printf("n=%4d, Pn=%12.4g, Pm1=%12.4g, Pm=%12.4g, cslms are %12.4g %12.4g\n",i,Pn,Pm1[i],Pm[i],mycslm_new,mycslm_old);
	Pm[i]=Pm1[i];
	Pm1[i]=Pn;
      }
    
      mycslm_old=mycslm_new;
    }
  }
#pragma omp parallel for
  for (int i=0;i<nx;i++) {
    //printf("Pm[%d]=%12.4g\n",i,Pm[i]);
    Pm[i]=Pm1[i];
  }
}

