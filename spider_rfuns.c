#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#include <gsl/gsl_interp.h>


//gcc-4.9 -I/Users/sievers/local/include -fopenmp -O3 -shared -fPIC -std=c99 -o libspider_rfuns.so spider_rfuns.c -L/Users/sievers/local/lib -lgsl -lm -lgomp

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
