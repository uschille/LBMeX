#ifndef LBM_FLUCTUATIONS_H_
#define LBM_FLUCTUATIONS_H_

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif
#include <AMReX_GpuComplex.H>

#include "LBM_d3q19.H"
#include "LBM_binary.H"

const int ncons = 2 + AMREX_SPACEDIM;
const int ndof = 2*nvel;

// Cholesky decomposition of matrix A
// result is stored in lower triangle of A
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cholesky_decomp(GpuArray<Real,ndof*ndof>& A, const int n, const int bstart) {
  // Cholesky-Banachiewicz algorithm
  Real sum;
  for (int i=bstart; i<n; ++i) {
    for (int j=bstart; j<=i; ++j) {
      sum = A[i*n+j];
      for (int k=j-1; k>=bstart; --k) {
	      sum -= A[i*n+k]*A[j*n+k];
      }
      if (i==j) {
	      if (sum>=0) {
	        A[i*n+j] = std::sqrt(sum);
	      } else {
	        A[i*n+j] = 0.0;
          Print() << "Row " << i << " matrix not positive definite! " << sum << std::endl;
          exit(-1);
	      }
      } else {
	      if (A[j*n+j]>0) {
	        A[i*n+j] = sum/A[j*n+j];
	      } else {
          Print() << "Cholesky decomposition should not reach " << __FILE__ <<":"<< __LINE__ << std::endl;
	        exit(-1);
	      }
      }
    }
  }
  for (int i=0; i<n; ++i) {
    for (int j=i+1; j<n; ++j) {
      A[i*n+j] = 0.0;
    }
  }
}

#if 0
// generate a pure sine wave in k-space
inline void create_kspace_sin(const Geometry& geom,
			      const MultiFab& noise_onegrid,
			      const Vector<std::unique_ptr<BaseFab<GpuComplex<Real>>>>& spectral_field) {
  const Box domain = geom.Domain();
  for (MFIter mfiter(noise_onegrid); mfiter.isValid(); ++mfiter) {
    IntVect fft_size = domain.length();
    fft_size[0] = fft_size[0]/2 + 1;
    Box fft_box = Box(IntVect(0), fft_size - IntVect(1));
    Array4<GpuComplex<Real>> const& xi = (*spectral_field[0]).array();
    ParallelFor(fft_box, [=] AMREX_GPU_DEVICE(int kx, int ky, int kz) {
      int kxloc = (kx == 0) ? 0 : domain.length(0) - kx;
      int kyloc = (ky == 0) ? 0 : domain.length(1) - ky;
      int kzloc = (kz == 0) ? 0 : domain.length(2) - kz;
      if (   (kx == 0)
	  && (ky == 1)
	  && (kz == 1) ) {
	xi(kx,ky,kz).m_real = 1.0;
	xi(kx,ky,kz).m_imag = 0.0;
      } else {
	xi(kx,ky,kz).m_real = 0.0;
	xi(kx,ky,kz).m_imag = 0.0;
      }
      if (kx > domain.length(0)/2) {
	Print() << "This should never execute sin" << std::endl;
	xi(kx,ky,kz).m_real = xi(kxloc,kyloc,kzloc).real();
	xi(kx,ky,kz).m_imag = xi(kxloc,kyloc,kzloc).imag();
      }
    });
  }
}
#endif

#if 0
// generate uncorrelated white noise in k-space
// requires whole domain without domain decomposition
inline void kspace_white_noise(const Geometry& geom,
			       MultiFab& kspace_noise_real_onegrid,
			       MultiFab& kspace_noise_imag_onegrid) {
  const Box domain = geom.Domain();
  for (MFIter mfi(kspace_noise_real_onegrid); mfi.isValid(); ++mfi) {
    const Box& box = mfi.fabbox();
    const Array4<Real>& xi_real = kspace_noise_real_onegrid.array(mfi);
    const Array4<Real>& xi_imag = kspace_noise_imag_onegrid.array(mfi);
    // construct noise in k-space
    ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int kx, int ky, int kz, RandomEngine const& engine) {
      int kxloc = (kx == 0) ? 0 : domain.length(0) - kx;
      int kyloc = (ky == 0) ? 0 : domain.length(1) - ky;
      int kzloc = (kz == 0) ? 0 : domain.length(2) - kz;
      // symmetry points are purely real
      if (((kx == 0) || (kx == domain.length(0) - kx))
	  && ((ky == 0) || (ky == domain.length(1) - ky))
	  && ((kz == 0) || (kz == domain.length(2) - kz))) {
	xi_real(kx,ky,kz) = RandomNormal(0., 1., engine);
	xi_imag(kx,ky,kz) = RandomNormal(0., 0., engine);
      } else {
	// complex conjugate symmetries
	if ((kx > domain.length(0)/2)
	    || ((ky > domain.length(1)/2) && (kx == kxloc))
	    || ((kz > domain.length(2)/2) && (ky == kyloc) && (kx == kxloc))) {
	  xi_real(kx,ky,kz) =  xi_real(kxloc,kyloc,kzloc);
	  xi_imag(kx,ky,kz) = -xi_imag(kxloc,kyloc,kzloc);
	} else {
	  // complex Gaussian random variables with zero mean and variance 0.5
	  xi_real(kx,ky,kz) = RandomNormal(0., std::sqrt(0.5), engine);
	  xi_imag(kx,ky,kz) = RandomNormal(0., std::sqrt(0.5), engine);
	}
      }
    });
  }
}
#endif

#if 0
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real structure_factor(int kx, int ky, int kz) {
  const Real rho0 = 1.0;
  const Real S = rho0*temperature/cs2;
  return S;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof*ndof> population_correlations(int kx, int ky, int kz) {
  GpuArray<Real,ndof*ndof> F = {};
  GpuArray<Real,nvel> fbar, gbar;
  const Real rho0 = 1.0;
  const Real mu0 = 0.0; // \todo
  const Real S_r = structure_factor(kx,ky,kz);
  const Real S_p = 0.0; // \todo
  const Real mu = S_r/rho0;
  for (int i=0; i<nvel; ++i) {
    fbar[i] = w[i]*rho0;
    gbar[i] = w[i]*Gamma*mu0;
  }
  /* density sector */
  for (int i=0; i<nvel; ++i) {
    for (int j=0; j<nvel; ++j) {
      F[i*ndof+j] = (S_r/rho0 - mu)/rho0*fbar[i]*fbar[j];
    }
    F[i*ndof+i] += mu*fbar[i];
  }
  /* order parameter sector */
  for (int i=nvel; i<ndof; ++i) {
    for (int j=nvel; j<ndof; ++j) {
    }
    F[i*ndof+i] += S_p*gbar[i-nvel];
  }
  return F;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof*ndof> moment_correlations(GpuArray<Real,ndof*ndof> F) {
  GpuArray<Real,ndof*ndof> G = {};
  // density sector
  for (int i=0; i<nvel; ++i) {
    for (int j=0; j<nvel; ++j) {
      G[i*ndof+j] = 0.0;
      for (int k=0; k<nvel; ++k) {
	for (int l=0; l<nvel; ++l) {
	  G[i*ndof+j] += e[i][k]*F[k*ndof+l]*e[j][l];
	}
      }
    }
  }
  // order parameter sector
  for (int i=nvel; i<ndof; ++i) {
    for (int j=nvel; j<ndof; ++j) {
      G[i*ndof+j] = 0.0;
      for (int k=0; k<nvel; ++k) {
	for (int l=0; l<nvel; ++l) {
	  G[i*ndof+j] += e[i][k]*F[(k+nvel)*ndof+(l+nvel)]*e[j][l];
	}
      }
    }
  }
  return G;
}

// construct noise covariance matrix
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof*ndof> noise_covariance(GpuArray<Real,ndof*ndof> const& G) {
  GpuArray<Real,ndof*ndof> Xi;
  for (int i=0; i<ndof; ++i) {
    for (int j=0; j<ndof; ++j) {
      Xi[i*ndof+j] = G[i*ndof+j];
      for (int k=0; k<ndof; ++k) {
	for (int l=0; l<ndof; ++l) {
	  Xi[i*ndof+j] -= Lambda[i][k]*G[k*ndof+l]*Lambda[j][l];
	}
      }
    }
  }
  return Xi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof*ndof> noise_covariance(int kx, int ky, int kz) {
  const GpuArray<Real,ndof*ndof> F = population_correlations(kx,ky,kz);
  const GpuArray<Real,ndof*ndof> G = moment_correlations(F);
  const GpuArray<Real,ndof*ndof> C = noise_covariance(G);
  return C;
}
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real fourier_laplace_operator(int ikx, int iky, int ikz, const Box& domain) {
  Real k2;

  IntVect n = domain.length();

  // FFTW convention for ordering of wave vectors
  Real kx = (ikx < (n[0]+1)/2) ? 2.*M_PI/n[0]*ikx : 2.*M_PI/n[0]*(ikx-n[0]);
  Real ky = (iky < (n[1]+1)/2) ? 2.*M_PI/n[1]*iky : 2.*M_PI/n[1]*(iky-n[1]);
  Real kz = (ikz < (n[2]+1)/2) ? 2.*M_PI/n[2]*ikz : 2.*M_PI/n[2]*(ikz-n[2]);

  Real cosx = cos(kx);
  Real cosy = cos(ky);
  Real cosz = cos(kz);

  Real expr1 = cosx + cosy + cosz;
  Real expr2 = cosx*cosy + cosy*cosz + cosx*cosz;
  k2 = -2./cs2*(1./9.*expr1 + 1./9.*expr2 - 2./3.);

  return k2;
}

// TODO: move to LBM_binary.H?
// thermodynamics of the binary fluid
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,4> thermodynamics(const Real rho0, const Real C0, const Real k2) {
  GpuArray<Real,4> thermovars;

  thermovars[0] = T + 2.*kappa*k2*(rho0-C0); // cs2
  thermovars[1] = 2.*kappa*k2*(2.*C0-rho0); // p_C
  thermovars[2] = - T/(rho0-C0) + 2.*chi*C0/(rho0*rho0) - 2.*kappa*k2; // mu_rho
  thermovars[3] = T*rho0/C0/(rho0-C0) - 2.*chi/rho0 + 4.*kappa*k2; // mu_C

  return thermovars;
}

// noise covariance matrix of the LB modes
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof*ndof> noise_covariance_matrix(const Real rho0, const Real C0, const Real k2) {
  const uint Q = nvel;

  const GpuArray<Real,4> thermovars = thermodynamics(rho0,C0,k2);
  const Real kT = temperature;
  const Real cs2 = thermovars[0];
  const Real p_C = thermovars[1];
  const Real mu_rho = thermovars[2];
  const Real mu_C = thermovars[3];

  // TODO: generalize to MRT
  const Real lambda_r = -1./tau_r;
  const Real lambda_p = -1./tau_p;

  // Noise covariance of the modes in k-space
  GpuArray<Real,ndof*ndof> Xi = {};
  Xi.fill(0);

  // TODO: discretized needs check!

  // diagonal part
  Xi[(   5)*ndof+(   5)] = -2.*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(   6)*ndof+(   6)] = -2.*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(   7)*ndof+(   7)] = -2.*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(   8)*ndof+(   8)] = -2.*kT*rho0*(5 - 9*cs2)*lambda_r*(2+lambda_r)/2;
  Xi[(   9)*ndof+(   9)] = -8.*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  10)*ndof+(  10)] = -(8.0/3.0)*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  11)*ndof+(  11)] = -(2.0/3.0)*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  12)*ndof+(  12)] = -(2.0/3.0)*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  13)*ndof+(  13)] = -(2.0/3.0)*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  14)*ndof+(  14)] = -4.*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  15)*ndof+(  15)] = -4.*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  16)*ndof+(  16)] = -4.*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  17)*ndof+(  17)] = -(4.0/3.0)*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(  18)*ndof+(  18)] = -(4.0/3.0)*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(Q+ 0)*ndof+(Q+ 0)] = -(4.0/3.0)*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(Q+ 1)*ndof+(Q+ 1)] = -18.*kT*rho0*(1 - cs2)*lambda_r*(2+lambda_r)/2;
  Xi[(Q+ 2)*ndof+(Q+ 2)] = -8.*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(Q+ 3)*ndof+(Q+ 3)] = -(8.0/3.0)*kT*rho0*lambda_r*(2+lambda_r)/2;
  Xi[(Q+ 4)*ndof+(Q+ 4)] = -2.*Gamma*kT/rho0*(-9*Gamma*mu_C + 5)*lambda_p*(2+lambda_p)/2;
  Xi[(Q+ 5)*ndof+(Q+ 5)] = -8.*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+ 6)*ndof+(Q+ 6)] = -(8.0/3.0)*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+ 7)*ndof+(Q+ 7)] = -(2.0/3.0)*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+ 8)*ndof+(Q+ 8)] = -(2.0/3.0)*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+ 9)*ndof+(Q+ 9)] = -(2.0/3.0)*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+10)*ndof+(Q+10)] = -4.*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+11)*ndof+(Q+11)] = -4.*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+12)*ndof+(Q+12)] = -4.*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+13)*ndof+(Q+13)] = -(4.0/3.0)*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+14)*ndof+(Q+14)] = -(4.0/3.0)*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+15)*ndof+(Q+15)] = -(4.0/3.0)*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+16)*ndof+(Q+16)] = -18.*Gamma*kT/rho0*(-Gamma*mu_C + 1)*lambda_p*(2+lambda_p)/2;
  Xi[(Q+17)*ndof+(Q+17)] = -8.*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;
  Xi[(Q+18)*ndof+(Q+18)] = -(8.0/3.0)*Gamma*kT/rho0*lambda_p*(2+lambda_p)/2;

  // rho-rho sector
  Xi[(   8)*ndof+(Q+ 1)] = -6.*kT*rho0*(3*cs2 - 1)*lambda_r*(2+lambda_r)/2;
  Xi[(Q+ 1)*ndof+(   8)] = -6.*kT*rho0*(3*cs2 - 1)*lambda_r*(2+lambda_r)/2;

  // rho-phi sector
  Xi[(   1)*ndof+(   8)] = 3.*kT*p_C/(mu_C*rho0)*lambda_r;
  Xi[(   8)*ndof+(   1)] = 3.*kT*p_C/(mu_C*rho0)*lambda_r;
  Xi[(   2)*ndof+(   5)] = C0*kT*lambda_p;
  Xi[(   3)*ndof+(   6)] = C0*kT*lambda_p;
  Xi[(   4)*ndof+(   7)] = C0*kT*lambda_p;
  Xi[(   5)*ndof+(   2)] = C0*kT*lambda_p;
  Xi[(   6)*ndof+(   3)] = C0*kT*lambda_p;
  Xi[(   7)*ndof+(   4)] = C0*kT*lambda_p;
  Xi[(   0)*ndof+(Q+ 4)] = 3.*Gamma*kT*rho0*mu_rho/cs2*lambda_p;
  Xi[(   0)*ndof+(Q+16)] = -3.*Gamma*kT*rho0*mu_rho/cs2*lambda_p;
  Xi[(Q+ 4)*ndof+(   0)] = 3.*Gamma*kT*rho0*mu_rho/cs2*lambda_p;
  Xi[(Q+16)*ndof+(   0)] = -3.*Gamma*kT*rho0*mu_rho/cs2*lambda_p;
  Xi[(   1)*ndof+(Q+ 1)] = -3.*kT*p_C/(mu_C*rho0)*lambda_r;
  Xi[(Q+ 1)*ndof+(   1)] = -3.*kT*p_C/(mu_C*rho0)*lambda_r;

  Xi[(   8)*ndof+(Q+ 4)] = 3.*kT*(Gamma*mu_C*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_p*(2+lambda_r)/2 + cs2*(3*Gamma*mu_C - 1)*p_C*lambda_r*(2+lambda_p)/2)/(cs2*mu_C*rho0);
  Xi[(   8)*ndof+(Q+16)] = -3.*kT*(Gamma*mu_C*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_p*(2+lambda_r)/2 + cs2*(3*Gamma*mu_C - 1)*p_C*lambda_r*(2+lambda_p)/2)/(cs2*mu_C*rho0);
  Xi[(Q+ 4)*ndof+(   8)] = 3.*kT*(Gamma*mu_C*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_p*(2+lambda_r)/2 + cs2*(3*Gamma*mu_C - 1)*p_C*lambda_r*(2+lambda_p)/2)/(cs2*mu_C*rho0);
  Xi[(Q+16)*ndof+(   8)] = -3.*kT*(Gamma*mu_C*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_p*(2+lambda_r)/2 + cs2*(3*Gamma*mu_C - 1)*p_C*lambda_r*(2+lambda_p)/2)/(cs2*mu_C*rho0);

  Xi[(Q+ 1)*ndof+(Q+ 4)] = -3.*kT*(Gamma*mu_C*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_p*(2+lambda_r)/2 + cs2*(3*Gamma*mu_C - 1)*p_C*lambda_r*(2+lambda_p)/2)/(cs2*mu_C*rho0);
  Xi[(Q+ 1)*ndof+(Q+16)] = 3.*kT*(Gamma*mu_C*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_p*(2+lambda_r)/2 + cs2*(3*Gamma*mu_C - 1)*p_C*lambda_r*(2+lambda_p)/2)/(cs2*mu_C*rho0);
  Xi[(Q+ 4)*ndof+(Q+ 1)] = -3.*kT*(Gamma*mu_C*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_p*(2+lambda_r)/2 + cs2*(3*Gamma*mu_C - 1)*p_C*lambda_r*(2+lambda_p)/2)/(cs2*mu_C*rho0);
  Xi[(Q+16)*ndof+(Q+ 1)] = 3.*kT*(Gamma*mu_C*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_p*(2+lambda_r)/2 + cs2*(3*Gamma*mu_C - 1)*p_C*lambda_r*(2+lambda_p)/2)/(cs2*mu_C*rho0);

  // phi-phi sector
  Xi[(Q+ 4)*ndof+(Q+16)] = -6.*Gamma*kT/rho0*(3*Gamma*mu_C - 1)*lambda_p*(2+lambda_p)/2;
  Xi[(Q+16)*ndof+(Q+ 4)] = -6.*Gamma*kT/rho0*(3*Gamma*mu_C - 1)*lambda_p*(2+lambda_p)/2;

  return Xi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<GpuComplex<Real>,ndof> kspace_white_noise(int kx, int ky, int kz, const Box& domain, RandomEngine const& engine) {
  GpuArray<GpuComplex<Real>,ndof> r = {};
  for (int i=ncons+1; i<ndof; ++i) {
    // symmetry points are purely real
    if (     ((kx == 0) || (kx == domain.length(0) - kx))
          && ((ky == 0) || (ky == domain.length(1) - ky))
          && ((kz == 0) || (kz == domain.length(2) - kz)) ) {
      // real Gaussian random variables with zero mean and variance 1
      r[i] = { RandomNormal(0., 1., engine), RandomNormal(0., 0., engine) };
    } else {
      // complex Gaussian random variables with zero mean and variance 0.5
      r[i] = { RandomNormal(0., std::sqrt(0.5), engine), RandomNormal(0., std::sqrt(0.5), engine) };
    }
  }
  return r;
}

// compute correlated noise vector from Gaussian random variables
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<GpuComplex<Real>,ndof> correlated_noise(int kx, int ky, int kz,
						   const Box& domain,
						   const RandomEngine& engine) {
  GpuArray<GpuComplex<Real>,ndof> r, xi;
  GpuArray<Real,ndof*ndof> C;

  const Real k2 = fourier_laplace_operator(kx, ky, kz, domain);

  // TODO: Generalize to non-homogeneous case?
  const Real rho0 = 1.0;
  const Real C0 = 0.5;

  // Cholesky decomposition of noise covariance matrix
  C = noise_covariance_matrix(rho0,C0,k2);
  cholesky_decomp(C,ndof,ncons+1);

  // need to generate the correct symmetries here? [uschill 07/25/2022]
  // possibly the case if C(k) != C(-k) [uschill 07/27/2022]
  r = kspace_white_noise(kx,ky,kz,domain,engine);

  // compute correlated noise vector from Gaussian random variables
  for (int i=0; i<ndof; ++i) {
    xi[i] = { 0, 0 };
    for (int j=0; j<=i; ++j) {
      xi[i] += C[i*ndof+j]*r[j];
    }
  }

  return xi;
}

// generate k-space noise for all non-conserved moments
// the required symmetries are not included here because of grid decomposition
// (this is to allow parallel generation of noise)
// the symmetries are handled when copying to one whole grid
inline void generate_kspace_noise(const Geometry& geom,
				  MultiFab& kspace_noise_real,
				  MultiFab& kspace_noise_imag) {
  const Box domain = geom.Domain();

  // include density? [uschill 07/26/2022]

  // generate noise in whole box because of grid decomposition
  // (generating noise without grid decomposition may be faster)
  for (MFIter mfi(kspace_noise_real); mfi.isValid(); ++mfi) {
    const Box& box = mfi.validbox();
    const Array4<Real>& xi_real = kspace_noise_real.array(mfi);
    const Array4<Real>& xi_imag = kspace_noise_imag.array(mfi);
    // construct noise in k-space
    ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int kx, int ky, int kz, RandomEngine const& engine) {
      if (kx <= domain.length(0)/2) { // need only half of k-space for c2r FFT
        GpuArray<GpuComplex<Real>,ndof> xi = {};

	      // compute correlated noise in k-space
	      xi = correlated_noise(kx,ky,kz,domain,engine);

        for (int i=0; i<ndof; ++i) {
          xi_real(kx,ky,kz,i) = xi[i].real();
          xi_imag(kx,ky,kz,i) = xi[i].imag();
        }
      }
    });
  }

}

#if 1
inline void check_kspace_symmetries(const Geometry& geom,
				    const MultiFab& noise_onegrid,
				    const Vector<std::unique_ptr<BaseFab<GpuComplex<Real>>>>& spectral_field) {
  const Box domain = geom.Domain();
  IntVect fft_size = domain.length();
  fft_size[0] = fft_size[0]/2 + 1;
  Box fft_box = Box(IntVect(0), fft_size - IntVect(1));
  Array4<GpuComplex<Real>> const& xi = (*spectral_field[0]).array();
  for (MFIter mfi(noise_onegrid); mfi.isValid(); ++mfi) {
    ParallelFor(fft_box, [=] AMREX_GPU_DEVICE(int kx, int ky, int kz) {
      bool symmetric = true;
      int kxloc = (kx == 0) ? 0 : domain.length(0) - kx;
      int kyloc = (ky == 0) ? 0 : domain.length(1) - ky;
      int kzloc = (kz == 0) ? 0 : domain.length(2) - kz;
      // symmetry points are purely real
      if (((kx == 0) || (kx == domain.length(0) - kx))
	  && ((ky == 0) || (ky == domain.length(1) - ky))
	  && ((kz == 0) || (kz == domain.length(2) - kz))) {
	symmetric &= (xi(kx,ky,kz).imag() == 0);
	if (!symmetric) {
	  Print() << "symmetry violation 1! ("
		  << kx << "," << ky << "," << kz << ") "
		  << xi(kx,ky,kz).imag() << " "
	          << symmetric << " " << (xi(kx,ky,kz).imag() == 0.)
		  << std::endl;
	  exit(0);
	}
      }
      if (kx > domain.length(0)/2) {
	Print() << "This should never execute" << std::endl;
	symmetric &= ((xi(kx,ky,kz).real() ==  xi(kxloc,kyloc,kzloc).real())
		   && (xi(kx,ky,kz).imag() == -xi(kxloc,kyloc,kzloc).imag()));
	if (!symmetric) {
	  Print() << "symmetry violation 2! ("
		  << kx << "," << ky << "," << kz << ")"
		  << std::endl;
	  exit(0);
	}
      } else {
	if ((ky > domain.length(1)/2) && (kx == kxloc)) {
	  symmetric &= ((xi(kx,ky,kz).real() ==  xi(kxloc,kyloc,kzloc).real())
		     && (xi(kx,ky,kz).imag() == -xi(kxloc,kyloc,kzloc).imag()));
	  if (!symmetric) {
	    Print() << "symmetry violation 3a! ("
		    << kx << "," << ky << "," << kz << ") ("
	            << kxloc << "," << kyloc << "," << kzloc << ")"
	            << " " << xi(kx,ky,kz).real() << " " << xi(kxloc,kyloc,kzloc).real()
		    << " " << xi(kx,ky,kz).imag() << " " << xi(kxloc,kyloc,kzloc).imag()
		    << std::endl;
	    exit(0);
	  }
	} else {
	  if ((kz > domain.length(2)/2) && (ky == kyloc) && (kx == kxloc)) {
	    symmetric &= ((xi(kx,ky,kz).real() ==  xi(kxloc,kyloc,kzloc).real())
		       && (xi(kx,ky,kz).imag() == -xi(kxloc,kyloc,kzloc).imag()));
	    if (!symmetric) {
	      Print() << "symmetry violation 3b! ("
		      << kx << "," << ky << "," << kz << ") ("
		      << kxloc << "," << kyloc << "," << kzloc << ")"
		      << " " << xi(kx,ky,kz).real() << " " << xi(kxloc,kyloc,kzloc).real()
		      << " " << xi(kx,ky,kz).imag() << " " << xi(kxloc,kyloc,kzloc).imag()
		      << std::endl;
	      exit(0);
	    }
	  } // if ((kz > domain.length(2)/2) ...
	} // if ((ky > domain.length(1)/2) ...
      } // if ((kx > domain.length(0)/2) ...
    });
  }
  return;
}
#endif

inline void compute_ifft(const Geometry& geom, MultiFab& realspace_noise,
			 const MultiFab& kspace_noise_real,
			 const MultiFab& kspace_noise_imag) {

#ifdef AMREX_USE_CUDA
    //Print() << "Using cuFFT\n";
    using FFTplan = cufftHandle;
    using FFTcomplex = cuDoubleComplex;
#else
    //Print() << "Using FFTW\n";
    using FFTplan = fftw_plan;
    using FFTcomplex = fftw_complex;
#endif

    // BoxArray and DistributionMapping for whole domain without decomposition
    Box domain = geom.Domain();
    BoxArray ba_onegrid(domain);
    DistributionMapping dm_onegrid(ba_onegrid);

    // FFT needs the whole grid without grid decomposition
    MultiFab noise_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    MultiFab kspace_noise_real_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    MultiFab kspace_noise_imag_onegrid(ba_onegrid, dm_onegrid, 1, 0);

    // number of sites for normalization of FFT
    long npts = domain.length(0)*domain.length(1)*domain.length(2);
    Real sqrtnpts = std::sqrt(npts);

    // Box for the complex conjugate spectral field (x-size is halved plus one)
    IntVect fft_size = domain.length();
    IntVect fft_adjust = { fft_size[0]/2 - 1, 0, 0 };
    Box fft_box = Box(IntVect(0), fft_size-fft_adjust-IntVect(1));

    // container to store the complex k-space noise for inverse FFT
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real>>>> spectral_field;

    // is this a memory leak? [uschill 07/24/2022]
    spectral_field.emplace_back(new BaseFab<GpuComplex<Real>>(fft_box,1,The_Device_Arena()));

    // for CUDA builds we only need to build the plan once; track whether we did
    bool built_plan = false;
    Vector<FFTplan> fftw_plans;
    FFTplan plan;

    for (int k=0; k<ndof; ++k) {

      // first copy the k-th conponent of the noise to one grid
      // may be faster to generate the noise here? [uschill 07/27/2022]
      kspace_noise_real_onegrid.ParallelCopy(kspace_noise_real,k,0,1);
      kspace_noise_imag_onegrid.ParallelCopy(kspace_noise_imag,k,0,1);

      // create the FFT plans
      if (!built_plan) {
	      for (MFIter mfi(noise_onegrid); mfi.isValid(); ++mfi) {

#ifdef AMREX_USE_CUDA
	        cufftResult result = cufftPlan3d(&plan, fft_size[2], fft_size[1], fft_size[0], CUFFT_C2R);
	        if (result != CUFFT_SUCCESS) {
	          amrex::AllPrint() << " cufftplan3d forward failed! Error: "
			        << cufftErrorToString(result) << "\n";
	        }
#else
	        plan = fftw_plan_dft_c2r_3d(fft_size[2], fft_size[1], fft_size[0],
				      reinterpret_cast<FFTcomplex*>
				      (spectral_field.back()->dataPtr()),
				      noise_onegrid[mfi].dataPtr(),
				      FFTW_ESTIMATE);
#endif
	        fftw_plans.push_back(plan);
	      }
	      built_plan = true;
      }

      ParallelDescriptor::Barrier(); // is this needed? [uschill 07/25/2022]

      // copy the complex noise to the spectral field for complex-to-real FFT
      // this takes care of the required k-space symmetries
      for (MFIter mfi(noise_onegrid); mfi.isValid(); ++mfi) {
	      Array4<Real> const& xi_real = kspace_noise_real_onegrid.array(mfi);
	      Array4<Real> const& xi_imag = kspace_noise_imag_onegrid.array(mfi);
	      Array4<GpuComplex<Real>> const& xi = (*spectral_field[0]).array();
	      ParallelFor(fft_box, [=] AMREX_GPU_DEVICE(int kx, int ky, int kz) {
	        //if (kx <= domain.length(0)/2)
	        {
	          // regular points
            xi(kx,ky,kz).m_real = xi_real(kx,ky,kz);
            xi(kx,ky,kz).m_imag = xi_imag(kx,ky,kz);
            // symmetry points (corners of first quadrant) are purely real
            //if (((kx == 0) || (kx == domain.length(0) - kx))
            //	&& ((ky == 0) || (ky == domain.length(1) - ky))
            //	&& ((kz == 0) || (kz == domain.length(2) - kz))) {
            //  xi(kx,ky,kz).m_imag = 0;
            //}
            // complex conjugate symmetries
            int kxloc = (kx == 0) ? 0 : domain.length(0) - kx;
            int kyloc = (ky == 0) ? 0 : domain.length(1) - ky;
            int kzloc = (kz == 0) ? 0 : domain.length(2) - kz;
            if (kx > domain.length(0)/2) {
              Print() << "This should never execute" << std::endl;
              xi(kx,ky,kz).m_real =  xi_real(kxloc,kyloc,kzloc);
              xi(kx,ky,kz).m_imag = -xi_imag(kxloc,kyloc,kzloc);
            }
            if  ( ( (ky > domain.length(1)/2) && (kx == kxloc) )
            || ( (kz > domain.length(2)/2) && (ky == kyloc) && (kx == kxloc) ) ) {
              xi(kx,ky,kz).m_real =  xi_real(kxloc,kyloc,kzloc);
              xi(kx,ky,kz).m_imag = -xi_imag(kxloc,kyloc,kzloc);
            }
          }
	      });
      }

      ParallelDescriptor::Barrier();
      check_kspace_symmetries(geom,noise_onegrid,spectral_field);

      // inverse FFT (complex to real)
      for (MFIter mfi(noise_onegrid); mfi.isValid(); ++mfi) {
	      int i = mfi.LocalIndex();
#ifdef AMREX_USE_CUDA
	      cufftSetStream(forward_plan[i], amrex::Gpu::gpuStream());
	      cufftResult result = cufftExecC2R(forward_plan[i],
					  reinterpret_case<FFTcomplex*>
					  (field[i]->dataPtr()),
					  noise_onegrid[mfi].dataPtr());
	      if (result != CUFFT_SUCCESS) {
	        amrex::AllPrint() << " forward transform using cufftExec failed! Error: "
			      << cufftErrorToString(result) << "\n";
	      }
#else
	      fftw_execute(fftw_plans[i]);
#endif
      }

      // copy the real-space noise back to the k-th component of the MultiFab
      realspace_noise.ParallelCopy(noise_onegrid,0,k,1);
      // normalization from FFT
      realspace_noise.mult(1./sqrtnpts,k,1);

    }

    // destroy fft plans
    for (int i=0; i<fftw_plans.size(); ++i) {
#ifdef AMREX_USE_CUDA
      cufftDestroy(fftw_plans[i]);
#else
      fftw_destroy_plan(fftw_plans[i]);
#endif
    }

}

// LB thermalization procedure for spatially correlated, non-diagonal noise
inline void generate_correlated_fluctuations(const Geometry& geom,
				  MultiFab& hydrovs,
				  MultiFab& noise) {
  BoxArray ba = noise.boxArray();
  DistributionMapping dm = noise.DistributionMap();
  MultiFab kspace_noise_real(ba, dm, ndof, 0);
  MultiFab kspace_noise_imag(ba, dm, ndof, 0);

  kspace_noise_real.setVal(0.);
  kspace_noise_imag.setVal(0.);

  // TODO: check if this takes care of ndof entries!?

  // generate noise in k-space
  generate_kspace_noise(geom, kspace_noise_real, kspace_noise_imag);

  // note that the k-space noise is generated without the required symmetries
  // (this is to allow for parallel generation of noise)
  // the k-space symmetries are handled when copying in compute_ifft

  // inverse Fourier transform noise vector to real space
  compute_ifft(geom, noise, kspace_noise_real, kspace_noise_imag);

}

// compute spatially uncorrelated noise vector from Gaussian random variables
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof> uncorrelated_noise(const Real rho0, const Real C0, const RandomEngine& engine) {
  GpuArray<Real,ndof> r, xi;
  GpuArray<Real,ndof*ndof> C;

  // Cholesky decomposition of noise covariance matrix
  C = noise_covariance_matrix(rho0,C0,0);
  cholesky_decomp(C,ndof,ncons+1);

  // random white noise
  for (int i=ncons+1; i<ndof; ++i) {
    r[i] = RandomNormal(0., 1., engine);
  }

  // compute noise vector from Gaussian random variables
  for (int i=0; i<ndof; ++i) {
    xi[i] = 0;
    for (int j=0; j<=i; ++j) {
      xi[i] += C[i*ndof+j]*r[j];
    }
  }

  return xi;
}

// generate real-space noise for all non-conserved moments
inline void generate_fluctuations(const Geometry& geom, MultiFab& hydrovs, MultiFab& noise) {
  for (MFIter mfi(noise); mfi.isValid(); ++mfi) {
    const Box& box = mfi.validbox();
    const Array4<Real>& xi = noise.array(mfi);
    const Array4<Real>& h = hydrovs.array(mfi);
    ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real rho0 = h(x,y,z,0);
      const Real C0 = h(x,y,z,1);
      GpuArray<Real,ndof> r = uncorrelated_noise(rho0,C0,engine);
      for (int i=0; i<ndof; ++i) {
        xi(x,y,z,i) = r[i];
      }
    });
  }
}
#endif
