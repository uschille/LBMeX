#ifndef LBM_FLUCTUATIONS_H_
#define LBM_FLUCTUATIONS_H_

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif
#include <AMReX_GpuComplex.H>
#include <math.h>

#include "LBM_d3q19.H"
#include "LBM_binary.H"

const int ncons = 1 + AMREX_SPACEDIM;

// Cholesky decomposition of matrix A
// result is stored in lower triangle of A
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cholesky_decomp(GpuArray<Real,nvel*nvel>& A, const int n, const int bstart) {
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
          // Print() << "Entry value: " << A[j*n+j] << " Row: " << i << " Column: " << j << "\n";
	      } else {
          // Print() << "Entry value: " << A[j*n+j] << " Row: " << i << " Column: " << j << "\n";
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

Real fourier_laplace_operator(int kx, int ky, int kz){
  // https://www.uni-muenster.de/Physik.TP/archive/fileadmin/lehre/NumMethoden/SoSe10/Skript/Ordening.pdf
  Real out;
  Real kxr = (Real)kx; Real kyr = (Real)ky; Real kzr = (Real)kz; Real nxr = (Real)nx;
  Real dpi = 2.*M_PI;

  if (kx < nx/2){kxr = kxr/nxr;}else{kxr = (kxr - nxr)/nxr;}
  if (ky < nx/2){kyr = kyr/nxr;}else{kyr = (kyr - nxr)/nxr;}
  if (kz < nx/2){kzr = kzr/nxr;}else{kzr = (kzr - nxr)/nxr;}

  Real cosx = cos(kxr*dpi);
  Real cosy = cos(kyr*dpi);
  Real cosz = cos(kzr*dpi);
  // Print() << "cosx: " << cos((Real)kx/(Real)nx*M_PI) << " cosy: " << cos((Real)ky/(Real)nx*M_PI) << " cosz: " << cos((Real)kz/(Real)nx*M_PI) << "\n";

  Real expr1 = cosx + cosy + cosz;
  Real expr2 = cosx*cosy + cosy*cosz + cosx*cosz;
  out = 2./9.*expr1 + 2./9.*expr2 - 4./3.;
  out /= -cs2;

  // Print() << "nx: " << nxr << "\n";
  // Print() << "kx: " << kx << " kxr:" << kxr << "\n";
  // Print() << "ky: " << ky << " kyr:" << kyr << "\n";
  // Print() << "kz: " << kz << " kzr:" << kzr << "\n";
  // Print() << "cs2k: " << out << "\n";
  // Print() << "kxr:" << kxr << " kyr:" << kyr << " kzr:" << kzr << " cs2k:" << out << "\n";
  return out;
}

Real thermal_cs2(Real rho){
  Real square_term = (rho - rhol)*(rho - rhov);
  Real expr1 = Beta*square_term*(rhol + rhov - 6.0*rho);
  Real expr2 = Beta*(rho - rhol)*(2.0*rho*(-2.0*rho + rhol + rhov) + square_term);
  Real expr3 = Beta*(rho - rhov)*(2.0*rho*(-2.0*rho + rhol + rhov) + square_term);
  Real out = -expr1 - expr2 - expr3;
  return out;
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
GpuArray<Real,nvel*nvel> population_correlations(int kx, int ky, int kz) {
  GpuArray<Real,nvel*nvel> F = {};
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
      F[i*nvel+j] = (S_r/rho0 - mu)/rho0*fbar[i]*fbar[j];
    }
    F[i*nvel+i] += mu*fbar[i];
  }
  /* order parameter sector */
  for (int i=nvel; i<nvel; ++i) {
    for (int j=nvel; j<nvel; ++j) {
    }
    F[i*nvel+i] += S_p*gbar[i-nvel];
  }
  return F;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,nvel*nvel> moment_correlations(GpuArray<Real,nvel*nvel> F) {
  GpuArray<Real,nvel*nvel> G = {};
  // density sector
  for (int i=0; i<nvel; ++i) {
    for (int j=0; j<nvel; ++j) {
      G[i*nvel+j] = 0.0;
      for (int k=0; k<nvel; ++k) {
	for (int l=0; l<nvel; ++l) {
	  G[i*nvel+j] += e[i][k]*F[k*nvel+l]*e[j][l];
	}
      }
    }
  }
  // order parameter sector
  for (int i=nvel; i<nvel; ++i) {
    for (int j=nvel; j<nvel; ++j) {
      G[i*nvel+j] = 0.0;
      for (int k=0; k<nvel; ++k) {
	for (int l=0; l<nvel; ++l) {
	  G[i*nvel+j] += e[i][k]*F[(k+nvel)*nvel+(l+nvel)]*e[j][l];
	}
      }
    }
  }
  return G;
}

// construct noise covariance matrix
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,nvel*nvel> noise_covariance(GpuArray<Real,nvel*nvel> const& G) {
  GpuArray<Real,nvel*nvel> Xi;
  for (int i=0; i<nvel; ++i) {
    for (int j=0; j<nvel; ++j) {
      Xi[i*nvel+j] = G[i*nvel+j];
      for (int k=0; k<nvel; ++k) {
	for (int l=0; l<nvel; ++l) {
	  Xi[i*nvel+j] -= Lambda[i][k]*G[k*nvel+l]*Lambda[j][l];
	}
      }
    }
  }
  return Xi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,nvel*nvel> noise_covariance(int kx, int ky, int kz) {
  const GpuArray<Real,nvel*nvel> F = population_correlations(kx,ky,kz);
  const GpuArray<Real,nvel*nvel> G = moment_correlations(F);
  const GpuArray<Real,nvel*nvel> C = noise_covariance(G);
  return C;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ncomp*ncomp> noise_covariance_ideal(int kx, int ky, int kz) {
  const Real gamma = 1. - 1./tau;
  const Real rho0 = rhol;

  // Noise covariance of the modes in k-space
  GpuArray<Real,ncomp*ncomp> Xi = {};
  Xi.fill(0);
	for (int i=ncons; i<ncomp; ++i) {
	  // noise covariance for ideal fluid
	  Xi[i*ncomp+i] = b[i]*temperature/cs2*(1.-gamma*gamma);
	}

  return Xi;
}
#endif

// construct noise covariance matrix of the modes in k-space
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,nvel*nvel> noise_covariance(int kx, int ky, int kz, Real rho) { // TODO: add function arguments: rho, phi, k, others?
  const Real rho0 = rhol;
  const Real lambda_s = -1.0/tau_r;
  const Real lambda_b = lambda_s;
  const Real lambda_o = lambda_s;
  const Real lambda_e = lambda_s;

  const Real k2 = fourier_laplace_operator(kx, ky, kz);
  // Print() << "kx: " << kx << " ky: " << ky << " kz: " << kz << " k2: " << k2 << "\n";
  // const Real k2 = 0.0;
  // Print() << "kx: " << kx_conv << " ky:" << ky_conv << " kz:" << kz_conv << "\n";

  // const Real system_cs2 = 2.*rho0*Beta*pow(rhol - rhov, 2);
  const Real system_cs2 = thermal_cs2(rho);
  const Real cs2_k = system_cs2 + rho*kappa*k2;
  // Print() << "k dependent cs2: " << cs2_k << "\n";
 
 // Noise covariance of the modes in k-space
  GpuArray<Real,nvel*nvel> Xi = {};
  Xi.fill(0);

  Xi[ 4*nvel+ 4] = -rho0*temperature*(5.-9.*cs2)*lambda_b*(lambda_b+2.);
  Xi[ 5*nvel+ 5] = -4.*rho0*temperature*lambda_s*(lambda_s+2.);
  Xi[ 6*nvel+ 6] = -4./3.*rho0*temperature*lambda_s*(lambda_s+2.);
  Xi[ 7*nvel+ 7] = -1./3.*rho0*temperature*lambda_s*(lambda_s+2.);
  Xi[ 8*nvel+ 8] = -1./3.*rho0*temperature*lambda_s*(lambda_s+2.);
  Xi[ 9*nvel+ 9] = -1./3.*rho0*temperature*lambda_s*(lambda_s+2.);
  // int idx = 4;
  // Xi[ idx*nvel+ idx] = -rho0*temperature*(5.-9.*cs2)*lambda_b*(lambda_b+2.); idx++;
  // Xi[ idx*nvel+ idx] = -1./3.*rho0*temperature*lambda_s*(lambda_s+2.);   idx++;
  // Xi[ idx*nvel+ idx] = -1./3.*rho0*temperature*lambda_s*(lambda_s+2.);   idx++;
  // Xi[ idx*nvel+ idx] = -4.*rho0*temperature*lambda_s*(lambda_s+2.);      idx++;
  // Xi[ idx*nvel+ idx] = -4./3.*rho0*temperature*lambda_s*(lambda_s+2.);   idx++;
  // Xi[ idx*nvel+ idx] = -1./3.*rho0*temperature*lambda_s*(lambda_s+2.);   idx++;

  Xi[10*nvel+10] = -2.*rho0*temperature*lambda_o*(lambda_o+2.);
  Xi[11*nvel+11] = -2.*rho0*temperature*lambda_o*(lambda_o+2.);
  Xi[12*nvel+12] = -2.*rho0*temperature*lambda_o*(lambda_o+2.);
  Xi[13*nvel+13] = -2./3.*rho0*temperature*lambda_o*(lambda_o+2.);
  Xi[14*nvel+14] = -2./3.*rho0*temperature*lambda_o*(lambda_o+2.);
  Xi[15*nvel+15] = -2./3.*rho0*temperature*lambda_o*(lambda_o+2.);
  Xi[16*nvel+16] = -9.*rho0*temperature*(1-cs2)*lambda_e*(lambda_e+2.);
  Xi[17*nvel+17] = -4.*rho0*temperature*lambda_e*(lambda_e+2.);
  Xi[18*nvel+18] = -4./3.*rho0*temperature*lambda_e*(lambda_e+2.);

  Xi[ 4*nvel+16] = -3.*rho0*temperature*(3.*cs2-1.)*(lambda_b*lambda_e+lambda_b+lambda_e);
  Xi[16*nvel+ 4] = Xi[ 4*nvel+16];

  return Xi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<GpuComplex<Real>,nvel> kspace_white_noise(int kx, int ky, int kz, const Box& domain, RandomEngine const& engine) {
  GpuArray<GpuComplex<Real>,nvel> r = {};
  for (int i=ncons; i<nvel; ++i) {
    // symmetry points are purely real
    if (     ((kx == 0) || (kx == domain.length(0) - kx))
          && ((ky == 0) || (ky == domain.length(1) - ky))
          && ((kz == 0) || (kz == domain.length(2) - kz)) ) {
      // real Gaussian random variables with zero mean and variance 1
      r[i] = { RandomNormal(0., 1., engine), RandomNormal(0., 0., engine) };
    } else {
      // complex Gaussian random variables with zero mean and variance 0.5
      r[i] = { RandomNormal(0., std::sqrt(0.5), engine), RandomNormal(0., std::sqrt(0.5), engine) };
      // r[i] = { RandomNormal(0., 1, engine), RandomNormal(0., 0, engine) };
    }
  }
  return r;
}

// compute correlated noise vector from Gaussian random variables
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<GpuComplex<Real>,nvel> correlated_noise(int kx, int ky, int kz,
						   const Box& domain,
						   const RandomEngine& engine,
               const Real rho) {
  GpuArray<GpuComplex<Real>,nvel> r, xi;
  GpuArray<Real,nvel*nvel> C;

  // Cholesky decomposition of noise covariance matrix
  C = noise_covariance(kx,ky,kz,rho);

  // Print() << "Xi diag 4: " << C[ 4*nvel+ 4] << "\n";
  // Print() << "Xi diag 5: " << C[ 5*nvel+ 5] << "\n";
  // Print() << "Xi diag 6: " << C[ 6*nvel+ 6] << "\n";
  // Print() << "Xi diag 7: " << C[ 7*nvel+ 7] << "\n";
  // Print() << "Xi diag 10: " << C[ 10*nvel+ 10] << "\n";
  // Print() << "Xi diag 13: " << C[ 13*nvel+ 13] << "\n";
  // Print() << "Xi diag 16: " << C[ 16*nvel+ 16] << "\n";
  // Print() << "Xi diag 17: " << C[ 17*nvel+ 17] << "\n";
  // Print() << "Xi diag 18: " << C[ 18*nvel+ 18] << "\n";
  // Print() << "Xi off diag 4,16: " << C[ 4*nvel+16] << "\n";

  cholesky_decomp(C,nvel,ncons);
  // Print() << "After decomposition" << "\n";
  // Print() << "Xi diag 4: " << C[ 4*nvel+ 4] << "\n";
  // Print() << "Xi diag 5: " << C[ 5*nvel+ 5] << "\n";
  // Print() << "Xi diag 6: " << C[ 6*nvel+ 6] << "\n";
  // Print() << "Xi diag 7: " << C[ 7*nvel+ 7] << "\n";
  // Print() << "Xi diag 10: " << C[ 10*nvel+ 10] << "\n";
  // Print() << "Xi diag 13: " << C[ 13*nvel+ 13] << "\n";
  // Print() << "Xi diag 16: " << C[ 16*nvel+ 16] << "\n";
  // Print() << "Xi diag 17: " << C[ 17*nvel+ 17] << "\n";
  // Print() << "Xi diag 18: " << C[ 18*nvel+ 18] << "\n";
  // Print() << "Xi off diag 4,16: " << C[ 4*nvel+16] << "\n";

  // need to generate the correct symmetries here? [uschill 07/25/2022]
  // possibly the case if C(k) != C(-k) [uschill 07/27/2022]
  r = kspace_white_noise(kx,ky,kz,domain,engine);

  // compute correlated noise vector from Gaussian random variables
  for (int i=0; i<nvel; ++i) {
    xi[i] = { 0, 0 };
    for (int j=0; j<=i; ++j) {
      xi[i] += C[i*nvel+j]*r[j];
    }
    // Print() << "xi " << i << " " << xi[i] << "\n";
  }
  return xi;
}

// generate k-space noise for all non-conserved moments
// the required symmetries are not included here because of grid decomposition
// (this is to allow parallel generation of noise)
// the symmetries are handled when copying to one whole grid
inline void generate_kspace_noise(const Geometry& geom,
				  MultiFab& kspace_noise_real,
				  MultiFab& kspace_noise_imag,
          MultiFab& hydrovs) {
  const Box domain = geom.Domain();

  // include density? [uschill 07/26/2022]
  // add hydrovs as an input to get density for multiphase/multicomponent mixtures
  // to calculate thermal speed of sound. [nkarthi 07/11/2024]

  // generate noise in whole box because of grid decomposition
  // (generating noise without grid decomposition may be faster)
  for (MFIter mfi(kspace_noise_real); mfi.isValid(); ++mfi) {
    const Box& box = mfi.validbox();
    const Array4<Real>& xi_real = kspace_noise_real.array(mfi);
    const Array4<Real>& xi_imag = kspace_noise_imag.array(mfi);
    const Array4<Real>& h = hydrovs.array(mfi);
    // construct noise in k-space
    ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int kx, int ky, int kz, RandomEngine const& engine) {
      if (kx <= domain.length(0)/2) { // need only half of k-space for c2r FFT
        GpuArray<GpuComplex<Real>,nvel> xi = {};

	      // compute correlated noise in k-space
        const Real rho = h(kx, ky, kz, 0);
	      xi = correlated_noise(kx,ky,kz,domain,engine,rho);

        for (int i=0; i<nvel; ++i) {
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

      for (int k=0; k<nvel; ++k) {

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
inline void generate_fluctuations(const Geometry& geom,
				  MultiFab& hydrovs,
				  MultiFab& noise) {
  BoxArray ba = noise.boxArray();
  DistributionMapping dm = noise.DistributionMap();
  MultiFab kspace_noise_real(ba, dm, nvel, 0);
  MultiFab kspace_noise_imag(ba, dm, nvel, 0);

  kspace_noise_real.setVal(0.);
  kspace_noise_imag.setVal(0.);

  // TODO: check if this takes care of nvel entries!?

  // generate noise in k-space
  generate_kspace_noise(geom, kspace_noise_real, kspace_noise_imag, hydrovs);
  // Print() << "k space noise generated\n";

  // note that the k-space noise is generated without the required symmetries
  // (this is to allow for parallel generation of noise)
  // the k-space symmetries are handled when copying in compute_ifft

  // inverse Fourier transform noise vector to real space
  compute_ifft(geom, noise, kspace_noise_real, kspace_noise_imag);
  // Print() << "kspace noise converted to real space noise \n";
}
#endif
