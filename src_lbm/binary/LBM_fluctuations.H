#ifndef LBM_FLUCTUATIONS_H_
#define LBM_FLUCTUATIONS_H_

#include <AMReX_FFT.H>

#include "LBM_binary.H"
#include "LBM_tests.H"

AMREX_GPU_MANAGED Real temperature = 1e-7;

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

// noise covariance matrix of the LB modes
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof*ndof> noise_covariance_matrix(const Real rho0, const Real phi0, const Real k2) {
  const uint Q = nvel;

  // TODO: generalize to MRT
  const Real lambdaLB_r = -1./tau_r;
  const Real lambdaLB_p = -1./tau_p;

  // TODO: needs check!
  const Real lambda_r = -lambdaLB_r*(2+lambdaLB_r)/2;
  const Real lambda_p = -lambdaLB_p*(2+lambdaLB_p)/2;
  const Real lambda_rp = -lambdaLB_r*(2+lambdaLB_p)/2;
  const Real lambda_pr = -lambdaLB_p*(2+lambdaLB_r)/2;

  // f = T/2*(rho+phi)*log((rho+phi)/2) + T/2*(rho-phi)*log((rho-phi)/2) - T*rho + chi/4*(rho**2-phi**2)/rho + kappa*(Drho**2+Dphi**2)
  // p = rho*T - kappa*(rho*D2rho+phi*D2phi)
  // mu =  T/2*log(rho+phi) - T/2*log(rho-phi) - chi/2*phi/rho - kappa*D2phi
  const Real kT = temperature;
  const Real cs2 = T + kappa*k2*rho0; // p_rho => modified speed of sound
  const Real p_phi = kappa*k2*phi0;
  const Real mu_rho = -T*phi0/(rho0*rho0-phi0*phi0) + chi/2*phi0/(rho0*rho0);
  const Real mu_phi = T*rho0/(rho0*rho0-phi0*phi0) - chi/2/rho0 + kappa*k2;

  // Noise covariance of the modes in k-space
  GpuArray<Real,ndof*ndof> Xi = {};
  Xi.fill(0);

  // diagonal part
  Xi[(   5)*ndof+(   5)] = 2.*Gamma*kT/rho0*lambda_p;
  Xi[(   6)*ndof+(   6)] = 2.*Gamma*kT/rho0*lambda_p;
  Xi[(   7)*ndof+(   7)] = 2.*Gamma*kT/rho0*lambda_p;
  Xi[(   8)*ndof+(   8)] = 2.*kT*rho0*(5 - 9*cs2)*lambda_r;
  Xi[(   9)*ndof+(   9)] = 8.*kT*rho0*lambda_r;
  Xi[(  10)*ndof+(  10)] = (8.0/3.0)*kT*rho0*lambda_r;
  Xi[(  11)*ndof+(  11)] = (2.0/3.0)*kT*rho0*lambda_r;
  Xi[(  12)*ndof+(  12)] = (2.0/3.0)*kT*rho0*lambda_r;
  Xi[(  13)*ndof+(  13)] = (2.0/3.0)*kT*rho0*lambda_r;
  Xi[(  14)*ndof+(  14)] = 4.*kT*rho0*lambda_r;
  Xi[(  15)*ndof+(  15)] = 4.*kT*rho0*lambda_r;
  Xi[(  16)*ndof+(  16)] = 4.*kT*rho0*lambda_r;
  Xi[(  17)*ndof+(  17)] = (4.0/3.0)*kT*rho0*lambda_r;
  Xi[(  18)*ndof+(  18)] = (4.0/3.0)*kT*rho0*lambda_r;
  Xi[(Q+ 0)*ndof+(Q+ 0)] = (4.0/3.0)*kT*rho0*lambda_r;
  Xi[(Q+ 1)*ndof+(Q+ 1)] = 18.*kT*rho0*(1 - cs2)*lambda_r;
  Xi[(Q+ 2)*ndof+(Q+ 2)] = 8.*kT*rho0*lambda_r;
  Xi[(Q+ 3)*ndof+(Q+ 3)] = (8.0/3.0)*kT*rho0*lambda_r;
  Xi[(Q+ 4)*ndof+(Q+ 4)] = 2.*Gamma*kT/rho0*(-9*Gamma*mu_phi + 5)*lambda_p;
  Xi[(Q+ 5)*ndof+(Q+ 5)] = 8.*Gamma*kT/rho0*lambda_p;
  Xi[(Q+ 6)*ndof+(Q+ 6)] = (8.0/3.0)*Gamma*kT/rho0*lambda_p;
  Xi[(Q+ 7)*ndof+(Q+ 7)] = (2.0/3.0)*Gamma*kT/rho0*lambda_p;
  Xi[(Q+ 8)*ndof+(Q+ 8)] = (2.0/3.0)*Gamma*kT/rho0*lambda_p;
  Xi[(Q+ 9)*ndof+(Q+ 9)] = (2.0/3.0)*Gamma*kT/rho0*lambda_p;
  Xi[(Q+10)*ndof+(Q+10)] = 4.*Gamma*kT/rho0*lambda_p;
  Xi[(Q+11)*ndof+(Q+11)] = 4.*Gamma*kT/rho0*lambda_p;
  Xi[(Q+12)*ndof+(Q+12)] = 4.*Gamma*kT/rho0*lambda_p;
  Xi[(Q+13)*ndof+(Q+13)] = (4.0/3.0)*Gamma*kT/rho0*lambda_p;
  Xi[(Q+14)*ndof+(Q+14)] = (4.0/3.0)*Gamma*kT/rho0*lambda_p;
  Xi[(Q+15)*ndof+(Q+15)] = (4.0/3.0)*Gamma*kT/rho0*lambda_p;
  Xi[(Q+16)*ndof+(Q+16)] = 18.*Gamma*kT/rho0*(-Gamma*mu_phi + 1)*lambda_p;
  Xi[(Q+17)*ndof+(Q+17)] = 8.*Gamma*kT/rho0*lambda_p;
  Xi[(Q+18)*ndof+(Q+18)] = (8.0/3.0)*Gamma*kT/rho0*lambda_p;

  // rho-rho sector [0, 2..4, 8..(Q+3)]
  Xi[(   8)*ndof+(Q+ 1)] = 6.*kT*rho0*(3*cs2 - 1)*lambda_r;
  Xi[(Q+ 1)*ndof+(   8)] = 6.*kT*rho0*(3*cs2 - 1)*lambda_r;

  // phi-phi sector [1, 5..7, (Q+4)..(Q-1)]
  Xi[(Q+ 4)*ndof+(Q+16)] = 6.*Gamma*kT/rho0*(3*Gamma*mu_phi - 1)*lambda_p;
  Xi[(Q+16)*ndof+(Q+ 4)] = 6.*Gamma*kT/rho0*(3*Gamma*mu_phi - 1)*lambda_p;

  // rho-phi sector
  Xi[(   8)*ndof+(Q+ 4)] = -3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
  Xi[(Q+ 4)*ndof+(   8)] = -3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
  Xi[(   8)*ndof+(Q+16)] = 3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
  Xi[(Q+16)*ndof+(   8)] = 3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);

  Xi[(Q+ 1)*ndof+(Q+ 4)] = 3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
  Xi[(Q+ 4)*ndof+(Q+ 1)] = 3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
  Xi[(Q+ 1)*ndof+(Q+16)] = -3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
  Xi[(Q+16)*ndof+(Q+ 1)] = -3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);

  Xi[(   0)*ndof+(Q+ 4)] = -3.*Gamma*kT*rho0*mu_rho/cs2*lambda_pr;
  Xi[(Q+ 4)*ndof+(   0)] = -3.*Gamma*kT*rho0*mu_rho/cs2*lambda_pr;
  Xi[(   0)*ndof+(Q+16)] = 3.*Gamma*kT*rho0*mu_rho/cs2*lambda_pr;
  Xi[(Q+16)*ndof+(   0)] = 3.*Gamma*kT*rho0*mu_rho/cs2*lambda_pr;
  Xi[(   1)*ndof+(Q+ 1)] = 3.*kT*p_phi/(mu_phi*rho0)*lambda_rp;
  Xi[(Q+ 1)*ndof+(   1)] = 3.*kT*p_phi/(mu_phi*rho0)*lambda_rp;
  Xi[(   1)*ndof+(   8)] = -3.*kT*p_phi/(mu_phi*rho0)*lambda_rp;
  Xi[(   8)*ndof+(   1)] = -3.*kT*p_phi/(mu_phi*rho0)*lambda_rp;
  Xi[(   2)*ndof+(   5)] = -phi0*kT*lambda_pr;
  Xi[(   3)*ndof+(   6)] = -phi0*kT*lambda_pr;
  Xi[(   4)*ndof+(   7)] = -phi0*kT*lambda_pr;
  Xi[(   5)*ndof+(   2)] = -phi0*kT*lambda_pr;
  Xi[(   6)*ndof+(   3)] = -phi0*kT*lambda_pr;
  Xi[(   7)*ndof+(   4)] = -phi0*kT*lambda_pr;

  return Xi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<GpuComplex<Real>,ndof> kspace_white_noise(int kx, int ky, int kz, const Box& domain, RandomEngine const& engine) {
  GpuArray<GpuComplex<Real>,ndof> r = {};
  for (int i=ncons; i<ndof; ++i) {
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
  const Real phi0 = 0.0;

  // Cholesky decomposition of noise covariance matrix
  C = noise_covariance_matrix(rho0,phi0,k2);
  cholesky_decomp(C,ndof,ncons);

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

inline void compute_ifft(const Geometry& geom,
    const MultiFab& kspace_noise_real, /* in */
    const MultiFab& kspace_noise_imag, /* in */
    MultiFab& realspace_noise /* out */) {

  Box domain = geom.Domain();
  BoxArray ba(domain);
  DistributionMapping dm(ba);

  // MultiFabs for one of the ndof components
  MultiFab realspace_noise_one(ba, dm, 1, 0);
  MultiFab kspace_noise_real_one(ba, dm, 1, 0);
  MultiFab kspace_noise_imag_one(ba, dm, 1, 0);

  FFT::R2C fft(domain);
  auto const& [cba, cdm] = fft.getSpectralDataLayout();
  FabArray<BaseFab<GpuComplex<Real>>> kspace_noise_one(cba, cdm, 1, 0);

  for (int k=0; k<ndof; ++k) {

    // first copy the k-th conponent of the noise
    // may be faster to generate the noise here? [uschill 07/27/2022]
    kspace_noise_real_one.ParallelCopy(kspace_noise_real,k,0,1);
    kspace_noise_imag_one.ParallelCopy(kspace_noise_imag,k,0,1);

    // copy real and imaginary part into the spectral MultiFab
    for (MFIter mfi(kspace_noise_one); mfi.isValid(); ++mfi) {

      const Array4<Real>& xi_real = kspace_noise_real_one.array(mfi);
      const Array4<Real>& xi_imag = kspace_noise_imag_one.array(mfi);
      Array4<GpuComplex<Real>> xi = kspace_noise_one.array(mfi);

      const Box& box = mfi.fabbox();
      ParallelFor(box, [=] AMREX_GPU_DEVICE (int kx, int ky, int kz) noexcept {
        //xi(kx,ky,kz) = { xi_real(kx,ky,kz), xi_imag(kx,ky,kz) };
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

    check_kspace_symmetries(geom, kspace_noise_one);

    fft.backward(kspace_noise_one, realspace_noise_one);

    // number of sites for normalization of FFT
    long n_sites = domain.length(0)*domain.length(1)*domain.length(2);
    realspace_noise_one.mult(1./std::sqrt(n_sites));
    // TODO Check whether square root needed [uschill 2024/12/29]

    // copy the real-space noise into the k-th component of the MultiFab
    realspace_noise.ParallelCopy(realspace_noise_one,0,k,1);

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
  compute_ifft(geom, kspace_noise_real, kspace_noise_imag, noise);

}

// compute spatially uncorrelated noise vector from Gaussian random variables
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof> uncorrelated_noise(const Real rho0, const Real phi0, const RandomEngine& engine) {
  GpuArray<Real,ndof> r, xi;
  GpuArray<Real,ndof*ndof> C;

  // Cholesky decomposition of noise covariance matrix
  C = noise_covariance_matrix(rho0,phi0,0);
  cholesky_decomp(C,ndof,ncons);

  // random white noise
  for (int i=ncons; i<ndof; ++i) {
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
      const Real phi0 = h(x,y,z,1);
      GpuArray<Real,ndof> r = uncorrelated_noise(rho0,phi0,engine);
      for (int i=0; i<ndof; ++i) {
        xi(x,y,z,i) = r[i];
      }
    });
  }
}

#if 0
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void thermalize_moments(Array1D<Real,0,nvel>& mf,
			Array1D<Real,0,nvel>& mg,
			RandomEngine const& engine) {
  const Real rho = mf(0);
  const Real phi = mg(0);
  const Real gamma_r = 1. - 1./tau_r;
  const Real gamma_p = 1. - 1./tau_p;
  const Real phi_r = sqrt(rho*temperature/cs2*(1.-gamma_r*gamma_r));
  const Real phi_p = sqrt(phi*temperature/cs2*(1.-gamma_p*gamma_p));
  Array<Real,nvel> r = {}; // {} to value initialize r
  for (int i=4; i<nvel; ++i) {
    mf(i) += sqrt(b[i])*phi_r*RandomNormal(0., 1., engine);
    mg(i) += sqrt(b[i])*phi_p*RandomNormal(0., 1., engine);
  }
}
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void thermalize_moments(int x, int y, int z,
      Array1D<Real,0,nvel>& mf, Array1D<Real,0,nvel>& mg,
      const Array4<Real>& noise) {
  // diffusive moments 1,2,3 in noise at 5,6,7
  for (int i=1; i<=AMREX_SPACEDIM; ++i) {
    mg(i) += noise(x,y,z,i+1+AMREX_SPACEDIM);
  }
  // nonconserved moments 4..18 in noise at 8..22 (f) and 23..37 (g)
  for (int i=1+AMREX_SPACEDIM; i<nvel; ++i) {
    mf(i) += noise(x,y,z,i+1+AMREX_SPACEDIM);
    mg(i) += noise(x,y,z,i+nvel);
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z,
	     const Array4<Real>& f,
	     const Array4<Real>& g,
	     const Array4<Real>& h,
       const Array4<Real>& noise) {
  Array1D<Real,0,nvel> mf, mg;

  mf = moments(x,y,z,f);
  mg = moments(x,y,z,g);

  relax_moments(x,y,z,mf,mg,h);

  thermalize_moments(x,y,z,mf,mg,noise);

  populations(x,y,z,f,mf);
  populations(x,y,z,g,mg);

}

// collide and stream (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide_stream(int x, int y, int z,
		    const Array4<Real>& fOld,  const Array4<Real>& gOld,
		    const Array4<Real>& fNew,  const Array4<Real>& gNew,
		    const Array4<Real>& h,
        const Array4<Real>& noise,
		    RandomEngine const& rng) {
  collide(x, y, z, fOld, gOld, h, noise); // collide populations on current site
  stream_push(x, y, z, fOld, gOld, fNew, gNew); // push populations to neighbors
}

// apply fluctuating lattice Boltzmann equation for whole lattice
inline void collide_stream(const Geometry& geom,
			   MultiFab& fold, MultiFab& gold,
			   MultiFab& fnew, MultiFab& gnew,
			   MultiFab& hydrovs, MultiFab& noise) {
  const int halo = 1; // need to push from first halo layer into domain
  fold.FillBoundary(geom.periodicity());
  gold.FillBoundary(geom.periodicity());
  noise.FillBoundary(geom.periodicity());
  hydrovs.FillBoundary(geom.periodicity());
  for (MFIter mfi(fold); mfi.isValid(); ++mfi) {
    const Array4<Real>& fOld = fold.array(mfi);
    const Array4<Real>& gOld = gold.array(mfi);
    const Array4<Real>& fNew = fnew.array(mfi);
    const Array4<Real>& gNew = gnew.array(mfi);
    const Array4<Real>& h = hydrovs.array(mfi);
    const Array4<Real>& xi = noise.array(mfi);
    const Box& valid_box = mfi.growntilebox(halo);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      collide_stream(x, y, z, fOld, gOld, fNew, gNew, h, xi, engine);
    });
  }
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);
}

// LB timestep for whole lattice
inline void LBM_timestep(const Geometry& geom,
			 MultiFab& fold, MultiFab& gold,
			 MultiFab& fnew, MultiFab& gnew,
			 MultiFab& hydrovs, MultiFab& noise) {
#if UNCORRELATED_NOISE
    generate_fluctuations(geom, hydrovs, noise);
#else
    generate_correlated_fluctuations(geom, hydrovs, noise);
#endif
  collide_stream(geom, fold, gold, fnew, gnew, hydrovs, noise);
  hydrovars(fold, gold, hydrovs);
}

#endif
