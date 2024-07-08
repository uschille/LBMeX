#ifndef LBM_FLUCTUATIONS_H_
#define LBM_FLUCTUATIONS_H_

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif
#include <AMReX_GpuComplex.H>

#include "d3q19.H"
#include "lbm.H"


// hard coded
extern const Real density;
extern const Real tau;

// user modifiable
extern Real temperature;
extern Real kappa;
extern Real Beta;
extern Real rhov;
extern Real rhol;

// extern AMREX_GPU_MANAGED Real density;
// extern AMREX_GPU_MANAGED Real temperature;
// extern AMREX_GPU_MANAGED Real tau;

// // construct noise covariance matrix of the modes in k-space
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// GpuArray<Real,ncomp*ncomp> noise_covariance(int kx, int ky, int kz) {
//   const Real gamma = 1. - 1./tau;

//   // Noise covariance of the modes in k-space
//   GpuArray<Real,ncomp*ncomp> Xi = {};
//   Xi.fill(0);

// 	for (int i=ncons; i<ncomp; ++i) {
// 	  // noise covariance for ideal fluid
// 	  Xi[i*ncomp+i] = b[i]*temperature/cs2*(1.-gamma*gamma);
// 	}

//   return Xi;
// }

// construct noise covariance matrix of the modes in k-space
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ncomp*ncomp> noise_covariance(int kx, int ky, int kz) {
  // const Real gamma = 1. - 1./tau;
  const Real lambda_a = 1./tau;
  const Real lambda_ee = 1./tau;
  const Real kT = temperature;
  const Real system_cs2 = 2.*Beta*pow(rhol - rhov, 2);

  const Real k2 = kx*kx + ky*ky + kz*kz;
  const Real cs2_k = system_cs2 + rho0*kappa*k2;
  const Real S_k = density*kT/cs2;

  // Noise covariance of the modes in k-space
  GpuArray<Real,ncomp*ncomp> Xi = {};
  Xi.fill(0);

  // diagonal part
  Xi[ 4*ncomp +  4] = 2.0*S_k*lambda_a*(5. - 9.*cs2_k)*b[4];
  // Print() << "noise matrix at row " << 4 << " and column " << 4 << " :" << Xi[ 4*ncomp +  4] << "\n";
  Xi[ 5*ncomp +  5] = 8.0*S_k*lambda_a*b[5];
  // Print() << "noise matrix at row " << 5 << " and column " << 5 << " :" << Xi[ 5*ncomp +  5] << "\n";
  // Print() << "noise matrix: " << Xi[ 5*ncomp +  5] << "\n";
  Xi[ 6*ncomp +  6] = 8.0/3.0*S_k*lambda_a*b[6];
  // Print() << "noise matrix at row " << 6 << " and column " << 6 << " :" << Xi[ 6*ncomp +  6] << "\n";
  Xi[ 7*ncomp +  7] = 2.0/3.0*S_k*lambda_a*b[7];
  // Print() << "noise matrix at row " << 7 << " and column " << 7 << " :" << Xi[ 7*ncomp +  7] << "\n";
  Xi[ 8*ncomp +  8] = Xi[ 7*ncomp +  7];
  Xi[ 9*ncomp +  9] = Xi[ 7*ncomp +  7];
  Xi[10*ncomp + 10] = 4.0*S_k*lambda_a*b[10];
  // Print() << "noise matrix at row " << 10 << " and column " << 10 << " :" << Xi[ 10*ncomp +  10] << "\n";
  Xi[11*ncomp + 11] = Xi[10*ncomp + 10];
  Xi[12*ncomp + 12] = Xi[10*ncomp + 10];
  Xi[13*ncomp + 13] = 4.0/3.0*S_k*lambda_a*b[13];
  // Print() << "noise matrix at row " << 13 << " and column " << 13 << " :" << Xi[ 13*ncomp +  13] << "\n";
  Xi[14*ncomp + 14] = Xi[13*ncomp + 13];
  Xi[15*ncomp + 15] = Xi[13*ncomp + 13];
  Xi[16*ncomp + 16] = 18.0*S_k*lambda_a*(1. - cs2_k)*b[16];
  // Print() << "noise matrix at row " << 16 << " and column " << 16 << " :" << Xi[ 16*ncomp +  16] << "\n";
  Xi[17*ncomp + 17] = 8.0*S_k*lambda_a*b[17];
  Xi[18*ncomp + 18] = 8.0/3.0*S_k*lambda_a*b[18];

  // off diagonal part
  Xi[16*ncomp +  4] = 6.*rho0*kT*lambda_ee*(3.*cs2_k - 1);
  // Print() << "noise matrix at row " << 16 << " and column " << 4 << " :" << Xi[ 16*ncomp +  4] << "\n";
  Xi[4*ncomp +  16] = Xi[16*ncomp +  4];

  return Xi;
}

// Cholesky decomposition of matrix A
// result is stored in lower triangle of A
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cholesky_decomp(GpuArray<Real,ncomp*ncomp>& A) {
  const int n = ncomp;
  GpuArray<Real,n> p;
  Real sum;
  bool positive_definite = true;

  // Cholesky-Banachiewicz algorithm
  for (int i=ncons; i<n; ++i) {
    for (int j=ncons; j<=i; ++j) {
      sum = A[i*n+j];
      for (int k=j-1; k>=ncons; --k) {
	      sum -= A[i*n+k]*A[j*n+k];
      }
      if (i==j) {
	      if (sum>FLT_EPSILON) {
	        p[i] = std::sqrt(sum);
	      } else {
	        p[i] = 0.0;
	        positive_definite = false;
          //Print() << "Row " << i << " matrix not positive definite!\n";
	      }
      } else {
	      if (p[j]>FLT_EPSILON) {
	        A[i*n+j] = sum/p[j];
	      } else {
	        A[i*n+j] = 0.0;
	      }
      }
    }
  }
  for (int i=0; i<n; ++i) {
    A[i*n+i] = p[i];
    for (int j=i+1; j<n; ++j) {
      A[i*n+j] = 0.0;
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<GpuComplex<Real>,ncomp> kspace_white_noise(int kx, int ky, int kz, const Box& domain, RandomEngine const& engine) {
  GpuArray<GpuComplex<Real>,ncomp> r = {};
  for (int i=ncons+1; i<ncomp; ++i) {
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
GpuArray<GpuComplex<Real>,ncomp> correlated_noise(int kx, int ky, int kz,
						   const Box& domain,
						   const RandomEngine& engine) {
  GpuArray<GpuComplex<Real>,ncomp> r, xi;
  GpuArray<Real,ncomp*ncomp> C;

  // Cholesky decomposition of noise covariance matrix
  C = noise_covariance(kx,ky,kz);
  cholesky_decomp(C);

  // need to generate the correct symmetries here? [uschill 07/25/2022]
  // possibly the case if C(k) != C(-k) [uschill 07/27/2022]
  r = kspace_white_noise(kx,ky,kz,domain,engine);

  // compute correlated noise vector from Gaussian random variables
  for (int i=0; i<ncomp; ++i) {
    xi[i] = { 0, 0 };
    for (int j=0; j<=i; ++j) {
      xi[i] += C[i*ncomp+j]*r[j];
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

  // generate noise in whole box because of grid decomposition
  // (generating noise without grid decomposition may be faster)
  for (MFIter mfi(kspace_noise_real); mfi.isValid(); ++mfi) {
    const Box& box = mfi.validbox();
    const Array4<Real>& xi_real = kspace_noise_real.array(mfi);
    const Array4<Real>& xi_imag = kspace_noise_imag.array(mfi);
    // construct noise in k-space
    ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int kx, int ky, int kz, RandomEngine const& engine) {
      if (kx <= domain.length(0)/2) { // need only half of k-space for c2r FFT
        GpuArray<GpuComplex<Real>,ncomp> xi = {};

	      // compute correlated noise in k-space
	      xi = correlated_noise(kx,ky,kz,domain,engine);

        for (int i=0; i<ncomp; ++i) {
          xi_real(kx,ky,kz,i) = xi[i].real();
          xi_imag(kx,ky,kz,i) = xi[i].imag();
        }
      }
    });
  }

}

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

    for (int k=0; k<ncomp; ++k) {

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
  MultiFab kspace_noise_real(ba, dm, ncomp, 0);
  MultiFab kspace_noise_imag(ba, dm, ncomp, 0);

  kspace_noise_real.setVal(0.);
  kspace_noise_imag.setVal(0.);

  // generate noise in k-space
  generate_kspace_noise(geom, kspace_noise_real, kspace_noise_imag);

  // note that the k-space noise is generated without the required symmetries
  // (this is to allow for parallel generation of noise)
  // the k-space symmetries are handled when copying in compute_ifft

  // inverse Fourier transform noise vector to real space
  compute_ifft(geom, noise, kspace_noise_real, kspace_noise_imag);

}
#endif
