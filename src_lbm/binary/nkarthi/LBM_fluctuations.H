#ifndef LBM_FLUCTUATIONS_H_NK
#define LBM_FLUCTUATIONS_H_NK

#ifdef AMREX_USE_CUDA
#include <culapacke.h> // this does not work yet. GPU support not implemented
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#include <lapack.h>
#endif

// #include <lapacke.h>
#include <AMReX_GpuComplex.H>
#include <math.h>
#include <vector>
#include <AMReX_Array.H>
#include "nkarthi/LBM_d3q19.H"
#include "nkarthi/LBM_FFT.H"
#include "nkarthi/LBM_thermodynamics.H"
#include "nkarthi/LBM_IO.H"

namespace nkarthi {

const int ncons = 2 + AMREX_SPACEDIM;

#if 0
void conv_3d_full(MultiFab& in_mf, MultiFab& kernel_mf, MultiFab& out_mf, int NVAR){

  const int row = nx; const int col = ny; const int dep = nz;
  const int krow = nx; const int kcol = ny; const int kdep = nz;
  const int orow = row+krow-1; const int ocol = col+kcol-1; const int odep = dep+kdep-1;
  // compute_fft(geom, ref_params, fft_real, fft_imag, nvar, true); //output is correct. Normalization is 1/boxVol

  // auto const & in = in_mf.arrays();
  // auto const & kernel = kernel_mf.arrays();
  auto const & out = out_mf.arrays();

  Box box(IntVect{0,0,0}, IntVect{orow,ocol,odep});
  FArrayBox onegrid_data(box, NVAR); onegrid_data.setVal(0.); //temporary data container made in case output is different size than input
  Array4<Real> const& ofabbox = onegrid_data.array();

  // for(int comp = 0; comp < NVAR; comp++){
  //   ParallelFor(in_mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
  //       for(int m = 0; m < krow; m++){
  //         for(int n = 0; n < kcol; n++){
  //           for(int l = 0; l < kdep; l++){
  //             const int ri = i - m; const int rj = j - n; const int rk = k - l;
  //               if (ri >= 0 && ri < row && rj >= 0 && rj < col && rk >= 0 && rk < dep){
  //                 ofabbox(i,j,k,comp) += ofabbox(i,j,k,comp) + in[nbx](rj,rj,rk,comp)*kernel[nbx](m,n,l,comp);
  //                 }
  //           }
  //         }
  //       }
  //   });

  //   ParallelFor(out_mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
  //         out[nbx](i,j,k,comp) = ofabbox(i,j,k,comp);
  //   });

  // }

  for(int comp = 0; comp < NVAR; comp++){
    for (MFIter mfi(in_mf); mfi.isValid(); ++mfi){
      const Array4<Real>& in = in_mf.array(mfi);
      const Array4<Real>& kernel = kernel_mf.array(mfi);
      for(int i = 0; i < orow; i++){
        for(int j = 0; j < ocol; j++){
          for(int k = 0; k < odep; k++){
            for(int m = 0; m < krow; m++){
              for(int n = 0; n < kcol; n++){
                for(int l = 0; l < kdep; l++){
                  const int ri = i - m; const int rj = j - n; const int rk = k - l;
                  if (ri >= 0 && ri < row && rj >= 0 && rj < col && rk >= 0 && rk < dep){
                    ofabbox(i,j,k,comp) += in(rj,rj,rk,comp)*kernel(m,n,l,comp);
                  }
                }
              }
            }
          }
        }
      }
    }
    ParallelFor(out_mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
          out[nbx](i,j,k,comp) = ofabbox(i,j,k,comp);
    });
  }
}

#if 0
void conv_3d_same(MultiFab& in_mf, MultiFab& kernel_mf, MultiFab& out_mf, int NVAR){

  //only works if its periodic, 1 grid(1 core) and kernel is the same size as the input

  const int row = nx; const int col = ny; const int dep = nz;
  const int krow = nx; const int kcol = ny; const int kdep = nz; // needs to change to indicate dimensions of response kernel
  const int orow = row; const int ocol = col; const int odep = dep; //same size as input
  const int kcenterx = krow/2; const int kcentery = kcol/2; const int kcenterz = kdep/2; // maybe subtracting by 1 may be better for all values when dimensions are even

  // wraparound may be the issue
  for (MFIter mfi(in_mf); mfi.isValid(); ++mfi){
    const Array4<Real>& in = in_mf.array(mfi);
    const Array4<Real>& kernel = kernel_mf.array(mfi);
    const Array4<Real>& out = out_mf.array(mfi);

    const Box& bx = mfi.tilebox();
    ParallelFor(bx, NVAR, [=] AMREX_GPU_DEVICE(int i, int j, int k, int comp) {
        for(int l = 0; l < krow; l++){
          for(int m = 0; m < kcol; m++){
            for(int n = 0; n < kdep; n++){
              const int ri = (i + (l - kcenterx) + nx)%nx;
              const int ri = (i + (l - kcenterx) + ny)%ny;
              const int ri = (i + (l - kcenterx) + nx)%nz;
              out(i,j,k,comp) += in(rj,rj,rk,comp)*kernel(m,n,l,comp);
            }
          }
        }
    });
  }

  // auto const & in = in_mf.arrays();
  // auto const & kernel = kernel_mf.arrays();
  // auto const & out = out_mf.arrays();

  // for(int comp = 0; comp < NVAR; comp++){
  //   ParallelFor(in_mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
  //       for(int m = 0; m < krow; m++){
  //         for(int n = 0; n < kcol; n++){
  //           for(int l = 0; l < kdep; l++){
  //             const int ri = i + (m - kcenterx); const int rj = j + (n - kcentery); const int rk = k + (l - kcenterz);
  //               if (ri >= 0 && ri < row && rj >= 0 && rj < col && rk >= 0 && rk < dep){
  //                 out[nbx](i,j,k,comp) += in[nbx](rj,rj,rk,comp)*kernel[nbx](m,n,l,comp);
  //               }
  //           }
  //         }
  //       }
  //   });
  // }



  // for(int comp = 0; comp < NVAR; comp++){
  //   for (MFIter mfi(in_mf); mfi.isValid(); ++mfi){
  //     const Array4<Real>& in = in_mf.array(mfi);
  //     const Array4<Real>& kernel = kernel_mf.array(mfi);
  //     const Array4<Real>& out = out_mf.array(mfi);
  //     for(int i = 0; i < orow; i++){
  //       for(int j = 0; j < ocol; j++){
  //         for(int k = 0; k < odep; k++){
  //           for(int m = 0; m < krow; m++){
  //             for(int n = 0; n < kcol; n++){
  //               for(int l = 0; l < kdep; l++){
  //                 const int ri = i + (m - kcenterx); const int rj = j + (n - kcentery); const int rk = k + (l - kcenterz);
  //                 const ri = ri%orow; //ensures that the index doesn't go out of bounds 
  //                 // output should match the input mf size
  //                 // allows for arbitrary kernel dimension up to and including up to input size
  //                 if (ri >= 0 && ri < row && rj >= 0 && rj < col && rk >= 0 && rk < dep){
  //                   out(i,j,k,comp) += in(ri,rj,rk,comp)*kernel(m,n,l,comp);
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void conv_3d_same(MultiFab& in_mf, MultiFab& kernel_mf, MultiFab& out_mf, int NVAR){
  const int row = 32; const int col = 32; const int dep = 32;
  const int krow = 32; const int kcol = 32; const int kdep = 32;
  const int orow = 32; const int ocol = 32; const int odep = 32;
  const int kcenterx = krow/2; const int kcentery = kcol/2; const int kcenterz = kdep/2;

  Array3D<Real, 0, row-1, 0, col-1, 0, dep-1, Order::C> sig;
  Array3D<Real, 0, krow-1, 0, kcol-1, 0, kdep-1, Order::C> ker;
  Array3D<Real, 0, orow-1, 0, ocol-1, 0, odep-1, Order::C> out;

  // auto const & in = in_mf.arrays();
  // auto const & kernel = kernel_mf.arrays();
  auto const & output = out_mf.arrays();

  for (MFIter mfi(in_mf); mfi.isValid(); ++mfi){
    const Array4<Real>& in = in_mf.array(mfi);
    const Array4<Real>& kernel = kernel_mf.array(mfi);
    for(int i = 0; i < row; i++){
      for(int j = 0; j < col; j++){
        for(int k = 0; k < dep; k++){
          sig(i,j,k) = in(i,j,k,NVAR);
          ker(i,j,k) = kernel(i,j,k,NVAR);
        }
      }
    }
  }


  // ParallelFor(in_mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
  //   sig(i,j,k) = in[nbx](i,j,k,NVAR);
  //   ker(i,j,k) = kernel[nbx](i,j,k,NVAR);
  // });

  // auto const & out = out_mf.arrays();
  for(int i = 0; i < orow; i++){
    for(int j = 0; j < ocol; j++){
      for(int k = 0; k < odep; k++){
        for(int m = 0; m < krow; m++){
          for(int n = 0; n < kcol; n++){
            for(int l = 0; l < kdep; l++){
              const int ri = i + (m - kcenterx); const int rj = j + (n - kcentery); const int rk = k + (l - kcenterz);
              if (ri >= 0 && ri < row && rj >= 0 && rj < col && rk >= 0 && rk < dep){
                out(i,j,k) += sig(rj,rj,rk)*ker(m,n,l);
              }
            }
          }
        }
      }
    }
  }

  ParallelFor(out_mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
    output[nbx](i,j,k,NVAR) = out(i,j,k);
  });
}
#endif

void conv_3d_same(MultiFab& in_mf, MultiFab& kernel_mf, 
                  MultiFab& out_mf, int NVAR){

   const int row = nx; const int col = ny; const int dep = nz;
   const int krow = nx; const int kcol = ny; const int kdep = nz;  //this line needs to change to indicate dimensions of response kernel
   const int orow = nx; const int ocol = ny; const int odep = nz; // forperiodic, orow must be the same as row
   const int kcenterx = krow/2; const int kcentery = kcol/2; const int kcenterz = kdep/2; // might consider krow/2-1, etc. if krow, kcol, and kdep are even

   out_mf.setVal(0.);

   for (MFIter mfi(in_mf); mfi.isValid(); ++mfi){
      const Array4<Real>& in = in_mf.array(mfi);
      const Array4<Real>& kernel = kernel_mf.array(mfi);
      const Array4<Real>& out = out_mf.array(mfi);

      const Box& bx = mfi.tilebox();

      ParallelFor(bx, NVAR, [=] AMREX_GPU_DEVICE(int i, int j, int k, int comp) { 
        for(int l = 0; l < krow; l++){
          for(int m = 0; m < kcol; m++){
            for(int n = 0; n < kdep; n++){
              const int ri = (i + (l - kcenterx) + nx)%nx;
              const int rj = (j + (m - kcentery) + ny)%ny;
              const int rk = (k + (n - kcenterz) + nz)%nz;
              out(i,j,k,comp) += in(ri,rj,rk,comp)*kernel(m,n,l,comp);
            }
          } 
        }
      });
    }
}

void convolve_3D(MultiFab& in_mf, MultiFab& kernel_mf, MultiFab& out_mf, 
                 std::string mode, int NVAR){

    if(mode == "same"){Print() << "Same mode convolution\n";conv_3d_same(in_mf, kernel_mf, out_mf, NVAR);}
    else if(mode == "full"){Print() << "Full mode convolution\n";conv_3d_full(in_mf, kernel_mf, out_mf, NVAR);}
    else{throw std::invalid_argument( "convolution mode is invalid" );exit(-1);}
}
#endif

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
GpuArray<GpuComplex<Real>,ndof> kspace_white_noise(int kx, int ky, int kz, const Box& domain, RandomEngine const& engine) {
  BL_PROFILE_VAR("kspace_white_noise()",kspace_white_noise);
  GpuArray<GpuComplex<Real>,ndof> r = {};
  for (int i=ncons; i<ndof; ++i) {
    // symmetry points are purely real
    if (     ((kx == 0) || (kx == domain.length(0) - kx))
          && ((ky == 0) || (ky == domain.length(1) - ky))
          && ((kz == 0) || (kz == domain.length(2) - kz)) ) {
      // real Gaussian random variables with zero mean and variance 1
      r[i] = { RandomNormal(0., 1., engine), RandomNormal(0., 0., engine) };
      r[i] = { 0.3456, 0.0 };
    } else {
      // complex Gaussian random variables with zero mean and variance 0.5
      r[i] = { RandomNormal(0., std::sqrt(0.5), engine), RandomNormal(0., std::sqrt(0.5), engine) };
      r[i] = { 0.1234, 0.2345 };
    }
  }
  return r;
}

// compute correlated noise vector from Gaussian random variables
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<GpuComplex<Real>,ndof> correlated_noise(int kx, int ky, int kz,
						   const Box& domain,
						   const RandomEngine& engine,
               const Array4<Real>& hydrovs_ref) {
  BL_PROFILE_VAR("correlated_noise()",correlated_noise);
  GpuArray<GpuComplex<Real>,ndof> r, xi;
  GpuArray<Real,ndof*ndof> C;
  const int Q = nvel;
  // Real k2;
  // if (use_correlated_noise){k2 = fourier_laplace_operator(kx, ky, kz, domain);}
  // else{k2 = 0.;}
  const Real k2 = fourier_laplace_operator(kx, ky, kz, domain);
  // // Cholesky decomposition of noise covariance matrix
  // const Real rho0 = hydrovs_ref(kx, ky, kz, 0);
  const Real rho0 = 1.;
  const Real phi0 = hydrovs_ref(kx, ky, kz, 1);
  
  C = noise_covariance(rho0, phi0, k2);
  #ifdef AMREX_USE_CUDA
    cholesky_decomp(C,ndof,ncons);
  #else
    // // LAPACK CHOLESKY DECOMP. COMMENT OUT IF THIS DOESN'T WORK
    const char uplo = 'L';
    const int lda = ndof-ncons;
    const int order = lda;
    int info = 0;
    GpuArray<Real,lda*lda> SLC_C; SLC_C.fill(0.);
    for (int i = ncons; i < ndof; i++){
      for (int j = ncons; j < ndof; j++){
        SLC_C[(i-ncons)*lda+(j-ncons)] = C[i*ndof+j];
      }
    }
    size_t arraySize = SLC_C.size();
    LAPACK_dpotrf(&uplo, &order, SLC_C.data(), &lda, &info);
    C.fill(0.);
    for (int i = ncons; i < ndof; i++){
      for (int j = ncons; j < i+1; j++){
        C[i*ndof+j] = SLC_C[(j-ncons)*lda+(i-ncons)];
      }
    }
    // // LAPACK CHOLESKY DECOMP. COMMENT OUT IF THIS DOESN'T WORK
  #endif

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
				  MultiFab& kspace_noise_imag,
          MultiFab& hydrovs) {
  BL_PROFILE_VAR("generate_kspace_noise()",generate_kspace_noise);
  
  const Box domain = geom.Domain();
  const int nvar = 2; const int nghost = 0; //const Real system_volume = domain.length(0)*domain.length(1)*domain.length(2);
  BoxArray ba = kspace_noise_real.boxArray();
  DistributionMapping dm = kspace_noise_real.DistributionMap();
  // Box domain2(IntVect{0,0,0}, IntVect{domain.length(0), domain.length(1), domain.length(2)});
  // Print() << "x:" << domain.length(0) << " y:" << domain.length(1) << " z:" << domain.length(2) << "\n";

  #if 0
  // FFT //
  MultiFab ref_params(ba, dm, nvar, nghost); ref_params.ParallelCopy(hydrovs, 0, 0, 2);
  MultiFab fft_real(ba, dm, nvar, nghost);
  MultiFab fft_imag(ba, dm, nvar, nghost);  
  compute_fft(geom, ref_params, fft_real, fft_imag, nvar, true); //output is correct. Normalization is 1/boxVol
  WriteOutput(0, fft_real, geom, "FFT_plt_real", nvar, 0);
  WriteOutput(0, fft_imag, geom, "FFT_plt_imag", nvar, 0);
  // FFT //
  
  // convolution //
  std::string mode = "same";
  // std::string mode = "full";
  MultiFab kernel(ba, dm, nvar, nghost); kernel.setVal(1.);
  MultiFab kref(ba, dm, nvar, nghost); kref.setVal(0.);
  MultiFab kref_imag(ba, dm, nvar, nghost); kref_imag.setVal(0.);
  MultiFab conv_real(ba, dm, nvar, nghost); conv_real.setVal(0.);
  MultiFab conv_imag(ba, dm, nvar, nghost); conv_imag.setVal(0.);
  convolve_3D(fft_real, kernel, conv_real, mode, nvar);
  convolve_3D(fft_imag, kernel, conv_imag, mode, nvar);

  // convolve_3D(fft_real, kernel, conv_real, mode, 0);
  // convolve_3D(fft_imag, kernel, conv_imag, mode, 0);
  // convolve_3D(fft_real, kernel, conv_real, mode, 1);
  // convolve_3D(fft_imag, kernel, conv_imag, mode, 1);

  WriteOutput(0, conv_real, geom, "conv_plt_real_", nvar, 0);
  WriteOutput(0, conv_imag, geom, "conv_plt_imag_", nvar, 0);
  // convolution //

  // conjugate multiplication //
  MultiFab::Copy(kref, conv_real, 0, 0, nvar, nghost); //copy works fine
  MultiFab::Copy(kref_imag, conv_imag, 0, 0, nvar, nghost); //copy works fine
  MultiFab::Multiply(kref, conv_real, 0, 0, nvar, nghost); //squaring. Checked to work
  MultiFab::Multiply(kref_imag, conv_imag, 0, 0, nvar, nghost); // squaring. Checked to work
  MultiFab::Subtract(kref, kref_imag, 0, 0, nvar, nghost); // multiplying convolution by its conjugate for reference state which acts as (a + b)*(a - b). Works
  // conjugate multiplication //
  #endif

  // generate noise in whole box because of grid decomposition
  // (generating noise without grid decomposition may be faster)
  for (MFIter mfi(kspace_noise_real); mfi.isValid(); ++mfi) {
    const Box& box = mfi.validbox();
    const Array4<Real>& xi_real = kspace_noise_real.array(mfi);
    const Array4<Real>& xi_imag = kspace_noise_imag.array(mfi);
    const Array4<Real>& r = hydrovs.array(mfi);
    // const Array4<Real>& kr = kref.array(mfi);

    // construct noise in k-space
    ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int kx, int ky, int kz, RandomEngine const& engine) {
      if (kx <= domain.length(0)/2) { // need only half of k-space for c2r FFT
        GpuArray<GpuComplex<Real>,ndof> xi = {};
	      // compute correlated noise in k-space
	      xi = correlated_noise(kx,ky,kz,domain,engine,r);
        // if (ky == 0 && kz == 0){Print() << "kx:" << kx << " rhok:" << kr(kx, ky, kz, 0) << " phik:" << kr(kx, ky, kz, 1) << "\n";}

        for (int i=0; i<ndof; ++i) {
          xi_real(kx,ky,kz,i) = xi[i].real();
          xi_imag(kx,ky,kz,i) = xi[i].imag();
        }
      }
    });
  }

}

// compute spatially uncorrelated noise vector from Gaussian random variables
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof> uncorrelated_noise(const Real rho0, const Real phi0, const RandomEngine& engine) {
  BL_PROFILE_VAR("uncorrelated_noise()",uncorrelated_noise);
  GpuArray<Real,ndof> r, xi; 
  GpuArray<Real,ndof*ndof> C;
  r.fill(0.); xi.fill(0.); C.fill(0.);

  // Cholesky decomposition of noise covariance matrix
  C = noise_covariance(rho0,phi0,0);
  #ifdef AMREX_USE_CUDA
    cholesky_decomp(C,ndof,ncons);
  #else
    // // LAPACK CHOLESKY DECOMP. COMMENT OUT IF THIS DOESN'T WORK
    const char uplo = 'L';
    const int lda = ndof-ncons;
    const int order = lda;
    int info = 0;
    GpuArray<Real,lda*lda> SLC_C; SLC_C.fill(0.);
    for (int i = ncons; i < ndof; i++){
      for (int j = ncons; j < ndof; j++){
        SLC_C[(i-ncons)*lda+(j-ncons)] = C[i*ndof+j];
      }
    }
    size_t arraySize = SLC_C.size();
    LAPACK_dpotrf(&uplo, &order, SLC_C.data(), &lda, &info);
    C.fill(0.);
    for (int i = ncons; i < ndof; i++){
      for (int j = ncons; j < i+1; j++){
        C[i*ndof+j] = SLC_C[(j-ncons)*lda+(i-ncons)]; //row and columns are inverted as output of dpotrf is in fortran column major not row major like C
      }
    }
    // // LAPACK CHOLESKY DECOMP. COMMENT OUT IF THIS DOESN'T WORK
  #endif

  // random white noise
  for (int i=ncons; i<ndof; ++i) {
    r[i] = RandomNormal(0., 1., engine);
    r[i] = 0.4567;
  }

  // compute noise vector from Gaussian random variables
  for (int i=0; i<ndof; ++i) {
    // xi[i] = 0;
    for (int j=0; j<=i; ++j) {
      xi[i] += C[i*ndof+j]*r[j];
    }
  }
  return xi;
}

// generate real-space noise for all non-conserved moments
inline void generate_realspace_noise(const Geometry& geom, MultiFab& hydrovs, MultiFab& noise) {
  BL_PROFILE_VAR("generate_realspace_noise()",generate_realspace_noise);
  for (MFIter mfi(noise); mfi.isValid(); ++mfi) {
    const Box& box = mfi.validbox();
    const Array4<Real>& xi = noise.array(mfi);
    const Array4<Real>& h = hydrovs.array(mfi);
    ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      // const Real rho0 = h(x,y,z,0);
      const Real rho0 = 1.;
      const Real phi0 = h(x,y,z,1);
      GpuArray<Real,ndof> r = uncorrelated_noise(rho0,phi0,engine);
      for (int i=0; i<ndof; ++i) {
        xi(x,y,z,i) = r[i];
      }
    });
  }
}

// LB thermalization procedure for spatially correlated, non-diagonal noise
inline void generate_fluctuations(const Geometry& geom,
				  MultiFab& hydrovs,
				  MultiFab& noise,
          const int use_correlated_noise) {
  BL_PROFILE_VAR("generate_fluctuations()",generate_fluctuations);

  // BoxArray ba = noise.boxArray();
  // DistributionMapping dm = noise.DistributionMap();
  // MultiFab kspace_noise_real(ba, dm, ndof, 0);
  // MultiFab kspace_noise_imag(ba, dm, ndof, 0);

  // kspace_noise_real.setVal(0.);
  // kspace_noise_imag.setVal(0.);
  // // generate noise in k-space
  // // Print() << "Correlated noise\n";
  // generate_kspace_noise(geom, kspace_noise_real, kspace_noise_imag, hydrovs);
  // // note that the k-space noise is generated without the required symmetries
  // // (this is to allow for parallel generation of noise)
  // // the k-space symmetries are handled when copying in compute_ifft

  // // inverse Fourier transform noise vector to real space
  // compute_ifft(geom, noise, kspace_noise_real, kspace_noise_imag, ndof);

  if (use_correlated_noise){
    BoxArray ba = noise.boxArray();
    DistributionMapping dm = noise.DistributionMap();
    MultiFab kspace_noise_real(ba, dm, ndof, 0); kspace_noise_real.setVal(0.);
    MultiFab kspace_noise_imag(ba, dm, ndof, 0); kspace_noise_imag.setVal(0.);
    
    // generate noise in k-space
    // Print() << "Correlated noise\n";
    // generate_kspace_noise(geom, kspace_noise_real, kspace_noise_imag, ref_params);
    generate_kspace_noise(geom, kspace_noise_real, kspace_noise_imag, hydrovs);
    // note that the k-space noise is generated without the required symmetries
    // (this is to allow for parallel generation of noise)
    // the k-space symmetries are handled when copying in compute_ifft

    // inverse Fourier transform noise vector to real space
    compute_ifft(geom, noise, kspace_noise_real, kspace_noise_imag, ndof);
  }
  else{
    // Print() << "Uncorrelated noise\n";
    // generate_realspace_noise(geom, ref_params, noise);
    generate_realspace_noise(geom, hydrovs, noise);
  }

}
}
#endif