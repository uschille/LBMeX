#ifndef AMREX_DFT_
#define AMREX_DFT_

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif

#include "AMReX_FileIO.H"
#include <AMReX_FabArray.H>

using FFTplan = fftw_plan;
using FFTcomplex = fftw_complex;

// Calculate DFT for component [dftcomp] of multifab [field] in 3D dimension;
// [field_dft_complex] contains real (comp=0) and imaginary (comp=1) coefficients for [dftcomp]'th component of input field [field]
AMREX_GPU_DEVICE
void amrex_fftw_r2c_3d(MultiFab& field_dft_complex, const MultiFab& field, int dftcomp=0, bool if_normalize=false){
    int ncomp = field.nComp();
    int ngrow = field.nGrow();
    DistributionMapping dm = field.DistributionMap();
    BoxArray ba = field.boxArray();
    Box domain = ba.minimalBox();
    IntVect domainSize = domain.length();
    if(dftcomp>ncomp-1){
        throw runtime_error("DFT component index out of range! (called from function 'amrex_fftw_r2c_3d')\n");
    }

    // create a new BoxArray and DistributionMapping for a MultiFab with 1 box only
    BoxArray ba_onegrid(domain);
    DistributionMapping dm_onegrid(ba_onegrid);

    // temp storage for field and the dft with 1 box
    MultiFab field_onegrid         (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab field_dft_real_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    MultiFab field_dft_imag_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    field_dft_complex.define(ba, dm, 2, ngrow);

    // copy field's dftcomp into field_onegrid (total 1 component)
    field_onegrid.ParallelCopy(field, dftcomp, 0, 1);

    // number of points in the domain
    long npts = domain.numPts();
    Real sqrtnpts = std::sqrt(npts);

    // contain to store FFT - note it is shrunk by approximately a half in x
    // FArrayBox <--> BaseFab<Real>
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;

    Vector<FFTplan> forward_plan;

    for (MFIter mfi(field_onegrid); mfi.isValid(); ++mfi) {
      // grab a single box
      Box realspace_bx = mfi.fabbox();  // Return the Box of the FAB at which we currently point.
      // size of box
      IntVect fft_size = realspace_bx.length(); // This will be different for FFTs of complex data
      // this is the size of the box, except the 0th component is 'halved plus 1'
      /* *****  In AMReX, the data arrangement obeys Fortran like column principal, 
      so here is 0th component rather than the original last component in FFTW;  */
      IntVect spectral_bx_size = fft_size;
      spectral_bx_size[0] = fft_size[0]/2 + 1;

      // spectral box
      Box spectral_bx = Box(IntVect(0), spectral_bx_size - IntVect(1));
      spectral_field.emplace_back(new BaseFab<GpuComplex<Real> >(spectral_bx,1, // Make BaseFab with desired domain (box) and number of components.
                                 The_Device_Arena()));
      spectral_field.back()->setVal<RunOn::Device>(-1.0); // touch the memory

      FFTplan fplan = fftw_plan_dft_r2c_3d(fft_size[2], fft_size[1], fft_size[0],
                   field_onegrid[mfi].dataPtr(),
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   FFTW_ESTIMATE);

      forward_plan.push_back(fplan);
    }

    ParallelDescriptor::Barrier();

    for (MFIter mfi(field_onegrid); mfi.isValid(); ++mfi) {
      int i = mfi.LocalIndex();
      fftw_execute(forward_plan[i]);
    }

    for (MFIter mfi(field_dft_real_onegrid); mfi.isValid(); ++mfi) {
      Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();

      Array4<Real> const& realpart = field_dft_real_onegrid.array(mfi);
      Array4<Real> const& imagpart = field_dft_imag_onegrid.array(mfi);

      Box bx = mfi.fabbox();
      /*
       Unpacking rules:
        For domains from (0,0,0) to (Nx-1,Ny-1,Nz-1)
        For any cells with i index > Nx/2, these values are complex conjugates of the corresponding
        entry where (Nx-i,Ny-j,Nz-k) UNLESS that index is zero, in which case you use 0.
        e.g. for an 8^3 domain, any cell with i index
        Cell (6,2,3) is complex conjugate of (2,6,5)
        Cell (4,1,0) is complex conjugate of (4,7,0)  (note that the FFT is computed for 0 <= i <= Nx/2)
      */
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        if (i <= bx.length(0)/2) {
              // copy value
              realpart(i,j,k) = spectral(i,j,k).real();
              imagpart(i,j,k) = spectral(i,j,k).imag();
        } else {
              // copy complex conjugate
              int iloc = bx.length(0)-i;
              int jloc = (j == 0) ? 0 : bx.length(1)-j;
              int kloc = (k == 0) ? 0 : bx.length(2)-k;

              realpart(i,j,k) =  spectral(iloc,jloc,kloc).real();
              imagpart(i,j,k) = -spectral(iloc,jloc,kloc).imag();
        }
        if(if_normalize){
          realpart(i,j,k) /= sqrtnpts;
          imagpart(i,j,k) /= sqrtnpts;
        }
      });
    }

    // destroy fft plan
    for (int i = 0; i < forward_plan.size(); ++i) {
        fftw_destroy_plan(forward_plan[i]);
    }

    // copy dft coefs back to output multifab [field_dft_complex] with same ba,dm as input [field];
    field_dft_complex.ParallelCopy(field_dft_real_onegrid, 0, 0, 1);
    field_dft_complex.ParallelCopy(field_dft_imag_onegrid, 0, 1, 1);
}

/*  (same for each direction)
f = [n/2, ..., n-1, 0, 1, ...,   n/2-1] / (d*n)   if n is even
f = [(n+1)/2, ..., n-1, 0, 1, ..., (n-1)/2] / (d*n)   if n is odd
*/
void amrex_shift_fft_3d(MultiFab& field_dft_coef, const int zero_avg=1){
  const BoxArray ba = field_dft_coef.boxArray();
  const int ncomp = field_dft_coef.nComp();
  Box box = ba.minimalBox();

  BoxArray ba_onegrid;
  // Initialize the boxarray "ba" from the single box "bx"
  ba_onegrid.define(box);
  DistributionMapping dmap_onegrid(ba_onegrid);
  
  MultiFab field_dft_coef_onegrid(ba_onegrid, dmap_onegrid, 1, 0);
  MultiFab field_dft_coef_onegrid_temp(ba_onegrid, dmap_onegrid, 1, 0);

  for(int n=0; n<ncomp; n++){
    field_dft_coef_onegrid_temp.ParallelCopy(field_dft_coef, n, 0, 1); // src, src_comp, dest_comp, num_comp,
    if(zero_avg==1){
      //  remove zero-freq components;
      for (MFIter mfi(field_dft_coef_onegrid_temp); mfi.isValid(); ++mfi) {
	      const Box& bx = mfi.tilebox();
	      const Array4<Real>& dft_temp = field_dft_coef_onegrid_temp.array(mfi);
	      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
  		    if (i == 0 && j == 0 && k == 0) {
		        dft_temp(i,j,k) = 0.;
		      }
        });
	    }
    }
    for (MFIter mfi(field_dft_coef_onegrid); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      const Array4<Real>& dft = field_dft_coef_onegrid.array(mfi);
      const Array4<Real>& dft_temp = field_dft_coef_onegrid_temp.array(mfi);
      int nx = bx.length(0);  int nxh = nx/2;
      int ny = bx.length(1);  int nyh = ny/2;
      int nz = bx.length(2);  int nzh = nz/2;

      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        int ip = (i+nxh)%nx;
        int jp = (j+nyh)%ny;
        int kp = (k+nzh)%nz;
        dft(ip,jp,kp) = dft_temp(i,j,k);
      });
    }
    field_dft_coef.ParallelCopy(field_dft_coef_onegrid, 0, n, 1);
  }
}

#endif