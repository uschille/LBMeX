#ifndef LBM_FFT_H_NK
#define LBM_FFT_H_NK

#ifdef AMREX_USE_CUDA
    #include <cufft.h>
#else
    #include <fftw3.h>
    #include <fftw3-mpi.h>
#endif

#include <AMReX_GpuComplex.H>

namespace nkarthi {

void compute_fft(const Geometry& geom, 
          const MultiFab& variables,
			    MultiFab& variables_dft_real, 
			    MultiFab& variables_dft_imag,
			    int NVAR,
          bool unpack){

    BL_PROFILE_VAR("compute_fft()", compute_fft);

    #ifdef AMREX_USE_CUDA
        // Print() << "Using cuFFT\n";
    #elif AMREX_USE_HIP
        // Print() << "Using rocFFT\n";
    #else
        // Print() << "Using FFTW\n";
    #endif

    bool is_flattened = false;

    long npts;

    // Initialize the boxarray "ba_onegrid" from the single box "domain"
    BoxArray ba_onegrid;
    {
      Box domain = geom.Domain();
      ba_onegrid.define(domain);
      npts = (domain.length(0)*domain.length(1)*domain.length(2));
    }

    Real sqrtnpts = std::sqrt(npts);

    DistributionMapping dmap_onegrid(ba_onegrid);

    // we will take one FFT at a time and copy the answer into the
    // corresponding component
    MultiFab variables_onegrid;
    MultiFab variables_dft_real_onegrid;
    MultiFab variables_dft_imag_onegrid;
    variables_onegrid.define(ba_onegrid, dmap_onegrid, 1, 0);
    variables_dft_real_onegrid.define(ba_onegrid, dmap_onegrid, 1, 0);
    variables_dft_imag_onegrid.define(ba_onegrid, dmap_onegrid, 1, 0);

    //    fftw_mpi_init();

    #ifdef AMREX_USE_CUDA
        using FFTplan = cufftHandle;
        using FFTcomplex = cuDoubleComplex;
    #elif AMREX_USE_HIP
        using FFTplan = rocfft_plan;
        using FFTcomplex = double2;
    #else
        using FFTplan = fftw_plan;
        using FFTcomplex = fftw_complex;
    #endif

    // contain to store FFT - note it is shrunk by "half" in x
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;

    Vector<FFTplan> forward_plan;

    // for CUDA builds we only need to build the plan once; track whether we did
    bool built_plan = false;
    
    for (int comp=0; comp<NVAR; comp++) {
        variables_onegrid.ParallelCopy(variables,comp,0,1);

        if (!built_plan) {

            for (MFIter mfi(variables_onegrid); mfi.isValid(); ++mfi) {

                // grab a single box including ghost cell range
                Box realspace_bx = mfi.fabbox();

                // size of box including ghost cell range
                IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

                // this is the size of the box, except the 0th component is 'halved plus 1'
                IntVect spectral_bx_size = fft_size;
                spectral_bx_size[0] = fft_size[0]/2 + 1;

                // spectral box
                Box spectral_bx = Box(IntVect(0), spectral_bx_size - IntVect(1));

                spectral_field.emplace_back(new BaseFab<GpuComplex<Real> >(spectral_bx,1,
                                                                       The_Device_Arena()));
                spectral_field.back()->setVal<RunOn::Device>(0.0); // touch the memory

                FFTplan fplan;

            #ifdef AMREX_USE_CUDA // CUDA
                if (is_flattened) {
                    cufftResult result = cufftPlan2d(&fplan, fft_size[1], fft_size[0], CUFFT_D2Z);
                    if (result != CUFFT_SUCCESS) {
                        amrex::AllPrint() << " cufftplan2d forward failed! Error: "
                                          << cufftErrorToString(result) << "\n";
                    }
                } 
                else {
                    cufftResult result = cufftPlan3d(&fplan, fft_size[2], fft_size[1], fft_size[0], CUFFT_D2Z);
                    if (result != CUFFT_SUCCESS) {
                        amrex::AllPrint() << " cufftplan3d forward failed! Error: "
                                          << cufftErrorToString(result) << "\n";
                    }
                }
            #elif AMREX_USE_HIP // HIP
                if (is_flattened) {
                    const std::size_t lengths[] = {std::size_t(fft_size[0]),std::size_t(fft_size[1])};
                    rocfft_status result = rocfft_plan_create(&fplan, rocfft_placement_notinplace, 
                                                              rocfft_transform_type_real_forward, rocfft_precision_double,
                                                              2, lengths, 1, nullptr);
                    assert_rocfft_status("rocfft_plan_create", result);
                } 
                else {
                    const std::size_t lengths[] = {std::size_t(fft_size[0]),std::size_t(fft_size[1]),std::size_t(fft_size[2])};
                    rocfft_status result = rocfft_plan_create(&fplan, rocfft_placement_notinplace, 
                                                              rocfft_transform_type_real_forward, rocfft_precision_double,
                                                              3, lengths, 1, nullptr);
                    assert_rocfft_status("rocfft_plan_create", result);
                }
            #else // host
                if (is_flattened) {
                    fplan = fftw_plan_dft_r2c_2d(fft_size[1], fft_size[0],
                                                 variables_onegrid[mfi].dataPtr(),
                                                 reinterpret_cast<FFTcomplex*>
                                                 (spectral_field.back()->dataPtr()),
                                                 FFTW_ESTIMATE);
                } 
                else {
                    fplan = fftw_plan_dft_r2c_3d(fft_size[2], fft_size[1], fft_size[0],
                                                 variables_onegrid[mfi].dataPtr(),
                                                 reinterpret_cast<FFTcomplex*>
                                                 (spectral_field.back()->dataPtr()),
                                                 FFTW_ESTIMATE);
                }
            #endif

                forward_plan.push_back(fplan);
            }

	    built_plan = true;
        
        }

        ParallelDescriptor::Barrier();

        // ForwardTransform
        for (MFIter mfi(variables_onegrid); mfi.isValid(); ++mfi) {
            int i = mfi.LocalIndex();
        
        #ifdef AMREX_USE_CUDA
            cufftSetStream(forward_plan[i], amrex::Gpu::gpuStream());
            cufftResult result = cufftExecD2Z(forward_plan[i],
                                              variables_onegrid[mfi].dataPtr(),
                                              reinterpret_cast<FFTcomplex*>
                                                  (spectral_field[i]->dataPtr()));
            if (result != CUFFT_SUCCESS) {
                amrex::AllPrint() << " forward transform using cufftExec failed! Error: "
                                  << cufftErrorToString(result) << "\n";
	    }
	    
        #elif AMREX_USE_HIP
            rocfft_execution_info execinfo = nullptr;
            rocfft_status result = rocfft_execution_info_create(&execinfo);
            assert_rocfft_status("rocfft_execution_info_create", result);

            std::size_t buffersize = 0;
            result = rocfft_plan_get_work_buffer_size(forward_plan[i], &buffersize);
            assert_rocfft_status("rocfft_plan_get_work_buffer_size", result);

            void* buffer = amrex::The_Arena()->alloc(buffersize);
            result = rocfft_execution_info_set_work_buffer(execinfo, buffer, buffersize);
            assert_rocfft_status("rocfft_execution_info_set_work_buffer", result);

            result = rocfft_execution_info_set_stream(execinfo, amrex::Gpu::gpuStream());
            assert_rocfft_status("rocfft_execution_info_set_stream", result);

	        amrex::Real* variables_onegrid_ptr = variables_onegrid[mfi].dataPtr();
	        FFTcomplex* spectral_field_ptr = reinterpret_cast<FFTcomplex*>(spectral_field[i]->dataPtr());
            result = rocfft_execute(forward_plan[i],
                                    (void**) &variables_onegrid_ptr, // in
                                    (void**) &spectral_field_ptr, // out
                                    execinfo);
            assert_rocfft_status("rocfft_execute", result);
            amrex::Gpu::streamSynchronize();
            amrex::The_Arena()->free(buffer);
            result = rocfft_execution_info_destroy(execinfo);
            assert_rocfft_status("rocfft_execution_info_destroy", result);
        #else
            fftw_execute(forward_plan[i]);
        #endif
        }

        // copy data to a full-sized MultiFab
        // this involves copying the complex conjugate from the half-sized field
        // into the appropriate place in the full MultiFab
        for (MFIter mfi(variables_dft_real_onegrid); mfi.isValid(); ++mfi) {

            Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();

            Array4<Real> const& realpart = variables_dft_real_onegrid.array(mfi);
            Array4<Real> const& imagpart = variables_dft_imag_onegrid.array(mfi);

            Box bx = mfi.fabbox();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                /*
                  Unpacking rules:
                  For domains from (0,0,0) to (Nx-1,Ny-1,Nz-1)
                  For any cells with i index > Nx/2, these values are complex conjugates of the corresponding
                  entry where (Nx-i,Ny-j,Nz-k) UNLESS that index is zero, in which case you use 0.
                  e.g. for an 8^3 domain, any cell with i index
                  Cell (6,2,3) is complex conjugate of (2,6,5)
                  Cell (4,1,0) is complex conjugate of (4,7,0)  (note that the FFT is computed for 0 <= i <= Nx/2)
                */
                if (i <= bx.length(0)/2) {
                    // copy value
                    realpart(i,j,k) = spectral(i,j,k).real();
                    imagpart(i,j,k) = spectral(i,j,k).imag();
                } else {
                    // copy complex conjugate
                    int iloc = bx.length(0)-i;
                    int jloc, kloc;
                    if (is_flattened) {
                        jloc = (j == 0) ? 0 : bx.length(1)-j;
                        kloc = 0;
                    } 
                    else {
                        jloc = (j == 0) ? 0 : bx.length(1)-j;
                        kloc = (k == 0) ? 0 : bx.length(2)-k;
                    }

                    if (unpack) {
                        realpart(i,j,k) =  spectral(iloc,jloc,kloc).real();
                        imagpart(i,j,k) = -spectral(iloc,jloc,kloc).imag();
                    }
                    else {
                        realpart(i,j,k) =  0.0;
                        imagpart(i,j,k) =  0.0;
                    }
                }

                realpart(i,j,k) /= npts;
                imagpart(i,j,k) /= npts;
            });

            /*
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                std::cout << "HACKFFT " << i << " " << j << " " << k << " "
                          << realpart(i,j,k) << " + " << imagpart(i,j,k) << "i"
                          << std::endl;
            });
            */
        }

        variables_dft_real.ParallelCopy(variables_dft_real_onegrid,0,comp,1);
        variables_dft_imag.ParallelCopy(variables_dft_imag_onegrid,0,comp,1);

    }

    // destroy fft plan
    for (int i = 0; i < forward_plan.size(); ++i) {
    #ifdef AMREX_USE_CUDA
        cufftDestroy(forward_plan[i]);
    #elif AMREX_USE_HIP
        rocfft_plan_destroy(forward_plan[i]);
    #else
        fftw_destroy_plan(forward_plan[i]);
    #endif
    }
//    fftw_mpi_cleanup();
}

#if 1
inline void check_kspace_symmetries(const Geometry& geom,
				    const MultiFab& noise_onegrid,
				    const Vector<std::unique_ptr<BaseFab<GpuComplex<Real>>>>& spectral_field) {
  BL_PROFILE_VAR("check_kspace_symmetries()",check_kspace_symmetries);
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
			 const MultiFab& kspace_noise_imag, int NVAR) {
  BL_PROFILE_VAR("compute_ifft",compute_ifft);
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

    for (int k=0; k<NVAR; ++k) {

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
            if  ( ( (ky > domain.length(1)/2) && (kz == kzloc) && (kx == kxloc) )
            || ( (kz > domain.length(2)/2) && (kx == kxloc) ) ) { // mind ParallelFor loop order!
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
}
#endif