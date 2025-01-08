#ifndef LBM_TESTS_H_
#define LBM_TESTS_H_

AMREX_GPU_HOST_DEVICE void cholesky_decomp(GpuArray<Real,ndof*ndof>& A, const int n, const int bstart);
void generate_kspace_noise(const Geometry& geom, FabArray<BaseFab<GpuComplex<Real>>>& kspace_noise);

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

inline bool check_kspace_symmetries(const Geometry& geom,
				    FabArray<BaseFab<GpuComplex<Real>>>& kspace_noise) {
  bool symmetric = true;

  Box domain = geom.Domain();
  BaseFab<GpuComplex<Real>> kspace_noise_whole(domain);
  Array4<GpuComplex<Real> const> const& xi = kspace_noise_whole.const_array();

  // we need the whole BaseFab for the global symmetry check
  kspace_noise.copyTo(kspace_noise_whole, 0);

  for (int kz=0; kz<domain.length(2); ++kz) {
    for (int ky=0; ky<domain.length(1); ++ky) {
      for (int kx=0; kx<domain.length(0); ++kx) {

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
          // Print() << "This should never execute" << std::endl;
          // symmetric &= ((xi(kx,ky,kz).real() ==  xi(kxloc,kyloc,kzloc).real())
          //   && (xi(kx,ky,kz).imag() == -xi(kxloc,kyloc,kzloc).imag()));
          // if (!symmetric) {
          //   Print() << "symmetry violation 2! ("
          //     << kx << "," << ky << "," << kz << ")"
          //     << std::endl;
          //   exit(0);
          // }
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
      }
    }
  }
  return symmetric;
}

bool symmetry_test(Geometry const& geom) {
  bool success = true;

  Box domain = geom.Domain();
  BoxArray ba(domain);
  DistributionMapping dm(ba);
  MultiFab realspace_noise(ba, dm, ndof, 0);
  FabArray<BaseFab<GpuComplex<Real>>> kspace_noise(ba, dm, ndof, 0);

  generate_kspace_noise(geom, kspace_noise);

  for (int k=0; k<ndof; ++k) {
    // alias MultiFabs for one of the ndof components
    MultiFab realspace_noise_one(realspace_noise, amrex::make_alias, k, 1);
    FabArray<BaseFab<GpuComplex<Real>>> kspace_noise_one(kspace_noise, amrex::make_alias, k, 1);

    success &= check_kspace_symmetries(geom, kspace_noise_one);

  }
  if (success) {
    Print() << "k-space symmetries of generated noise ok" << std::endl;
  }
  return success;
}

GpuArray<Real,ndof*ndof> square_matrix(GpuArray<Real,ndof*ndof> const& L) {
    GpuArray<Real,ndof*ndof> C = {};
    C.fill(0.);
    for (int i=0; i<ndof; ++i) {
        for (int j=0; j<ndof; ++j) {
            for (int k=0; k<ndof; ++k) {
                C[i*ndof+j] += L[i*ndof+k]*L[j*ndof+k];
            }
        }
    }
    return C;
}

GpuArray<Real,ndof*ndof> spd_matrix(const int n, const int bstart) {
    GpuArray<Real,ndof*ndof> A, C;
    C.fill(0.);
    for (int i=bstart; i<n; ++i) {
        for (int j=bstart; j<n; ++j) {
            A[i*ndof+j] = 0.1*(Random()-0.5);
        }
    }
    for (int i=bstart; i<n; ++i) {
        for (int j=bstart; j<n; ++j) {
            C[i*ndof+j] = 0.5*(A[i*ndof+j]+A[j*ndof+i]);
        }
    }
    for (int i=bstart; i<n; ++i) {
        C[i*ndof+i] += 1.;
    }
    return C;
}

bool test_matrix(GpuArray<Real,ndof*ndof> const& A, GpuArray<Real,ndof*ndof> const& B) {
    bool test = true;
    for (int i=0; i<ndof; ++i) {
        for (int j=0; j<=i; ++j) {
            Real a = A[i*ndof+j];
            Real b = B[i*ndof+j];
            if ((fabs(a-b)>FLT_EPSILON) && (fabs(a+b)>FLT_EPSILON)) {
                Print() << "Test failed: " << i << " " << j << " " << "A = " << A[i*ndof+j] << "\tB = " << B[i*ndof+j] << "\t" << fabs(a)-fabs(b) << "\n";
                test = false;
            }
        }
    }
    return test;
}

bool cholesky_test(const int ntrial) {
    bool status = true;

    GpuArray<Real,ndof*ndof> A, B, C;
    for (int t=0; t<ntrial; ++t) {
        C = spd_matrix(ndof, ncons);
        A = C;
        cholesky_decomp(C, ndof, ncons);
        B = square_matrix(C);
        status = test_matrix(A,B);
        if (!status) {
            Print() << "Trial " << t << " failed." << std::endl;
            break;
        }
    }

    if (status) {
        Print() << "Cholesky test ok, tested " << ntrial << " matrices." << std::endl;
    } else {
        Print() << "ERROR: Cholesky test failed" << std::endl;
        ParallelDescriptor::Abort(-1, false);
    }

    return status;
}

bool equilibrium_test(int x, int y, int z, const Array4<Real>& h) {
    bool test = true;

    // test that the direct computation of the equilibrium moments
    // equals the transformation of the equilibrium distributions

    const Array1D<Real,0,2> fields = { h(x,y,z,0), h(x,y,z,1) };
    const RealVect u = { h(x,y,z,2), h(x,y,z,3), h(x,y,z,4) };

    const Array1D<RealVect,0,2> gradients = {
        gradient(x,y,z,h,0),
        gradient(x,y,z,h,1),
    };

    const Array1D<Real,0,2> laplacians = {
        laplacian(x,y,z,h,0),
        laplacian(x,y,z,h,1),
    };

    const Array1D<Real,0,2> mu = chemical_potentials(fields, gradients, laplacians);

    Array1D<Real,0,nvel> f = fequilibrium(fields, gradients, laplacians, u);
    Array1D<Real,0,nvel> mf = fequilibrium_moments(fields, gradients, laplacians, u);
    Array1D<Real,0,nvel> g = gequilibrium(fields(1), mu(1), u);
    Array1D<Real,0,nvel> mg = gequilibrium_moments(fields, mu, u);

    Array1D<Real,0,nvel> mf_transformed, mg_transformed;
    for (int k=0; k<nvel; k++) {
        mf_transformed(k) = 0.0;
        mg_transformed(k) = 0.0;
        for (int i=0; i<nvel; i++) {
            mf_transformed(k) += e[k][i] * f(i);
            mg_transformed(k) += e[k][i] * g(i);
        }
    }
    for (int k=0; k<nvel; k++) {
        if (fabs(mf(k)-mf_transformed(k)) > FLT_EPSILON || fabs(mg(k)-mg_transformed(k) > FLT_EPSILON)) {
            test = false;
            break;
        }
    }

    return test;
}

bool equilibrium_test(const Geometry& geom, MultiFab& hydrovs) {
    for (MFIter mfi(hydrovs); mfi.isValid(); ++mfi) {
        const Array4<Real>& h = hydrovs.array(mfi);
        const Box& valid_box = mfi.growntilebox(1);
        ParallelFor(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z) {
            bool test = true;
            test = equilibrium_test(x, y, z, h);
            if (!test) {
                Print() << "equilibrium test failed!" << std::endl;
                exit(-1);
            }
        });
    }
    return false;
}

bool unit_tests(const Geometry& geom, MultiFab& hydrovs) {
    bool status = true;
    status &= cholesky_test(100);
    status &= equilibrium_test(geom, hydrovs);
    status &= symmetry_test(geom);
    return status;
}

#endif
