#include "nkarthi/LBM_binary.H"
#include "nkarthi/LBM_fluctuations.H"

bool regression_test_noise_covariance_matrix() {
  bool test = true;

  Real rho0 = 0.9876;
  Real phi0 = 0.2345;
  Real k2 = 1.5432;

  GpuArray<Real,ndof*ndof> C_uschill, C_nkarthi;
  C_uschill = noise_covariance_matrix(rho0, phi0, k2);
  C_nkarthi = nkarthi::noise_covariance(rho0, phi0, k2);

  //Print() << "C_uschill - C_nkarthi: " << std::endl;
  for (int i=0; i<ndof; ++i) {
    //Print() << "row " << i << ": ";
    for (int j=0; j<ndof; ++j) {
      int k = i*ndof + j;
      //Print() << C_uschill[k]-C_nkarthi[k] << " ";
      if (fabs(C_uschill[k]-C_nkarthi[k])>FLT_EPSILON) {
        Print() << "covariance matrix element (" << i << ", " << j << ") differs" << std::endl;
        test = false;
      }
    }
    //Print() << std::endl;
  }

  if (test) Print() << "OK: coavriance matrix matches" << std::endl;
  return test;

}

bool regression_test_cholesky() {
  bool success = true;

  Real rho0 = 0.9876;
  Real phi0 = 0.2345;
  Real k2 = 1.5432;

  GpuArray<Real,ndof*ndof> C_uschill, C_nkarthi;
  C_uschill = noise_covariance_matrix(rho0, phi0, k2);
  C_nkarthi = nkarthi::noise_covariance(rho0, phi0, k2);

  cholesky_decomp(C_uschill, ndof, ncons);
  nkarthi::cholesky_decomp(C_nkarthi, ndof, ncons);

  //Print() << "C_uschill - C_nkarthi: " << std::endl;
  for (int i=0; i<ndof; ++i) {
    //Print() << "row " << i << ": ";
    for (int j=0; j<ndof; ++j) {
      int k = i*ndof + j;
      //Print() << C_uschill[k]-C_nkarthi[k] << " ";
      if (fabs(C_uschill[k]-C_nkarthi[k])>FLT_EPSILON) {
        Print() << "covariance matrix element (" << i << ", " << j << ") differs" << std::endl;
        success = false;
      }
    }
    //Print() << std::endl;
  }


  if (success) Print() << "OK: Cholesky decomposition matches" << std::endl;
  return success;

}

bool regression_test_ifft(const Geometry& geom) {
    bool success = true;

    Box domain = geom.Domain();
    BoxArray ba(domain);
    DistributionMapping dm(ba);

    MultiFab kspace_noise_real(ba, dm, ndof, 0);
    MultiFab kspace_noise_imag(ba, dm, ndof, 0);
    MultiFab noise_uschill(ba, dm, ndof, 0);
    MultiFab noise_nkarthi(ba, dm, ndof, 0);
    MultiFab differences(ba, dm, 1, 0);

    differences.setVal(0.0);
    kspace_noise_real.setVal(0.);
    kspace_noise_imag.setVal(0.);

    generate_kspace_noise(geom, kspace_noise_real, kspace_noise_imag);

    compute_ifft(geom, kspace_noise_real, kspace_noise_imag, noise_uschill);
    nkarthi::compute_ifft(geom, noise_nkarthi, kspace_noise_real, kspace_noise_imag, ndof);

    auto const & xi_uschill = noise_uschill.arrays();
    auto const & xi_nkarthi = noise_nkarthi.arrays();
    auto const & diff = differences.arrays();
    ParallelFor(noise_uschill, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
      for (int k=0; k<ndof; k++) {
        Real xi_us = xi_uschill[nbx](x,y,z,k);
        Real xi_nk = xi_nkarthi[nbx](x,y,z,k);
        if (fabs(xi_us-xi_nk) > FLT_EPSILON) {
          Print() << "ifft ("<<x<<","<<y<<","<<z<<","<<k<<"): "<<xi_us<<" vs "<<xi_nk<<std::endl;
          diff[nbx](x,y,z) += fabs(xi_us-xi_nk);
        }
      }
    });

    if (differences.sum() > 0.0) {
      success = false;
    } else {
      Print() << "OK: Inverse FFT matches" << std::endl;
    }

    return success;
}

bool regression_test_correlated_fluctuations(const Geometry& geom, MultiFab& hydrovs) {
    bool success = true;

    Box domain = geom.Domain();
    BoxArray ba(domain);
    DistributionMapping dm(ba);

    MultiFab noise_uschill(ba, dm, 2*nvel, 2);
    MultiFab noise_nkarthi(ba, dm, 2*nvel, 2);
    MultiFab differences(ba, dm, 2*nvel, 2);

    generate_correlated_fluctuations(geom, hydrovs, noise_uschill);
    nkarthi::generate_fluctuations(geom, hydrovs, noise_nkarthi, 1);

    const auto & xi_uschill = noise_uschill.arrays();
    const auto & xi_nkarthi = noise_nkarthi.arrays();
    const auto & diff = differences.arrays();
    ParallelFor(noise_uschill, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
      for (int k=0; k<ndof; k++) {
        Real xi_us = xi_uschill[nbx](x,y,z,k);
        Real xi_nk = xi_nkarthi[nbx](x,y,z,k);
        if (fabs(xi_us-xi_nk) > FLT_EPSILON) {
          Print() << "noise ("<<x<<","<<y<<","<<z<<","<<k<<"): "<<xi_us<<" vs "<<xi_nk<<std::endl;
          diff[nbx](x,y,z) += fabs(xi_us-xi_nk);
        }

      }
    });

    if (differences.sum() > 0.0) {
      success = false;
    } else {
      Print() << "OK: Correlated noise matches" << std::endl;
    }

    return success;
}

bool regression_test_uncorrelated_fluctuations(const Geometry& geom, MultiFab& hydrovs) {
    bool success = true;

    Box domain = geom.Domain();
    BoxArray ba(domain);
    DistributionMapping dm(ba);

    MultiFab noise_uschill(ba, dm, 2*nvel, 2);
    MultiFab noise_nkarthi(ba, dm, 2*nvel, 2);
    MultiFab differences(ba, dm, 2*nvel, 2);

    generate_fluctuations(geom, hydrovs, noise_uschill);
    nkarthi::generate_fluctuations(geom, hydrovs, noise_nkarthi, 0);

    const auto & xi_uschill = noise_uschill.arrays();
    const auto & xi_nkarthi = noise_nkarthi.arrays();
    const auto & diff = differences.arrays();
    ParallelFor(noise_uschill, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
      for (int k=0; k<ndof; k++) {
        Real xi_us = xi_uschill[nbx](x,y,z,k);
        Real xi_nk = xi_nkarthi[nbx](x,y,z,k);
        if (fabs(xi_us-xi_nk) > FLT_EPSILON) {
          Print() << "noise ("<<x<<","<<y<<","<<z<<","<<k<<"): "<<xi_us<<" vs "<<xi_nk<<std::endl;
          diff[nbx](x,y,z) += fabs(xi_us-xi_nk);
        }

      }
    });

    if (differences.sum() > 0.0) {
      success = false;
    } else {
      Print() << "OK: Uncorrelated noise matches" << std::endl;
    }

    return success;
}

bool regression_test_equilibria(const Geometry& geom, MultiFab& hydrovs) {
  bool success = true;

  Box domain = geom.Domain();
  BoxArray ba(domain);
  DistributionMapping dm(ba);

  MultiFab differences(ba, dm, 1, 0);
  differences.setVal(0.);

  const auto & hv = hydrovs.arrays();
  const auto & diff = differences.arrays();

  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Array4<Real>& h = hv[nbx];

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

    Array1D<Real,0,2> mu_uschill, mu_nkarthi;
    Array1D<Real,0,nvel> f_uschill, g_uschill, f_nkarthi, g_nkarthi;

    mu_uschill = chemical_potentials(fields, gradients, laplacians);
    f_uschill = fequilibrium_moments(fields, gradients, laplacians, u);
    g_uschill = gequilibrium_moments(fields, mu_uschill, u);

    nkarthi::bulk_free_energy curr_state(h(x,y,z,0), h(x,y,z,1));
    mu_nkarthi = nkarthi::chemical_potentials(curr_state, gradients, laplacians);
    f_nkarthi = fequilibrium_moments(curr_state, gradients, laplacians, u);
    g_nkarthi = gequilibrium_moments(curr_state, mu_nkarthi, u);

    Real us, nk;
    for (int i=0; i<nvel; ++i) {
      us = f_uschill(i);
      nk = f_nkarthi(i);
      if (fabs(us-nk) > FLT_EPSILON) {
        Print() << "f_eq ("<<x<<","<<y<<","<<z<<"): "<<us<<" vs "<<nk<<std::endl;
        diff[nbx](x,y,z) += fabs(us-nk);
      }
      us = g_uschill(i);
      nk = g_nkarthi(i);
      if (fabs(us-nk) > FLT_EPSILON) {
        Print() << "g_eq ("<<x<<","<<y<<","<<z<<"): "<<us<<" vs "<<nk<<std::endl;
        diff[nbx](x,y,z) += fabs(us-nk);
      }
    }

  });

  if (differences.sum() > 0.0) {
    Print() << "FAIL: Equilibria do not match" << std::endl;
    success = false;
  } else {
    Print() << "OK: Equilibria match" << std::endl;
  }

  return success;
}

bool regression_test_LBM_timestep(const Geometry& geom,
    MultiFab& fold,
    MultiFab& gold,
    MultiFab& fnew,
    MultiFab& gnew,
    MultiFab& hydrovs,
    MultiFab& noise) {
  bool success = true;

  Box domain = geom.Domain();
  BoxArray ba(domain);
  DistributionMapping dm(ba);

  MultiFab f_uschill(ba, dm, nvel, 2);
  MultiFab g_uschill(ba, dm, nvel, 2);
  MultiFab f_nkarthi(ba, dm, nvel, 2);
  MultiFab g_nkarthi(ba, dm, nvel, 2);
  MultiFab h_uschill(ba, dm, 2*nvel, 2);
  MultiFab h_nkarthi(ba, dm, 2*nvel, 2);
  MultiFab differences(ba, dm, 2*nvel, 2);

  Copy(f_uschill, fold, 0, 0, nvel, 2);
  Copy(g_uschill, gold, 0, 0, nvel, 2);
  Copy(h_uschill, hydrovs, 0, 0, 2*nvel, 2);

  Copy(f_nkarthi, fold, 0, 0, nvel, 2);
  Copy(g_nkarthi, gold, 0, 0, nvel, 2);
  Copy(h_nkarthi, hydrovs, 0, 0, 2*nvel, 2);

  LBM_timestep(geom, f_uschill, g_uschill, fnew, gnew, h_uschill, noise);

  nkarthi::LBM_timestep(geom, f_nkarthi, g_nkarthi, fnew, gnew, h_nkarthi, noise);

  const auto & f_us = f_uschill.arrays();
  const auto & f_nk = f_nkarthi.arrays();
  const auto & g_us = g_uschill.arrays();
  const auto & g_nk = g_nkarthi.arrays();
  const auto & h_us = h_uschill.arrays();
  const auto & h_nk = h_nkarthi.arrays();
  const auto & diff = differences.arrays();

  ParallelFor(f_uschill, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    for (int i=0; i<nvel; i++) {
      Real us, nk;
      us = f_us[nbx](x,y,z,i);
      nk = f_nk[nbx](x,y,z,i);
      if (fabs(us-nk) > FLT_EPSILON) {
        Print() << "f ("<<x<<","<<y<<","<<z<<","<<i<<"): "<<us<<" vs "<<nk<<std::endl;
        diff[nbx](x,y,z) += fabs(us-nk);
      }
      us = g_us[nbx](x,y,z,i);
      nk = g_nk[nbx](x,y,z,i);
      if (fabs(us-nk) > FLT_EPSILON) {
        Print() << "g ("<<x<<","<<y<<","<<z<<","<<i<<"): "<<us<<" vs "<<nk<<std::endl;
        diff[nbx](x,y,z) += fabs(us-nk);
      }
      us = h_us[nbx](x,y,z,i);
      nk = h_nk[nbx](x,y,z,i);
      if (fabs(us-nk) > FLT_EPSILON) {
        Print() << "h ("<<x<<","<<y<<","<<z<<","<<i<<"): "<<us<<" vs "<<nk<<std::endl;
        diff[nbx](x,y,z) += fabs(us-nk);
      }
    }
  });

  if (differences.sum() > 0.0) {
    success = false;
  } else {
    Print() << "OK: LBM timestep matches" << std::endl;
  }

  return success;
}

void regression_tests(const Geometry& geom,
  MultiFab& fold,
  MultiFab& gold,
  MultiFab& fnew,
  MultiFab& gnew,
  MultiFab& hydrovs,
  MultiFab& noise) {
    bool status = true;
    status &= regression_test_noise_covariance_matrix();
    status &= regression_test_cholesky();
    status &= regression_test_ifft(geom);
    status &= regression_test_correlated_fluctuations(geom, hydrovs);
    status &= regression_test_uncorrelated_fluctuations(geom, hydrovs);
    status &= regression_test_equilibria(geom, hydrovs);
    status &= regression_test_LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, noise);
}
