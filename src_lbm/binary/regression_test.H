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

void regression_tests(const Geometry& geom, MultiFab& hydrovs) {
    bool status = true;
    status &= regression_test_noise_covariance_matrix();
    status &= regression_test_cholesky();
    status &= regression_test_ifft(geom);
}
