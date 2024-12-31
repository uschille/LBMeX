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

void regression_tests() {
    bool status = true;
    status &= regression_test_noise_covariance_matrix();
    status &= regression_test_cholesky();
}
