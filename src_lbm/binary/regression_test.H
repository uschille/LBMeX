#include "nkarthi/LBM_fluctuations.H"

void regression_test_noise_covariance_matrix() {

  Real rho0 = 1.0;
  Real phi0 = 0.0;
  Real k2 = 0.0;

  GpuArray<Real,ndof*ndof> C_uschill, C_nkarthi;
  C_uschill = noise_covariance_matrix(rho0, phi0, k2);
  C_nkarthi = nkarthi::noise_covariance(rho0, phi0, k2);

  Print() << "C_uschill - C_nkarthi:" << std::endl;
  for (int i=0; i<ndof; ++i) {
    Print() << "row " << i << ": ";
    for (int j=0; j<ndof; ++j) {
      int k = i*ndof + j;
      Print() << C_uschill[k]-C_nkarthi[k] << " ";
    }
    Print() << std::endl;
  }

}

void regression_tests() {
    regression_test_noise_covariance_matrix();
}
