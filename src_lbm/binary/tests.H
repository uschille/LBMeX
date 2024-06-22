#include "LBM_fluctuations.H"

void cholesky_test() {
    const int n = ndof;
    GpuArray<Real,n*n> L = {};
    GpuArray<Real,n*n> C = {};

    for (int i=0; i<2*ncons; ++i) {
        for (int j=0; j<n; ++j) {
            L[i*n+j] = 0.;
            L[j*n+i] = 0.;
        }
    }

    for (int i=2*ncons; i<n; ++i) {
        for (int j=i+1; j<n; ++j) {
            L[i*n+j] = 0.0;
        }
        for (int j=2*ncons; j<i; ++j) {
            L[i*n+j] = 0.5;
        }
        L[i*n+i] = 2./3.;
    }

    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            C[i*n+j] = 0.;
            if (i>=ncons && j>=ncons)
            for (int k=0; k<n; ++k) {
                C[i*n+j] += L[i*n+k]*L[j*n+k];
            }
        }
    }

    cholesky_decomp(C,n);

    bool success = true;

    for (int i=0; i<n; i++) {
        for (int j=0; j<=i; ++j) {
            Real c = C[i*n+j];
            Real l = L[i*n+j];
            if (fabs(l-c>FLT_EPSILON)) {
                success = false;
                Print() << i << " " << j << " " << "C = " << C[i*n+j] << "\tL = " << L[i*n+j] << "\t" << l-c << "\n";
            }
        }
    }

    if (success)
        Print() << "Cholesky test ok." << std::endl;
    else
        exit(-1);

}