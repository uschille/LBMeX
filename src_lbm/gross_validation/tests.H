#include "LBM_fluctuations.H"

GpuArray<Real,nvel*nvel> square_matrix(GpuArray<Real,nvel*nvel> const& L) {
    GpuArray<Real,nvel*nvel> C = {};
    C.fill(0.);
    for (int i=0; i<nvel; ++i) {
        for (int j=0; j<nvel; ++j) {
            for (int k=0; k<nvel; ++k) {
                C[i*nvel+j] += L[i*nvel+k]*L[j*nvel+k];
            }
        }
    }
    return C;
}

GpuArray<Real,nvel*nvel> spd_matrix(const int n, const int bstart) {
    GpuArray<Real,nvel*nvel> A, C;
    C.fill(0.);
    for (int i=bstart; i<n; ++i) {
        for (int j=bstart; j<n; ++j) {
            A[i*nvel+j] = 0.1*(Random()-0.5);
        }
    }
    for (int i=bstart; i<n; ++i) {
        for (int j=bstart; j<n; ++j) {
            C[i*nvel+j] = 0.5*(A[i*nvel+j]+A[j*nvel+i]);
        }
    }
    for (int i=bstart; i<n; ++i) {
        C[i*nvel+i] += 1.;
    }
    return C;
}

bool test_matrix(GpuArray<Real,nvel*nvel> const& A, GpuArray<Real,nvel*nvel> const& B) {
    bool test = true;
    for (int i=0; i<nvel; ++i) {
        for (int j=0; j<=i; ++j) {
            Real a = A[i*nvel+j];
            Real b = B[i*nvel+j];
            if ((fabs(a-b)>FLT_EPSILON) && (fabs(a+b)>FLT_EPSILON)) {
                Print() << "Test failed: " << i << " " << j << " " << "A = " << A[i*nvel+j] << "\tB = " << B[i*nvel+j] << "\t" << fabs(a)-fabs(b) << "\n";
                test = false;
            }
        }
    }
    return test;
}

bool cholesky_test(const int ntrial) {
    bool status = true;

    GpuArray<Real,nvel*nvel> A, B, C;
    for (int t=0; t<ntrial; ++t) {
        C = spd_matrix(nvel, ncons);
        A = C;
        cholesky_decomp(C, nvel, ncons);
        B = square_matrix(C);
        status = test_matrix(A,B);
        if (!status) {
            Print() << "Trial " << t << " failed." << std::endl;
            break;
        }
    }

    if (status) {
        Print() << "Cholesky test ok, tested " << ntrial << " matrices." << std::endl;
    }
    else
        exit(-1);

    return status;
}
