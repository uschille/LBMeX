#include "LBM_fluctuations.H"

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
            A[i*ndof+j] = arc4random_uniform(0.5);
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
    }
    else
        exit(-1);

    return status;
}
