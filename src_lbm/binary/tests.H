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
    }
    else
        exit(-1);

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
    return status;
}