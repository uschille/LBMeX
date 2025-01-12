#ifndef LBM_TESTS_H_
#define LBM_TESTS_H_

bool equilibrium_test(int x, int y, int z, const Array4<Real>& h) {
    bool test = true;

    // test that the direct computation of the equilibrium moments
    // equals the transformation of the equilibrium distributions

    const Array1D<Real,0,2> fields = { h(x,y,z,0), h(x,y,z,1) };
    const RealVect u = { h(x,y,z,2), h(x,y,z,3), h(x,y,z,4) };

    Array1D<Real,0,nvel> f = equilibrium(fields(0), u);
    Array1D<Real,0,nvel> mf = equilibrium_moments(fields(0), u);
    Array1D<Real,0,nvel> g = equilibrium(fields(1), u);
    Array1D<Real,0,nvel> mg = equilibrium_moments(fields(1), u);

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
    status &= equilibrium_test(geom, hydrovs);
    return status;
}

#endif
