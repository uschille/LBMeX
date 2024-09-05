#include "LBM_fluctuations.H"
#include <math.h> 

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


int fft_test(){
    int nx = 32;
    int ny = 32;
    int nz = 32;
    int nghost = 1;
    int NVAR = 1;
    bool unpack = false;
    const Real A = 1.0;

    // set up Box and Geomtry
    IntVect dom_lo(0, 0, 0);
    IntVect dom_hi(nx-1, ny-1, nz-1);
    Array<int,3> periodicity({1,1,1});
    Box domain(dom_lo, dom_hi);
    RealBox real_box({0.,0.,0.},{1.,1.,1.});
    Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
    BoxArray ba(domain);
    ba.maxSize(32);
    DistributionMapping dm(ba);

    MultiFab variables(ba, dm, 1, nghost);
    MultiFab variables_dft_real(ba, dm, 1, nghost);
    MultiFab variables_dft_imag(ba, dm, 1, nghost);
    MultiFab output(ba, dm, 1, nghost);
    
    variables.setVal(0, 0, 1, nghost);
    variables_dft_real.setVal(0, 0, 1, nghost);
    variables_dft_imag.setVal(0, 0, 1, nghost);

    // initialize a sin wave in the x direction
    for (MFIter mfi(variables); mfi.isValid(); ++mfi) {

    const Box& valid_box = mfi.validbox();
    const Array4<Real>& v = variables.array(mfi);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
        Real check = (Real(x))/Real(nx)*2.*M_PI;
        // add in sin wave here. Previous implementation was wrong
        });
    }
    
    int step = 0;
    Real time = step;
    std::string pltfile = amrex::Concatenate("FFT_test",step,5);
    Vector<std::string> var_names = {"FFT and ifft conversion"};
    WriteSingleLevelPlotfile(pltfile, variables, var_names, geom, time, step);

    compute_fft(variables,
			    variables_dft_real, 
			    variables_dft_imag,
			    geom,
			    NVAR,
                unpack);

    compute_ifft(geom, output,
			 variables_dft_real,
			 variables_dft_imag);
    
    step++;
    time = step;
    WriteSingleLevelPlotfile(pltfile, output, var_names, geom, time, step);
    return 1;
}