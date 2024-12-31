#include "nkarthi/LBM_fluctuations.H"
#include "nkarthi/LBM_IO.H"
#include <math.h> 

namespace nkarthi;

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
    // Real diff, sum;
    for (int i=0; i<ndof; ++i) {
        for (int j=0; j<=i; ++j) {
            const Real a = A[i*ndof+j];
            const Real b = B[i*ndof+j];
            const Real diff = a - b;
            const Real sum  = a + b;

            if ((fabs(diff)>FLT_EPSILON) && (fabs(sum)>FLT_EPSILON)) {
                Print() << "Test failed: " << i << " " << j << " " << "A = " << A[i*ndof+j] << "\tB = " << B[i*ndof+j] << "\t" << diff << "\n";
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


void generate_ifft_test_noise(const Geometry& geom, MultiFab& kspace_noise_real, MultiFab& kspace_noise_imag, int nx, int k){
    const Box domain = geom.Domain();
    Real set_val = 1.;

    for (MFIter mfi(kspace_noise_real); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const Array4<Real>& xi_real = kspace_noise_real.array(mfi);
        const Array4<Real>& xi_imag = kspace_noise_imag.array(mfi);
        int idx = 0;

        // sin in <1 0 0> direction
        xi_real(1*k, 0, 0, idx) = set_val;
        // Print() << xi_real(1*k, 0, 0, idx) << "\n";
        idx++;
        // sin in <0 1 0> direction
        xi_real(0, 1*k, 0, idx) = set_val;
        idx++;
        // sin in <0 0 1> direction
        xi_real(0, 0, 1*k, idx) = set_val;
        idx++;
        // sin in <1 1 0> direction
        xi_real(1*k, 1*k, 0, idx) = set_val;
        idx++;
        // sin in <1 0 1> direction
        xi_real(1*k, 0, 1*k, idx) = set_val;
        idx++;
        // sin in <0 1 1> direction
        xi_real(0, 1*k, 1*k, idx) = set_val;
        idx++;
        // sin in <1 1 1> direction
        xi_real(1*k, 1*k, 1*k, idx) = set_val;
        idx++;
        // sin in <-1 1 1> direction
        xi_real(nx - 1*k, 1*k, 1*k, idx) = set_val;
        idx++;
        // sin in <1 -1 1> direction
        xi_real(1*k, nx - 1*k, 1*k, idx) = set_val;
        idx++;
        // sin in <1 1 -1> direction
        xi_real(1*k, 1*k, nx - 1*k, idx) = set_val;
        idx++;
  }
}

void test_case_fft(){
    int L = 128; int reps = 1;
    int nghost = 1;
    int NVAR = 38;
    bool unpack = false;
    int step = 0;
    Real time = step;

    Vector<std::string> var_names(NVAR);
    for(int i = 0; i < NVAR; i++){var_names[i] = "Test case " + std::to_string(i + 1);}

    // set up Box and Geomtry
    IntVect dom_lo(0, 0, 0);
    IntVect dom_hi(L - 1, L - 1, L - 1);
    Array<int,3> periodicity({1,1,1});
    Box domain(dom_lo, dom_hi);
    RealBox real_box({0.,0.,0.},{1.,1.,1.});
    Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
    BoxArray ba(domain);
    ba.maxSize(L);
    DistributionMapping dm(ba);

    Print() << "Box created \n";

    MultiFab noise(ba, dm, NVAR, nghost);
    MultiFab variables_dft_real(ba, dm, NVAR, 0);
    MultiFab variables_dft_imag(ba, dm, NVAR, 0);
    MultiFab output(ba, dm, NVAR, nghost);
    noise.setVal(0.);variables_dft_real.setVal(0.);variables_dft_imag.setVal(0.);output.setVal(0.);
    // int kx = 0; int ky = 0; int kz = 0;
    
    generate_ifft_test_noise(geom, variables_dft_real, variables_dft_imag, L, reps);
    WriteSingleLevelPlotfile("ifft_test_plt_input", variables_dft_real, var_names, geom, 0, 0);
    Print() << "data generated \n";
    compute_ifft(geom, noise, variables_dft_real, variables_dft_imag, NVAR);
    WriteSingleLevelPlotfile("ifft_test_plt", noise, var_names, geom, 0, 0);
    Print() << "IFFT completed \n";

    variables_dft_real.setVal(0.);variables_dft_imag.setVal(0.);output.setVal(0.);
    compute_fft(geom, noise, variables_dft_real, variables_dft_imag, NVAR, unpack);
    
    noise.setVal(0.);
    compute_ifft(geom, noise, variables_dft_real, variables_dft_imag, NVAR);
    WriteSingleLevelPlotfile("ifft_to_fft_to_ifft_test_plt", noise, var_names, geom, 0, 0);
    Print() << "FFT_to_iFFT completed \n";
}