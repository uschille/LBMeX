# include "LBM_binary.H"
#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <math.h>

void generate_ifft_test_noise(const Geometry& geom, MultiFab& kspace_noise_real, MultiFab& kspace_noise_imag, int nx, int k){
    const Box domain = geom.Domain();
    Real set_val = 1;

    for (MFIter mfi(kspace_noise_real); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const Array4<Real>& xi_real = kspace_noise_real.array(mfi);
        const Array4<Real>& xi_imag = kspace_noise_imag.array(mfi);
        int idx = 0;
        
        // sin in <1 0 0> direction
        xi_real(1*k, 0, 0, idx) = set_val;
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

Vector<std::string> test_case_var_names(){
    Vector<std::string> var_names(ndof);
    int cnt = 0;
    var_names[cnt++] += "<1 0 0>";
    var_names[cnt++] += "<0 1 0>";
    var_names[cnt++] += "<0 0 1>";

    var_names[cnt++] += "<1 1 0>";
    var_names[cnt++] += "<1 0 1>";
    var_names[cnt++] += "<0 1 1>"; 

    var_names[cnt++] += "<1 1 1>"; 

    var_names[cnt++] += "<-1 1 1>"; 
    var_names[cnt++] += "<1 -1 1>"; 
    var_names[cnt++] += "<1 1 -1>";

    for(int i = cnt; i < ndof; i ++){var_names[i] = "Test case " + std::to_string(i + 1);}
    return var_names;
}

void test_case_ifft(const Geometry& geom, MultiFab& noise, int nx, int k = 1){

    BoxArray ba = noise.boxArray();
    DistributionMapping dm = noise.DistributionMap();
    
    MultiFab kspace_noise_real(ba, dm, ndof, 0);
    MultiFab kspace_noise_imag(ba, dm, ndof, 0);

    kspace_noise_real.setVal(0.);
    kspace_noise_imag.setVal(0.);

    generate_ifft_test_noise(geom, kspace_noise_real, kspace_noise_imag, nx, k);

    compute_ifft(geom, noise, kspace_noise_real, kspace_noise_imag);
    
    Vector<std::string> var_names = test_case_var_names();
    // for(int i = 0; i < ndof; i++){var_names[i] = "Test case " + std::to_string(i + 1);}

    WriteSingleLevelPlotfile("ifft_test_plt", noise, var_names, geom, 0, 0);
}