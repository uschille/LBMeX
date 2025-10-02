#ifndef LBM_EXTERN_
#define LBM_EXTERN_
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include "AMReX_FileIO.H"
#include <AMReX_FabArray.H>
#include "AMReX_FileIO.H"
#include "AMReX_Analysis.H"
#include "LBM_hydrovs.H"

// These library function are directly converted from python, see "CodingNote_analysis_supp.ipynb" for details; 

const int NumOfTerms = 20;

AMREX_GPU_HOST_DEVICE
double inv_acosh(double x) {
    if (fabs(x) < 710.4) {
        return 1 / cosh(x);
    } else {
        return 0;
    }
}

//  calculate C_n^k, binomial coefficients;
AMREX_GPU_HOST_DEVICE
double comb(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    double result = 1;
    for (int i = 1; i <= k; i++) {
        result *= (double)(n - i + 1) / i;
    }
    return result;
}

AMREX_GPU_HOST_DEVICE
double factorial(int n) {
    if (n == 0) return 1;
    double result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

AMREX_GPU_HOST_DEVICE
std::vector<Real> getCoefS(int N) {
    std::vector<Real> coefSVec(N, 0.);
    double* A_array = (double*)malloc(N * sizeof(double));
    A_array[0] = 1.0;

    for (int k = 1; k < N; k++) {
        double A2k = 0.0;
        for (int kp = 0; kp < k; kp++) {
            A2k += A_array[kp] * comb(2 * k, 2 * kp);
        }
        A2k = -A2k;
        A_array[k] = A2k;
    }

    double* A_array_prime = (double*)malloc(N * sizeof(double));

    for (int k = 0; k < N; k++) {
        A_array_prime[k] = A_array[k] / factorial(2 * k);
        double Sk = 0.0;
        for (int k1 = 0; k1 <= k; k1++) {
            for (int k2 = 0; k2 <= k - k1; k2++) {
                for (int k3 = 0; k3 <= k - k1 - k2; k3++) {
                    int k4 = k - k1 - k2 - k3;
                    Sk += A_array_prime[k1] * A_array_prime[k2] * A_array_prime[k3] * A_array_prime[k4];
                }
            }
        }
        coefSVec[k] = Sk;
    }

    free(A_array);
    free(A_array_prime);
    
    return coefSVec;
}

// get given fixed size 2D matrix for binomial coefficients; same with function comb(), just for efficiency;
AMREX_GPU_HOST_DEVICE
std::vector<std::vector<Real>> getCombNomial(int n_max=4) {
    std::vector<std::vector<Real>> comb_nomial_array;

    for (int i = 0; i <= n_max; i++) {
        std::vector<Real> comb_row(n_max+1, 0.);    // ATTENTION! size of vector is n_max+1!
        for(int j=0; j<=n_max; j++){
            comb_row[j] = comb(i, j);
        }
        comb_nomial_array.push_back(comb_row);
    }
    return comb_nomial_array;
}

//  Total terms of series summation is equal to the size of vector [S_array]; More terms, more accurate;
//  Input [n] can only be 2, 3 & 4 three cases; 
AMREX_GPU_HOST_DEVICE
Real integral_func2_series(int n, double d, double c, const std::vector<std::vector<Real>>& comb_nomial_array, const std::vector<Real>& S_array, double delta=0.4) {

    long double int_val = 0.0;
    const long double factor = 16.0/6.0;
    long double ratio_cd = c/d;
    int N = S_array.size();

    for (int k = 0; k < N; k++) {
        long double exponent = (2*k+4)*d;
        long double inv_exponent = 1.0/exponent;
        long double inv_exponent2 = pow(inv_exponent, 2.);
        long double inv_exponent3 = pow(inv_exponent, 3.);
        long double inv_exponent4 = pow(inv_exponent, 4.);
        long double inv_exponent5 = pow(inv_exponent, 5.);

        long double val_term1, val_term2;

        // different cases for n = 2, 3 & 4
        if (n == 4) {
            val_term1 = (inv_exponent*pow(ratio_cd-delta, 4.) - 4*inv_exponent2*pow(ratio_cd-delta, 3.) + 12*inv_exponent3*pow(ratio_cd-delta, 2.) - 24*inv_exponent4*(ratio_cd-delta)
                         + 24*inv_exponent5)*exp(-exponent*delta) - 24*inv_exponent5*exp(-(2*k+4)*c);
            val_term2 = (inv_exponent*pow(ratio_cd+delta, 4.) + 4*inv_exponent2*pow(ratio_cd+delta, 3.) + 12*inv_exponent3*pow(ratio_cd+delta, 2.) + 24*inv_exponent4*(ratio_cd+delta)
                         + 24*inv_exponent5)*exp(-exponent*delta);
        } else if (n == 3) {
            val_term1 = (inv_exponent*pow(ratio_cd-delta, 3.) - 3*inv_exponent2*pow(ratio_cd-delta, 2.) + 6*inv_exponent3*(ratio_cd-delta) - 6*inv_exponent4
                         )*exp(-exponent*delta) + 6*inv_exponent4*exp(-(2*k+4)*c);
            val_term2 = (inv_exponent*pow(ratio_cd+delta, 3.) + 3*inv_exponent2*pow(ratio_cd+delta, 2.) + 6*inv_exponent3*(ratio_cd+delta) + 6*inv_exponent4)*exp(-exponent*delta);
        } else if (n == 2) {
            val_term1 = (inv_exponent*pow(ratio_cd-delta, 2.) - 2*inv_exponent2*(ratio_cd-delta) + 2*inv_exponent3)*exp(-exponent*delta)
                - 2*inv_exponent3*exp(-(2*k+4)*c);
            val_term2 = (inv_exponent*pow(ratio_cd+delta, 2.) + 2*inv_exponent2*(ratio_cd+delta) + 2*inv_exponent3)*exp(-exponent*delta);
        }
        //Print() << val_term1 + val_term2 << '\n'; // same with python code
        if(k%2==0){
            int_val += factor*(k+1)*(k+2)*(k+3)*(long double)(val_term1 + val_term2);
        }else{
            int_val = int_val - factor*(k+1)*(k+2)*(k+3)*(long double)(val_term1 + val_term2);
        }
        //Print() << int_val << '\n'; //  DIFFER with python code ??
        long double val_n = 0.0;
        for (int l = 0; l <= n; l++) {
            long double val_tmp = pow(delta, 2*k+l+1) - pow(-delta, 2*k+l+1);
            val_n += comb_nomial_array[n][l]*pow(c/d, n-l)*val_tmp/(2*k+l+1);
        }
        int_val += S_array[k]*pow(d, 2*k)*val_n;
    }

    return int_val;
}

AMREX_GPU_HOST_DEVICE
double integral_func3_series(int n, double c, int N=50) {
    double int_val = 0;
    for (int k = 1; k <= N; k++) {
        double k2 = k * k;
        double k3 = k2 * k;
        if (n == 3) {
            int_val += 6 * pow(-1, k+1) * (c/k2 + 1.0/4/k3 * exp(-2*k*c));
        } else if (n == 2) {
            int_val += pow(-1, k) * exp(-2*k*c)/k2 + pow(-1, k+1) * 2.0/k2;
        }
    }
    int_val += 2 * pow(c, n);
    return int_val;
}

// here n is restricted to only two cases: n = 2 & n=3
AMREX_GPU_HOST_DEVICE
double integral_func1_series(int n, double a, int N=100) {
    double int_val = 0.0;
    if (n == 3) {
        double val_sum_1 = 0.0;
        double val_sum_2 = 0.0;
        for (int k = 1; k <= N; k++) {
            double k2 = k * k;
            val_sum_1 += pow(-1, k+1)/k2 * exp(-2*k*a);
            val_sum_2 += pow(-1, k+1)/k2;
        }
        int_val = 1.5 * val_sum_1 - 3 * val_sum_2 - 3 * a * a;
    } else {
        int_val = -a - log(2) - log(cosh(a));
    }
    return int_val;
}

/*
std::vector<std::vector<Real>> combNomial = getCombNomial(4);
    std::vector<Real> S_array = getCoefS(NumOfTerms);
*/
AMREX_GPU_HOST_DEVICE
double JRn_Rn(const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
        double Wn, double Rn, double eta_R, double dt, double C0=1.) {
    double param_c = Rn / sqrt(2 * Wn);
    double param_d = 1.0;
    
    double JRn_Rn_val = -C0*eta_R * dt * sqrt(2 * Wn) * M_PI * 
                        integral_func2_series(2, param_d, param_c, combNomial, S_array, 1.0/param_d);
    return JRn_Rn_val;
}

AMREX_GPU_HOST_DEVICE
double JRn_Wn(const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
        double Wn, double Rn, double eta_R, double dt, double C0=1.) {
    double param_c = Rn / sqrt(2 * Wn);
    double param_d = 1.0;
    double Wn2 = Wn * Wn;
    double factor = C0*0.25 * eta_R * dt * M_PI / Wn2;
    return factor * (Rn * 2 * Wn * sqrt(2 * Wn) * 
           integral_func2_series(2, param_d, param_c, combNomial, S_array, 1.0/param_d) -
           4 * Wn2 * integral_func2_series(3, param_d, param_c, combNomial, S_array, 1.0/param_d));
}

AMREX_GPU_HOST_DEVICE
double JWn_Rn(const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
        double Wn, double Rn, double eta_W, double dt, double C0=1.) {
    double param_c = Rn / sqrt(2 * Wn);
    double param_d = 1.0;
    double factor = C0*0.25 * eta_W * dt;
    return factor * (2 * sqrt(2) * M_PI * Rn / sqrt(Wn) * 
           integral_func2_series(2, param_d, param_c, combNomial, S_array, 1.0/param_d) -
           4 * M_PI * integral_func2_series(3, param_d, param_c, combNomial, S_array, 1.0/param_d));
}

AMREX_GPU_HOST_DEVICE
double JWn_Wn(const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
        double Wn, double Rn, double eta_W, double dt, double C0=1.) {
    double param_c = Rn / sqrt(2 * Wn);
    double param_d = 1.0;
    double factor = -C0*0.125 * eta_W * dt * M_PI / pow(Wn, 3);
    double sqrt2Wn = sqrt(2. * Wn);
    double val_term1 = pow(sqrt2Wn, 3) * Rn * Rn * 
                      integral_func2_series(2, param_d, param_c, combNomial, S_array, 1.0/param_d);
    double val_term2 = pow(sqrt2Wn, 5) * 
                      integral_func2_series(4, param_d, param_c, combNomial, S_array, 1.0/param_d);
    double val_term3 = 2 * Rn * pow(sqrt2Wn, 4) * 
                      integral_func2_series(3, param_d, param_c, combNomial, S_array, 1.0/param_d);
    return factor * (val_term1 + val_term2 - val_term3);
}

AMREX_GPU_HOST_DEVICE
Real func_MfWn_mult(Real x, Real y, Real z, Real sqrt2Wn, Real Rn, const RealVect& r0){
    Real dx = x-r0[0];  Real dy = y-r0[1];  Real dz = z - r0[2];
    Real r_prime = sqrt(dx*dx + dy*dy + dz*dz);
    Real dist = Rn - r_prime;
    Real inv_cosh_val = inv_acosh(dist/sqrt2Wn);
    return dist*inv_cosh_val*inv_cosh_val;
}

AMREX_GPU_HOST_DEVICE
double MfWn(Function3DAMReX& func_rho, double Wn, double Rn, const RealVect& r0) {
    Real sqrt2Wn = sqrt(2.*Wn);

    const BoxArray& ba = func_rho.getBoxArray();
    const Geometry& geom = func_rho.getGeometry();
    const DistributionMapping& dm = func_rho.getDistributionMapping();
    int ncomp = func_rho.getnComp();
    int ngrow = func_rho.getnGrow();

    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    IndexType itype = ba.ixType(); // using Box or BoxArray;
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }
    //Print() << cell_size << '\n';
    MultiFab mfab_multiplicand(ba, dm, ncomp, ngrow);
    for (MFIter mfi(mfab_multiplicand); mfi.isValid(); ++mfi) {
        const Box& valid_box = mfi.validbox();
        Array4<Real> const& mult_array = mfab_multiplicand.array(mfi); // This function include ghost layers by default;
        ParallelFor(valid_box, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k){
            mult_array(i,j,k,0) = func_MfWn_mult((i+0.5*(1-itype[0]))*cell_size[0],(j+0.5*(1-itype[1]))*cell_size[1],
                                    (k+0.5*(1-itype[2]))*cell_size[2], sqrt2Wn, Rn, r0);
        });
    }
    
    Function3DAMReX func_multiplicand(mfab_multiplicand, geom);
    Real MfWn_val = func_rho.integral3D(func_multiplicand);

    return 1./pow(sqrt2Wn, 3.)*MfWn_val;
}

AMREX_GPU_HOST_DEVICE
Real func_MfRn_mult(Real x, Real y, Real z, Real sqrt2Wn, Real Rn, const RealVect& r0){
    Real dx = x-r0[0];  Real dy = y-r0[1];  Real dz = z - r0[2];
    Real r_prime = sqrt(dx*dx + dy*dy + dz*dz);
    Real dist = Rn - r_prime;
    Real inv_cosh_val = inv_acosh(dist/sqrt2Wn);
    return inv_cosh_val*inv_cosh_val;
}

AMREX_GPU_HOST_DEVICE
double MfRn(Function3DAMReX& func_rho, Real Wn, Real Rn, const RealVect& r0){
    Real sqrt2Wn = sqrt(2.*Wn);

    const BoxArray& ba = func_rho.getBoxArray();
    const Geometry& geom = func_rho.getGeometry();
    const DistributionMapping& dm = func_rho.getDistributionMapping();
    int ncomp = func_rho.getnComp();
    int ngrow = func_rho.getnGrow();

    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    IndexType itype = ba.ixType(); // using Box or BoxArray;
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }

    MultiFab mfab_multiplicand(ba, dm, ncomp, ngrow);
    for (MFIter mfi(mfab_multiplicand); mfi.isValid(); ++mfi) {
        const Box& valid_box = mfi.validbox();
        Array4<Real> const& mult_array = mfab_multiplicand.array(mfi); // This function include ghost layers by default;
        ParallelFor(valid_box, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k){
            mult_array(i,j,k,0) = func_MfRn_mult((i+0.5*(1-itype[0]))*cell_size[0],(j+0.5*(1-itype[1]))*cell_size[1],
                                    (k+0.5*(1-itype[2]))*cell_size[2], sqrt2Wn, Rn, r0);
        });
    }
    
    Function3DAMReX func_multiplicand(mfab_multiplicand, geom);
    Real MfRn_val = func_rho.integral3D(func_multiplicand);
    return MfRn_val/sqrt2Wn;
}

AMREX_GPU_HOST_DEVICE
Real KWn(Real Wn, Real Rn){
    Real sqrtWn = sqrt(Wn);
    Real sqrt2Wn = sqrt(2*Wn);
    Real param_c = Rn/sqrt2Wn;
    Real factor = sqrt(2)*M_PI/pow(sqrtWn, 3.);
    Real val_term1 = Rn*pow(sqrt2Wn, 3.)*integral_func3_series(2, param_c);
    Real val_term2 = 4.*Wn*Wn*integral_func3_series(3, param_c);
    Real val_term3 = Rn*pow(sqrt2Wn, 3.)*integral_func1_series(2, param_c);
    Real val_term4 = 4.*Wn*Wn*integral_func1_series(3, param_c);
    return factor*(val_term1 - val_term2 + val_term3 - val_term4);
}
    
AMREX_GPU_HOST_DEVICE
Real KRn(Real Wn, Real Rn){
    Real sqrt2Wn = sqrt(2.*Wn);
    Real param_c = Rn/sqrt2Wn;
    Real param_d = 1.;
    Real factor = 4.*M_PI*2.*Wn;
    Real val_term1 = integral_func3_series(2, param_c);
    Real val_term2 = integral_func1_series(2, param_c);
    return factor*(val_term1 + val_term2);
}

AMREX_GPU_HOST_DEVICE
void paramsVariations(Real* param_vars, const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
    Function3DAMReX& func_rho, Real Wn, Real Rn, Real eta_W, Real eta_R, Real dt, double C0=1.){

    Real matA[2][2] = { {0., 0.}, 
                        {0., 0.}};
    Real matB[2][2] = { {0., 0.}, 
                        {0., 0.}};
    Real vecC[2] = {0., 0.};
    Real coef_JRn_Rn = JRn_Rn(combNomial, S_array, Wn, Rn, eta_R, dt, C0);
    Real coef_JWn_Rn = JWn_Rn(combNomial, S_array, Wn, Rn, eta_W, dt, C0);
    Real coef_JRn_Wn = JRn_Wn(combNomial, S_array, Wn, Rn, eta_R, dt, C0);
    Real coef_JWn_Wn = JWn_Wn(combNomial, S_array, Wn, Rn, eta_W, dt, C0);
    //Print() << coef_JWn_Wn << '\n';
    matA[0][0] = 1 - coef_JRn_Rn;
    matA[0][1] = coef_JWn_Rn;
    matA[1][0] = coef_JRn_Wn;
    matA[1][1] = 1 - coef_JWn_Wn;
    // matrix B assembly
    matB[0][0] = -eta_W*dt;
    matB[1][1] = eta_R*dt;
    // vector C assembly
    RealVect vec_r0 = {0., 0., 0.};
    getCenterOfMass(vec_r0, func_rho);  //  of size 3*1, center of mass position vector

    vecC[0] = MfWn(func_rho,Wn,Rn,vec_r0) - 0.5*KWn(Wn,Rn);
    vecC[1] = MfRn(func_rho,Wn,Rn,vec_r0) - 0.5*KRn(Wn,Rn);
    //  calculate determinant
    Real detD = (1.-coef_JWn_Wn)*(1.-coef_JRn_Rn) - coef_JWn_Rn*coef_JRn_Wn;
    param_vars[0] = 0.; param_vars[1] = 0.; //  param_vars = 1/detD * matA @ matB @ vecC
    for(int k=0; k<2; k++){
        for(int l=0; l<2; l++){
            param_vars[0] += matA[0][k] * matB[k][l] * vecC[l];
            param_vars[1] += matA[1][k] * matB[k][l] * vecC[l];
        }
    }
    param_vars[0] = param_vars[0]/detD;
    param_vars[1] = param_vars[1]/detD;
}


#endif


