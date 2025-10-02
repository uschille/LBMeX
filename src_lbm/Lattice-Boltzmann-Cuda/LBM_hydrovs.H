#ifndef LBM_HYDRO_
#define LBM_HYDRO_
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sys/stat.h>
#include "LBM_binary.H"
#include "Debug.H"
#include "externlib.H"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

const Real MIN_LEN_SCALE = 1e-6;

extern void paramsVariations(Real* param_vars, const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
    Function3DAMReX& func_rho, Real Wn, Real Rn, Real eta_W, Real eta_R, Real dt, double C0);

extern const int NumOfTerms;
extern std::vector<Real> getCoefS(int N);
extern std::vector<std::vector<Real>> getCombNomial(int n_max);

AMREX_GPU_HOST_DEVICE
void getCenterOfMass(RealVect& vec_com, Function3DAMReX& func_rho, Real*** wt=NULL, bool freeMem=true){
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

    MultiFab mfab_X(ba, dm, ncomp, ngrow);
    MultiFab mfab_Y(ba, dm, ncomp, ngrow);
    MultiFab mfab_Z(ba, dm, ncomp, ngrow);
    for (MFIter mfi(mfab_X); mfi.isValid(); ++mfi) {
        const Box& valid_box = mfi.validbox();
        Array4<Real> const& X_array = mfab_X.array(mfi); // This function include ghost layers by default;
        Array4<Real> const& Y_array = mfab_Y.array(mfi);
        Array4<Real> const& Z_array = mfab_Z.array(mfi);
        ParallelFor(valid_box, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k){
            X_array(i,j,k,0) = (i+0.5*(1-itype[0]))*cell_size[0];
            Y_array(i,j,k,0) = (j+0.5*(1-itype[1]))*cell_size[1];
            Z_array(i,j,k,0) = (k+0.5*(1-itype[2]))*cell_size[2];
        });
    }
    Function3DAMReX func_X(mfab_X, geom);
    Function3DAMReX func_Y(mfab_Y, geom);
    Function3DAMReX func_Z(mfab_Z, geom);


    if(wt==NULL){
        Integration::trapezoid3DWeightTensor(geom.Domain(), &wt);
    }
    Real mass = func_rho.integral3D(wt);
    amrex::ParallelDescriptor::Barrier();
    vec_com[0] = func_rho.integral3D(func_X, wt)/mass;
    amrex::ParallelDescriptor::Barrier();
    vec_com[1] = func_rho.integral3D(func_Y, wt)/mass;
    amrex::ParallelDescriptor::Barrier();
    vec_com[2] = func_rho.integral3D(func_Z, wt)/mass;
    amrex::ParallelDescriptor::Barrier();

    if(freeMem){    Integration::free_3d_array(wt, box[0], box[1], box[2]); }
}

AMREX_GPU_HOST_DEVICE
void fittingDroplet(Function3DAMReX& func_rho, std::vector<Array<Real, 2>>& param_vec,
    const std::vector<std::vector<Real>>& combNomial, const std::vector<Real>& S_array,
    Real W0=0.02, Real R0=0.3, Real eta_W=0.2, Real eta_R=0.2, Real dt=0.02, int Nstep= 500, bool printDetails=false){    // kappa := 2W;
    
    param_vec.resize(Nstep);
    Array<Real, 2> params;  params[0] = W0; params[1] = R0;
    param_vec[0] = params;
    Real param_arr[2];
    Real Wn = W0;   Real Rn = R0;
    // get the density variation range
    const MultiFab& rho_mfab = func_rho.getMultiFab();
    Real C0 = rho_mfab.max(0) - rho_mfab.min(0);
    for(int k=1; k<Nstep; k++){
        paramsVariations(param_arr, combNomial, S_array, func_rho, Wn, Rn, eta_W, eta_R, dt, C0);
        Wn = Wn + param_arr[0];
        Rn = Rn + param_arr[1];
        if(Wn<=0){  // !! handling exceptional case where Wn < 0 due to large updating time step !!
            Wn = Wn - param_arr[0]; // reset to positive values & update learning rate for Wn
            dt = dt/5.;
        }
        if(std::abs(Wn) < MIN_LEN_SCALE){ //  reset if cannot update anymore;
            Wn = W0;    
        }
        Array<Real, 2> params;  params[0] = Wn; params[1] = Rn;
        param_vec[k] = params;
        if(printDetails){
            printf("(Wn=%.3e,Rn=%f)\n", Wn, Rn);
        }
    }
}

//  param_vec[k]: k'the step parameter pair: (Wk, Rk);
AMREX_GPU_HOST_DEVICE
void fittingDroplet(Function3DAMReX& func_rho, std::vector<Array<Real, 2>>& param_vec,
    Real W0=0.02, Real R0=0.3, Real eta_W=0.2, Real eta_R=0.2, Real dt=0.02, int Nstep=300){

    const std::vector<std::vector<Real>> combNomial = getCombNomial(4); // verified
    const std::vector<Real> S_array = getCoefS(NumOfTerms); // verified
    fittingDroplet(func_rho, param_vec, combNomial, S_array, W0, R0, eta_W, eta_R, dt, Nstep);
}

// last [step_window] steps ensemble;   return [W, R, err];
AMREX_GPU_HOST_DEVICE
Array<Real, 3> fittingDropletParams(Function3DAMReX& func_rho, int step_window=30, Real undul_ratio=0.005,
    int Nstep=400, Real W0=0.02, Real R0=0.3, Real eta_W=0.2, Real eta_R=0.2, Real dt=0.02){

    std::vector<Array<Real, 2>> param_vec;

    amrex::ParallelDescriptor::Barrier();
    fittingDroplet(func_rho, param_vec, W0, R0, eta_W, eta_R, dt, Nstep);
    amrex::ParallelDescriptor::Barrier();

    int startIdx = Nstep - step_window;
    int endIdx = Nstep; // not included by default;    
    Array<Real, 2> param_mean = getArrayStatistics<2>(param_vec, 0, startIdx, endIdx);
    Array<Real, 2> param_max = getArrayStatistics<2>(param_vec, 2, startIdx, endIdx);
    Array<Real, 2> param_min = getArrayStatistics<2>(param_vec, 3, startIdx, endIdx);
    Real undul_ratio_data_W = (param_max[0]-param_min[0])/param_mean[0];
    Real undul_ratio_data_R = (param_max[1]-param_min[1])/param_mean[1];
    //printf("param_mean_W = %f\tparam_mean_R = %f\n", param_mean[0], param_mean[1]);
    //printf("delta_param_W = %f\tdelta_param_R = %f\n", param_max[0]-param_min[0], param_max[1]-param_min[1]);
    int Max_Iter = 10;   int iter_idx = 1;
    Real dt_new = dt/5.;
    while(iter_idx<=Max_Iter && !(undul_ratio_data_W<=undul_ratio && undul_ratio_data_R<=undul_ratio)){
        if(iter_idx == 1){
            Print() << "Fitting Parameters does not fall below convergent bound, try at most " << Max_Iter << " times..." << '\n';
        }
        printf("trial times %d with W0 = %f, R0 = %f, dt = %f ...\n", iter_idx, param_mean[0], param_mean[1], dt_new);

        amrex::ParallelDescriptor::Barrier();
        fittingDroplet(func_rho, param_vec, param_mean[0], param_mean[1], eta_W, eta_R, dt_new, Nstep);
        amrex::ParallelDescriptor::Barrier();

        param_mean = getArrayStatistics<2>(param_vec, 0, startIdx, endIdx);
        param_max = getArrayStatistics<2>(param_vec, 2, startIdx, endIdx);
        param_min = getArrayStatistics<2>(param_vec, 3, startIdx, endIdx);
        undul_ratio_data_W = (param_max[0]-param_min[0])/param_mean[0];
        undul_ratio_data_R = (param_max[1]-param_min[1])/param_mean[1];
        iter_idx++;
        dt_new = dt_new/5.;
    }
    param_mean = getArrayStatistics<2>(param_vec, 0, startIdx, endIdx);
    param_max = getArrayStatistics<2>(param_vec, 2, startIdx, endIdx);
    param_min = getArrayStatistics<2>(param_vec, 3, startIdx, endIdx);
    undul_ratio_data_W = (param_max[0]-param_min[0])/param_mean[0];
    undul_ratio_data_R = (param_max[1]-param_min[1])/param_mean[1];
    if(!(undul_ratio_data_W<=undul_ratio && undul_ratio_data_R<=undul_ratio)){
        char str_err[200];
        sprintf(str_err, "statistical undulation (%.2e, %.2e) out of bounds! (called from function 'fittingDropletParams')\n",
        undul_ratio_data_W, undul_ratio_data_R);
        throw runtime_error(str_err);
    }
    Array<Real, 3> param_arr;
    param_arr[0] = param_mean[0];   param_arr[1] = param_mean[1];   param_arr[2] = std::max(undul_ratio_data_W, undul_ratio_data_R);
    return param_arr;
}

// generate density multifab based on "tanh" profile:
//                      rho := \rho_{f}\left(\bm{r}\right)\sim\frac{1}{2}\left(1+\tanh\frac{R-\left|\bm{r}-\bm{r}_{0}\right|}{\sqrt{2W}}\right)
AMREX_GPU_HOST_DEVICE
void fitting_density_mfab(MultiFab& rho_eq, MultiFab& phi_eq, MultiFab& rhot_eq, const Geometry geom,
    Real W, Real R, const RealVect& r0){
    
    Real sqrt2W = sqrt(2.*W);
    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    IndexType itype = rho_eq.boxArray().ixType(); // using Box or BoxArray;
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }

    for (MFIter mfi(rho_eq); mfi.isValid(); ++mfi) {
        const Box& valid_box = mfi.validbox();
        Array4<Real> const& rho_array = rho_eq.array(mfi); // This function include ghost layers by default;
        Array4<Real> const& phi_array = phi_eq.array(mfi);
        Array4<Real> const& rhot_array = rhot_eq.array(mfi);
        ParallelFor(valid_box, [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k){
            Real dx = (i+0.5*(1-itype[0]))*cell_size[0] - r0[0];
            Real dy = (j+0.5*(1-itype[1]))*cell_size[1] - r0[1];
            Real dz = (k+0.5*(1-itype[2]))*cell_size[2] - r0[2];
            Real dist = dx*dx + dy*dy + dz*dz;
            rho_array(i,j,k,0) = 0.5*(1 + tanh((R-dist)/sqrt2W));
            phi_array(i,j,k,0) = 1. - rho_array(i,j,k,0);
            rhot_array(i,j,k,0) = 1.;
        });
    }
}

/*
    [rho_f]:    each component represents one frame;
    [R0]:       equilibrium radius, i.e., V = 4\pi/3*R0^3;
*/
AMREX_GPU_HOST_DEVICE
std::vector<Array<Real, 3>> fittingDropletCovariance(const Geometry& geom, const MultiFab& rho_f, Real R0){
    std::vector<Array<Real, 3>> CovarMat;
    // get cell size based on data index types;
    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    IndexType itype = rho_f.boxArray().ixType(); // using Box or BoxArray;
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }

    int nframes = rho_f.nComp();
    const BoxArray ba = rho_f.boxArray();
    const DistributionMapping dm = rho_f.DistributionMap();
    const int ngrow = rho_f.nGrow();
    MultiFab rho_f_frame(ba, dm, 1, ngrow);
    RealVect vec_com = {0., 0., 0.};    //  center-of-mass
    
    Real*** wt;
    Integration::trapezoid3DWeightTensor(geom.Domain(), &wt);
    Eigen::MatrixXd covar_mat(3, 3);
    //std::array<std::array<Real, 3>, 3> covar_mat = {0.};
    for(int n=0; n<nframes; n++){
        amrex::MultiFab::Copy(rho_f_frame, rho_f, n, 0, 1, ngrow);
        Real mass = rho_f_frame.sum(0); // no ghost cells are included for sum(0)
        Function3DAMReX func3D_rho_f(rho_f_frame, geom);
        getCenterOfMass(vec_com, func3D_rho_f, wt, false);  // "false" --> do not free weight matrix [wt];
        //Print() << vec_com << '\n';
        for (MFIter mfi(rho_f_frame); mfi.isValid(); ++mfi) {
            const Box& valid_box = mfi.validbox();
            Array4<Real> const& rho_f_frame_ptr = rho_f_frame.array(mfi); // This function include ghost layers by default;
            ParallelFor(valid_box, [&] AMREX_GPU_HOST_DEVICE(int i, int j, int k){
                Real m_i = rho_f_frame_ptr(i,j,k,0);
                RealVect pos = {(i+0.5*(1-itype[0]))*cell_size[0], (j+0.5*(1-itype[1]))*cell_size[1], (k+0.5*(1-itype[2]))*cell_size[2]};
                for(int alpha=0; alpha<3; alpha++){
                    for(int beta=alpha; beta<3; beta++){
                        covar_mat(alpha, beta) += m_i*(pos[alpha] - vec_com[alpha])*(pos[beta] - vec_com[beta]);
                    }
                }
            });
        }
        for(int alpha=0; alpha<3; alpha++){
            for(int beta=alpha; beta<3; beta++){
                covar_mat(alpha, beta) = covar_mat(alpha, beta)/mass;
            }
        }
        for(int alpha=0; alpha<3; alpha++){
            for(int beta=0; beta<alpha; beta++){
                covar_mat(alpha, beta) = covar_mat(beta, alpha);
            }
        }
        /*for(int alpha=0; alpha<3; alpha++){
            printf("(%f,%f,%f)\n", covar_mat[alpha][0], covar_mat[alpha][1], covar_mat[alpha][2]);
        }*/    
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(covar_mat);
        std::cout << "The eigenvalues of A are:" << std::endl << es.eigenvalues() << std::endl;
        if(Eigen::Success == es.info()){
            Real residue = (covar_mat*es.eigenvectors().col(0) - es.eigenvalues()[0]*es.eigenvectors().col(0)).norm();
            printf("successful with residue norm = %.2e\n", residue);
        }
        
    }
    
    return CovarMat;
}


/*std::vector<Array<Real, 2>> fittingDropletRadiusAnsys(const std::vector<Array<Real, 2>>& param_vec, int step_window=50){
    int Nstep = param_vec.size();
    for(int k=0; k<Nstep; k++){

    }
    
}*/





    


#endif
