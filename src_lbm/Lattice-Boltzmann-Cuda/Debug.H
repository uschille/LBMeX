#ifndef LBM_DEB_
#define LBM_DEB_
#include "LBM_binary.H"
#include <cmath>

const Real PI = 3.1415926;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void loopNaN(int x, int y, int z, int ncomp, const Array4<Real>& mfab_multi_array4D,
    bool* iftrue_device_ptr, bool* ifPrintNANGrid){

    if(std::isnan(mfab_multi_array4D(x,y,z,ncomp))){
        if(*ifPrintNANGrid){
            printf("mfab(%d,%d,%d,%d) = %f\t", x,y,z,ncomp, mfab_multi_array4D(x,y,z,ncomp));
        }
        *iftrue_device_ptr = true;
    }
}


AMREX_GPU_HOST
bool loopNaN(MultiFab& mfab, bool ifPrintNANGrid){
    auto const & mfab_multi_array4D = mfab.arrays();
    int ncomp = mfab.nComp();
    amrex::Gpu::DeviceScalar<bool> ifPrintNANGrid_device(ifPrintNANGrid); 
    bool iftrue_host = false;   // default return result is "is not NaN"
    amrex::Gpu::DeviceScalar<bool> iftrue_device(false); // Device-side boolean
    bool* iftrue_device_ptr = iftrue_device.dataPtr();
    bool* ifPrintNANGrid_device_ptr = ifPrintNANGrid_device.dataPtr();

    for(int n=0; n<ncomp; n++){
        // reference [&] use instead of [=] value in Lambda expression;
        ParallelFor(mfab, IntVect(0), [=] AMREX_GPU_HOST_DEVICE(int nbx, int x, int y, int z) {
            loopNaN(x,y,z,n,mfab_multi_array4D[nbx],iftrue_device_ptr,ifPrintNANGrid_device_ptr);
        });
    }
    // Copy back the result
    amrex::Gpu::synchronize();
    iftrue_host = iftrue_device.dataValue();

    return iftrue_host;
}

// check if [mfab] has NAN errors for all elements; 
// print lattice points where NAN values appear if ifPrintNANGrid = true; 
// if nstep>0 information is not provided, it will only check the NAN existence without printing step info;
AMREX_GPU_HOST AMREX_FORCE_INLINE
bool MultiFabNANCheck(MultiFab& mfab, bool ifPrintNANGrid, int nstep = -1) {
    if(loopNaN(mfab, ifPrintNANGrid)){
        if(nstep>=0){
            Print() << "NAN accurs at step " << nstep << " stop and check!" << '\n';
            exit(0);
        }else{
            Print() << "The MultiFab has NAN elements" << '\n'; return true;
        }
    }else{
        return false;
    }
}


AMREX_GPU_HOST AMREX_FORCE_INLINE
Array1D<Real,0,2> compute_multifab_fluctuation(MultiFab& mfab, const int ncomp){
    // calculate the fluctuation of component [ncomp]
    amrex::Gpu::DeviceScalar<amrex::Real> mean_device(0.0);
    amrex::Gpu::DeviceScalar<amrex::Real> var_device(0.0);
    amrex::Gpu::DeviceScalar<int> N_device(0);

    // Get device pointers for reduction
    amrex::Real* mean_ptr = mean_device.dataPtr();
    amrex::Real* var_ptr = var_device.dataPtr();
    int* N_ptr = N_device.dataPtr();

    // Get the array of arrays for the MultiFab
    auto const& mfab_multi_array4D = mfab.arrays();

    // Launch the kernel to compute mean and count
    amrex::ParallelFor(mfab, amrex::IntVect(0), [=] AMREX_GPU_HOST_DEVICE(int nbx, int x, int y, int z) {
        amrex::Real value = mfab_multi_array4D[nbx](x, y, z, ncomp);
        amrex::Gpu::Atomic::Add(mean_ptr, value);
        amrex::Gpu::Atomic::Add(N_ptr, 1);
    });
    // Synchronize to ensure the kernel has completed
    amrex::Gpu::synchronize();
    // Copy results back to the host
    amrex::Real mean = mean_device.dataValue();
    int N = N_device.dataValue();
    // Compute the mean
    if (N > 0) {
        mean /= N;
    }

    // Launch the kernel to compute variance
    amrex::ParallelFor(mfab, amrex::IntVect(0), [=] AMREX_GPU_HOST_DEVICE(int nbx, int x, int y, int z) {
        amrex::Real value = mfab_multi_array4D[nbx](x, y, z, ncomp);
        amrex::Real diff = value - mean;
        amrex::Gpu::Atomic::Add(var_ptr, diff * diff);
    });
    // Synchronize to ensure the kernel has completed
    amrex::Gpu::synchronize();
    // Copy the variance result back to the host
    amrex::Real var = var_device.dataValue();
    // Compute the variance
    if (N > 0) {
        var /= N;
    }
    Array1D<Real,0,2> density_info = {mean, sqrt(var)};
    return density_info;
}


// [densityIdx] contains index of density needed to print; set -1 if empty for that site, i.e., {5, -1, -1};
// Print step info if [nstep] is provided;
// Compute the total fluctuation for all grid points of a single multifab; measure of the data uniformity
// return Array2D object: (comp_index={k:densityIdx[k], k=0,1,2}, info_index={0: mean; 1: standard deviation})
AMREX_GPU_HOST AMREX_FORCE_INLINE
Array2D<Real,0,3,0,2> PrintDensityFluctuation(MultiFab& hydrovs, Vector<std::string> VariableNames,
    int nstep = -1, const IntVect densityIdx = {0,1,5}){
    Array2D<Real,0,3,0,2> hydrovs_info;
    for(int n=0; n<3; n++){
        if(densityIdx[n]>=0){
            Array1D<Real,0,2> rho_info = compute_multifab_fluctuation(hydrovs, densityIdx[n]);
            Print() << VariableNames[densityIdx[n]] << ": " << "(mean = " << rho_info(0) << ", ";
            Print() << "standard deviation = " << rho_info(1) << ") ";
            hydrovs_info(n, 0) = rho_info(0);
            hydrovs_info(n, 1) = rho_info(1);
            if(nstep>=0){
                Print() << "at step " << nstep << '\n';
            }else{
                Print() << '\n';
            }
        }
    }
    return hydrovs_info;
}

// [densityIdx] contains index of density needed to print; set -1 if empty for that site, i.e., {5, -1, -1};
// Generally called at last step
AMREX_GPU_HOST AMREX_FORCE_INLINE
void PrintMassConservation(MultiFab& hydrovs, Vector<std::string> VariableNames, 
    Real L, Real radius, const IntVect densityIdx = {0,1,5}, int tag = 0){
    for(int n=0; n<3; n++){
        if(n == tag){
            Array1D<Real,0,2> rho_info;
            if(densityIdx[n]>=0){
                rho_info = compute_multifab_fluctuation(hydrovs, densityIdx[n]);
                Print() << VariableNames[densityIdx[n]] << ": " << "(mean = " << rho_info(0) << ", ";
                Print() << "standard deviation = " << rho_info(1) << ") ";
            }
            if(tag == 0){
                Real ratio_rho_f = rho_info(0)*L*L*L/(4./3.*PI*radius*radius*radius);
                Print() << "(rho_f*L^3)/(droplet volume): " << ratio_rho_f << '\n';
            }
        }
    }
}

/*  
    amrex::MultiFabFileFullPrefix():    e.g., return "** /plt[nstep]/Level_[nlevel]/Cell"
    plotfilename (e.g., "** /plt[nstep]") is assembled by function std::string amrex::Concatenate(const std::string& root, int num, int mindigits)
    (returns rootNNNN where NNNN == num) using input "plot_file_root"
    This function calculates the p-like norm of the density variations (ensemble average mean substracted) matrix from [step1] ~ [step2] (both included).
    ifSave: indicator of whether save the ensemble average hydrovs component into [mfab_ref]; false (do not save) by default;
*/		
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void print_convergence_mean(int x, int y, int z, int n,
    Array4<Real> const& hydrovs_readin_frames_array4, Array4<Real> const& hydrovs_mean_array4){
    hydrovs_mean_array4(x,y,z,0) = hydrovs_mean_array4(x,y,z,0) + hydrovs_readin_frames_array4(x,y,z,n);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void print_convergence_deviation(int x, int y, int z, int n,
    Array4<Real> const& hydrovs_readin_frames_array4, Array4<Real> const& hydrovs_mean_array4,
    Array4<Real> const& hydrovs_deviation_array4){
    hydrovs_deviation_array4(x,y,z,0) = hydrovs_deviation_array4(x,y,z,0)
        + std::abs(hydrovs_readin_frames_array4(x,y,z,n) - hydrovs_mean_array4(x,y,z,0));
}

AMREX_GPU_HOST AMREX_FORCE_INLINE
void PrintConvergence(std::string plot_file_root, int step1, int step2, int step_int, MultiFab& mfab_ref, int compIdx=0,int p_tag=1, bool ifSave=false, 
            int nlevel=0, int mindigits=5, const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){
    std::string File(amrex::Concatenate(plot_file_root,step1,mindigits) + "/Header");
    //VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr); // char stream of [File]
    /*for (int i = 0; i < fileCharPtr.size(); ++i) {
        std::cout << fileCharPtr[i] << " ";
    }*/
    std::string fileCharPtrString(fileCharPtr.dataPtr()); // get access to the underlying data pointer, ONLY for Vector in AMRex; same with .data() here;
    std::istringstream is(fileCharPtrString, std::istringstream::in);
    std::string line, word;
    // read in title line, meaningless generally;
    std::getline(is, line);
    // read in finest_level
    int numComp;
    is >> numComp;
    Print() << "num of components: " << numComp << '\n';
    
    Print() << "Convergence test emsemble selection: from step " << step1 << " to step " << step2 << " with step interval " << step_int << '\n';
    int Nsteps = (step2 - step1)/step_int + 1;

    // [hydrovs_readin_frames] used for storing read-in multifabs' component [compIdx] (for frames ensemble average)
    MultiFab hydrovs_readin_frames(mfab_ref.boxArray(), mfab_ref.DistributionMap(), Nsteps, mfab_ref.nGrow());
    MultiFab hydrovs_readin_tmp(mfab_ref.boxArray(), mfab_ref.DistributionMap(), numComp, mfab_ref.nGrow());
    MultiFab hydrovs_mean(mfab_ref.boxArray(), mfab_ref.DistributionMap(), 1, mfab_ref.nGrow());
    hydrovs_mean.setVal(0., 0, 1); // set values 0. from comp 0, total 1 component;
    MultiFab hydrovs_deviation(mfab_ref.boxArray(), mfab_ref.DistributionMap(), 1, mfab_ref.nGrow());
    hydrovs_deviation.setVal(0., 0 , 1);
    // Read in each frame from Disk;
    for(int n=0; n<Nsteps; n++){
        int nstep = step1 + step_int*n;
        // construct multifab object with same params with given [mfab_ref];
        const std::string& checkpointname = amrex::Concatenate(plot_file_root,nstep,mindigits);
        // read in each frame and temperally save it into [hydrovs_readin_tmp]
        VisMF::Read(hydrovs_readin_tmp, amrex::MultiFabFileFullPrefix(nlevel, checkpointname, levelPrefix, mfPrefix));
        // copy each frame's component [compIdx] from temp multifab to [hydrovs_readin_array]
        hydrovs_readin_frames.ParallelCopy(hydrovs_readin_tmp, compIdx, n, 1); 
    }
    // calculate the mean & standard deviation multifab for given component index [compIdx]
    for (MFIter mfi(hydrovs_readin_frames,TilingIfNotGPU()); mfi.isValid(); ++mfi){
        const Box& bx = mfi.tilebox();
        Array4<Real> const& hydrovs_readin_frames_array4 = hydrovs_readin_frames[mfi].array();
        Array4<Real> const& hydrovs_mean_array4 = hydrovs_mean[mfi].array();
        ParallelFor(bx, hydrovs_readin_frames.nComp(), [=] AMREX_GPU_HOST_DEVICE (int x, int y, int z, int n){
            print_convergence_mean(x, y, z, n, hydrovs_readin_frames_array4, hydrovs_mean_array4);
        });
    }
    // multiplication for component 0, total 1 components with [mfab_ref.nGrow()] ghost layer;
    hydrovs_mean.mult(1./Nsteps, 0, 1, mfab_ref.nGrow()); 
    for (MFIter mfi(hydrovs_readin_frames,TilingIfNotGPU()); mfi.isValid(); ++mfi){
        const Box& bx = mfi.tilebox();
        Array4<Real> const& hydrovs_readin_frames_array4 = hydrovs_readin_frames[mfi].array();
        Array4<Real> const& hydrovs_mean_array4 = hydrovs_mean[mfi].array();
        Array4<Real> const& hydrovs_deviation_array4 = hydrovs_deviation[mfi].array();
        ParallelFor(bx, hydrovs_readin_frames.nComp(), [=] AMREX_GPU_HOST_DEVICE (int x, int y, int z, int n){
            print_convergence_deviation(x, y, z, n, hydrovs_readin_frames_array4, hydrovs_mean_array4, hydrovs_deviation_array4);
        });
    }
    hydrovs_deviation.mult(1./Nsteps);
    // if passing in true, then using the ensemble average mean as equilibrium state solution and save it into [mfab_ref]
    if(ifSave){
        mfab_ref.copy(hydrovs_mean, 0, 0, 1);   // derived from BaseFab member function;
    }

    // calculate the p-residues (p=1 or \infty) for ensemble MultiFabs stored in array [hydrovs_readin_array];
    // Here mfab data are considered as reshaped vectors;
    // ||A||_1: 1/N*\sum |aijk| and ||A||_\infty: max_{ijk} |aijk|
    const BoxArray& ba = mfab_ref.boxArray();
    Box domain_box = ba.minimalBox();
    IntVect total_size = domain_box.size();
    switch(p_tag) {
        case 1:
            Print() << hydrovs_deviation.sum()/(total_size[0]*total_size[1]*total_size[2]) << '\n';
            break;
        case -1:    // for p=\infty
            Print() << hydrovs_deviation.max(0) << '\n';    // maximum value for component 0;
            break;
    }
}

AMREX_GPU_HOST AMREX_FORCE_INLINE
void WriteVectorToFile(const std::vector<double>& data, const std::string& filename) {
    // Open the file for writing
    std::ofstream outFile(filename, std::ios::app | std::ios::out);
    // Check if the file was opened successfully
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    // Write the vector elements to the file, separated by '\t'
    for (size_t i = 0; i < data.size(); ++i) {
        outFile << data[i];
        //if (i < data.size() - 1) {    // auto add '\t' at the end of the file; for append mode
            outFile << '\t';  // Add tab separator between elements
        //}
    }
    // Close the file
    outFile.close();
}

AMREX_GPU_HOST AMREX_FORCE_INLINE
void WriteOutNoise(string plot_file_root, int step,
			const MultiFab& fnoise, const MultiFab& gnoise,
			const Geometry& geom, const int Ndigits,
            const Vector<std::string>& var_names_f = {},
            const Vector<std::string>& var_names_g = {}) {

  const Real time = step;
  int pltlen = plot_file_root.length();
  const int ncomp = fnoise.nComp();
  string plot_file_root_fnoise = plot_file_root.substr(0, pltlen-3); // 15; substract the "plt" length;
  plot_file_root_fnoise = plot_file_root_fnoise + "data_fnoise/fn";
  const std::string& pltfile_fn = amrex::Concatenate(plot_file_root_fnoise,step,Ndigits);
  Vector <std::string> vname;
  for(int n=0; n<ncomp; n++){
    string compname = var_names_f.empty()? format("fa%d", n) : var_names_f[n];
    vname.push_back(compname);
  }
  WriteSingleLevelPlotfile(pltfile_fn, fnoise, vname, geom, time, step);

  string plot_file_root_gnoise = plot_file_root.substr(0, pltlen-3);
  plot_file_root_gnoise = plot_file_root_gnoise + "data_gnoise/gn";
  const std::string& pltfile_gn = amrex::Concatenate(plot_file_root_gnoise,step,Ndigits);
  vname.clear();
  for(int n=0; n<ncomp; n++){
    string compname = var_names_g.empty()? format("ga%d", n) : var_names_g[n];
    vname.push_back(compname);
  }
  WriteSingleLevelPlotfile(pltfile_gn, gnoise, vname, geom, time, step);
}

#ifdef AMREX_USE_CUDA
AMREX_GPU_HOST AMREX_FORCE_INLINE
void PrintCudaMem(string str_tag){
    size_t free_mem, total_mem, used_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    used_mem = total_mem - free_mem;

    std::cout << str_tag << " ----- GPU Memory Usage:\n";
    std::cout << "Free: " << free_mem / (1024 * 1024) << " MB\n";
    std::cout << "Total: " << total_mem / (1024 * 1024) << " MB\n";
    std::cout << "Used: " << used_mem / (1024 * 1024) << " MB\n";
}
#endif


#endif




/*





AMREX_GPU_HOST AMREX_FORCE_INLINE
Array1D<Real,0,2> compute_multifab_fluctuation(MultiFab& mfab, const int ncomp){
    // calculate the fluctuation of component [ncomp]
    amrex::Gpu::DeviceScalar<amrex::Real> mean_device(0.0);
    amrex::Gpu::DeviceScalar<amrex::Real> var_device(0.0);
    amrex::Gpu::DeviceScalar<int> N_device(0);

    // Get device pointers for reduction
    amrex::Real* mean_ptr = mean_device.dataPtr();
    amrex::Real* var_ptr = var_device.dataPtr();
    int* N_ptr = N_device.dataPtr();

    // Get the array of arrays for the MultiFab
    auto const& mfab_multi_array4D = mfab.arrays();

    // Launch the kernel to compute mean and count
    amrex::ParallelFor(mfab, amrex::IntVect(0), [=] AMREX_GPU_HOST_DEVICE(int nbx, int x, int y, int z) {
        amrex::Real value = mfab_multi_array4D[nbx](x, y, z, ncomp);
        amrex::Gpu::Atomic::Add(mean_ptr, value);
        amrex::Gpu::Atomic::Add(N_ptr, 1);
    });
    // Synchronize to ensure the kernel has completed
    amrex::Gpu::synchronize();
    // Copy results back to the host
    amrex::Real mean = mean_device.dataValue();
    int N = N_device.dataValue();
    // Compute the mean
    if (N > 0) {
        mean /= N;
    }

    // Launch the kernel to compute variance
    amrex::ParallelFor(mfab, amrex::IntVect(0), [=] AMREX_GPU_HOST_DEVICE(int nbx, int x, int y, int z) {
        amrex::Real value = mfab_multi_array4D[nbx](x, y, z, ncomp);
        amrex::Real diff = value - mean;
        amrex::Gpu::Atomic::Add(var_ptr, diff * diff);
    });
    // Synchronize to ensure the kernel has completed
    amrex::Gpu::synchronize();
    // Copy the variance result back to the host
    amrex::Real var = var_device.dataValue();
    // Compute the variance
    if (N > 0) {
        var /= N;
    }
    Array1D<Real,0,2> density_info = {mean, sqrt(var)};
    return density_info;
}


*/