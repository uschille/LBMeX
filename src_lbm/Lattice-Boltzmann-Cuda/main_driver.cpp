#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
//#include <format>
/*
  The definition of the correlation <,> is opposite to the paper, in which the conjugate operator is on left <var^*(k), var(k)>;
  In FHDeX: 1. calculate rho(k) for each frame and sum them with frames increasing;
            2. then remove the zero-freq components by Shift func
            3. compute the mean <rho(k)> for selected frames;
            2. compute the magnitude of <rho(k)> ?????
*/

//  data backup cmd:  cp -rp ./data_droplet* /Volumes/Extreme\ SSD/Lattice-Boltzmann-Data/

#include <StructFact.H>


using namespace amrex;

#include "LBM_binary.H"
#include "Debug.H"
#include "AMReX_FileIO.H"
//#include "externlib.H"
//#include "AMReX_Analysis.H"
//#include "LBM_hydrovs.H"
//#include "AMReX_DFT.H"

/*
    change "MAIN PARAMS SETTING" part and "GENERAL SETTINGS" for each job
    define Macro [MAIN_SOLVER] to run LBM evolution
    define Macro [POST_PROCESS] to run post data processing
*/


#define MAIN_SOLVER
//#define POST_PROCESS_DFT
//#define POST_PROCESS_DROPLET


extern Real alpha0; //  %.2f format
extern Real T;      //  %.1e format

//  **************************************    GENERAL SETTINGS     *****************************
const bool tagHDF5 = false; 

const bool is_flate_interface = false;
const bool is_droplet = false;
const bool is_mixture = true;
const int Ndigits = 7;
const string root_path = ".";//"/home/xdengae/Lattice-Boltzmann";
//  ********************************************************************************************

inline void WriteOutput(string plot_file_root, int step,
			const MultiFab& hydrovs,
			const Vector<std::string>& var_names,
			const Geometry& geom, StructFact& structFact, int plot_SF=0,
      bool tagHDF5 = false) {

  const Real time = step;
  const std::string& pltfile = amrex::Concatenate(plot_file_root,step,Ndigits);

  if(tagHDF5){
    //WriteSingleLevelPlotfileHDF5(pltfile, hydrovs, var_names, geom, time, step);
  }else{
    WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, time, step);
  }
  const int zero_avg = 1;
  if (plot_SF > 0) {
    string plot_file_root_SF = plot_file_root + "_SF";
    structFact.WritePlotFile(step, static_cast<Real>(step), plot_file_root_SF, zero_avg);
  }
}

#ifdef AMREX_USE_CUDA
__global__ void kernel_example() {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id == 0) {
      printf("Block size: %d, Number of threads: %d\n", blockDim.x, gridDim.x * blockDim.x);
  }
}
#endif

//nsys profile --trace=cuda,osrt,syscall --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --cuda-memory-usage=true --output=profile_output ./main3d.gnu.MPI.CUDA.ex

int main(int argc, char* argv[]) {

  amrex::Initialize(argc, argv);
  amrex::Arena::PrintUsage();
  bool noiseSwitch;
  if(T==0){
    noiseSwitch = false;
  }else{
    noiseSwitch = true;
  }
  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();
  int my_rank = amrex::ParallelDescriptor::MyProc(); 
  int nprocs = amrex::ParallelDescriptor::NProcs();
  // ***********************************  Basic AMReX Params ***********************************************************
  // default grid parameters
  int nx = 16; //40;
  int max_grid_size = nx/2;//4;

  
                  // ****************************************************************************
          // ***************************************************************************************************
  // ************************************************  MAIN PARAMS SETTING   ******************************************************
  // *******************************************  change for each job *******************************************************************
  int step_continue = 400;//1300400;//3500400; // set to be 0 for noise=0 case; set to be the steps number of the checkpoint file when noise != 0;
  bool continueFromNonFluct = true;//false; // true for first time switching on noise; setting the suffix of the chkpoint file to be loaded (true->0 or false->T);
                                    // set "false" only if hope to continue from chkpoint WITH noise case;
  int nsteps = 400000;//100000;
  int Out_Step = noiseSwitch? step_continue: step_continue;// + nsteps/2;
  int plot_int = 2000;
  int print_int = 100;
  int Out_Noise_Step = plot_int;
  int plot_SF_window = 50000; // not affected by [plot_int]; out freq controlled by [Out_SF_Step]
  int Out_SF_Step = 10;
  // default output parameters
  int plot_SF = noiseSwitch? plot_int: 0; // switch on writting Structure Factor for noise!=0 case only;
  if(plot_SF_window == 0){
    Print() << "No stuct factor will be calculated\n";
    plot_SF = 0;  // only for >0 values it will be output to disk;
  }
  const int t_window = 10*plot_int;   //  specifying time window for calculating the equilibrium state solution;
                              //    must be multiples of [plot_int]
                              //  from step [last_step_index-t_window] to [last_step_index] (total frames determined by [plot_int])
  // ****************************************************************************************************************************************
      // **********************************************************************************************************************  
                  // ****************************************************************************


  // default droplet radius (% of box size)
  Real radius = 0.35;
  // set up Box and Geomtry
  IntVect dom_lo(0, 0, 0);
  // **********************************************************************************************


  // ********************  Box Domain Setting, change with the system  ************************
  IntVect dom_hi;
  if(is_mixture){
    //dom_hi = IntVect(64-1, 2-1, 2-1);     // for mixture
    dom_hi = IntVect(nx-1, nx-1, nx-1);
  }else if(is_droplet){
    dom_hi = IntVect(nx-1, nx-1, nx-1); // for droplet 
  }else if(is_flate_interface){

  }
  
  Array<int,3> periodicity({1,1,1});
  Box domain(dom_lo, dom_hi);
  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
  BoxArray ba(domain);
  // split BoxArray into chunks no larger than "max_grid_size" along a direction
  ba.maxSize(max_grid_size);
  DistributionMapping dm(ba);
  // need two halo layers for gradients
  int nghost = 2; //2;
  // number of hydrodynamic fields to output
  int nhydro = 15; //6; 
  // **********************************************************************************************


  // ************************************  File directory settings  ********************************
  string check_point_root_f, check_point_root_g;
  char plot_file_root_cstr[200];
  if(is_flate_interface){
    check_point_root_f = root_path + "/data_interface/f_checkpoint";
    check_point_root_g = root_path + "/data_interface/g_checkpoint";
    if(noiseSwitch){
      sprintf(plot_file_root_cstr, "%s/data_interface/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d_continue/plt",
        root_path.c_str(), alpha0, T, dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }else{
      sprintf(plot_file_root_cstr, "%s/data_interface/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d/plt",
      root_path.c_str(), alpha0, T, dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }
  }else if(is_droplet){
    check_point_root_f = root_path + "/data_droplet/f_checkpoint";
    check_point_root_g = root_path + "/data_droplet/g_checkpoint";
    if(noiseSwitch){
      sprintf(plot_file_root_cstr, "%s/data_droplet/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d_continue/plt",
        root_path.c_str(), alpha0, T, dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }else{
      sprintf(plot_file_root_cstr, "%s/data_droplet/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d/plt",
      root_path.c_str(), alpha0, T, dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }
  }else if(is_mixture){
    check_point_root_f = root_path + "/data_mixture/f_checkpoint";
    check_point_root_g = root_path + "/data_mixture/g_checkpoint";
    if(noiseSwitch){
      sprintf(plot_file_root_cstr, "%s/data_mixture/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d_continue/plt",
      root_path.c_str(), alpha0, T, dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }else{
      sprintf(plot_file_root_cstr, "%s/data_mixture/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d/plt",
      root_path.c_str(), alpha0, T, dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    }
  }
  string plot_file_root(plot_file_root_cstr);
  // **********************************************************************************************************

  //  Set up equilibirum state solutions
  MultiFab rho_eq(ba, dm, 1, nghost); rho_eq.setVal(0.);  // default values applies for 0 noise cases
  MultiFab phi_eq(ba, dm, 1, nghost); phi_eq.setVal(0.);
  MultiFab rhot_eq(ba, dm, 1, nghost);  rhot_eq.setVal(1.);
  string rho_eq_file, phi_eq_file, rhot_eq_file;
  if(is_droplet){
    rho_eq_file = root_path + "/data_droplet/equilibrium_rho";
    phi_eq_file = root_path + "/data_droplet/equilibrium_phi";
    rhot_eq_file = root_path + "/data_droplet/equilibrium_rhot";
  }else if(is_mixture){
    rho_eq_file = root_path + "/data_mixture/equilibrium_rho";
    phi_eq_file = root_path + "/data_mixture/equilibrium_phi";
    rhot_eq_file = root_path + "/data_mixture/equilibrium_rhot";
  }else if(is_flate_interface){

  }
  rho_eq_file = rho_eq_file + "_alpha0_" + format("%.2f", alpha0) 
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  phi_eq_file = phi_eq_file + "_alpha0_" + format("%.2f", alpha0)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  rhot_eq_file = rhot_eq_file + "_alpha0_" + format("%.2f", alpha0)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  
  #ifdef MAIN_SOLVER    // **********************************   MICRO for time evolution  ****************************
  // *************************************  Set up Physical MultiFab Variables  *********************************

  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, nhydro, nghost);
  MultiFab hydrovsbar(ba, dm, 2, nghost); // modified hydrodynamic variables, only contains rho & phi two hydrovars used for grad & laplacian
  MultiFab fnoisevs(ba, dm, nvel, nghost); // thermal noise storage of fluid f for each step;
  MultiFab gnoisevs(ba, dm, nvel, nghost);

  // set up variable names for output
  const Vector<std::string> var_names = VariableNames(nhydro);

  if(noiseSwitch){
    Print() << "Noise switch on\n";
    LoadSingleMultiFab(rho_eq_file, rho_eq);  // the ghost layers values are meaningless
    LoadSingleMultiFab(phi_eq_file, phi_eq);
    LoadSingleMultiFab(rhot_eq_file, rhot_eq);
  }else{
    Print() << "Noise switch off\n";
  }
  if(noiseSwitch){  printf("Numerical equilibrium state solution lower bound:\tmin rho_eq: %f\tmin phi_eq: %f\n", rho_eq.min(0), phi_eq.min(0));  }

  // *********************************** pre-processing numerical solution data for the droplet-equilibrium state  ************************************
  
  // ***********************************************  INITIALIZE **********************************************************************  

  bool if_continue_from_last_frame = noiseSwitch? true: false;
  string str_step_continue = "";
  if(step_continue>0){  str_step_continue = amrex::Concatenate(str_step_continue,step_continue,Ndigits);  }
  std::string pltfile_f, pltfile_g;

  // continue running from a checkpoint ...
  if(if_continue_from_last_frame){
    MultiFab f_last_frame(ba, dm, nvel, nghost);
    MultiFab g_last_frame(ba, dm, nvel, nghost);
    Print() << "Load in last frame checkpoint ....\n";
    pltfile_f = check_point_root_f + str_step_continue;
    pltfile_g = check_point_root_g + str_step_continue;
    Real chk_temp = continueFromNonFluct? 0.: T;
    pltfile_f = pltfile_f + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", chk_temp)
      + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    pltfile_g = pltfile_g + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", chk_temp)
      + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    if(is_flate_interface){
      LoadSingleMultiFab(pltfile_f, f_last_frame);
      LoadSingleMultiFab(pltfile_g, g_last_frame);
    }else if(is_droplet){
      LoadSingleMultiFab(pltfile_f, f_last_frame);
      LoadSingleMultiFab(pltfile_g, g_last_frame);
    }else if(is_mixture){
      LoadSingleMultiFab(pltfile_f, f_last_frame);
      LoadSingleMultiFab(pltfile_g, g_last_frame);
    }
    // continue from given initial populations f & g
    LBM_init(geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, 
    f_last_frame, g_last_frame, rho_eq, phi_eq, rhot_eq);

    PrintDensityFluctuation(hydrovs, var_names, -1); // check the data uniformity
  }else{  // continuing running from initial default states
    if(is_flate_interface){
      
    }else if(is_droplet){
      LBM_init_droplet(radius, geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
    }else if(is_mixture){
      Print() << "Init mixture system ...\n";
      LBM_init_mixture(geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
    }
  }
  Print() << "check modified hydrodynamic quantities validity ...\n";
  MultiFabNANCheck(hydrovsbar, false, 0);
  Print() << "check real hydrodynamic quantities validity ...\n";
  MultiFabNANCheck(hydrovs, false, 0);

  // set up StructFact
  int nStructVars = 2;
  //const Vector<std::string> var_names = hydrovars_names(nStructVars);
  const Vector<int> pairA = { 0, 1, 2, 3, 4, 0, 2, 3, 4};
  const Vector<int> pairB = { 0, 1, 2, 3, 4, 1, 3, 4, 2};
  const Vector<Real> var_scaling = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  StructFact structFact(ba, dm, var_names, var_scaling, pairA, pairB);
  structFact.Reset();

  // Write a plotfile of the initial data if plot_int > 0 and starting from initial default states
  if (plot_int > 0 && step_continue == 0)
    WriteOutput(plot_file_root, 0, hydrovs, var_names, geom,  structFact, 0); 
  Print() << "LB initialized with alpha0 = " << alpha0 << " and T = " << T << '\n';
  if(is_droplet && my_rank == 0){  printf("Current case is droplet system with initial radius = %f\n", radius);   }

  // *****************************************************  TIMESTEP  *********************************************************
  int SF_start = step_continue + nsteps - plot_SF_window;
  std::vector<Real> radius_frames;
  std::vector<Real> rho_mean_frames;  //  rho mean value for each frame
  std::vector<Real> rho_sigma_frames; //  rho standard deviation for each frame
  for (int step=step_continue+1; step <= step_continue+nsteps; ++step) {
    if(step%print_int == 0){
      Print() << "LB step " << step << " info:\n";
    }
    //amrex::ParallelDescriptor::Barrier();
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
    //amrex::ParallelDescriptor::Barrier();

    if(noiseSwitch && step>=SF_start && step%Out_SF_Step == 0){
      amrex::ParallelDescriptor::Barrier();
      structFact.FortStructure(hydrovs, 0); // default reset value = 0 used here for accumulating correlations for each frame >= [SF_start] 
      amrex::ParallelDescriptor::Barrier();
      //WriteOutNoise(plot_file_root, step, fnoisevs, gnoisevs, geom, Ndigits);
    }
    if(noiseSwitch && step%Out_Noise_Step == 0){
      WriteOutNoise(plot_file_root, step, fnoisevs, gnoisevs, geom, Ndigits);
    }
    if (plot_int > 0 && step%plot_int == 0){
      Print() << "*****\tLB step output at step = " << step << "\t*******\n";
      // ******************************* Running Process Monitor *******************************************
      if(is_mixture && nprocs == 1 && (!noiseSwitch)){
        Array2D<Real,0,3,0,2> density_info = PrintDensityFluctuation(hydrovs, var_names, step);
        Print() << "Call host density function\n";
        rho_mean_frames.push_back(density_info(0, 0));  // (0: density rho; 0: mean)
        rho_sigma_frames.push_back(density_info(0, 1));  // (0: density rho; 1: standard deviation)
      }
      if(step >= Out_Step && step!=step_continue+nsteps){
        WriteOutput(plot_file_root, step, hydrovs, var_names, geom, structFact, 0); // do not output [structFact] during running time;
      }
    }
    if(step == step_continue+nsteps){
      WriteOutput(plot_file_root, step, hydrovs, var_names, geom, structFact, plot_SF);
    }
  }
  if(is_mixture && amrex::ParallelDescriptor::IOProcessor() && nprocs == 1){ // vector [radius_frames] is written by I/O rank so must output it in same rank;
    int idx_end = plot_file_root.length() - 3;
    string rho_mean_file = plot_file_root.substr(0, idx_end);
    rho_mean_file = rho_mean_file + "rho_mean_steps";
    Print() << "write out density rho mean for each frame to file " << rho_mean_file << '\n';
    WriteVectorToFile(rho_mean_frames, rho_mean_file);
    Print() << "write out density rho standard deviation for each frame to file " << rho_mean_file << '\n';
    string rho_sigma_file = plot_file_root.substr(0, idx_end);
    rho_sigma_file = rho_sigma_file + "rho_sigma_steps";
    WriteVectorToFile(rho_sigma_frames, rho_sigma_file);
  }

  // *****************************************************  Post-Processing  *********************************************************
  // write out last frame checkpoint
  pltfile_f = amrex::Concatenate(check_point_root_f, step_continue+nsteps, Ndigits);
  pltfile_g = amrex::Concatenate(check_point_root_g, step_continue+nsteps, Ndigits);
  pltfile_f = pltfile_f + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", T)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  pltfile_g = pltfile_g + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", T)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  if(is_flate_interface){
     
  }else if(is_droplet){
    Vector< std::string > varname_chk;  varname_chk.push_back("rho_chk");
    WriteSingleLevelPlotfile(pltfile_f, fold, varname_chk, geom, 0, 0);
    varname_chk.clear();  varname_chk.push_back("phi_chk");
    WriteSingleLevelPlotfile(pltfile_g, gold, varname_chk, geom, 0, 0);
  }else if(is_mixture){
    Vector< std::string > varname_chk;  varname_chk.push_back("rho_chk");
    WriteSingleLevelPlotfile(pltfile_f, fold, varname_chk, geom, 0, 0); // time & step = 0 just for simplicity; meaningless here;
    varname_chk.clear();  varname_chk.push_back("phi_chk");
    WriteSingleLevelPlotfile(pltfile_g, gold, varname_chk, geom, 0, 0); // time & step = 0 just for simplicity; meaningless here;
  }
  
  const IntVect box = geom.Domain().length();
  if(is_droplet && amrex::ParallelDescriptor::IOProcessor()){ PrintMassConservation(hydrovs, var_names, box[0], radius*box[0]); }

  // Call the timer again and compute the maximum difference between the start time
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);
  amrex::Print() << "Run time = " << stop_time << std::endl;

  // Extract the equilibrium state solution;
  MultiFab mfab_rho_eq(ba, dm, 1, nghost);
  MultiFab mfab_phi_eq(ba, dm, 1, nghost);
  MultiFab mfab_rhot_eq(ba, dm, 1, nghost);
  int step1 = step_continue + nsteps - t_window; int step2 = step_continue + nsteps;

  // copy the ensemble averaged solution to [mfab_*_eq];
  if(!noiseSwitch){
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_rho_eq, 0, 1/*lp,p=1 norm*/, (!noiseSwitch), 0, Ndigits);
    Vector< std::string > vec_varname;  vec_varname.push_back("rho_eq");
    WriteSingleLevelPlotfile(rho_eq_file, mfab_rho_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
    vec_varname.clear();  vec_varname.push_back("phi_eq");
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_phi_eq, 1, 1/*lp,p=1 norm*/, (!noiseSwitch), 0, Ndigits);  // [phi] at index 1;
    WriteSingleLevelPlotfile(phi_eq_file, mfab_phi_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
    vec_varname.clear();  vec_varname.push_back("rhot_eq");
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_rhot_eq, 5, 1, (!noiseSwitch), 0, Ndigits); // total density at index 5; nlevel=0;
    WriteSingleLevelPlotfile(rhot_eq_file, mfab_rhot_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
  }

  #endif  // **********************************   END MICRO for time evolution  ****************************

  amrex::Finalize();
}