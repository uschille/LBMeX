#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <StructFact.H>

using namespace amrex;

#include "LBM_binary.H"
#include "Debug.H"
#include "AMReX_FileIO.H"
#include "AMReX_Analysis.H"
#include "LBM_hydrovs.H"
#include "externlib.H"

#define STRUCT_HYDROVARS
//#define STRUCT_LB_HYDROVARS

#define PLOT_POPULATIONS


#define SYS_DROPLET
//#define SYS_MIXTURE
//#define SYS_FLATE_INTERFACE

const string root_path = ".";   //"/home/xdengae/Binary-Fluctuating-Lattice-Boltzmann/";
const int Ndigits = 7;

extern Real alpha0; //  %.2f format

AMREX_GPU_MANAGED Real init_frac = 0.5; // fractional size of droplet or stripe

inline void WriteOutput(string plot_file_root, int step,
			const MultiFab& hydrovs,
			const Vector<std::string>& var_names,
			const Geometry& geom, StructFact& structFact, int plot_SF=0,
      bool tagHDF5 = false) {

  const Real time = step;
  const std::string& pltfile = amrex::Concatenate(plot_file_root,step,Ndigits);

  if(tagHDF5){
    // write HDF5 format plotfile; to be implemented;
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


int main(int argc, char* argv[]) {

  amrex::Initialize(argc, argv);
  amrex::Arena::PrintUsage();
  // T=0: no noise, obtain the equilibrium state solutions; T>0: with noise;
  bool noiseSwitch = (kBT==0) ? false : true;
  // store the current time so we can later compute total run time.
  Real start_time = ParallelDescriptor::second();
  int my_rank = amrex::ParallelDescriptor::MyProc(); 
  int nprocs = amrex::ParallelDescriptor::NProcs();
  amrex::InitRandom(12345);

  // ***********************************  Basic AMReX Params ***********************************************************
  int nx = 32; //16; //40;
  int ny = 0;   int nz = 0; // for different size system;
  int max_grid_size = nx/2;//4;

                  // ****************************************************************************
          // ***************************************************************************************************
  // ************************************************  MAIN PARAMS SETTING   ******************************************************
  // *******************************************  change for each job *******************************************************************
  // set to be 0 for noise=0 case; set to be the step's number of the checkpoint file when noise != 0;
  int step_continue = 40000;//500;//1300400;//3500400;

  // [true] for a: kBT=0; b: kBT>0 && switching on noise for the FIRST time;
  // [false] only if hope to continue from chkpoint in which noise>0;
  bool continueFromNonFluct = true; //false;

  // Total number of steps to run; MUST be integer multiples of [plot_int];
  int nsteps = 1200000;//1500000;//3000;
  // ouput trajectories from step >= Out_Step; by default, it is set to be the same as [step_continue];
  int out_step = noiseSwitch? step_continue + 3*nsteps/5: step_continue;
  int plot_int = 100;//100;//2000;//10; // output configurations every [plot_int] steps;
  int print_int = 100;     // print out info every [print_int] steps;
  /*specifying time window for calculating the equilibrium state solution;
    usually be set as multiples of [plot_int], from step [last_step_index-t_window] to [last_step_index];
    i.e., [numOfFrames-1]*[plot_int], in which last_step_index is also multiples of [plot_int] */
  const int t_window = 50*plot_int;
  int out_noise_step = plot_int;    // output noise terms every [out_noise_step] steps;

  // [plot_SF_window] is the time window for calculating the structure factor;
  int plot_SF_window = 0;//200000; // not affected by [plot_int]; out freq controlled by [out_SF_step]
  int out_SF_step = 100; // 50*2, select every 100 steps to calcualte Structure Factor;
  // default output parameters
  int plot_SF = noiseSwitch? plot_SF_window: 0; // switch on writting Structure Factor for noise!=0 case only;
  if(plot_SF_window == 0)   Print() << "plot_SF_window = 0 and No stuct factor will be calculated\n";
  // ****************************************************************************************************************************************
      // **********************************************************************************************************************  
                  // *******************************************************************************

#ifdef SYS_DROPLET
  // default droplet radius (% of box size), %.2f format
  const Real radius = 0.2;
  bool if_print_radius = false; // print fitted droplet radius for each output frame;
  Vector<RealVect> com_ref; // reference center of mass position for equilibrium state;
  /*com_ref.push_back(RealVect{AMREX_D_DECL(0.,0.,0.)}); 
  com_ref.push_back(RealVect{AMREX_D_DECL(0.,0.,0.)}); 
  com_ref.push_back(RealVect{AMREX_D_DECL(0.,0.,0.)});*/ 
  com_ref.push_back(RealVect{AMREX_D_DECL(nx/2.,nx/2.,nx/2.)}); 
  com_ref.push_back(RealVect{AMREX_D_DECL(nx/2.,nx/2.,nx/2.)}); 
  com_ref.push_back(RealVect{AMREX_D_DECL(nx/2.,nx/2.,nx/2.)}); 
#endif
  // set up Box and Geomtry
  IntVect dom_lo(0, 0, 0);
  // ********************  Box Domain Setting, change with the system  ************************
  IntVect dom_hi;
#ifdef SYS_DROPLET
  dom_hi = IntVect(nx-1, nx-1, nx-1);     // for droplet
#endif
#ifdef SYS_FLATE_INTERFACE
  nx = 8; ny = 256; nz = 64; // for flate interface with height = 64 lbu and stripe width = 8 lbu & length = 256 lbu
  //max_grid_size = 8;
  dom_hi = IntVect(nx-1, ny-1, nz-1);
#endif
#ifdef SYS_MIXTURE
  dom_hi = IntVect(nx-1, nx-1, nx-1);     // for mixture
#endif

  const Array<int,3> periodicity({1,1,1});
  Box domain(dom_lo, dom_hi);
  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
  BoxArray ba(domain);
  // split BoxArray into chunks no larger than "max_grid_size" along a direction
  ba.maxSize(max_grid_size);
  DistributionMapping dm(ba);
  // need two halo layers for laplacian operator
  int nghost = 2;
  // number of hydrodynamic fields to output
  int nhydro = 22;//18; //6;
  // **********************************************************************************************

  // ************************************  File directory settings  ********************************
  string check_point_root_f, check_point_root_g;
  char plot_file_root_cstr[200];
  string plot_file_dir;
#ifdef SYS_FLATE_INTERFACE
    char plot_file_dir_cstr[60];
    sprintf(plot_file_dir_cstr, "data_interface_alpha0_%.2f", alpha0);
    plot_file_dir.assign(plot_file_dir_cstr);
#endif
#ifdef SYS_DROPLET
    char plot_file_dir_cstr[60];
    sprintf(plot_file_dir_cstr, "data_droplet_alpha0_%.2f_r%.2f_size%d-%d-%d", alpha0, radius,
                          dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    plot_file_dir.assign(plot_file_dir_cstr);
#endif

#ifdef SYS_MIXTURE
    plot_file_dir = "data_mixture";
    #ifdef STRUCT_HYDROVARS
    plot_file_dir = plot_file_dir + "_hydrovars";
    #endif
    #ifdef STRUCT_LB_HYDROVARS
    plot_file_dir = plot_file_dir + "_lb_hydrovars";
    #endif
#endif
  // check point file root name;
  check_point_root_f = root_path + "/" + plot_file_dir + "/f_checkpoint";
  check_point_root_g = root_path + "/" + plot_file_dir + "/g_checkpoint";
  if(noiseSwitch){
    sprintf(plot_file_root_cstr, "%s/%s/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d_continue/plt",
        root_path.c_str(), plot_file_dir.c_str(), alpha0, kBT, dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  }else{
    sprintf(plot_file_root_cstr, "%s/%s/lbm_data_shshan_alpha0_%.2f_xi_%.1e_size%d-%d-%d/plt",
        root_path.c_str(), plot_file_dir.c_str(), alpha0, kBT, dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  }
  // plot file root name of each frame's configuration;
  string plot_file_root(plot_file_root_cstr);
  // **********************************************************************************************************

  //  Set up equilibirum state solutions
  MultiFab rho_eq(ba, dm, 1, nghost); rho_eq.setVal(0., nghost);  // default values applies for 0 noise cases
  MultiFab phi_eq(ba, dm, 1, nghost); phi_eq.setVal(0., nghost);
  MultiFab rhot_eq(ba, dm, 1, nghost);  rhot_eq.setVal(1., nghost);
  // set up file names for equilibrium state solutions
  string rho_eq_file = root_path + "/" + plot_file_dir + "/equilibrium_rho";
  string phi_eq_file = root_path + "/" + plot_file_dir + "/equilibrium_phi";
  string rhot_eq_file = root_path + "/" + plot_file_dir + "/equilibrium_rhot";
  rho_eq_file = rho_eq_file + "_alpha0_" + format("%.2f", alpha0) 
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  phi_eq_file = phi_eq_file + "_alpha0_" + format("%.2f", alpha0)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  rhot_eq_file = rhot_eq_file + "_alpha0_" + format("%.2f", alpha0)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  
  // *************************************  Set up Physical MultiFab Variables  *********************************
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, nhydro, nghost);
  MultiFab hydrovsbar(ba, dm, 15/*nhydro*/, nghost); // modified hydrodynamic variables, only contains rho & phi two hydrovars used for grad & laplacian
  MultiFab fnoisevs(ba, dm, nvel, nghost); // thermal noise storage of fluid f for each step;
  MultiFab gnoisevs(ba, dm, nvel, nghost);
  // set up hydro-variable names for output
  const Vector<std::string> var_names = VariableNames(nhydro, true);

  if(noiseSwitch){
    Print() << "Noise switch on\n";
    LoadSingleMultiFab(rho_eq_file, rho_eq);  // the ghost layers values are meaningless
    LoadSingleMultiFab(phi_eq_file, phi_eq);
    LoadSingleMultiFab(rhot_eq_file, rhot_eq);
    //rho_eq.FillBoundary(geom.periodicity());  // fill the ghost layers with the equilibrium state solution, why cannot work ??
    //phi_eq.FillBoundary(geom.periodicity());
    //rhot_eq.FillBoundary(geom.periodicity());
    Print() << "Mass rho_eq (Multifab sum() function) = " << rho_eq.sum(0) << '\n';
    Print() << "Mass rho_eq (usr defined function MultiFabValidSum()) = " << MultiFabValidSum(rho_eq) << '\n';
    Print() << "Mass rho_eq (usr defined function MultiFabValidSumAtomic()) = " << MultiFabValidSumAtomic(rho_eq) << '\n';
    Print() << "Mass phi_eq (Multifab sum() function) = " << phi_eq.sum(0) << '\n';
    Print() << "Mass rhot_eq (Multifab sum() function) = " << rhot_eq.sum(0) << '\n';
#ifdef SYS_DROPLET
    RealVect com_rho, com_phi, com_rhot;
    update_com(geom, com_rho,  rho_eq,  true);
    update_com(geom, com_phi,  phi_eq,  true);
    update_com(geom, com_rhot, rhot_eq, true);
    com_ref[0] = com_rho;
    com_ref[1] = com_phi;
    com_ref[2] = com_rhot;
#endif
    //printf("Numerical equilibrium state solution lower bound:\tmin rho_eq: %f\tmin phi_eq: %f\n", rho_eq.min(0), phi_eq.min(0));
  }else{
    Print() << "Noise switch off, calculating the equilibrium state solutions...\n";
  }

#ifdef SYS_DROPLET
  // *********************************** pre-processing numerical solution data for the droplet-equilibrium state  ************************************
#endif

  // ***********************************************  INITIALIZE **********************************************************************  
  /* determine whether initial states read from chkpoint files;
    [if_continue_from_last_frame]=true: read from chkpoint files, otherwise not.*/
  bool if_continue_from_last_frame = noiseSwitch? true: false;
  string str_step_continue = "";
  if(step_continue>0){  str_step_continue = amrex::Concatenate(str_step_continue,step_continue,Ndigits);  }
  std::string pltfile_f, pltfile_g;
  // continue running from a checkpoint ...
  if(if_continue_from_last_frame){
    MultiFab f_last_frame(ba, dm, nvel, nghost);
    MultiFab g_last_frame(ba, dm, nvel, nghost);
    Print() << "Loading in last frame checkpoint files....\n";
    pltfile_f = check_point_root_f + str_step_continue;
    pltfile_g = check_point_root_g + str_step_continue;
    Real chk_temp = continueFromNonFluct? 0.: kBT;  // check point file temperature;
    pltfile_f = pltfile_f + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", chk_temp)
      + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    pltfile_g = pltfile_g + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", chk_temp)
      + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
    LoadSingleMultiFab(pltfile_f, f_last_frame);
    LoadSingleMultiFab(pltfile_g, g_last_frame);
    /* continue from given initial populations f & g;
       filling ghost layers before computing [hydrovs], [hydrovsbar];
       Ensure ghost layers of all variables are fulfilled before next step.*/
    LBM_init(geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, 
    f_last_frame, g_last_frame, rho_eq, phi_eq, rhot_eq, com_ref);
#ifdef SYS_MIXTURE
    //PrintDensityFluctuation(hydrovs, var_names, -1); // check the data uniformity for mixture system only;
#endif
  }else{  // running from initial default states;
#ifdef SYS_MIXTURE
    Print() << "Init mixture system ...\n";
    LBM_init_mixture(geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
#endif
#ifdef SYS_FLATE_INTERFACE
    Print() << "Init flate interface system ...\n";
    LBM_init_stripe(init_frac, geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
#endif
#ifdef SYS_DROPLET
    Print() << "Init droplet system ...\n";
    LBM_init_droplet(radius, geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
#endif
  }

  Print() << "check initial modified hydrodynamic quantities validity ...\n";
  MultiFabNANCheck(hydrovsbar, true, 0);
  Print() << "check initial real hydrodynamic quantities validity ...\n";
  MultiFabNANCheck(hydrovs, true, 0);

  // set up StructFact
  // ufbar(20) ugbar(21) nf(18), 
  const Vector<int> pairA = { 0,  1,  0,  2,  3,  4,  6,  7,  8,  2,
                              9,  15, 16, 17, 15, 18, 19, 20, 21, 
                              20, 20, 21}; 
  const Vector<int> pairB = { 0,  1,  1,  2,  3,  4,  6,  7,  8,  6,
                              9,  15, 16, 17, 16, 18, 19, 20, 21, 
                              21, 18, 18};
  const Vector<Real> var_scaling = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0};
  StructFact structFact(ba, dm, var_names, var_scaling, pairA, pairB);
  //structFact.Reset();

  // Write a plotfile of the initial data if plot_int > 0 and starting from initial default states
  if (plot_int > 0 && step_continue == 0)
    WriteOutput(plot_file_root, 0, hydrovs, var_names, geom, structFact, 0); 
  Print() << "LB initialized with alpha0 = " << alpha0 << " and T = " << kBT << '\n';
#ifdef SYS_DROPLET
  Print() << "Initial droplet radius = " << radius << '\n';
#endif

// *****************************************************  TIMESTEP  *********************************************************
  int SF_start = step_continue + nsteps - plot_SF_window;
  std::vector<Real> radius_frames;
  std::vector<Real> rho_mean_frames;  //  rho mean value for each frame
  std::vector<Real> rho_sigma_frames; //  rho standard deviation for each frame
  MultiFab rhof(ba, dm, 1, nghost);
  for (int step=step_continue+1; step <= step_continue+nsteps; ++step) {
    if(step%print_int == 0){
      Print() << "LB step " << step << " info:\n";
    }
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, hydrovsbar, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq, com_ref);

    //PrintMultiFabComp(fold, 3, 0);
    if(noiseSwitch && step>=SF_start && step%out_SF_step == 0){
      structFact.FortStructure(hydrovs/*hydrovs*/, 0); // default reset value = 0 used here for accumulating correlations for each frame >= [SF_start] 
    }
    if(noiseSwitch && step%out_noise_step == 0){
      WriteOutNoise(plot_file_root, step, fnoisevs, gnoisevs, geom, Ndigits);
    }
    if (plot_int > 0 && step%plot_int == 0){
      //PrintMultiFabComp(fold, 3, 0);
      Print() << "\t**************************************\t" << std::endl;
      Print() << "\tLB step " << step << " & Output" << std::endl;
      Print() << "\t**************************************\t" << std::endl;
      // ************************************* Running Process Monitor *******************************************
      #ifdef SYS_DROPLET
        amrex::ParallelCopy(rhof, hydrovs, 0/*srccomp*/, 0/*dstcomp*/, 1); // copy rho from hydrovs's 0th comp to rhof, i.e., density fluid f;
        Function3DAMReX func_rho(rhof, geom);
        RealVect vec_com = {0., 0., 0.};
        update_com(geom, vec_com, rhof, true);
        // Fitting equilibrirum droplet radius; \frac{1}{2}\left(1+\tanh\frac{R-\left|\bm{r}-\bm{r}_{0}\right|}{\sqrt{2W}}\right)
        if(if_print_radius && nprocs == 1){ // fitting radius only for single processor case;
          Array<Real, 3> param_arr = fittingDropletParams(func_rho, 20, 0.01, 400, kappa, radius);  // last 20 steps ensemble mean, relative error < 0.005 bound;
          printf("fitting parameters for equilibrium density rho: (W=%f, R=%f)\n", param_arr[0], param_arr[1]);
          radius_frames.push_back(param_arr[1]); // store the radius for each frame;
        }
      #endif
      if(step >= out_step && step!=step_continue+nsteps){
        WriteOutput(plot_file_root, step, hydrovs, var_names, geom, structFact, 0); // do not output [structFact] during running time;
      }
    }
    if(step == step_continue+nsteps){
      WriteOutput(plot_file_root, step, hydrovs, var_names, geom, structFact, plot_SF);
    }
  }

  #ifdef SYS_DROPLET
  if(amrex::ParallelDescriptor::IOProcessor()){ // vector [radius_frames] is written by I/O rank so must output it in same rank;
    string radius_frames_file = plot_file_root.substr(0, plot_file_root.length()-3); // remove the last 3 characters "plt"
    radius_frames_file = radius_frames_file + "radius_steps_out";
    Print() << "write out radius for each frame to file " << radius_frames_file << '\n';
    WriteVectorToFile(radius_frames, radius_frames_file);
  }
  #endif

  // *****************************************************  Post-Processing *********************************************************
  // write out last frame checkpoint
  pltfile_f = amrex::Concatenate(check_point_root_f, step_continue+nsteps, Ndigits);
  pltfile_g = amrex::Concatenate(check_point_root_g, step_continue+nsteps, Ndigits);
  pltfile_f = pltfile_f + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", kBT)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  pltfile_g = pltfile_g + "_alpha0_" + format("%.2f", alpha0) + "_xi_" + format("%.1e", kBT)
    + "_size" + format("%d-%d-%d", dom_hi[0]-dom_lo[0]+1, dom_hi[1]-dom_lo[1]+1, dom_hi[2]-dom_lo[2]+1);
  Vector< std::string > varname_chk;  varname_chk.push_back("rho_chk");
  WriteSingleLevelPlotfile(pltfile_f, fold, varname_chk, geom, 0, 0); // time & step = 0 just for simplicity; meaningless here;
  varname_chk.clear();  varname_chk.push_back("phi_chk");
  WriteSingleLevelPlotfile(pltfile_g, gold, varname_chk, geom, 0, 0); // time & step = 0 just for simplicity; meaningless here;
  
  const IntVect box = geom.Domain().length();
#ifdef SYS_DROPLET
  // notice that here [radius] is the ratio of droplet radius to the box size;
  if(amrex::ParallelDescriptor::IOProcessor()){ PrintMassConservation(hydrovs, var_names, box[0], radius*box[0]); }
#endif
  // Call the timer again and compute the maximum difference between the start time
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - start_time;
  ParallelDescriptor::ReduceRealMax(stop_time);
  amrex::Print() << "Run time = " << stop_time << std::endl;

  // Extract the equilibrium state solution;
  MultiFab mfab_rho_eq(ba, dm, 1, nghost);
  MultiFab mfab_phi_eq(ba, dm, 1, nghost);
  MultiFab mfab_rhot_eq(ba, dm, 1, nghost);
  int step1 = step_continue + nsteps - t_window; int step2 = step_continue + nsteps;
  // copy the ensemble averaged solution to [mfab_*_eq];
  if(!noiseSwitch){
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_rho_eq, 
        0/*read-in comp index*/, 1/*lp,p=1 norm*/, (!noiseSwitch), 0/*nlevel=0*/, Ndigits);
    Vector< std::string > vec_varname;  vec_varname.push_back("rho_eq");
    WriteSingleLevelPlotfile(rho_eq_file, mfab_rho_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
    vec_varname.clear();  vec_varname.push_back("phi_eq");
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_phi_eq, 1, 1/*lp,p=1 norm*/, (!noiseSwitch), 0, Ndigits);  // [phi] at index 1;
    WriteSingleLevelPlotfile(phi_eq_file, mfab_phi_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
    vec_varname.clear();  vec_varname.push_back("rhot_eq");
    PrintConvergence(plot_file_root, step1, step2, plot_int, mfab_rhot_eq, 5, 1, (!noiseSwitch), 0, Ndigits); // total density at index 5; nlevel=0;
    WriteSingleLevelPlotfile(rhot_eq_file, mfab_rhot_eq, vec_varname, geom, 0, 0);  // time & step = 0 just for simplicity; meaningless here;
  }

  amrex::Finalize();
}