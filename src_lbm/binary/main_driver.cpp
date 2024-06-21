#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include "StructFact.H"

using namespace amrex;

#include "LBM_binary.H"

inline void WriteOutput(int step,
			const MultiFab& hydrovs,
			const Vector<std::string>& var_names,
			const Geometry& geom) {
  const Real time = step;
  const std::string& pltfile = amrex::Concatenate("hydro_plt_",step,5);
  WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, time, step);
}

inline void WriteDist(int step, 
      const MultiFab& fold,
      const MultiFab& gold, 
      const Vector<std::string>& var_names,
      const Geometry& geom){
      
      const Real time = step;
      std::string pltfile = amrex::Concatenate("f_plt_",step,5);
      WriteSingleLevelPlotfile(pltfile, fold, var_names, geom, time, step);

      pltfile = amrex::Concatenate("g_plt_",step,5);
      WriteSingleLevelPlotfile(pltfile, gold, var_names, geom, time, step);
      }

inline Vector<std::string> VariableNames(const int numVars) {
  // set variable names for output
  Vector<std::string> var_names(numVars);
  std::string name;
  int cnt = 0;
  // rho, phi, psi
  var_names[cnt++] = "density";
  var_names[cnt++] = "phi";
  // velx, vely, velz
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "u";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // pxx, pxy, pxz, pyy, pyz, pzz
  for (int i=0; i<AMREX_SPACEDIM; ++i) {
    for (int j=i; j<AMREX_SPACEDIM; ++j) {
      name = "p";
      name += (120+i);
      name += (120+j);
      var_names[cnt++] = name;
    }
  }
  // kinetic moments
  for (; cnt<nvel+1;) {
    name = "mf";
    name += std::to_string(cnt-1);
    var_names[cnt++] = name;
  }
  for (; cnt<numVars;) {
    name = "mg";
    name += std::to_string(cnt-nvel);
    var_names[cnt++] = name;
  }
  return var_names;
}

inline Vector<std::string> distribution_Variable_names() {
  // output names
  Vector<std::string> var_names(nvel);
  std::string name;

  for (int i = 0; i < nvel; i++){
    name = "e";
    name += std::to_string(i);
    var_names[i] = name;  
  }
  return var_names;
}

void main_driver(const char* argv) {

  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();
    
  // default grid parameters
  int nx = 16;
  int max_grid_size = 8;

  // default time stepping parameters
  int nsteps = 100;
  int plot_int = 10;

  // input parameters
  ParmParse pp;
  pp.query("nx", nx);
  pp.query("max_grid_size", max_grid_size);
  pp.query("nsteps", nsteps);
  pp.query("plot_int", plot_int);
  pp.query("kappa", kappa);
  pp.query("lambda", lambda);
  pp.query("T", T);
  pp.query("temperature", temperature);

  // set up Box and Geomtry
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(nx-1, nx-1, nx-1);
  Array<int,3> periodicity({1,1,1});

  Box domain(dom_lo, dom_hi);

  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);

  BoxArray ba(domain);

  // split BoxArray into chunks no larger than "max_grid_size" along a direction
  ba.maxSize(max_grid_size);

  DistributionMapping dm(ba);

  // need two halo layers for gradients
  int nghost = 2;

  // number of hydrodynamic fields to output
  int nhydro = 6;

  // set up MultiFabs
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, 2*nvel, nghost);
  MultiFab noise(ba, dm, 2*nvel, nghost);

  // set up variable names for output
  const Vector<std::string> var_names = VariableNames(2*nvel);
  const Vector<std::string> distribution_var_names = distribution_Variable_names();

  int nStructVars = hydrovs.nComp();

  Vector<Real> var_scaling(nStructVars*(nStructVars+1)/2);
  for (int d=0; d<var_scaling.size(); ++d) {
    if (temperature>0) var_scaling[d] = temperature; else var_scaling[d] = 1.;
  }

  // StructFact structFact(ba, dm, var_names, var_scaling);

  // INITIALIZE
  LBM_init_mixture(fold, gold, hydrovs);
  // Write a plotfile of the initial data if plot_int > 0
  if (plot_int > 0) {
    WriteOutput(0, hydrovs, var_names, geom);
    // WriteDist(0, fold, gold, distribution_var_names, geom);
    // structFact.FortStructure(hydrovs, geom);
    // structFact.WritePlotFile(0, 0., geom, "SF_plt_");
  }
  Print() << "LB initialized\n";

  // TIMESTEP
  for (int step=1; step <= nsteps; ++step) {
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, noise);
    // structFact.FortStructure(hydrovs, geom);
    if (plot_int > 0 && step%plot_int ==0){
      WriteOutput(step, hydrovs, var_names, geom);
      // WriteDist(step, fold, gold, distribution_var_names, geom);
    }
    Print() << "LB step " << step << "\n";
  }

  // structFact.WritePlotFile(nsteps, nsteps, geom, "SF_plt");

  // Call the timer again and compute the maximum difference between the start time 
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);
  amrex::Print() << "Run time = " << stop_time << std::endl;
  
}
