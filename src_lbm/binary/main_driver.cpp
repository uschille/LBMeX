#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <StructFact.H>

using namespace amrex;

#include "LBM_binary.H"
#include "tests.H"

inline Vector<std::string> VariableNames(const int numVars) {
  // set variable names for output
  Vector<std::string> var_names(numVars);
  std::string name;
  int cnt = 0;
  // rho, phi
  if (cnt<numVars) var_names[cnt++] = "density";
  if (cnt<numVars) var_names[cnt++] = "phi";
  // velx, vely, velz
  for (int d=0; d<AMREX_SPACEDIM, cnt<numVars; d++) {
    name = "u";
    name += (120+d);
    var_names[cnt++] = name;
  }
  for (int d=0; d<AMREX_SPACEDIM, cnt<numVars; d++) {
    name = "phi*u";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // pxx, pxy, pxz, pyy, pyz, pzz
  // for (int i=0; i<AMREX_SPACEDIM; ++i) {
  //   for (int j=i; j<AMREX_SPACEDIM; ++j) {
  //     name = "p";
  //     name += (120+i);
  //     name += (120+j);
  //     var_names[cnt++] = name;
  //   }
  // }
  // remaining moments
  for (; cnt<nvel+ncons, cnt<numVars;) {
    name = "mf";
    name += std::to_string(cnt-ncons);
    var_names[cnt++] = name;
  }
  for (; cnt<numVars;) {
    name = "mg";
    name += std::to_string(cnt-nvel);
    var_names[cnt++] = name;
  }
  return var_names;
}

inline void WriteOutput(int step,
			const MultiFab& hydrovs,
			const Geometry& geom) {
  // set up variable names for output
  const Vector<std::string> var_names = VariableNames(2*nvel);
  const std::string& pltfile = amrex::Concatenate("plt",step,5);
  WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, Real(step), step);
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
  pp.query("lambda", chi);
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

  // set up MultiFabs
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, 2*nvel, nghost);
  MultiFab noise(ba, dm, 2*nvel, nghost);

  int nStructVars = 5;
  const Vector<std::string> var_names = VariableNames(nStructVars);
  Vector<Real> var_scaling(nStructVars*(nStructVars+1)/2);
  for (int i=0; i<var_scaling.size(); ++i) {
    if (temperature>0) var_scaling[i] = temperature; else var_scaling[i] = 1.;
  }
  StructFact structFact(ba, dm, var_names, var_scaling);

  // INITIALIZE
  LBM_init_mixture(fold, gold, hydrovs);
  // Write a plotfile of the initial data if plot_int > 0
  if (plot_int > 0) WriteOutput(0, hydrovs, geom);
  Print() << "LB initialized\n";

  unit_tests(geom, hydrovs);

  // TIMESTEP
  for (int step=1; step <= nsteps; ++step) {
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, noise);
    structFact.FortStructure(hydrovs, geom);
    if (plot_int > 0 && step%plot_int ==0) {
      WriteOutput(step, hydrovs, geom);
      structFact.WritePlotFile(step, static_cast<Real>(step), geom, "plt_SF", 0);
    }
    Print() << "LB step " << step << "\n";
  }

  // Call the timer again and compute the maximum difference between the start time 
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);
  amrex::Print() << "Run time = " << stop_time << std::endl;
  
}
