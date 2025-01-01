#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <StructFact.H>

using namespace amrex;

#include "LBM_binary.H"
#include "LBM_fluctuations.H"
//#include "LBM_tests.H"

// default grid parameters
int nx = 16;

// default time stepping parameters
int nsteps = 10;

inline void ReadInput() {
  ParmParse pp;
  /* grid parameters */
  pp.query("nx", nx);

  /* time stepping and output parameters */
  pp.query("nsteps", nsteps);
  pp.query("plot_int", plot_int);

  /* binary fluid parameters */
  pp.query("lambda", chi);
  pp.query("T", T);
  pp.query("kappa", kappa);
  pp.query("gamma", Gamma);

  /* noise parameters */
  pp.query("temperature", temperature);

}

inline void WriteOutput(int step,
      const Geometry& geom,
			const MultiFab& hydrovs,
      StructFact& structFact) {
  // set up variable names for output
  const int zero_avg = 1;
  const int nvars = 5;
  const Vector<std::string> var_names = hydrovars_names(nvars);
  const std::string& pltfile = amrex::Concatenate("plt",step,5);
  WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, Real(step), step);
  structFact.WritePlotFile(step, static_cast<Real>(step), geom, "plt_SF", zero_avg);
}

void main_driver(const char* argv) {

  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();

  // read input parameters
  ReadInput();

  // set up Box and Geomtry
  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(nx-1, nx-1, nx-1);
  Array<int,3> periodicity({1,1,1});
  int nghost = 2; // need two halo layers for gradients

  Box domain(dom_lo, dom_hi);
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
  BoxArray ba(domain);
  DistributionMapping dm(ba);

  // set up MultiFabs
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, 2*nvel, nghost);
  MultiFab noise(ba, dm, 2*nvel, nghost);

  // set up StructFact
  int nStructVars = 5;
  const Vector<std::string> var_names = hydrovars_names(nStructVars);
  const Vector<int> pairA = { 0, 1, 2, 3, 4 };
  const Vector<int> pairB = { 0, 1, 2, 3, 4 };
  const Vector<Real> var_scaling = { 1.0, 1.0, 1.0, 1.0, 1.0 };
  StructFact structFact(ba, dm, var_names, var_scaling, pairA, pairB);

  // INITIALIZE
  LBM_init_mixture(fold, gold, hydrovs);
  if (plot_int > 0) WriteOutput(0, geom, hydrovs, structFact);
  Print() << "LB initialized\n";

  //unit_tests(geom, hydrovs);

  // TIMESTEP
  for (int step=1; step <= nsteps; ++step) {
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, noise);
    structFact.FortStructure(hydrovs, geom);
    if (plot_int > 0 && step%plot_int ==0) {
      WriteOutput(step, geom, hydrovs, structFact);
      Print() << "LB step " << step << "\n";
    }
  }

  // Call the timer again and compute the maximum difference between the start time 
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);
  amrex::Print() << "Run time = " << stop_time << " s" << std::endl;
  
}
