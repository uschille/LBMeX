#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <StructFact.H>

using namespace amrex;

#include "LBM_shan-chen.H"
#include "LBM_tests.H"

// default grid parameters
IntVect domain_size(32);
IntVect max_box_size(16);

// default time stepping parameters
int nsteps;
int plot_print = 100;

// default output parameters
int plot_SF = 0;


inline void ReadInput() {
  ParmParse pp;

  /* grid parameters */
  pp.query("nx", domain_size[0]);
  domain_size[2] = domain_size[1] = domain_size[0]; // default to cubic box
  pp.query("ny", domain_size[1]);
  pp.query("nz", domain_size[2]);

  pp.query("max_grid_size_x", max_box_size[0]);
  max_box_size[2] = max_box_size[1] = max_box_size[0]; // default to same maxSize in all directions
  pp.query("max_grid_size_y", max_box_size[1]);
  pp.query("max_grid_size_z", max_box_size[2]);

  // initial condition
  init_cond = "droplet";
  pp.query("init_cond", init_cond);
  pp.query("init_frac", init_frac);

  // thermodynamic parameters
  pp.query("G", G);
  pp.query("temperature", temperature);

  /* time stepping and output parameters */
  pp.query("nsteps", nsteps);
  pp.query("plot_int", plot_int);
  pp.query("plot_SF", plot_SF);

}

inline void WriteOutput(int step,
      const Geometry& geom,
			const MultiFab& hydrovs,
      StructFact& structFact) {
  // set up variable names for output
  const int zero_avg = 1;
  const Vector<std::string> var_names = hydrovars_names(hydrovs.nComp());
  const std::string& pltfile = amrex::Concatenate("./data_droplet/plt",step,7);
  WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, Real(step), step);
  if (plot_SF > 0) structFact.WritePlotFile(step, static_cast<Real>(step), "plt_SF", zero_avg);
}

int main(int argc, char* argv[]) {
  amrex::Initialize(argc, argv);
  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();
  amrex::InitRandom(12345);
  // read input parameters
  ReadInput();
  int plot_start = 0;//nsteps/10*6;

  // set up Box and Geomtry
  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(domain_size-1);
  Array<int,3> periodicity({1,1,1});
  int nghost = 2; // need two halo layers for gradients

  Box domain(dom_lo, dom_hi);
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
  BoxArray ba(domain);
  ba.maxSize(max_box_size); // chop domain into boxes
  DistributionMapping dm(ba);

  // set up MultiFabs
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, 2*nvel, nghost);
  MultiFab refstate(ba, dm, 2, nghost);
  MultiFab noise(ba, dm, 2*nvel, nghost);

  // set up StructFact
  int nStructVars = 8;
  const Vector<std::string> var_names = hydrovars_names(nStructVars);
  const Vector<int> pairA = { 0, 1, 2, 3, 4, 5, 6, 7 };
  const Vector<int> pairB = { 0, 1, 2, 3, 4, 5, 6, 7 };
  const Vector<Real> var_scaling = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  StructFact structFact(ba, dm, var_names, var_scaling, pairA, pairB);

  // INITIALIZE
  Print() << "Total time steps N = " << nsteps << "\n";
  LBM_init(geom, fold, gold, hydrovs, refstate);
  PrintMFComponent(fold, 0, 0);
  if (plot_int > 0) WriteOutput(0, geom, hydrovs, structFact);
  Print() << "LB initialized lattice " << domain <<"\n" << ba << dm << std::endl;

  // unit_tests(geom, hydrovs);

  // TODO: for nonhomogeneous systems perform equilibration before copying reference state

  // copy the reference state
  ParallelCopy(refstate, hydrovs, 0, 0, 2, IntVect(nghost), IntVect(nghost));

  // TIMESTEP
  Print() << "Steps >= " << plot_start << " will be output \n";
  Print() << plot_int << " is the plot interval \n";
  for (int step=1; step <= nsteps; ++step) {
    if(step%plot_print==0){
      Print() << "LB step " << step << std::endl;
    }
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, refstate);
    PrintMFComponent(fold, 0, 0); // print density component
    if (plot_SF > 0) structFact.FortStructure(hydrovs);
    if (plot_int > 0 && step%plot_int ==0) {
      if(step >= plot_start) {
        Print() << "Writing output for step " << step << std::endl;
        WriteOutput(step, geom, hydrovs, structFact);
      }
    }
  }

  Print() << "LB completed " << nsteps << " time steps" << std::endl;

  // Call the timer again and compute the maximum difference between the start time 
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);
  amrex::Print() << "Run time = " << stop_time << " s (" << domain.numPts()*nsteps/stop_time << " LUP/s)" << std::endl;

  amrex::Finalize();
}
