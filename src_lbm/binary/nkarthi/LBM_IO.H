#ifndef LBM_IO_H_NK
#define LBM_IO_H_NK

#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_PlotFileDataImpl.H>

#include <sys/stat.h>
#include <chrono>
using namespace std::chrono;

#include "nkarthi/LBM_d3q19.H"

namespace nkarthi {

// // relaxation parameters
// AMREX_GPU_MANAGED Real tau_r = 0.7886751345948129; // tau=0.5*(1.+1./sqrt(3.)) minimizes spurious currents according to Swift et al.
// AMREX_GPU_MANAGED Real tau_p = 1.0;
// AMREX_GPU_MANAGED int nx = 16;
// AMREX_GPU_MANAGED int ny = 16;
// AMREX_GPU_MANAGED int nz = 16;
// AMREX_GPU_MANAGED int ic = 0;
// AMREX_GPU_MANAGED Real R = 0;

// // thermodynamic model
// AMREX_GPU_MANAGED Real kappa = 0.03;
// AMREX_GPU_MANAGED Real Gamma = 1.0;
// AMREX_GPU_MANAGED Real chi = 1.1;
// AMREX_GPU_MANAGED Real T = 0.6;

// // system temperature for fluctations
// AMREX_GPU_MANAGED Real temperature = 0.0;
AMREX_GPU_MANAGED int use_correlated_noise = 1;

//trying to move all input params to one place to clean code up
#if 0
struct input_params {
    // default grid parameters
    const int nx;
    const int ny;
    const int nz;
    IntVect max_grid_size;
    const int ic;
    const Real R;

    // default time stepping parameters
    int n_sci_start;
    int nsteps;
    int dump_hydro;
    int n_hydro;
    int dump_SF;
    int n_SF;
    int start_step;
    int output_hdf;

    // AMREX_GPU_MANAGED Real kappa = 0.03;
    const Real Gamma;
    const Real chi;
    const Real T;

    // Default constructor
    bulk_free_energy() : nx(16), ny(16), nz(16), max_grid_size({16, 16, 16}), ic(0), R(0.3), 
                         n_sci_start(0), nsteps(100), 
                         dump_hydro(1), n_hydro(10),  
                         dump_SF(0), n_SF(10)
                         start_step(0), output_hdf(0){}

};
#endif

namespace {
    void GotoNextLine (std::istream& is)
    {
        constexpr std::streamsize bl_ignore_max { 100000 };
        is.ignore(bl_ignore_max, '\n');
    }
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
  // rho, phi
  if (cnt<numVars) var_names[cnt++] = "density";
  if (cnt<numVars) var_names[cnt++] = "phi";
  // velx, vely, velz
  if (numVars == 2){return var_names;}
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "u";
    name += (120+d);
    var_names[cnt++] = name;
  }
  if (numVars == 5){return var_names;}
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    name = "phi*u";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // remaining moments
  for (int d=4; d < nvel; d++){
    name = "mf";
    name += std::to_string(d);
    var_names[cnt++] = name;
  }
  for (int d=4; d < nvel; d++){
    name = "mg";
    name += std::to_string(d);
    var_names[cnt++] = name;
  }
  return var_names;
}

inline void WriteOutput(int step,
			const MultiFab& hydrovs,
			const Geometry& geom,
      const std::string pltname,
      int nvar,
      int output_hdf) {
  // set up variable names for output
  const Vector<std::string> var_names = VariableNames(nvar);
  const std::string& pltfile = amrex::Concatenate(pltname,step,7);
  if(output_hdf)
    ;//WriteSingleLevelPlotfileHDF5(pltfile, hydrovs, var_names, geom, Real(step), step);
  else
    WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, Real(step), step);
}

inline bool file_exists(const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

void WriteCheckPoint(int step, MultiFab& hydrovs, const std::string pltname){
    // adapted from FHDeX/exec/src/hydro/Checkpoint.cpp
    // If wanting to run on GPU's need to comment out the RNG output. Maybe don't output it at all?
  
    // // timer for profiling
    // BL_PROFILE_VAR("WriteCheckPoint()",WriteCheckPoint);

    // checkpoint file name, e.g., chk0000010
    const std::string& checkpointname = amrex::Concatenate(pltname,step,10);

    amrex::Print() << "Writing checkpoint " << checkpointname << "\n";
    BoxArray ba = hydrovs.boxArray();

    // single level problem
    int nlevels = 1;

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);
    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    // write Header file
    if (ParallelDescriptor::IOProcessor()) {

        std::ofstream HeaderFile;
        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        std::string HeaderFileName(checkpointname + "/Header");
        HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                        std::ofstream::trunc |
                        std::ofstream::binary);

        if( !HeaderFile.good()) {
            amrex::FileOpenFailed(HeaderFileName);
        }
        HeaderFile.precision(17);
        // write out title line
        HeaderFile << "Checkpoint file for LBMeX\n";
        // write out the time step number
        HeaderFile << step << "\n";
        // write the BoxArray
        ba.writeOn(HeaderFile);
        HeaderFile << '\n';
    }

    int comm_rank = 0;
    int n_ranks = 1;
    #if AMREX_USE_MPI
        // C++ random number engine
        // have each MPI process write its random number state to a different file
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    #endif

    // don't write out all the rng states at once (overload filesystem)
    // one at a time write out the rng states to different files, one for each MPI rank
    for (int rank=0; rank<n_ranks; ++rank) {

        if (comm_rank == rank) {

            std::ofstream rngFile;
            rngFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

            // create filename, e.g. chk0000005/rng0000002
            const std::string& rngFileNameBase = (checkpointname + "/rng");
            const std::string& rngFileName = amrex::Concatenate(rngFileNameBase,comm_rank,7);
        
            rngFile.open(rngFileName.c_str(), std::ofstream::out   |
                         std::ofstream::trunc |
                         std::ofstream::binary);

            if( !rngFile.good()) {
                amrex::FileOpenFailed(rngFileName);
            }
    
            amrex::SaveRandomState(rngFile);

        }
        ParallelDescriptor::Barrier();
    }
    
    // write the MultiFab data to, e.g., chk00010/Level_0/
    VisMF::Write(hydrovs, amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "hydrovars"));
}

void ReadFile (const std::string& filename, Vector<char>& charBuf, bool bExitOnError){
    enum { IO_Buffer_Size = 262144 * 8 };

    #ifdef BL_SETBUF_SIGNED_CHAR
        typedef signed char Setbuf_Char_Type;
    #else
        typedef char Setbuf_Char_Type;
    #endif

    Vector<Setbuf_Char_Type> io_buffer(IO_Buffer_Size);

    Long fileLength(0), fileLengthPadded(0);

    std::ifstream iss;

    iss.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
    iss.open(filename.c_str(), std::ios::in);
    if ( ! iss.good()) {
        if(bExitOnError) {
            amrex::FileOpenFailed(filename);
        } else {
            fileLength = -1;
        }
    } else {
        iss.seekg(0, std::ios::end);
        fileLength = static_cast<std::streamoff>(iss.tellg());
        iss.seekg(0, std::ios::beg);
    }

    if(fileLength == -1) {
      return;
    }

    fileLengthPadded = fileLength + 1;
    // fileLengthPadded += fileLengthPadded % 8;
    charBuf.resize(fileLengthPadded);

    iss.read(charBuf.dataPtr(), fileLength);
    iss.close();

    charBuf[fileLength] = '\0';
}

void ReadCheckPoint(int& step, MultiFab& hydrovs, const std::string pltname, BoxArray& ba, DistributionMapping& dmap){
    // adapted from FHDeX/exec/src/hydro/Checkpoint.cpp
    // If wanting to run on GPU's need to comment out the RNG output. Maybe don't output it at all?
    
    // timer for profiling
    // BL_PROFILE_VAR("ReadCheckPoint()",ReadCheckPoint);

    // checkpoint file name, e.g., chk0000010
    // const std::string& checkpointname = amrex::Concatenate(chk_base_name,restart,7);
    const std::string& checkpointname = amrex::Concatenate(pltname,step,10);

    amrex::Print() << "Restart from checkpoint " << checkpointname << "\n";
    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());
    std::string line, word;

    // Header
    {
        std::string File(checkpointname + "/Header");
        Vector<char> fileCharPtr;
        ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
        std::string fileCharPtrString(fileCharPtr.dataPtr());
        std::istringstream is(fileCharPtrString, std::istringstream::in);

        // read in title line
        std::getline(is, line);

        // read in time step number
        is >> step;
        GotoNextLine(is);
        ++step;
        // read in level 'lev' BoxArray from Header
        ba.readFrom(is);
        GotoNextLine(is);

        // create a distribution mapping
        dmap.define(ba, ParallelDescriptor::NProcs());

        // build MultiFab data
        // int nghost = 2;
        // int ncomp = 38; // 38 is hard coded for now just for testing, should be 2*nvel
        // hydrovs.define(ba, dmap, ncomp, nghost);
      }

    int comm_rank = 0;
    int n_ranks = 1;
    #if AMREX_USE_MPI
        // C++ random number engine
        // each MPI process reads in its own file
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    #endif
    int seed = 0;
    if (seed < 0) {

    #ifdef AMREX_USE_CUDA
            Abort("Restart with negative seed not supported on GPU");
    #endif

        // read in rng state from checkpoint
        // don't read in all the rng states at once (overload filesystem)
        // one at a time write out the rng states to different files, one for each MPI rank
    for (int rank=0; rank<n_ranks; ++rank) {
        if (comm_rank == rank) {
            // create filename, e.g. chk0000005/rng0000002
            std::string FileBase(checkpointname + "/rng");
            std::string File = amrex::Concatenate(FileBase,comm_rank,7);

            // read in contents
            Vector<char> fileCharPtr;
            ReadFile(File, fileCharPtr, true);
            std::string fileCharPtrString(fileCharPtr.dataPtr());
            std::istringstream is(fileCharPtrString, std::istringstream::in);

            // restore random state
            amrex::RestoreRandomState(is, 1, 0);

        }
        ParallelDescriptor::Barrier();
      }
    } else if (seed == 0) { 
        // initializes the seed for C++ random number calls based on the clock
        auto now = time_point_cast<nanoseconds>(system_clock::now());
        int randSeed = now.time_since_epoch().count();
        // broadcast the same root seed to all processors
        ParallelDescriptor::Bcast(&randSeed,1,ParallelDescriptor::IOProcessorNumber());
        
        InitRandom(randSeed+ParallelDescriptor::MyProc(),
                   ParallelDescriptor::NProcs(),
                   randSeed+ParallelDescriptor::MyProc());
    }
    else {
        // initializes the seed for C++ random number calls
        InitRandom(seed+ParallelDescriptor::MyProc(),
                   ParallelDescriptor::NProcs(),
                   seed+ParallelDescriptor::MyProc());
    }

    // read in the MultiFab data
    VisMF::Read(hydrovs, amrex::MultiFabFileFullPrefix(0, checkpointname, "Level_", "hydrovars"));
}

void checkpointRestart(int& step, MultiFab& hydrovs, const std::string pltname, MultiFab& fold, MultiFab& gold, BoxArray& ba, DistributionMapping& dmap){
  ReadCheckPoint(step, hydrovs, pltname, ba, dmap); //writes checkpoint file to hydrovars

  // this loop converts from hydrovars to the f and g distributions requires to initiate the LBM timestepping
  for (MFIter mfi(fold); mfi.isValid(); ++mfi) {
    const Box& box = mfi.validbox();
    const Array4<Real>& f = fold.array(mfi);
    const Array4<Real>& g = gold.array(mfi);
    const Array4<Real>& h = hydrovs.array(mfi);
    // ordering of output is
    // rho phi ux uy uz phi*ux phi*uy phi*uz mf4...mf18 mg4...mg18
    ParallelForRNG(box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      Array1D<Real,0,nvel> mf; Array1D<Real,0,nvel> mg;
      Array1D<Real,0,nvel> f_dist; Array1D<Real,0,nvel> g_dist;
      mf(0) = h(x,y,z,0);
      mg(0) = h(x,y,z,1);
      for(int i = 1; i < 4; i++){
        mf(i) = h(x,y,z,i+1);
        mg(i) = h(x,y,z,i+4);
      }
      for(int i = 4; i < nvel; i++){
        mf(i) = h(x,y,z,i+4);
        mg(i) = h(x,y,z,i+nvel);
      }
      populations(x,y,z,f,mf);
      populations(x,y,z,g,mg);
    });
  }
}
}
#endif
