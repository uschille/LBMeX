#ifndef LBM_IO_H_
#define LBM_IO_H_

#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>

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
      const std::string pltname) {
  // set up variable names for output
  const Vector<std::string> var_names = VariableNames(2*nvel);
  const std::string& pltfile = amrex::Concatenate(pltname,step,7);
  WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, Real(step), step);
}

void WriteCheckpointFile(int step, const MultiFab& hydrovs){
    // std::string pltfile = amrex::Concatenate("checkpoint",step,7);
    const std::string& checkpointname = amrex::Concatenate("checkpoint_", step, 8);
    const int nlevels = 1;

    bool callBarrier = true;
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, callBarrier);

    if (ParallelDescriptor::IOProcessor()) {

        VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
        std::ofstream HeaderFile;
        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        std::string HeaderFileName(checkpointname + "/Header");
        HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                                                std::ofstream::trunc |
                                                std::ofstream::binary);

        if( ! HeaderFile.good()) {amrex::FileOpenFailed(HeaderFileName);}
        HeaderFile.precision(17);
        int finest_level = nlevels - 1;

        // write out title line
        HeaderFile << "Checkpoint file for LBMeX\n";

        // write the BoxArray at each level
        for (int lev = 0; lev <= finest_level; ++lev) {
            boxArray(lev).writeOn(HeaderFile);
            HeaderFile << '\n';
        }

        for (int lev = 0; lev <= finest_level; ++lev) {
            VisMF::Write(hydrovs[lev],
                            amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", "hydrovars"));
        }
    }
}

// utility to skip to next line in Header
void GotoNextLine (std::istream& is){
    constexpr std::streamsize bl_ignore_max { 100000 };
    is.ignore(bl_ignore_max, '\n');
}

void ReadCheckpointFile(int step, const MultiFab& hydrovs){
    const std::string& restart_chkfile = amrex::Concatenate("checkpoint_", step, 8);
    const std::string File(restart_chkfile + "/Header");
    const int nlevels = 1;
    int finest_level = 0;
    int ncomp = 38;
    int nghost = 2;

    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line, word;

    // read in title line
    std::getline(is, line);

    // read in finest_level
    is >> finest_level;
    GotoNextLine(is);

    for (int lev = 0; lev <= finest_level; ++lev) {
        // read in level 'lev' BoxArray from Header
        BoxArray ba;
        ba.readFrom(is);
        GotoNextLine(is);

        // create a distribution mapping
        DistributionMapping dm { ba, ParallelDescriptor::NProcs() };

        // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);

        // build MultiFab and FluxRegister data
        hydrovs[lev].define(grids[lev], dmap[lev], ncomp, nghost);
        if (lev > 0 && do_reflux) {
            flux_reg[lev] = std::make_unique<FluxRegister>(grids[lev], dmap[lev], refRatio(lev-1), lev, ncomp);
        }
    }

    // read in the MultiFab data
    for (int lev = 0; lev <= finest_level; ++lev) {
        VisMF::Read(hydrovs[lev],
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "phi"));
    }


}

#endif