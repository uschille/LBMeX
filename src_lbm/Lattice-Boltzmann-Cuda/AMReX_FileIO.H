#ifndef LBM_IO_
#define LBM_IO_
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sys/stat.h>
#include "LBM_binary.H"

using namespace std;

/*
    [filename]: directory of the folder which contains: folder $levelPrefix_$nlevel (e.g., Level_0) and Header file;
    Load single MultiFab variable from disk to [mfab_in];
*/
AMREX_FORCE_INLINE
bool LoadSingleMultiFab(std::string filename, MultiFab& mfab_in, int nlevel=0,
    const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){

    if(!mfab_in.ok()) {
        Print() << "MultiFab passed in is not defined\n";
        Print() << "the BoxArray on the disk will be used and a new DistributionMapping will be made (called from function 'LoadSingleMultiFab').\n";
    }
    if(!filesystem::exists(filename)){
        Print() << "Cannot load in MultiFab " << filename << '\n';
        return false;
    }else{
        VisMF::Read(mfab_in, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
        Print() << "Load in MultiFab " << filename << " successfully\n";
        return true;
    }
}

/*
    [filename]: directory of the folder which contains: folder $levelPrefix_$nlevel (e.g., Level_0) and Header file;
    Load specific component [loadcomp] of MultiFab variable from disk to [mfab_in];
*/
AMREX_FORCE_INLINE
bool LoadSlicedMultiFab(std::string filename, MultiFab& mfab_in, int loadcomp=0, int nlevel=0,
    const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){
    
    MultiFab mfab_readin;
    if(!mfab_in.ok()) {
        Print() << "MultiFab passed in is not defined\n";
        Print() << "the BoxArray on the disk will be used and a new DistributionMapping will be made (called from function 'LoadSlicedMultiFab').\n";
    }else{
        VisMF::Read(mfab_readin, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
        int ncomp = mfab_readin.nComp();
        mfab_readin.define(mfab_in.boxArray(), mfab_in.DistributionMap(), ncomp, mfab_in.nGrow());
    }
    if(!filesystem::exists(filename)){
        Print() << "Cannot load in MultiFab " << filename << '\n';
        return false;
    }else{
        VisMF::Read(mfab_readin, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
        Print() << "Load in MultiFab " << filename << " successfully\n";
        mfab_in.ParallelCopy(mfab_readin, loadcomp, 0, 1);    // load from srccomp [loadcomp] to destcomp [0] of [mfab_readin], total 1 component;
        return true;
    }
}

/*
    [plot_file_root]:   stands for root name of plotfile; it consists of the directory name which contains all multifabs 
    and each multifab folder's prefix, i.e., dir/plt***, then plot_file_root=dir/plt; *** denotes time step info;
    [ndigits]:  number of digits 
*/
AMREX_FORCE_INLINE
bool LoadSetOfMultiFabs(std::string plot_file_root, MultiFab& mfab_in, int step1, int step2, int plot_int,
    int loadcomp=0, bool if_print=false,
    int ndigits=5, int nlevel=0, const std::string &levelPrefix="Level_", const std::string &mfPrefix="Cell"){

    MultiFab mfab_readin;
    std::string filename;
    int nstep = (step2-step1)/plot_int + 1;
    if(!mfab_in.ok()) {
        Print() << "MultiFab passed in is not defined\n";
        Print() << "the BoxArray on the disk will be used and a new DistributionMapping will be made (called from function 'LoadSetOfMultiFabs').\n";
        filename = amrex::Concatenate(plot_file_root,step1,ndigits);
        if(!filesystem::exists(filename)){
            Print() << "Cannot load in MultiFab " << filename << '\n';
            throw runtime_error("called by function 'LoadSetOfMultiFabs'!\n");
            return false;
        }else{
            VisMF::Read(mfab_readin, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
            mfab_in.define(mfab_readin.boxArray(), mfab_readin.DistributionMap(), nstep, mfab_readin.nGrow());
        }
    }else{
        if(mfab_in.nComp()<nstep){
            printf("Input multifab has less components than required!\n");
            throw runtime_error("called by function 'LoadSetOfMultiFabs'!\n");
            return false;
        }
    }
    int nfile=0;
    for(int step=step1; step<=step2; step=step+plot_int){
        filename = amrex::Concatenate(plot_file_root,step,ndigits);
        if(!filesystem::exists(filename)){
            Print() << "Cannot load in MultiFab " << filename << '\n';
            throw runtime_error("called by function 'LoadSetOfMultiFabs'!\n");
            return false;
        }else{
            VisMF::Read(mfab_readin, amrex::MultiFabFileFullPrefix(nlevel, filename, levelPrefix, mfPrefix));
            if(if_print){
                printf("Read in file %s ...\n", filename.c_str());
            }
            mfab_in.ParallelCopy(mfab_readin, loadcomp, nfile, 1);
            nfile++;
        }
    }
    return true;
}

AMREX_FORCE_INLINE
void PrintMultiFab(const MultiFab& mfab, int ngrow=-1){
    int ncomp = mfab.nComp();
    auto const & mfab_ptr = mfab.arrays();

    IntVect halo;
    if(ngrow<0){
        halo = IntVect(mfab.nGrow());
    }else{
        halo = IntVect(ngrow);
    }
    ParallelFor(mfab, halo, [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) {
        for(int c=0; c<ncomp; c++){
            printf("(%d,%d,%d,%d)-(%f)\t", i, j, k, c, mfab_ptr[nbx](i,j,k,c));
        }
        //Print() << '\n';
    });
    Print() << '\n';
}

// Variadic template to handle multiple parameters
template <typename... Args>
std::string format(const std::string& format_str, Args... args) {
    char buffer[256]; // Adjust buffer size as needed
    sprintf(buffer, format_str.c_str(), args...);
    return std::string(buffer);
}

// Hydrodynamic Variable Names Vectors; used for saving data
inline Vector<std::string> VariableNames(const int numVars) {
    // set variable names for output
    Vector<std::string> var_names(numVars);
    std::string name;
    int cnt = 0;
    // rho, phi
    var_names[cnt++] = "rho";
    var_names[cnt++] = "phi";
    // velx, vely, velz of fluid f
    for (int d=0; d<AMREX_SPACEDIM; d++) {
      name = "uf";
      name += (120+d);
      var_names[cnt++] = name;
    }
    var_names[cnt++] = "p_bulk";
    // velx, vely, velz of fluid g
    for (int d=0; d<AMREX_SPACEDIM; d++) {
      name = "ug";
      name += (120+d);
      var_names[cnt++] = name;
    }
    // acceleration_{x,y,z} of fluid f
    for (int d=0; d<AMREX_SPACEDIM; d++) {
      name = "af";
      name += (120+d);
      var_names[cnt++] = name;
    }
    // acceleration_{x,y,z} of fluid g
    for (int d=0; d<AMREX_SPACEDIM; d++) {
      name = "ag";
      name += (120+d);
      var_names[cnt++] = name;
    }
  
    // The following part is cutoff due to [cnt]>=variables number we input
    // pxx, pxy, pxz, pyy, pyz, pzz
    for (int i=0; i<AMREX_SPACEDIM, cnt<numVars; ++i) {
      for (int j=i; j<AMREX_SPACEDIM, cnt<numVars; ++j) {
        name = "p";
        name += (120+i);
        name += (120+j);
        var_names[cnt++] = name;
      }
    }
    // kinetic moments
    for (; cnt<numVars;) {
      name = "m";
      name += std::to_string(cnt);
      var_names[cnt++] = name;
    }
  
    // Print all variable names
    Print() << "Variable names list: ";
    for(int n=0; n<cnt; n++){
      Print() << var_names[n] << " ";
    }
    Print() << '\n';
  
    return var_names;
  }

/*
  MultiFab testfunc(ba, dm, 5, nghost);
  MultiFab testfunc_cpy(ba, dm, 5, nghost);
  
  auto const & testFunc = testfunc.arrays();
  ParallelFor(testfunc, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    for(int ncomp=0; ncomp<5; ncomp++){
      testFunc[nbx](x,y,z,ncomp) = x+y+z+ncomp;
    }
  });
  std::vector<int> g1, g2;
  for (int i = 0; i < 5; i++)
    g1.push_back(i);

  WriteMultiFabToPlainText('X', testfunc, ".", "MFab/");
  //SliceWriteMultiFabToPlainText('Y', g1, testfunc, g1, ".", "MFab/");
  //SliceReadMultiFabToPlainText('Y', g1, testfunc_cpy, g1, ".", "MFab/");
  ReadMultiFabFromPlainText('X', testfunc_cpy, ".", "MFab/");

  auto const & testFunc_cpy = testfunc_cpy.arrays();
  ParallelFor(testfunc, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,0) - testFunc[nbx](x,y,z,0) << '\t';
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,1) - testFunc[nbx](x,y,z,1) << '\t';
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,2) - testFunc[nbx](x,y,z,2) << '\t';
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,3) - testFunc[nbx](x,y,z,3) << '\t';
    Print() << "(" << x << "," << y << "," << z << ")--" << testFunc_cpy[nbx](x,y,z,4) - testFunc[nbx](x,y,z,4) << '\t';
  });

*/


#endif