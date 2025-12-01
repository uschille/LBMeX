#ifndef LBM_DEBUG_H_
#define LBM_DEBUG_H_

#include "LBM_shan-chen.H"

void PrintMFComponent(MultiFab& mfab, int compIdx=0, int ngrow=-1){
    int ncomp = mfab.nComp();
    auto const & mfab_ptr = mfab.arrays();
    if(compIdx >= ncomp) {
        Print() << "Error: compIdx >= ncomp (" << compIdx << " >= " << ncomp << ") in function [PrintMFComponent] \n";
        return;
    }
    IntVect halo;
    if(ngrow<0){
        halo = IntVect(mfab.nGrow());
    }else{
        halo = IntVect(ngrow);
    }
    // loop over all boxes in the MultiFab
    for (MFIter mfi(mfab); mfi.isValid(); ++mfi){
        const Box& bx = grow(mfi.validbox(), halo);
        const Array4<Real>& fab_arr = mfab.array(mfi);
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        Print() << "====== Box " << mfi.index() << " component " << compIdx << " ======\n";
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    Print() << "(" << i << "," << j << "," << k << ") = " << fab_arr(i,j,k,compIdx) << "  ";
                }
                Print() << "\n";
            }
            Print() << "\n";
        }
        Print() << "===========================\n";
    }
}


#endif