/*  *****************************************************************************************************
    *************************** Based on AMReX-MultiFab arithmetic operations ***************************
    *****************************************************************************************************
  // The function returns a single scalar (Real) value, which is the sum of the products of corresponding elements in the specified components of x and y;
  Print() << "multifab dot(): " << amrex::MultiFab::Dot(rho_eq, 0, Imfab, 0, 1, 0) << '\n';
  // Returns the sum of component "comp" (0 by default) over the MultiFab; :sum(int comp = 0, bool local = false)
  Print() << "multifab sum() " << rho_eq.sum() << '\n';
  // element-wise scaling amrex::MultiFab::Multiply(MultiFab& dst, const MultiFab& src, int srccomp, int dstcomp, int numcomp, int nghost)
  amrex::MultiFab::Multiply(rho_eq, Imfab, 0, 0, 1, 0);
  Print() << "after multifab Multiply() " << rho_eq.sum() << '\n';
  //Scales the value of each cell in the valid region of each component of the MultiFab by the scalar val
  void amrex::MultiFab::mult(Real val, int nghost=0)
  //Adds the scalar value val to the value of each cell in the specified subregion of the MultiFab which consists of the num_comp components starting at component comp. 
  void amrex::MultiFab::plus(Real val, int 	comp, int num_comp, int nghost=0)
*/

#ifndef AMREX_ANALY_
#define AMREX_ANALY_
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include "AMReX_FileIO.H"
#include <AMReX_FabArray.H>

AMREX_GPU_HOST_DEVICE
Real func3D_test(Real x, Real y, Real z, int n){
    return x*x * y*y * z*z;
}

AMREX_GPU_HOST_DEVICE
Real func3D_rho_test(Real x, Real y, Real z, int n){
    Real R = 0.1;   Real kappa = 0.02;
    return 0.5*(1+tanh((R-sqrt((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5)))/sqrt(kappa)));
}

class Function3DAMReX;  // forward declaration allows you to use pointers (and references) to that class without yet having the whole class definition.

AMREX_GPU_HOST_DEVICE
class Integration{
    public:
        static double*** create_3d_array(int Nx, int Ny, int Nz, Real initVal=1.) {    // by default, all values are initialized to be 1.;
            double*** arr = (double***)malloc(Nx * sizeof(double**));
            for (int i = 0; i < Nx; i++) {
                arr[i] = (double**)malloc(Ny * sizeof(double*));
                for (int j = 0; j < Ny; j++) {
                    arr[i][j] = (double*)malloc(Nz * sizeof(double));
                }
            }
            // Initialize all elements to 1
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    for (int k = 0; k < Nz; k++) {
                        arr[i][j][k] = 1.0;
                    }
                }
            }
            return arr;
        }

        // Static member function shared with external funcion;
        static Real trapezoid3D(const Function3DAMReX& func3D){
            return 0.;
        }

        static void trapezoid3DWeightTensor(const Box& box, Real**** wPtr){
            IntVect boxSize = box.length();
            int Nx = boxSize[0];    int Ny = boxSize[1];    int Nz = boxSize[2];
            (*wPtr) = create_3d_array(boxSize[0], boxSize[1], boxSize[2]);
            // Corner points, total 8 points
            (*wPtr)[0][0][0] = 1.0/8;
            (*wPtr)[Nx-1][0][0] = 1.0/8;
            (*wPtr)[0][Ny-1][0] = 1.0/8;
            (*wPtr)[Nx-1][Ny-1][0] = 1.0/8;
            (*wPtr)[0][0][Nz-1] = 1.0/8;
            (*wPtr)[Nx-1][0][Nz-1] = 1.0/8;
            (*wPtr)[0][Ny-1][Nz-1] = 1.0/8;
            (*wPtr)[Nx-1][Ny-1][Nz-1] = 1.0/8;

            // Boundary edge points, total 12 edges
            // 4 vertical edges
            for (int k = 1; k < Nz-1; k++) {
                (*wPtr)[0][0][k] = 1.0/4;
                (*wPtr)[Nx-1][0][k] = 1.0/4;
                (*wPtr)[0][Ny-1][k] = 1.0/4;
                (*wPtr)[Nx-1][Ny-1][k] = 1.0/4;
            }
            // 4 x-parallel horizontal edges
            for (int i = 1; i < Nx-1; i++) {
                (*wPtr)[i][0][0] = 1.0/4;
                (*wPtr)[i][0][Nz-1] = 1.0/4;
                (*wPtr)[i][Ny-1][Nz-1] = 1.0/4;
                (*wPtr)[i][Ny-1][0] = 1.0/4;
            }
            // 4 y-parallel horizontal edges
            for (int j = 1; j < Ny-1; j++) {
                (*wPtr)[0][j][0] = 1.0/4;
                (*wPtr)[0][j][Nz-1] = 1.0/4;
                (*wPtr)[Nx-1][j][Nz-1] = 1.0/4;
                (*wPtr)[Nx-1][j][0] = 1.0/4;
            }
            // Interior points of the bounding surfaces, total 6 surfaces
            // Surfaces parallel with xy-plane
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    if ((*wPtr)[i][j][0] == 1.0) {
                        (*wPtr)[i][j][0] = 1.0/2;
                    }
                    if ((*wPtr)[i][j][Nz-1] == 1.0) {
                        (*wPtr)[i][j][Nz-1] = 1.0/2;
                    }
                }
            }
            // Surfaces parallel with xz-plane
            for (int i = 0; i < Nx; i++) {
                for (int k = 0; k < Nz; k++) {
                    if ((*wPtr)[i][0][k] == 1.0) {
                        (*wPtr)[i][0][k] = 1.0/2;
                    }
                    if ((*wPtr)[i][Ny-1][k] == 1.0) {
                        (*wPtr)[i][Ny-1][k] = 1.0/2;
                    }
                }
            }
            // Surfaces parallel with yz-plane
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    if ((*wPtr)[0][j][k] == 1.0) {
                        (*wPtr)[0][j][k] = 1.0/2;
                    }
                    if ((*wPtr)[Nx-1][j][k] == 1.0) {
                        (*wPtr)[Nx-1][j][k] = 1.0/2;
                    }
                }
            }
        }

        static void free_3d_array(Real*** dataPtr, int Nx, int Ny, int Nz){
            for (int i = 0; i < Nx; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    free(dataPtr[i][j]);
                }
                // Free the array of row pointers
                free(dataPtr[i]);
            }
            // Free the outermost array of 2D pointers
            free(dataPtr);
        }
};


class Function3DAMReX{
    private:
        BoxArray ba;
        Geometry geom;               
        DistributionMapping dm;    
        int ncomp;                 // Number of components of the 3D function
        int ngrow;                 // Number of ghost cells

        MultiFab func_mfab;
    public:
        AMREX_GPU_DEVICE Function3DAMReX(const BoxArray ba_, const Geometry geom_, const DistributionMapping dm_, Real (*func) (Real, Real, Real, int), int ncomp_, int ngrow_);

        AMREX_GPU_DEVICE Function3DAMReX(const BoxArray ba_, const Geometry geom_, const DistributionMapping dm_, int ngrow_, Real*** dataPtr); // 1 component total, constructed from 3D array with same size of Box;

        AMREX_GPU_DEVICE Function3DAMReX(const MultiFab& mfab, const Geometry& geom_) {
            ba = mfab.boxArray();
            geom = geom_;
            dm = mfab.DistributionMap();
            ncomp = mfab.nComp();
            ngrow = mfab.nGrow();

            func_mfab.define(ba, dm, ncomp, ngrow); 
            // MUST FIRST define the MultiFab with same DistributionMapping, otherwise dest mfab has empty [dm] property; SAME for boxArray;
            func_mfab.ParallelCopy(mfab);
        }

        // Accessors for member variables, cannot MODIFY them;
        // In this way, BoxArray ba_func = func3D.getBoxArray(); [ba_func] will be a new object which violates the purpose of "const"
        //  BoxArray getBoxArray() const { return ba; }  
        // providing two choices: return const or non-const reference of MultiFab members 
        AMREX_GPU_DEVICE const MultiFab& getMultiFab() const { return func_mfab; }   // Call it by using "CONST reference" to void using the deleted constructor of MultiFab;
        AMREX_GPU_DEVICE MultiFab& getMultiFab() { return func_mfab; }
        
        
        AMREX_GPU_DEVICE const BoxArray getBoxArray() const { return ba; }
        AMREX_GPU_DEVICE const Geometry getGeometry() const { return geom; }
        AMREX_GPU_DEVICE const DistributionMapping getDistributionMapping() const { return dm; }
        AMREX_GPU_DEVICE const int getnComp() const { return ncomp; }
        AMREX_GPU_DEVICE const int getnGrow() const { return ngrow; }
        
        // Modify elementary properties;
        AMREX_GPU_DEVICE void setBoxArray(const BoxArray& ba_) { ba = ba_; } // does not affect ba in multifab;
        AMREX_GPU_DEVICE void setGeometry(const Geometry& geom_) { geom = geom_; }
        AMREX_GPU_DEVICE void setDistributionMapping(const DistributionMapping& dm_) { dm = dm_; }
        AMREX_GPU_DEVICE void setnComp(int ncomp_) { ncomp = ncomp_; }
        AMREX_GPU_DEVICE void setnGrow(int ngrow_) { ngrow = ngrow_; }

        // ******Ghost layers are not included by default******;
        // Modify element at grid point (x,y,z) of component k
        AMREX_GPU_DEVICE void setElement(Real val, int x, int y, int z, int n);
        // Get element at grid point (x,y,z) of component k
        AMREX_GPU_DEVICE Real getElement(int x, int y, int z, int n);

        // *********** Basic ELEMENT-WISE arithmetic operation ***********
        /* cannot add "const" at the ending, in that case the member variable [func_mfab] declared const, you cannot bind its Array4<const amrex::Real> result
        to a variable of type Array4<amrex::Real> or const Array4<amrex::Real> */
        AMREX_GPU_DEVICE void add(const Function3DAMReX& func3D);
        AMREX_GPU_DEVICE void add(const Real scalar);    
        AMREX_GPU_DEVICE void mult(const Function3DAMReX& func3D);
        AMREX_GPU_DEVICE void mult(const Real scalar);
        /*  Return the sum of component "compIdx" (-1 by default) over the MultiFab; if compIdx<0, then summation over all components
        are returned */
        AMREX_GPU_DEVICE Real sum(const int compIdx);

        // member functions provided for numerical computations
        /* This function is build on [integral3D] without multiplicand function, which consumes double data memory size when calling constructor; 
            Both functions DO NOT modify the original data  */
        AMREX_GPU_DEVICE Real integral3D(const Function3DAMReX& func_multiplicand, Real*** weightTensor=NULL);    // calculate the integral of \int (*this)\cdot func_weight dV

        AMREX_GPU_DEVICE Real integral3D(Real*** weightTensor=NULL);

        AMREX_GPU_DEVICE void fillPeriodicBC(){  func_mfab.FillBoundary(geom.periodicity()); }
        
        AMREX_GPU_DEVICE void convertNodal2Cell();
        AMREX_GPU_DEVICE void convertCell2Nodal(bool ifPeriodic);

        // Remove Overlapping Regions for nodal-centered data; 
        //  This function WILL MODIFY original boxArray passed in for the constructor; 
        //  i.e., when calling "Function3DAMReX(BoxArray ba_, ...)", though ba_ is passed in by value, it will be affected by this function;
        AMREX_GPU_DEVICE void makeDivision();    
        AMREX_GPU_DEVICE void invmakeDivision(); // inverse operation of function makeDivision();

};

AMREX_GPU_DEVICE
Function3DAMReX::Function3DAMReX(const BoxArray ba_, const Geometry geom_, const DistributionMapping
    dm_, int ngrow_, Real*** dataPtr){
    
    ba = ba_;
    geom = geom_;
    dm = dm_;
    ngrow = ngrow_;
    ncomp = 1;
    func_mfab.define(ba, dm, ncomp, ngrow); // Allocating multifab object;

    // Get the dimensions of the Box
    const amrex::Box& domain = geom.Domain();
    int nx = domain.length(0); // Size in x-direction
    int ny = domain.length(1); // Size in y-direction
    int nz = domain.length(2); // Size in z-direction

    // Flatten the 3D array into a contiguous 1D array
    amrex::Gpu::DeviceVector<Real> dataPtr_d(nx * ny * nz);
    Real* dataPtr_d_ptr = dataPtr_d.data();
    // Copy dataPtr to the flattened 1D array
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                int idx = i + nx * j + nx * ny * k; // Correct linear index
                dataPtr_d_ptr[idx] = dataPtr[i][j][k];
            }
        }
    }

    for (amrex::MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);

        // Use ParallelFor to copy data from dataPtr to the MultiFab
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int idx = (i - domain.smallEnd(0)) + nx * (j - domain.smallEnd(1)) + nx * ny * (k - domain.smallEnd(2));
            mf_array(i, j, k, 0) = dataPtr_d_ptr[idx];
        });
    }
}

AMREX_GPU_DEVICE
Function3DAMReX::Function3DAMReX(const BoxArray ba_, const Geometry geom_, const DistributionMapping
    dm_, Real (*func) (Real, Real, Real, int), int ncomp_=1, int ngrow_=0) : ncomp(ncomp_), ngrow(ngrow_){
            
    ba = ba_;
    geom = geom_;
    dm = dm_; 
    ngrow = ngrow_;
    ncomp = ncomp_;
    //  IndexType typ({AMREX_D_DECL(1,1,1)}); // nodal
    // ************     to be consistent with Trapezoid formula     ************
    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    IndexType itype = ba.ixType(); // using Box or BoxArray;
    //Print() << itype[0] << '-' << itype[1] << '\n';
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        Print() << "cell centered\n";
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
        //cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
        Print() << "nodal centered\n";
    }
    
    func_mfab.define(ba, dm, ncomp, ngrow); // Allocating multifab object;
    for(amrex::MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);

        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            //mf_array(i,j,k,n) = func(i*cell_size[0],j*cell_size[1],k*cell_size[2],n);
            mf_array(i,j,k,n) = func((i+0.5*(1-itype[0]))*cell_size[0],(j+0.5*(1-itype[1]))*cell_size[1],(k+0.5*(1-itype[2]))*cell_size[2],n);
            //printf("Nodal(%d,%d,%d) = %f\t",i,j,k,mf_array(i,j,k,n));
        });
    }
}

/*
    copy constructor of 'Function3DAMReX' is implicitly deleted because field 'func_mfab' has a deleted copy constructor
        MultiFab func_mfab;
*/
AMREX_GPU_DEVICE
void Function3DAMReX::add(const Function3DAMReX& func3D) {
    const MultiFab& mfab_add = func3D.getMultiFab();

    for(amrex::MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<amrex::Real>& mf_array = func_mfab.array(mfi);
        /* function to be added; Since it is declared to be const, so here use type "Array4<const amrex::Real>" instead;
        see comments in class definition */
        const amrex::Array4<const amrex::Real>& mf_array_add = mfab_add.array(mfi);   
        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            mf_array(i,j,k,n) = mf_array(i,j,k,n) + mf_array_add(i,j,k,n);
        });
    }
}

AMREX_GPU_DEVICE
void Function3DAMReX::add(const Real scalar){
    func_mfab.plus(scalar, 0, ncomp);
}

// overlapping points in nodal-centered data will NOT affect the amrex::Multiply operations;
AMREX_GPU_DEVICE
void Function3DAMReX::mult(const Function3DAMReX& func3D){
    const MultiFab& mfab_mult = func3D.getMultiFab();
    amrex::MultiFab::Multiply(func_mfab, mfab_mult, 0, 0, ncomp, 0); // using library provided by AMReX;
}

AMREX_GPU_DEVICE
void Function3DAMReX::mult(const Real scalar){
    func_mfab.mult(scalar);
}

AMREX_GPU_DEVICE
Real Function3DAMReX::sum(const int compIdx){
    IndexType itype = ba.ixType();
    if(itype.nodeCentered()){ // remove overlapping grid points' effect
    //  Basic idea is:  remove overlaps & restore the original amrex data structure;
        //Print() << ba << '\n';
        this->makeDivision();
        //Print() << ba << '\n';
        //Print() << func_mfab.boxArray() << '\n';
    }else{
        if(!itype.cellCentered()){
            throw runtime_error("sum() only supports either nodal-centered or cell-centered data!\n");
        }
    }

    Real sumVal = 0.;
    amrex::Gpu::DeviceScalar<Real> sumVal_device(sumVal);
    Real* sumVal_ptr = sumVal_device.dataPtr();
    if (compIdx >= 0) {
        for (MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& valid_box = mfi.validbox();
            Array4<Real> const& nodal_arr = func_mfab.array(mfi);
            amrex::ParallelFor(valid_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // CUDA kernel code
                atomicAdd(sumVal_ptr, nodal_arr(i, j, k, compIdx));
            });
        }
    } else {
        for (MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& valid_box = mfi.validbox();
            Array4<Real> const& nodal_arr = func_mfab.array(mfi);
            amrex::ParallelFor(valid_box, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                // CUDA kernel code
                atomicAdd(sumVal_ptr, nodal_arr(i, j, k, n));
            });
        }
    }
    
    if(itype.nodeCentered()){ // recover overlapping grid points' effect if possible;
        this->invmakeDivision();
    }
    // Copy the result back to the host
    sumVal = sumVal_device.dataValue();
    return sumVal;
}

AMREX_GPU_DEVICE
void Function3DAMReX::setElement(Real val, int x, int y, int z, int n){
    for (MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        FArrayBox& fab = func_mfab[mfi];
        if (bx.contains(IntVect(x, y, z))) {
            fab(IntVect(x, y, z), n) = val;
            break;
        }
    }
}

AMREX_GPU_DEVICE
Real Function3DAMReX::getElement(int x, int y, int z, int n){
    Real val = -1e-30;
    for (MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        FArrayBox& fab = func_mfab[mfi];
        auto fab_array = fab.array();
        if (bx.contains(IntVect(x, y, z))) {
            val = fab_array(x,y,z,n);
            break;
        }
    }
    return val;
}

AMREX_GPU_DEVICE
Real Function3DAMReX::integral3D(const Function3DAMReX& func_multiplicand, Real*** weightTensor){
    // First compare box size if they are same;
    // Build copy for function itself to avoid directly modifying the data during integral calculations;
    Function3DAMReX func_copy(func_mfab, geom); 
    
    Real*** wt = weightTensor;
    if(wt == NULL){
        Integration::trapezoid3DWeightTensor(geom.Domain(), &wt);
    }
    func_copy.mult(func_multiplicand);
    Real intVal = func_copy.integral3D(wt);
    return intVal;
}

AMREX_GPU_DEVICE
Real Function3DAMReX::integral3D(Real*** weightTensor){
    Function3DAMReX func_copy(func_mfab, geom);
    IndexType itype = ba.ixType();
    Real*** wt = weightTensor;
    bool ifFreeWeightMat = false;
    if(itype.nodeCentered()){
        if(wt == NULL){
            Print() << "weight matrix initialized" << '\n'; ifFreeWeightMat = true;
            Integration::trapezoid3DWeightTensor(geom.Domain(), &wt);
        }
        Function3DAMReX func_weight(ba, geom, dm, ngrow, wt); // cannot use reference here;
        func_copy.mult(func_weight);    //MultiFab& func_mfab_copy = func_copy.getMultiFab();
    }
    //WriteMultiFabToPlainText('z', weight_mfab, "./Deb");
    
    const IntVect box = geom.Domain().length();
    RealBox real_box = geom.ProbDomain();
    Real Lx = real_box.hi(0) - real_box.lo(0);
    Real Ly = real_box.hi(1) - real_box.lo(1);
    Real Lz = real_box.hi(2) - real_box.lo(2);
    //Print() << itype << '-' << box << '\n';
    Array<Real, 3> cell_size;
    if(itype.cellCentered()){
        //cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
        cell_size = {Lx/box[0], Ly/box[1], Lz/box[2]};
    }else if(itype.nodeCentered()){
        cell_size = {Lx/(box[0]-1), Ly/(box[1]-1), Lz/(box[2]-1)};
    }
    Real cell_volume = cell_size[0]*cell_size[1]*cell_size[2];
    //Print() << "Cell Size: " << cell_size << '\n';
    Real val = func_copy.sum(0);
    // Or implemented by definition
    /*Real val = 0.;
    for(amrex::MFIter mfi(func_mfab); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);

        ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k){
            val += mf_array(i,j,k,0)*wt[i][j][k];
        });
    }*/
    if(ifFreeWeightMat){
        Integration::free_3d_array(wt, box[0], box[1], box[2]);
    }
    return val*cell_volume;
}

AMREX_GPU_DEVICE
void Function3DAMReX::convertNodal2Cell(){
    // MultiFab copy for original data; By default it is nodal-centered;
    MultiFab func_mfab_copy;
    func_mfab_copy.define(ba, dm, ncomp, ngrow);
    func_mfab_copy.ParallelCopy(func_mfab);

    // convert(func_mfab.boxArray(), IntVect::TheNodeVector()); // does not change dm;
    // !!!!!!!!!!!!!!!  UPDATING boxArray & Geometry    !!!!!!!!!!!!!!!
    ba = convert(ba, IntVect::TheCellVector());
    dm.define(ba);
    // Get the physical domain (RealBox)
    const RealBox& physical_domain = geom.ProbDomain();
    const Array<int, AMREX_SPACEDIM> is_periodic = {geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2)};
    geom.define(ba.minimalBox(), physical_domain, CoordSys::cartesian, is_periodic);

    func_mfab.define(ba, dm, ncomp, ngrow);
    Print() << "node to cell boxArray: " << func_mfab.boxArray() << '\n';
    for (MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& valid_box = mfi.validbox();
    
        Array4<Real> const& node_arr = func_mfab_copy.array(mfi); // This function include ghost layers by default;
        Array4<Real> const& cell_arr = func_mfab.array(mfi);

        ParallelFor(valid_box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            cell_arr(i, j, k, n) = 0.125 * (node_arr(i, j, k, n) + node_arr(i + 1, j, k, n) + node_arr(i, j + 1, k, n) +
                node_arr(i, j, k + 1, n) + node_arr(i + 1, j + 1, k, n) + node_arr(i + 1, j, k + 1, n) + node_arr(i, j + 1, k + 1, n) + node_arr(i + 1, j + 1, k + 1, n));
        });
    }
}

AMREX_GPU_DEVICE
void Function3DAMReX::convertCell2Nodal(bool ifPeriodic=true){
    // MultiFab copy for original data; By default it is cell-centered;
    MultiFab func_mfab_copy;
    func_mfab_copy.define(ba, dm, ncomp, ngrow+1);  // !!! to ensure the cell-centered data will include ghost cells !!!

    func_mfab_copy.ParallelCopy(func_mfab);
    func_mfab_copy.FillBoundary(geom.periodicity());
    //PrintMultiFab(func_mfab_copy);

    // !!!!!!!!!! UPDATING boxArray & Geometry  !!!!!!!!!!!!!!!
    ba = convert(ba, IntVect::TheNodeVector());
    dm.define(ba);
    // Get the physical domain (RealBox)
    const RealBox& physical_domain = geom.ProbDomain();
    const Array<int, AMREX_SPACEDIM> is_periodic = {geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2)};
    geom.define(ba.minimalBox(), physical_domain, CoordSys::cartesian, is_periodic);

    // convert(func_mfab.boxArray(), IntVect::TheNodeVector()); // does not change dm;
    func_mfab.define(ba, dm, ncomp, ngrow);
    //Print() << func_mfab.boxArray() << "ghost layer = " << ngrow << '\n';

    for (MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        //const Box& node_box = amrex::convert(mfi.validbox(), IntVect::TheNodeVector()); // here the input params are constant reference &;
        const Box& valid_box = mfi.validbox();
        const amrex::Box& ghost_box = amrex::grow(valid_box, 1); 
    
        Array4<Real> const& cell_arr = func_mfab_copy.array(mfi); // This function include ghost layers by default;
        Array4<Real> const& node_arr = func_mfab.array(mfi);

        ParallelFor(valid_box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            node_arr(i, j, k, n) = 0.125 * (cell_arr(i, j, k, n) + cell_arr(i - 1, j, k, n) + cell_arr(i, j - 1, k, n) +
                cell_arr(i, j, k - 1, n) + cell_arr(i - 1, j - 1, k, n) + cell_arr(i - 1, j, k - 1, n) + cell_arr(i, j - 1, k - 1, n) + cell_arr(i - 1, j - 1, k - 1, n));
        });
    }
}

AMREX_GPU_DEVICE
void Function3DAMReX::makeDivision(){
    // ********* these lines solves Problem I ?? ****************************
    MultiFab func_mfab_copy;
    func_mfab_copy.define(ba, dm, ncomp, ngrow);
    func_mfab_copy.ParallelCopy(func_mfab);
    // **************************************

    IndexType itype = ba.ixType();
    Box domain_nodal = geom.Domain();
    if(itype.nodeCentered()){ // remove overlapping grid points' effect
    //  Basic idea is:  remove overlaps by shrinking the bigEnd points of each box within boxArray;
        for (int i = 0; i < ba.size(); ++i) {
            Box box = ba[i];   // operator [] return temporary object, so CANNOT directly modify the boxes within boxArray;
            // Box& boxref = ba[i]; THIS CANNOT be compiled; Binding non-constant reference to temporary object box;
            // Remove the overlap by shrinking the box on its upper boundaries
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
                if (box.bigEnd(dim) == domain_nodal.bigEnd(dim)){
                    continue; // Keep domain boundary
                }else{
                    box.growHi(dim, -1);  // Shrink the upper boundary to avoid overlap
                    ba.set(i, box);
                }
            }
        }
    }else{
        throw runtime_error("makeDivision() only supports nodal-centered data!\n");
    }
    const RealBox& physical_domain = geom.ProbDomain();
    const Array<int, AMREX_SPACEDIM> is_periodic = {geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2)};
    geom.define(ba.minimalBox(), physical_domain, CoordSys::cartesian, is_periodic);
    dm.define(ba);  func_mfab.define(ba, dm, ncomp, ngrow);
    for(amrex::MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);
        amrex::Array4<amrex::Real> const& mf_array_old = func_mfab_copy.array(mfi);
        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            mf_array(i,j,k,n) = mf_array_old(i,j,k,n);
        });
    }
    func_mfab.FillBoundary(geom.periodicity());
}

AMREX_GPU_DEVICE
void Function3DAMReX::invmakeDivision(){
    // **************************************
    MultiFab func_mfab_copy;
    func_mfab_copy.define(ba, dm, ncomp, ngrow+1);
    func_mfab_copy.ParallelCopy(func_mfab);
    func_mfab_copy.FillBoundary(geom.periodicity());  
    // ****************************************

    IndexType itype = ba.ixType();
    Box domain_nodal = geom.Domain();
    if(itype.nodeCentered()){ // recover overlapping grid points
        for (int i = 0; i < ba.size(); ++i) {
            Box box = ba[i];   // operator [] return temporary object, so CANNOT directly modify the boxes within boxArray;
            // Box& boxref = ba[i]; THIS CANNOT be compiled; Binding non-constant reference to temporary object box;
            // Remove the overlap by shrinking the box on its upper boundaries
            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
                if (box.bigEnd(dim) == domain_nodal.bigEnd(dim)){
                    continue; // Keep domain boundary
                }else{
                    box.growHi(dim, 1);  // Shrink the upper boundary to avoid overlap
                    ba.set(i, box);
                }
            }
        }
    }else{
        throw runtime_error("makeDivision() only supports nodal-centered data!\n");
    }
    const RealBox& physical_domain = geom.ProbDomain();
    const Array<int, AMREX_SPACEDIM> is_periodic = {geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2)};
    geom.define(ba.minimalBox(), physical_domain, CoordSys::cartesian, is_periodic);
    dm.define(ba);  func_mfab.define(ba, dm, ncomp, ngrow);
    for(amrex::MFIter mfi(func_mfab, TilingIfNotGPU()); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& mf_array = func_mfab.array(mfi);
        amrex::Array4<amrex::Real> const& mf_array_old = func_mfab_copy.array(mfi);
        ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n){
            mf_array(i,j,k,n) = mf_array_old(i,j,k,n);
        });
    }
    func_mfab.FillBoundary(geom.periodicity());
}

// each vector element stores [Ndim] variables' mean, variance, maximum, minimum, abs_max, abs_min respectively, total 6;
//  eg: return_vector[k][ncomp]:    k'th property (like mean, variance...) of ncomp'th variable, of size 7*Ndim;
//  start_index & end_index:    selected partition for calculating statistical property; end_index=-1 indicates last element index (NOT included);  
//  selected vector range:  [start_index, end_index)  
template<int Ndim>
Array<Real, Ndim> getArrayStatistics(const std::vector<Array<Real, Ndim>>& vec_data_in, int ncomp,
    int start_index=0, int end_index=-1){
    
    int end_index_eff = end_index;
    if(end_index_eff<0){
        end_index_eff = vec_data_in.size();
    }
    if(end_index_eff <= start_index){
        throw runtime_error("subvector index out of range! (called from function 'getArrayStatistics')\n");
    }
    // subvector data we need
    std::vector<Array<Real, Ndim>> vec_data(vec_data_in.begin() + start_index, vec_data_in.begin() + end_index_eff);
    int N = vec_data.size();
    Array<Real, Ndim> stat_arr;
    if(ncomp>=0 && ncomp<=5){
        switch(ncomp) {
            case 0:{
                for(int ndim=0; ndim<Ndim; ndim++){
                    Real mean = 0.;
                    for(int n=0; n<N; n++){
                        //printf("vec_data[%d][%d]=%f\n", n, ndim, vec_data[n][ndim]);
                        mean += vec_data[n][ndim];
                    }
                    mean = mean/N;
                    stat_arr[ndim] = mean;
                }
                break;
            }
            case 1:{
                for(int ndim=0; ndim<Ndim; ndim++){
                    Real mean = 0.; Real var = 0.;
                    for(int n=0; n<N; n++){
                        mean += vec_data[n][ndim];
                    }
                    mean = mean/N;
                    for(int n=0; n<N; n++){
                        var += (vec_data[n][ndim] - mean)*(vec_data[n][ndim] - mean);
                    }
                    var = var/N;
                    stat_arr[ndim] = var;
                }
                break;
            }
            case 2:{
                for(int ndim=0; ndim<Ndim; ndim++){
                    Real max = vec_data[0][ndim];
                    for(int n=1; n<N; n++){
                        if(vec_data[n][ndim] >= max){
                            max = vec_data[n][ndim];
                        }
                    }
                    stat_arr[ndim] = max;
                }
                break;
            }
            case 3:{
                for(int ndim=0; ndim<Ndim; ndim++){
                    Real min = vec_data[0][ndim];
                    for(int n=1; n<N; n++){
                        if(vec_data[n][ndim] <= min){
                            min = vec_data[n][ndim];
                        }
                    }
                    stat_arr[ndim] = min;
                }
                break;
            }
            case 4:{
                for(int ndim=0; ndim<Ndim; ndim++){
                    Real max = std::abs(vec_data[0][ndim]);
                    for(int n=1; n<N; n++){
                        Real abs_data = std::abs(vec_data[n][ndim]);
                        if(abs_data >= max){
                            max = abs_data;
                        }
                    }
                    stat_arr[ndim] = max;
                }
                break;
            }
            case 5:{
                for(int ndim=0; ndim<Ndim; ndim++){
                    Real min = std::abs(vec_data[0][ndim]);
                    for(int n=1; n<N; n++){
                        Real abs_data = vec_data[n][ndim];
                        if(abs_data <= min){
                            min = abs_data;
                        }
                    }
                    stat_arr[ndim] = min;
                }
                break;
            }
            default:
                throw runtime_error("statistic index out of range! (called from function 'getArrayStatistics')\n");
        }
        return stat_arr;
    }else{
        throw runtime_error("statistic property index out of range! (called from function 'getArrayStatistics')\n");
    }
}



#endif