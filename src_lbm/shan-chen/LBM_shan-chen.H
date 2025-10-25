#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"
#include <iostream>
#include <random>
#include <cmath>
#include "AMReX_FileIO.H"
#include "Debug.H"
#include "LBM_hydrovs.H"

//#define USE_REF_STATE

//#define RANDRAW
//const IntVect print_grid(0, 0, 0); // grid to print out

AMREX_GPU_MANAGED unsigned long long seed = 12345ULL;  // A fixed seed for reproducibility in each independent CUDA program;
AMREX_GPU_MANAGED Real tau_f = 1./2.; // tau=0.5*(1.+1./sqrt(3.)) minimizes spurious currents according to Swift et al.
AMREX_GPU_MANAGED Real tau_g = 1./2.;
AMREX_GPU_MANAGED Real alpha0 = 1.5;//4.;//1.5;//0.;//0.95; also = \mathcal{G} in paper;
AMREX_GPU_MANAGED Real alpha1 = 0.;//alpha1;
const Real epsilon = 1e-14;

AMREX_GPU_MANAGED Real kappa = 0.1;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real equivDensityInv(Real density){
  //return 0.;
  if(density < epsilon){
    return 0.;
  }else{
    return 1./density;
  }
}

#ifdef AMREX_USE_CUDA
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
#else
AMREX_GPU_HOST AMREX_FORCE_INLINE
#endif
Real draw_from_pdf_normal(int i, int j, int k) {

#ifdef RANDRAW // it's very slow!!!!
  curandStateMRG32k3a_t  state; 
  unsigned long long sequence = i + j * 1e5 + k * 1e10;  // Unique per thread
  unsigned long long offset = 0; // Can be used for additional decorrelation
  curand_init(::seed, sequence, offset, &state);  // Initialize curand state with a seed based on indices
  
  // Generate and return the normal random number
  return curand_normal(&state) * 1. + 0.;
  
#else
  // Use amrex::RandomNormal library to generate a normally distributed random number
  amrex::RandomEngine engine;
  return 0. + amrex::RandomNormal(0.0, std::sqrt(1.), engine);    
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure(Real rho, Real phi) {

  Real p0 = rho*kBT;

  return p0;
}

AMREX_GPU_HOST AMREX_FORCE_INLINE
void thermal_noise(const Geometry& geom, MultiFab& fnoisevs, MultiFab& gnoisevs,
  MultiFab& rho_eqvs, MultiFab& phi_eqvs, MultiFab& rhot_eqvs, 
  MultiFab& hydrovsbar, const RealVect& pos_com_relative){ // additional 3 equilibrium state

  const Box domain = geom.Domain();
  const Real tau_f_bar = 1./(tau_f+0.5);
  const Real tau_g_bar = tau_f_bar;
  const Real tau_f_bar2 = tau_f_bar*tau_f_bar;
  const Real tau_g_bar2 = tau_g_bar*tau_g_bar;
  for (MFIter mfi(fnoisevs); mfi.isValid(); ++mfi) {
    const Array4<Real>& fnoise = fnoisevs.array(mfi);
    const Array4<Real>& gnoise = gnoisevs.array(mfi);
    const Array4<Real>& rho_eq = rho_eqvs.array(mfi);
    const Array4<Real>& phi_eq = phi_eqvs.array(mfi);
    const Array4<Real>& rhot_eq = rhot_eqvs.array(mfi);
    const Array4<Real>& hydrovs = hydrovsbar.array(mfi);
    const Box& valid_box = mfi.validbox();
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, const RandomEngine& engine) {
      #ifdef USE_REF_STATE
      // shifted coordinates for center of mass frame;
      int x_shift = x - static_cast<int>(pos_com_relative[0]);
      int y_shift = y - static_cast<int>(pos_com_relative[1]);
      int z_shift = z - static_cast<int>(pos_com_relative[2]);
      // periodic BC for shifted coordinates;
      if(x_shift < domain.smallEnd(0)) x_shift += domain.length(0);
      if(x_shift > domain.bigEnd(0))   x_shift -= domain.length(0);
      if(y_shift < domain.smallEnd(1)) y_shift += domain.length(1);
      if(y_shift > domain.bigEnd(1))   y_shift -= domain.length(1);
      if(z_shift < domain.smallEnd(2)) z_shift += domain.length(2);
      if(z_shift > domain.bigEnd(2))   z_shift -= domain.length(2);
      Real rho = rho_eq(x_shift, y_shift, z_shift, 0);
      Real phi = phi_eq(x_shift, y_shift, z_shift, 0);
      Real rhot = rhot_eq(x_shift, y_shift, z_shift, 0);  // equilibrium density
      #else
      //Print() << "Not using reference state for thermal noise generation!" << '\n';
      Real rho = hydrovs(x, y, z, 0);
      Real phi = hydrovs(x, y, z, 1);
      Real rhot = rho + phi;  // current density
      #endif
      fnoise(x,y,z,0) = 0.;
      gnoise(x,y,z,0) = 0.;
      for (int a=1; a<=AMREX_SPACEDIM; a++) {
        // \sqrt(0.5)*random normal distribution;
        fnoise(x,y,z,a) = sqrt(2.*(tau_f_bar - 0.5*tau_f_bar2)*kBT*std::abs(rho*phi/rhot))*RandomNormal(0., 1., engine);
        gnoise(x,y,z,a) = -fnoise(x,y,z,a);
        /*if(std::isnan(fnoise(x,y,z,a))) {
          Print() << "current validbox is " << valid_box << '\t';
          printf("fnoise NAN at (%d,%d,%d,%d) with shifted (%d,%d,%d), rho: %f, phi: %f, rhot: %f\n", x,y,z,a, x_shift, y_shift, z_shift, rho, phi, rhot);
        }*/
      }
      for (int a=AMREX_SPACEDIM+1; a<nvel; a++) {
        fnoise(x,y,z,a) = sqrt(2.*(tau_f_bar - 0.5*tau_f_bar2)*kBT/cs2*b[a]*std::abs(rho))*RandomNormal(0., 1., engine);
        gnoise(x,y,z,a) = sqrt(2.*(tau_g_bar - 0.5*tau_g_bar2)*kBT/cs2*b[a]*std::abs(phi))*RandomNormal(0., 1., engine);
      }
    });
  }
  fnoisevs.FillBoundary(geom.periodicity());
  gnoisevs.FillBoundary(geom.periodicity());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect gradient(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    for (int dir=0; dir<3; dir++) {
      gradient[dir] += w[i]/cs2*field(xp,yp,zp,icomp)*c[i][dir]; // icomp is the input field component index we hope to calculate
    }
    /*
    For D3Q19, \nabla_\alpha [field] = 1/cs^2*\sum_i *w_i n(r+e_i\Delta t)e_{i\alpha}
    */
  }
  return gradient;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real laplacian(int x, int y, int z, const Array4<Real>& field, int icomp) {
  Real laplacian = 0.0;
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    laplacian += 2.*w[i]/cs2*(field(xp,yp,zp,icomp)-field(x,y,z,icomp));
  }
  /*
    \nabla^2 [field] = \sum_i w_i*(n(r+e_i\Delta t) - n(r))/(0.5*cs^2*(\Delta t)^2)
  */
  return laplacian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect grad_laplacian_2nd(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  for(int dir=0; dir<AMREX_SPACEDIM; dir++){
    for (int i=0; i<nvel; i++) {
      for(int j=0; j<nvel; j++){
        int xc = x + c[j][0];
        int yc = y + c[j][1];
        int zc = z + c[j][2];

        int xp = xc + c[i][0];
        int yp = yc + c[i][1];
        int zp = zc + c[i][2];
        gradient[dir] += 2.*w[i]*w[j]/cs4*(field(xp,yp,zp,icomp)-field(xc,yc,zc,icomp))*c[j][dir];
      }
    }
  }
  /*  2nd order accuracy scheme of laplacian is used;
    \partial_\alpha \nabla^2 [field] = \sum_i w_i*w_j*(n(r+e_j\Delta t+e_i\Delta t) - n(r+e_j\Delta t))*c[j][\alpha]/(0.5*cs^4*(\Delta t)^3)
  */
  return gradient;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hydrovars(int x, int y, int z, // real density & momentum;
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& h,
         const Array4<Real>& hbar, // here modified density & momentum in [h] are passed in 
         const Array4<Real>& nf,
         const Array4<Real>& ng) {

  // hydrovars_bar(x, y, z, f, g, h);
  Array1D<Real,0,AMREX_SPACEDIM> ufbar; // ONLY WORKS for 3D;
  Array1D<Real,0,AMREX_SPACEDIM> ugbar;
  Array1D<Real,0,AMREX_SPACEDIM> afbar;
  Array1D<Real,0,AMREX_SPACEDIM> agbar;
  const Real rho = hbar(x,y,z,0);
  const Real phi = hbar(x,y,z,1);

  Real jf[3], jg[3];
  jf[0] = jf[1] = jf[2] = jg[0] = jg[1] = jg[2] = 0.0;
  
  h(x,y,z,0) = rho; // rho
  h(x,y,z,1) = phi;  // phi
  for (int i=0; i<nvel; ++i) {
    Real fi, gi;
    fi = f(x,y,z,i);
    gi = g(x,y,z,i);
    jf[0] += fi*c[i][0];  // 2
    jf[1] += fi*c[i][1];  // 3
    jf[2] += fi*c[i][2];  // 4
    jg[0] += gi*c[i][0];  // 6
    jg[1] += gi*c[i][1];  // 7
    jg[2] += gi*c[i][2];  // 8
  }
  // preparation for the acceleration of fluid; we should know that the acceleration field is same with the modified version;
  const Array1D<RealVect,0,2> grad_laplacians = {
    grad_laplacian_2nd(x,y,z,hbar,0),
    grad_laplacian_2nd(x,y,z,hbar,1),
  };

  const Array1D<RealVect,0,2> grad = {
    gradient(x,y,z,hbar,0),
    gradient(x,y,z,hbar,1),
  };

  for(int k=0; k<AMREX_SPACEDIM; k++){
    ufbar(k) = (std::abs(rho) > FLT_EPSILON) ? jf[k]/rho : 0.; // 2~4
    ugbar(k) = (std::abs(phi) > FLT_EPSILON) ? jg[k]/phi : 0.; // 6~8
    // acceleration of fluid f;
    const RealVect grad_rho = grad(0);
    const RealVect grad_laplacian_rho = grad_laplacians(0);
    // acceleration of fluid g;
    const RealVect grad_phi = grad(1);
    const RealVect grad_laplacian_phi = grad_laplacians(1);
    afbar(k) = -cs2*(alpha0*grad_phi[k] + alpha1*grad_laplacian_phi[k]); // 9~11
    agbar(k) = -cs2*(alpha0*grad_rho[k] + alpha1*grad_laplacian_rho[k]); // 12~14
  }
  // next, update the modified quantites to real quantities;
  RealVect nfvel = {nf(x,y,z,1), nf(x,y,z,2), nf(x,y,z,3)};
  RealVect ngvel = {ng(x,y,z,1), ng(x,y,z,2), ng(x,y,z,3)};

  nfvel = (std::abs(rho) > FLT_EPSILON) ? nfvel/rho : RealVect(0.);
  ngvel = (std::abs(phi) > FLT_EPSILON) ? ngvel/phi : RealVect(0.);
  // 0.75 ufbar + 0.25 ugbar + 0.5*nfvel; 
  h(x,y,z,2) = ufbar(0) + 0.5*afbar(0) - 0.5/(tau_f+0.5)*phi/(rho+phi)*(ufbar(0)-ugbar(0) + 0.5*(afbar(0)-agbar(0))) + 0.5*nfvel[0];//*equivDensityInv(rho); 
  h(x,y,z,3) = ufbar(1) + 0.5*afbar(1) - 0.5/(tau_f+0.5)*phi/(rho+phi)*(ufbar(1)-ugbar(1) + 0.5*(afbar(1)-agbar(1))) + 0.5*nfvel[1];//*equivDensityInv(rho);
  h(x,y,z,4) = ufbar(2) + 0.5*afbar(2) - 0.5/(tau_f+0.5)*phi/(rho+phi)*(ufbar(2)-ugbar(2) + 0.5*(afbar(2)-agbar(2))) + 0.5*nfvel[2];//*equivDensityInv(rho);

  h(x,y,z,6) = ugbar(0) + 0.5*agbar(0) - 0.5/(tau_g+0.5)*rho/(rho+phi)*(ugbar(0)-ufbar(0) + 0.5*(agbar(0)-afbar(0))) + 0.5*ngvel[0];//*equivDensityInv(phi);
  h(x,y,z,7) = ugbar(1) + 0.5*agbar(1) - 0.5/(tau_g+0.5)*rho/(rho+phi)*(ugbar(1)-ufbar(1) + 0.5*(agbar(1)-afbar(1))) + 0.5*ngvel[1];//*equivDensityInv(phi);
  h(x,y,z,8) = ugbar(2) + 0.5*agbar(2) - 0.5/(tau_g+0.5)*rho/(rho+phi)*(ugbar(2)-ufbar(2) + 0.5*(agbar(2)-afbar(2))) + 0.5*ngvel[2];//*equivDensityInv(phi);

  Real rho_tot = rho + phi; // total density;
  h(x,y,z,5) = rho_tot;  // total density;

  // acceleration of fluid f; equal to modified quantities;
  h(x,y,z,9) = afbar(0);
  h(x,y,z,10) = afbar(1);
  h(x,y,z,11) = afbar(2);
  // acceleration of fluid g; equal to modified quantities;
  h(x,y,z,12) = agbar(0);
  h(x,y,z,13) = agbar(1);
  h(x,y,z,14) = agbar(2);

  h(x,y,z,15) = (rho*ufbar(0) + phi*ugbar(0) + 0.5*(rho*afbar(0) + phi*agbar(0)))/rho_tot; // barcentric velocity in x direction;
  h(x,y,z,16) = (rho*ufbar(1) + phi*ugbar(1) + 0.5*(rho*afbar(1) + phi*agbar(1)))/rho_tot; // barcentric velocity in y direction;
  h(x,y,z,17) = (rho*ufbar(2) + phi*ugbar(2) + 0.5*(rho*afbar(2) + phi*agbar(2)))/rho_tot; // barcentric velocity in z direction;

  h(x,y,z,18) = nfvel[0]; // noise f related terms;
  h(x,y,z,19) = ngvel[0]; // noise g related terms;

  h(x,y,z,20) = ufbar(0); // lb velocity f in x direction;
  h(x,y,z,21) = ugbar(0); // lb velocity g in x direction;
}

// calculate real hydrodynamic moments for whole lattice
inline void LBM_hydrovars(const Geometry& geom, 
  MultiFab& mf, MultiFab& mg, MultiFab& hydrovs, MultiFab& hydrovsbar,
  MultiFab& fnoisevs, MultiFab& gnoisevs) {

  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & h = hydrovs.arrays();
  auto const & hbar = hydrovsbar.arrays();
  auto const & nf = fnoisevs.arrays(); // multiple noise Array4D arrays of size [box_index, geometry_size, nvel], i.e., nf[nbx](i,j,k,ncomp); 
  auto const & ng = gnoisevs.arrays();
  // 0 ghost cells for hydrovars ???? need to be further checked;
  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    hydrovars(x, y, z, f[nbx], g[nbx], h[nbx], hbar[nbx], nf[nbx], ng[nbx]);
  });
  hydrovs.FillBoundary(geom.periodicity());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hydrovars_bar_density(int x, int y, int z,
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& hb) {
  Real rho, phi, jf[3], jg[3];
  rho = phi = jf[0] = jf[1] = jf[2] = jg[0] = jg[1] = jg[2] = 0.0;
  for (int i=0; i<nvel; ++i) {
    Real fi, gi;
    fi = f(x,y,z,i);
    gi = g(x,y,z,i);
    rho  += fi;
    phi  += gi;
  }
  hb(x,y,z,0) = rho;
  hb(x,y,z,1) = phi;

  // for test of structure factor of modified hydrodynamic moments; not necessary for the real simulation;
  const Array1D<Real,0,nvel> mf = moments(x,y,z,f);
  const Array1D<Real,0,nvel> mg = moments(x,y,z,g);
  for (int i=1; i<=AMREX_SPACEDIM; ++i) {     // momentum/velocity i=1..3
    hb(x,y,z,i+1) = (std::abs(mf(0)) > FLT_EPSILON) ? mf(i)/mf(0) : 0.;               // u_f to be stored in 2..4
    hb(x,y,z,i+2+AMREX_SPACEDIM) = (std::abs(mg(0)) > FLT_EPSILON) ? mg(i)/mg(0) : 0.; // u_g to be stored in 6..8
  }
  hb(x,y,z,2+AMREX_SPACEDIM) = mf(0) + mg(0); // total density to be stored in 5;
}

// calculate modified hydrodynamic moments for whole lattice
inline void LBM_hydrovars_density(const Geometry& geom, MultiFab& mf, MultiFab& mg, MultiFab& hydrovsbar) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & hbar = hydrovsbar.arrays();
  const IntVect halo = IntVect(0);

  //  First step: update the density first (due to the fact that the accerleration field depends on a cluster of grids rather than single grid)
  ParallelFor(hydrovsbar, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) { 
    hydrovars_bar_density(x, y, z, f[nbx], g[nbx], hbar[nbx]);
  });
  hydrovsbar.FillBoundary(geom.periodicity());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> equilibrium_moments(const Array1D<Real,0,2>& fields, // rho&phi
			    const int field_index, const RealVect& u) {  // u is real velocity field
  Array1D<Real,0,nvel> mEq;

  const Real rho = fields(field_index); // density distribution on current site only;

  const Real coefA = rho;
  const Real coefB = 1.;
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { u[0], u[1], u[2] };
  //Print() << coefE << '\n';
  Real coefAD[3][3];   // \frac{\bm{v}\bm{v}}{2c^4}, tensor of size 3*3;
  coefAD[0][0] = (coefA*u[0]*u[0])/2./cs4; 
  coefAD[0][1] = (coefA*u[0]*u[1])/2./cs4;  
  coefAD[0][2] = (coefA*u[0]*u[2])/2./cs4;
  coefAD[1][0] = coefAD[0][1]; 
  coefAD[1][1] = (coefA*u[1]*u[1])/2./cs4; 
  coefAD[1][2] = (coefA*u[1]*u[2])/2./cs4;
  coefAD[2][0] = coefAD[0][2]; coefAD[2][1] = coefAD[1][2]; 
  coefAD[2][2] = (coefA*u[2]*u[2])/2./cs4;
  const Real trace_coefAD = coefAD[0][0] + coefAD[1][1] + coefAD[2][2];

  mEq( 0) = coefAB;

  mEq( 1) = coefC*cs2*(coefA*u[0]);
  mEq( 2) = coefC*cs2*(coefA*u[1]);
  mEq( 3) = coefC*cs2*(coefA*u[2]);

  mEq( 4) = 2.*cs4*trace_coefAD;
  mEq( 5) = 6.*cs4*coefAD[0][0] - 2.*cs4*trace_coefAD;
  mEq( 6) = 2.*cs4*(coefAD[1][1] - coefAD[2][2]);
  mEq( 7) = cs4*(coefAD[0][1] + coefAD[1][0]);
  mEq( 8) = cs4*(coefAD[1][2] + coefAD[2][1]);
  mEq( 9) = cs4*(coefAD[0][2] + coefAD[2][0]);

  mEq(10) = 0.; mEq(11) = 0.; mEq(12) = 0.; mEq(13) = 0.; mEq(14) = 0.; mEq(15) = 0.;

  mEq(16) = 0.;//coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*trace_coefAD*(6.*cs4+cs2) - trace_coefAD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  //+ trace_coefAD*cs2*(18.*cs2+1) - cs2*trace_coefAD;
  mEq(17) = 0.;//2.*cs2*(2.*coefAD[0][0] - coefAD[1][1] - coefAD[2][2]) - 18.*cs4*coefAD[0][0] + 6.*cs4*trace_coefAD;
  mEq(18) = 0.;

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> phi_moments(const Array1D<Real,0,2>& fields, const int field_index,
				   const RealVect& u, const RealVect& a) {  // u,a are REAL velocity & acceleration field of fluid f excluding NOISE impact;
  Array1D<Real,0,nvel> mEq;

  const Real rho = fields(field_index);

  const Real coefA = rho;
  const Real coefB = 0.; 
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { a[0], a[1], a[2] };
  Real coefAD[3][3];   // \frac{\bm{a}\bm{v}}{c^4}, tensor of size 3*3; a,v are real hydrodynamic variables;
  coefAD[0][0] = a[0]*(coefA*u[0])/cs4; coefAD[0][1] = a[0]*(coefA*u[1])/cs4; coefAD[0][2] = a[0]*(coefA*u[2])/cs4;
  coefAD[1][0] = a[1]*(coefA*u[0])/cs4; coefAD[1][1] = a[1]*(coefA*u[1])/cs4; coefAD[1][2] = a[1]*(coefA*u[2])/cs4;
  coefAD[2][0] = a[2]*(coefA*u[0])/cs4; coefAD[2][1] = a[2]*(coefA*u[1])/cs4; coefAD[2][2] = a[2]*(coefA*u[2])/cs4;
  const Real trace_coefAD = coefAD[0][0] + coefAD[1][1] + coefAD[2][2];

  // Correction factor 1./(1.+1./(2.*tau_f)) should be multiplied before each moment's expression;
  const Real modifactor = 1./(1.+1./(2.*tau_f)); // 0.5
  mEq( 0) = modifactor*coefAB;

  mEq( 1) = modifactor*coefAC*cs2*coefE[0];
  mEq( 2) = modifactor*coefAC*cs2*coefE[1];
  mEq( 3) = modifactor*coefAC*cs2*coefE[2];

  mEq( 4) = modifactor*2.*cs4*trace_coefAD;///2.;
  mEq( 5) = modifactor*(6.*cs4*coefAD[0][0] - 2.*cs4*trace_coefAD);///2.;
  mEq( 6) = modifactor*2.*cs4*(coefAD[1][1] - coefAD[2][2]);///2.;
  mEq( 7) = modifactor*cs4*(coefAD[0][1] + coefAD[1][0]);///2.;
  mEq( 8) = modifactor*cs4*(coefAD[1][2] + coefAD[2][1]);///2.;
  mEq( 9) = modifactor*cs4*(coefAD[0][2] + coefAD[2][0]);///2.;

  mEq(10) = 0.; mEq(11) = 0.; mEq(12) = 0.; mEq(13) = 0.; mEq(14) = 0.; mEq(15) = 0.;

  //mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*trace_coefAD*(6.*cs4+cs2) - trace_coefAD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  //+ trace_coefAD*cs2*(18.*cs2+1) - cs2*trace_coefAD;
  mEq(16) = 0.;//modifactor*mEq(16);
  
  //mEq(17) = 2.*cs2*(2.*coefAD[0][0] - coefAD[1][1] - coefAD[2][2]) - 18.*cs4*coefAD[0][0] + 6.*cs4*trace_coefAD;
  mEq(17) = 0.;//modifactor*mEq(17);
  mEq(18) = 0.;

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z,
	     Array4<Real> f,  Array4<Real> g,
	     Array4<Real> h,  //Array4<Real> hbar,
       Array4<Real> fn, Array4<Real> gn) {

  const Array1D<Real,0,2> fields = { h(x,y,z,0), h(x,y,z,1) }; // 0: rho; 1: phi
  const RealVect uf = { h(x,y,z,2), h(x,y,z,3), h(x,y,z,4) }; // velocity of f
  const RealVect ug = { h(x,y,z,6), h(x,y,z,7), h(x,y,z,8) }; // velocity of g
  const RealVect af = { h(x,y,z,9), h(x,y,z,10), h(x,y,z,11) }; // acceleration of f
  const RealVect ag = { h(x,y,z,12), h(x,y,z,13), h(x,y,z,14) }; // acceleration of g
  const RealVect noisef = { fn(x,y,z,1), fn(x,y,z,2), fn(x,y,z,3) }; // noise moments of f, x,y & z directions respectively;
  const RealVect noiseg = { gn(x,y,z,1), gn(x,y,z,2), gn(x,y,z,3) }; // noise moments of f, x,y & z directions respectively;

  Array1D<Real,0,nvel> mf, mg, mfEq, mgEq, mPhif, mPhig;

  mf = moments(x,y,z,f); // here is the modified moments since all program is based on modified distribution functions f,g;
  mg = moments(x,y,z,g);

  // mf_eq and mg_eq, moments of equilibirum distribution function
  RealVect v_b = (fields(0)*uf + fields(1)*ug)/(fields(0) + fields(1));
  // these two lines are equivalent;
  //Print() << "v_b_h = " << v_b << '\n';
  //Print() << "v_b_m = " << ((RealVect(mf(1), mf(2), mf(3)) + 0.5*mf(0)*af) + (RealVect(mg(1), mg(2), mg(3)) + 0.5*mg(0)*ag))/(mf(0) + mg(0)) << '\n';
  
  mfEq = equilibrium_moments(fields, 0, v_b); // !!!!!!!!!!!!??????????!!!!!!!!!!!
  mgEq = equilibrium_moments(fields, 1, v_b);

  // moments of force term, modified version with pre-factor!
  mPhif = phi_moments(fields, 0, uf, af);
  mPhig = phi_moments(fields, 1, ug, ag);

  /*if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]) {
    printf("moments in collide() = \n");
    for(int i=0; i<nvel; ++i) {
      printf("mfEq(%d) = %.8f\t", i, mfEq(i));
      printf("mPhif(%d) = %.8f\t", i, mPhif(i));
      printf("mgEq(%d) = %.8f\t", i, mgEq(i));
      printf("mPhig(%d) = %.8f\t", i, mPhig(i));
    }
    printf("\ndensity in collide() = \n");
    printf("h(%d) = %.8f\t", 0, h(x,y,z,0));
    printf("h(%d) = %.8f\t", 1, h(x,y,z,1));
    printf("\ncenter-of-mass velocity & acceleration in collide() = \n");
    for(int i=0; i<3; ++i) {
      printf("ub(%d) = %.8f\t", i, v_b[i]);
      printf("af(%d) = %.8f\t", i, af[i]);
      printf("ag(%d) = %.8f\t", i, ag[i]);
    }
    printf("\n");
  }*/

  // thermal noise in moment space: fn(x,y,z,ncomp) & gn(x,y,z,ncomp);
  const Real tau_f_bar = tau_f*(1.+0.5/tau_f);
  const Real tau_g_bar = tau_g*(1.+0.5/tau_g);
  for (int a=0; a<nvel; ++a) {  // the correction factor for the force term is included in the function [Phif_equilibrium_moments];
    Real Raf = 1./tau_f_bar * (mfEq(a) - mf(a)) + mPhif(a) + fn(x,y,z,a);
    Real Rag = 1./tau_g_bar * (mgEq(a) - mg(a)) + mPhig(a) + gn(x,y,z,a);
    mf(a) = mf(a) + Raf;
    mg(a) = mg(a) + Rag;
  }

  populations(x,y,z,f,mf);
  populations(x,y,z,g,mg);

}

// stream from \vec{x} to \vec{x}+\vec{c}_i (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stream_push(int x, int y, int z,
		 const Array4<Real>& fOld, const Array4<Real>& gOld,
		 const Array4<Real>& fNew, const Array4<Real>& gNew) {
  for (int i=0; i<nvel; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = fOld(x,y,z,i);
    gNew(xp,yp,zp,i) = gOld(x,y,z,i);
    //printf("streaming from (%d, %d, %d) to (%d, %d, %d) is %.8f for comp = %d\n", x, y, z, xp, yp, zp, fOld(x,y,z,i), i);
  }
}

// collide and stream (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide_stream(int x, int y, int z,
		    const Array4<Real>& fOld,  const Array4<Real>& gOld,
		    const Array4<Real>& fNew,  const Array4<Real>& gNew,
		    const Array4<Real>& h,     //const Array4<Real>& hbar,
			  const Array4<Real>& fn,    const Array4<Real>& gn ) {
  collide(x, y, z, fOld, gOld, h, fn, gn); // collide populations on current site
  stream_push(x, y, z, fOld, gOld, fNew, gNew); // push populations to neighbors
}

// LB timestep for whole lattice
inline void LBM_timestep(const Geometry& geom,
  MultiFab& fold, MultiFab& gold,
  MultiFab& fnew, MultiFab& gnew,
  MultiFab& hydrovs, MultiFab& hydrovsbar,
  MultiFab& fnoisevs, MultiFab& gnoisevs,
  MultiFab& rho_eq, MultiFab& phi_eq, MultiFab& rhot_eq,
  amrex::Vector<RealVect>& com_ref) {

  fold.FillBoundary(geom.periodicity());
  gold.FillBoundary(geom.periodicity());  // MUST fullfil the periodic boundary condition at EACH STEP !!!
  hydrovs.FillBoundary(geom.periodicity());
  auto const & fOld = fold.arrays();
  auto const & gOld = gold.arrays();
  auto const & fNew = fnew.arrays();
  auto const & gNew = gnew.arrays();
  auto const & h = hydrovs.arrays();
  //auto const & hbar = hydrovsbar.arrays();
  auto const & fn = fnoisevs.arrays();
  auto const & gn = gnoisevs.arrays();

  IntVect halo = IntVect(1); // need to push from first halo layer into domain
  //IntVect halo = IntVect(2);
  //printf("ParallelFor loop start: = \n");
  ParallelFor(fold, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    //printf("(%d,%d,%d)\t", x, y, z);
    collide_stream(x, y, z,
      fOld[nbx], gOld[nbx],
      fNew[nbx], gNew[nbx], h[nbx], fn[nbx], gn[nbx]);
  });
  //printf("ParallelFor loop end\n");
  /*
  Swap (MultiFab &dst, MultiFab &src, int srccomp, int dstcomp, int numcomp, int nghost)
  Swap from src to dst including nghost ghost cells. 
  */
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);

  // here all Multifab variables;
  LBM_hydrovars_density(geom, fold, gold, hydrovsbar);
  //amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  // update center of mass (COM) position (relative to reference state COM position) collision & streaming;
  RealVect pos_com_relative;
  update_com(geom, pos_com_relative, hydrovsbar);
  pos_com_relative -= com_ref[0];
  //Print() << "COM is " << pos_com << '\n';
  thermal_noise(geom, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq, hydrovsbar, pos_com_relative);
  //amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  LBM_hydrovars(geom, fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
  //amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
}

// *****************************************************  Initialization  ********************************************************* //

inline void LBM_init_mixture(const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs,
           MultiFab& hydrovsbar,
           MultiFab& fnoisevs,
           MultiFab& gnoisevs,
           MultiFab& rho_eq, MultiFab& phi_eq, MultiFab& rhot_eq) {
  const Real C1 = 0.5;
  const Real C2 = 0.5;
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Box& valid_box = mfi.validbox();
    const Array4<Real>& f = mf.array(mfi);
    const Array4<Real>& g = mg.array(mfi);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real rho = 2.*C1;// + (x+y+z)/500.;// + C2;
      const Real phi = 2.*C2;// - (x+y+z)/500.;//(C1-C2)+0.01*RandomNormal(0., 1., engine);
      for (int i=0; i<nvel; i++) {
	      f(x,y,z,i) = w[i]*rho;
	      g(x,y,z,i) = w[i]*phi;
      }
    });
  }
  LBM_hydrovars_density(geom, mf, mg, hydrovsbar);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  RealVect pos_com;
  update_com(geom, pos_com, hydrovsbar);
  thermal_noise(geom, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq, hydrovsbar, pos_com);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  LBM_hydrovars(geom, mf, mg, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
}

// set initial condition to be given fluid f density [rho_0] and fluid g density [phi_0];
inline void LBM_init(const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs,
           MultiFab& hydrovsbar,
           MultiFab& fnoisevs,
           MultiFab& gnoisevs,
           MultiFab& mf0, MultiFab& mg0, // given initial state for f, g;
           MultiFab& rho_eq, MultiFab& phi_eq, MultiFab& rhot_eq,
           amrex::Vector<RealVect>& com_ref) {

  mf.ParallelCopy(mf0); mg.ParallelCopy(mg0);
  mf.FillBoundary(geom.periodicity());
  mg.FillBoundary(geom.periodicity());
  //hydrovs.FillBoundary(geom.periodicity());
  //hydrovsbar.FillBoundary(geom.periodicity());

  LBM_hydrovars_density(geom, mf, mg, hydrovsbar);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  RealVect pos_com_relative;
  update_com(geom, pos_com_relative, hydrovsbar);
  pos_com_relative -= com_ref[0];
  thermal_noise(geom, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq, hydrovsbar, pos_com_relative);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  /*Print() << "fnoisevs NAN check: \n";
  MultiFabNANCheck(fnoisevs, true, 0);
  Print() << "fnoisevs NAN check complete: \n";*/
  LBM_hydrovars(geom, mf, mg, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
}

// flat interface 
void LBM_init_stripe(const Real frac, const Geometry& geom,
				  MultiFab& mf, MultiFab& mg,
			    MultiFab& hydrovs, MultiFab& hydrovsbar,
          MultiFab& fnoisevs, MultiFab& gnoisevs,
          MultiFab& rho_eq, MultiFab& phi_eq, MultiFab& rhot_eq) {
  
  const Real rho_lo = 0.1;
  const Real rho_hi = 3.;
  GpuArray<Real,3> dx = geom.CellSizeArray();
  const IntVect box = geom.Domain().length();
  const Real rho_t = rho_hi + rho_lo;
  const Real pos_lo = (-0.5*frac)*box[2]; // position of the lower interface
  const Real pos_hi = (0.5*frac)*box[2]; // position of the upper interface
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real pos = z-box[2]/2;
    const Real rho =  (rho_hi - rho_lo)*0.5*(std::tanh((pos-pos_lo)/std::sqrt(kappa)) + std::tanh((pos_hi-pos)/std::sqrt(kappa))) + rho_lo;
    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*(rho_t-rho);
    }
  });
  
  LBM_hydrovars_density(geom, mf, mg, hydrovsbar);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  RealVect pos_com_relative = {0., 0., 0.}; // initial flat interface does not need to be shifted;
  thermal_noise(geom, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq, hydrovsbar, pos_com_relative); // rho_eq, phi_eq, rhot_eq here have no meaningful values, since kBT=0.;
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  LBM_hydrovars(geom, mf, mg, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
}


// binary droplet of C1 and C2
inline void LBM_init_droplet(const Real r,
			     const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs,
           MultiFab& hydrovsbar,
           MultiFab& fnoisevs,
           MultiFab& gnoisevs,
           MultiFab& rho_eq, MultiFab& phi_eq, MultiFab& rhot_eq) {

  const IntVect box = geom.Domain().length();
  /*
  const Box& amrex::Geometry::Domain() const: Returns our rectangular domain.
  AMREX_GPU_HOST_DEVICE IntVect amrex::Box::length() const: Return the length of the Box.
  */
  const Real R = r*box[0];

  auto const & f = mf.arrays(); 
  auto const & g = mg.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) { 
    const Real rx = x-box[0]/2.;
    const Real ry = y-box[1]/2.;
#if (AMREX_SPACEDIM == 2)
    const Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    const Real rz = z-box[0]/2;
    const Real r2 = rx*rx+ry*ry+rz*rz;
#endif
    const Real r = sqrt(r2);

    const Real rho_lo = 0.;
    const Real rho_hi = 3.;//1.;
    const Real rho_tot = rho_hi + rho_lo;
    // tanh profile varying from [rho_lo] to [rho_hi], used for the density of fluid f;
    const Real rho = (rho_hi - rho_lo)*(1.+tanh((R-r)/sqrt(kappa)))/2. + rho_lo;
    // droplet is fluid f
    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*(rho_tot - rho);
    }
  });
  LBM_hydrovars_density(geom, mf, mg, hydrovsbar);
  RealVect pos_com_relative = {0., 0., 0.}; // initial droplet does not need to be shifted;
  thermal_noise(geom, fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq, hydrovsbar, pos_com_relative);
  LBM_hydrovars(geom, mf, mg, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
}

#endif
