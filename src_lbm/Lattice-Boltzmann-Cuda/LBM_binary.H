#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"
#include <iostream>
#include <random>
#include <cmath>
#include "AMReX_FileIO.H"
#include "Debug.H"


AMREX_GPU_MANAGED unsigned long long seed = 12345ULL;  // A fixed seed for reproducibility in each independent CUDA program;
AMREX_GPU_MANAGED Real tau_f = 1.; // tau=0.5*(1.+1./sqrt(3.)) minimizes spurious currents according to Swift et al.
AMREX_GPU_MANAGED Real tau_g = 1.;
AMREX_GPU_MANAGED Real alpha0 = 0.;//0.95; also = \mathcal{G} in paper;
AMREX_GPU_MANAGED Real alpha1 = 0.;//alpha1;
const Real epsilon = 1e-14;

AMREX_GPU_MANAGED Real kappa = 0.001;

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

#ifdef AMREX_USE_CUDA // it's very slow!!!!
  curandStateMRG32k3a_t  state; 
  unsigned long long sequence = i + j * 1e5 + k * 1e10;  // Unique per thread
  unsigned long long offset = 0; // Can be used for additional decorrelation
  curand_init(::seed, sequence, offset, &state);  // Initialize curand state with a seed based on indices
  
  // Generate and return the normal random number
  return curand_normal(&state) * 1. + 0.;
  
#else
  // Use amrex::RandomNormal to generate a normally distributed random number
  amrex::RandomEngine engine;
  return 0. + amrex::RandomNormal(0.0, std::sqrt(1.), engine);    
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure(Real rho, Real phi) {

  Real p0 = rho*kBT;

  return p0;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void thermal_noise(int x, int y, int z, const Array4<Real>& fnoise, const Array4<Real>& gnoise,
  const Array4<Real>& rho_eq, const Array4<Real>& phi_eq, const Array4<Real>& rhot_eq) {
    
    const Real tau_f_bar = 1./(tau_f+0.5);
    const Real tau_g_bar = tau_f_bar;
    const Real tau_f_bar2 = tau_f_bar*tau_f_bar;
    const Real tau_g_bar2 = tau_g_bar*tau_g_bar;
    Real rho = rho_eq(x, y, z, 0);
    Real phi = phi_eq(x, y, z, 0);
    Real rhot = rhot_eq(x, y, z, 0);  // equilibrium density
    fnoise(x,y,z,0) = 0.;
    gnoise(x,y,z,0) = 0.;

    for (int a=1; a<=AMREX_SPACEDIM; a++) {
      fnoise(x,y,z,a) = sqrt(2.*(tau_f_bar - 0.5*tau_f_bar2)*kBT*fabs(rho*phi/rhot))*draw_from_pdf_normal(x,y,z); //*draw_from_pdf_normal(0., 1.);
      gnoise(x,y,z,a) = -fnoise(x,y,z,a);
      if(rhot<0||std::isnan(gnoise(x,y,z,a))){
        printf("fnoise-xyza(%d, %d, %d, %d)=%f\t", x, y, z, a, fnoise(x,y,z,a));
      }
    }
    for (int a=AMREX_SPACEDIM+1; a<nvel; a++) {
      fnoise(x,y,z,a) = sqrt(2.*(tau_f_bar - 0.5*tau_f_bar2)*kBT/cs2*b[a]*std::fabs(rho))*draw_from_pdf_normal(x,y,z);
      gnoise(x,y,z,a) = sqrt(2.*(tau_g_bar - 0.5*tau_g_bar2)*kBT/cs2*b[a]*std::fabs(phi))*draw_from_pdf_normal(x,y,z);
      if(std::isnan(gnoise(x,y,z,a))){
        printf("gnoise-xyza(%d, %d, %d, %d)=%f\t", x, y, z, a, gnoise(x,y,z,a));
      }
      if(std::isnan(fnoise(x,y,z,a))){
        printf("fnoise-xyza(%d, %d, %d, %d)=%f\t", x, y, z, a, fnoise(x,y,z,a));
      }
    }
}

AMREX_GPU_HOST AMREX_FORCE_INLINE
void thermal_noise(MultiFab& fnoisevs, MultiFab& gnoisevs,
  MultiFab& rho_eqvs, MultiFab& phi_eqvs, MultiFab& rhot_eqvs){ // additional 3 equilibrium state
  auto const & fnoise = fnoisevs.arrays();
  auto const & gnoise = gnoisevs.arrays();
  //auto const & hydrobar = hydrovsbar.arrays();
  auto const & rho_eq = rho_eqvs.arrays();
  auto const & phi_eq = phi_eqvs.arrays();
  auto const & rhot_eq = rhot_eqvs.arrays();

  ParallelFor(fnoisevs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) { // IntVect(0): no ghost cells when loop over the grid sites;
    //printf("Thread (x, y, z) = (%d, %d, %d)\n", x, y, z);
    thermal_noise(x,y,z, fnoise[nbx], gnoise[nbx], rho_eq[nbx], phi_eq[nbx], rhot_eq[nbx]);
  });
  //PrintCudaMem("inner side thermal_noise");
}

AMREX_GPU_HOST AMREX_FORCE_INLINE
void thermal_noise_new(MultiFab& fnoisevs, MultiFab& gnoisevs,
  MultiFab& rho_eqvs, MultiFab& phi_eqvs, MultiFab& rhot_eqvs){ // additional 3 equilibrium state

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
    const Box& valid_box = mfi.validbox();
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, const RandomEngine& engine) {
      Real rho = rho_eq(x, y, z, 0);
      Real phi = phi_eq(x, y, z, 0);
      Real rhot = rhot_eq(x, y, z, 0);  // equilibrium density
      fnoise(x,y,z,0) = 0.;
      gnoise(x,y,z,0) = 0.;
      for (int a=1; a<=AMREX_SPACEDIM; a++) {
        fnoise(x,y,z,a) = sqrt(2.*(tau_f_bar - 0.5*tau_f_bar2)*kBT*std::abs(rho*phi/rhot))*RandomNormal(0., 1., engine);
        gnoise(x,y,z,a) = -fnoise(x,y,z,a);
      }
      for (int a=AMREX_SPACEDIM+1; a<nvel; a++) {
        fnoise(x,y,z,a) = sqrt(2.*(tau_f_bar - 0.5*tau_f_bar2)*kBT/cs2*b[a]*std::abs(rho))*RandomNormal(0., 1., engine);
        gnoise(x,y,z,a) = sqrt(2.*(tau_g_bar - 0.5*tau_g_bar2)*kBT/cs2*b[a]*std::abs(phi))*RandomNormal(0., 1., engine);
      }
    });
  }
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
  // next, update the modified quantiites to real quantities;
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
    ufbar(k) = jf[k]/rho; // 2~4
    ugbar(k) = jg[k]/phi; // 6~8
    // acceleration of fluid f;
    const RealVect grad_rho = grad(0);
    const RealVect grad_laplacian_rho = grad_laplacians(0);
    afbar(k) = -cs2*(alpha0*grad_rho[0] + alpha1*grad_laplacian_rho[0]); // 9~11
    // acceleration of fluid g;
    const RealVect grad_phi = grad(1);
    const RealVect grad_laplacian_phi = grad_laplacians(1);
    agbar(k) = -cs2*(alpha0*grad_phi[k] + alpha1*grad_laplacian_phi[k]); // 12~14
  }
  // **********************   WITHOUT NOISE TERM   **********************;
  h(x,y,z,2) = ufbar(0) + 0.5*afbar(0) - 0.5/(tau_f+0.5)*phi/(rho+phi)*(ufbar(0)-ugbar(0) + 0.5*(afbar(0)-agbar(0))) + 0.5*nf(x,y,z,1)*equivDensityInv(rho); 
  h(x,y,z,3) = ufbar(1) + 0.5*afbar(1) - 0.5/(tau_f+0.5)*phi/(rho+phi)*(ufbar(1)-ugbar(1) + 0.5*(afbar(1)-agbar(1))) + 0.5*nf(x,y,z,2)*equivDensityInv(rho);
  h(x,y,z,4) = ufbar(2) + 0.5*afbar(2) - 0.5/(tau_f+0.5)*phi/(rho+phi)*(ufbar(2)-ugbar(2) + 0.5*(afbar(2)-agbar(2))) + 0.5*nf(x,y,z,3)*equivDensityInv(rho);

  h(x,y,z,6) = ugbar(0) + 0.5*agbar(0) - 0.5/(tau_g+0.5)*rho/(rho+phi)*(ugbar(0)-ufbar(0) + 0.5*(agbar(0)-afbar(0))) + 0.5*ng(x,y,z,1)*equivDensityInv(phi);
  h(x,y,z,7) = ugbar(1) + 0.5*agbar(1) - 0.5/(tau_g+0.5)*rho/(rho+phi)*(ugbar(1)-ufbar(1) + 0.5*(agbar(1)-afbar(1))) + 0.5*ng(x,y,z,2)*equivDensityInv(phi);
  h(x,y,z,8) = ugbar(2) + 0.5*agbar(2) - 0.5/(tau_g+0.5)*rho/(rho+phi)*(ugbar(2)-ufbar(2) + 0.5*(agbar(2)-afbar(2))) + 0.5*ng(x,y,z,3)*equivDensityInv(phi);

  h(x,y,z,5) = rho + phi;  // total density;

  // acceleration of fluid f; equal to modified quantities;
  h(x,y,z,9) = afbar(0);
  h(x,y,z,10) = afbar(1);
  h(x,y,z,11) = afbar(2);
  // acceleration of fluid g; equal to modified quantities;
  h(x,y,z,12) = agbar(0);
  h(x,y,z,13) = agbar(1);
  h(x,y,z,14) = agbar(2);
}

// calculate real hydrodynamic moments for whole lattice
inline void LBM_hydrovars(MultiFab& mf, MultiFab& mg, MultiFab& hydrovs, MultiFab& hydrovsbar, MultiFab& fnoisevs, MultiFab& gnoisevs) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & h = hydrovs.arrays();
  auto const & hbar = hydrovsbar.arrays();
  auto const & nf = fnoisevs.arrays(); // multiple noise Array4D arrays of size [box_index, geometry_size, nvel], i.e., nf[nbx](i,j,k,ncomp); 
  auto const & ng = gnoisevs.arrays();
  //Print() << "Call function kernel 'hydrovars' ...\n";
  ParallelFor(hydrovs, IntVect(2), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    hydrovars(x, y, z, f[nbx], g[nbx], h[nbx], hbar[nbx], nf[nbx], ng[nbx]);
  });
  //Print() << "function kernel 'hydrovars' ends ...\n";
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
}

// calculate modified hydrodynamic moments for whole lattice
inline void LBM_hydrovars_modified(const Geometry& geom, MultiFab& mf, MultiFab& mg, MultiFab& hydrovsbar) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & hbar = hydrovsbar.arrays();
  const IntVect halo = IntVect(2);

  //  First step: update the density first (due to the fact that the accerleration field depends on a cluster of grids rather than single grid)
  ParallelFor(hydrovsbar, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) { 
    hydrovars_bar_density(x, y, z, f[nbx], g[nbx], hbar[nbx]);
  });
  hydrovsbar.FillBoundary(geom.periodicity());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> fequilibrium_moments(const Array1D<Real,0,2>& fields, // rho&phi
			      const RealVect& u/*, const RealVect& fn*/) {  // u is real velocity field of fluid f (without noise terms effects);
  Array1D<Real,0,nvel> mEq;

  const Real rho = fields(0); // density distribution on current site only;
  //const Real phi = fields(1);

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

  mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*trace_coefAD*(6.*cs4+cs2) - trace_coefAD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  + trace_coefAD*cs2*(18.*cs2+1) - cs2*trace_coefAD;
  mEq(17) = 2.*cs2*(2.*coefAD[0][0] - coefAD[1][1] - coefAD[2][2]) - 18.*cs4*coefAD[0][0] + 6.*cs4*trace_coefAD;
  mEq(18) = 0.;
  /*for(int i=0;i<nvel;i++){
    Print() << mEq(i) << '\t';
  }
  Print() << '\n';*/
  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> gequilibrium_moments(const Array1D<Real,0,2>& fields,
				   const RealVect& u/*, const RealVect& gn*/) { // u is real velocity field of fluid g;
  Array1D<Real,0,nvel> mEq;

  const Real phi = fields(1);

  const Real coefA = phi;
  const Real coefB = 1.;
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { u[0], u[1], u[2] };
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

  mEq( 1) = coefC*cs2*coefA*u[0];
  mEq( 2) = coefC*cs2*coefA*u[1];
  mEq( 3) = coefC*cs2*coefA*u[2];

  mEq( 4) = 2.*cs4*trace_coefAD;
  mEq( 5) = 6.*cs4*coefAD[0][0] - 2.*cs4*trace_coefAD;
  mEq( 6) = 2.*cs4*(coefAD[1][1] - coefAD[2][2]);
  mEq( 7) = cs4*(coefAD[0][1] + coefAD[1][0]);
  mEq( 8) = cs4*(coefAD[1][2] + coefAD[2][1]);
  mEq( 9) = cs4*(coefAD[0][2] + coefAD[2][0]);

  mEq(10) = 0.; mEq(11) = 0.; mEq(12) = 0.; mEq(13) = 0.; mEq(14) = 0.; mEq(15) = 0.;

  mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*trace_coefAD*(6.*cs4+cs2) - trace_coefAD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  + trace_coefAD*cs2*(18.*cs2+1) - cs2*trace_coefAD;
  mEq(17) = 2.*cs2*(2.*coefAD[0][0] - coefAD[1][1] - coefAD[2][2]) - 18.*cs4*coefAD[0][0] + 6.*cs4*trace_coefAD;
  mEq(18) = 0.;

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> Phif_equilibrium_moments(const Array1D<Real,0,2>& fields,
				   const RealVect& u, const RealVect& a/*, const RealVect& fn*/) {  // u,a are REAL velocity & acceleration field of fluid f excluding NOISE impact;
  //  with noise terms, we need to add the additional terms "0.5*nf(x,y,z,*)/rho" effects by modifying the corresponding velocity related terms;
  Array1D<Real,0,nvel> mEq;

  const Real rho = fields(0);

  const Real coefA = rho;
  const Real coefB = 0.;
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { a[0], a[1], a[2] };
  Real coefAD[3][3];   // \frac{\bm{a}\bm{v}}{c^4}, tensor of size 3*3; a,v are real hydrodynamic variables;
  coefAD[0][0] = a[0]*(coefA*u[0] /*+ 0.5*fn[0]*/)/cs4; coefAD[0][1] = a[0]*(coefA*u[1] /*+ 0.5*fn[1]*/)/cs4; coefAD[0][2] = a[0]*(coefA*u[2] /*+ 0.5*fn[2]*/)/cs4;
  coefAD[1][0] = a[1]*(coefA*u[0] /*+ 0.5*fn[0]*/)/cs4; coefAD[1][1] = a[1]*(coefA*u[1] /*+ 0.5*fn[1]*/)/cs4; coefAD[1][2] = a[1]*(coefA*u[2] /*+ 0.5*fn[2]*/)/cs4;
  coefAD[2][0] = a[2]*(coefA*u[0] /*+ 0.5*fn[0]*/)/cs4; coefAD[2][1] = a[2]*(coefA*u[1] /*+ 0.5*fn[1]*/)/cs4; coefAD[2][2] = a[2]*(coefA*u[2] /*+ 0.5*fn[2]*/)/cs4;
  const Real trace_coefAD = coefAD[0][0] + coefAD[1][1] + coefAD[2][2];

  // Correction factor 1./(1.+1./(2.*tau_f)) should be multiplied before each moment's expression;
  const Real modifactor = 1./(1.+1./(2.*tau_f));
  mEq( 0) = modifactor*coefAB;

  mEq( 1) = modifactor*coefAC*cs2*coefE[0];
  mEq( 2) = modifactor*coefAC*cs2*coefE[1];
  mEq( 3) = modifactor*coefAC*cs2*coefE[2];

  mEq( 4) = modifactor*2.*cs4*trace_coefAD;
  mEq( 5) = modifactor*6.*cs4*coefAD[0][0] - 2.*cs4*trace_coefAD;
  mEq( 6) = modifactor*2.*cs4*(coefAD[1][1] - coefAD[2][2]);
  mEq( 7) = modifactor*cs4*(coefAD[0][1] + coefAD[1][0]);
  mEq( 8) = modifactor*cs4*(coefAD[1][2] + coefAD[2][1]);
  mEq( 9) = modifactor*cs4*(coefAD[0][2] + coefAD[2][0]);

  mEq(10) = 0.; mEq(11) = 0.; mEq(12) = 0.; mEq(13) = 0.; mEq(14) = 0.; mEq(15) = 0.;

  mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*trace_coefAD*(6.*cs4+cs2) - trace_coefAD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  + trace_coefAD*cs2*(18.*cs2+1) - cs2*trace_coefAD;
  mEq(16) = modifactor*mEq(16);
  
  mEq(17) = 2.*cs2*(2.*coefAD[0][0] - coefAD[1][1] - coefAD[2][2]) - 18.*cs4*coefAD[0][0] + 6.*cs4*trace_coefAD;
  mEq(17) = modifactor*mEq(17);

  mEq(18) = 0.;

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> Phig_equilibrium_moments(const Array1D<Real,0,2>& fields,
				   const RealVect& u, const RealVect& a/*, const RealVect& gn*/) {  // u,a are real velocity & acceleration field of fluid g;
  Array1D<Real,0,nvel> mEq;

  const Real phi = fields(1);

  const Real coefA = phi;
  const Real coefB = 0.;
  const Real coefAB = coefA*coefB;
  const Real coefC = 1./cs2;
  const Real coefAC = coefA*coefC;
  RealVect coefE = { a[0], a[1], a[2] };
  Real coefAD[3][3];   // \frac{\bm{a}\bm{v}}{c^4}, tensor of size 3*3; a,v are real hydrodynamic variables;
  coefAD[0][0] = a[0]*(coefA*u[0] /*+ 0.5*gn[0]*/)/cs4; coefAD[0][1] = a[0]*(coefA*u[1] /*+ 0.5*gn[1]*/)/cs4; coefAD[0][2] = a[0]*(coefA*u[2] /*+ 0.5*gn[2]*/)/cs4;
  coefAD[1][0] = a[1]*(coefA*u[0] /*+ 0.5*gn[0]*/)/cs4; coefAD[1][1] = a[1]*(coefA*u[1] /*+ 0.5*gn[1]*/)/cs4; coefAD[1][2] = a[1]*(coefA*u[2] /*+ 0.5*gn[2]*/)/cs4;
  coefAD[2][0] = a[2]*(coefA*u[0] /*+ 0.5*gn[0]*/)/cs4; coefAD[2][1] = a[2]*(coefA*u[1] /*+ 0.5*gn[1]*/)/cs4; coefAD[2][2] = a[2]*(coefA*u[2] /*+ 0.5*gn[2]*/)/cs4;
  const Real trace_coefAD = coefAD[0][0] + coefAD[1][1] + coefAD[2][2];

  // Correction factor 1./(1.+1./(2.*tau_g)) should be multiplied before each moment's expression;
  const Real modifactor = 1./(1.+1./(2.*tau_g));
  mEq( 0) = modifactor*coefAB;

  mEq( 1) = modifactor*coefAC*cs2*coefE[0];
  mEq( 2) = modifactor*coefAC*cs2*coefE[1];
  mEq( 3) = modifactor*coefAC*cs2*coefE[2];

  mEq( 4) = modifactor*2.*cs4*trace_coefAD;
  mEq( 5) = modifactor*6.*cs4*coefAD[0][0] - 2.*cs4*trace_coefAD;
  mEq( 6) = modifactor*2.*cs4*(coefAD[1][1] - coefAD[2][2]);
  mEq( 7) = modifactor*cs4*(coefAD[0][1] + coefAD[1][0]);
  mEq( 8) = modifactor*cs4*(coefAD[1][2] + coefAD[2][1]);
  mEq( 9) = modifactor*cs4*(coefAD[0][2] + coefAD[2][0]);

  mEq(10) = 0.; mEq(11) = 0.; mEq(12) = 0.; mEq(13) = 0.; mEq(14) = 0.; mEq(15) = 0.;

  mEq(16) = coefAB*(3.*(3.*cs2 + 6.*cs4) - 18.*cs2 + 1) + 3.*trace_coefAD*(6.*cs4+cs2) - trace_coefAD*(6.*cs2+12.*cs4+9.*cs4+18.*cs4*cs2)
  + trace_coefAD*cs2*(18.*cs2+1) - cs2*trace_coefAD;
  mEq(16) = modifactor*mEq(16);
  
  mEq(17) = 2.*cs2*(2.*coefAD[0][0] - coefAD[1][1] - coefAD[2][2]) - 18.*cs4*coefAD[0][0] + 6.*cs4*trace_coefAD;
  mEq(17) = modifactor*mEq(17);

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
  RealVect v_b = (fields(0)*uf + fields(1)*ug)/(fields(0) + fields(1)); // add the acceleration terms!!!!
  mfEq = fequilibrium_moments(fields, v_b/*, noisef*/); // !!!!!!!!!!!!??????????!!!!!!!!!!!
  mgEq = gequilibrium_moments(fields, v_b/*, noiseg*/);

  // moments of force term, modified version with pre-factor!
  mPhif = Phif_equilibrium_moments(fields, uf, af/*, noisef*/);
  mPhig = Phig_equilibrium_moments(fields, ug, ag/*, noiseg*/);

  // thermal noise in moment space: fn(x,y,z,ncomp) & gn(x,y,z,ncomp);

  const Real tau_f_bar = tau_f*(1.+0.5/tau_f);
  const Real tau_g_bar = tau_g*(1.+0.5/tau_g);
  for (int a=0; a<nvel; ++a) {  // the correction factor for the force term is included in the function [Phif_equilibrium_moments];
    Real Raf = 1./tau_f_bar * (mfEq(a) - mf(a)) + mPhif(a) + fn(x,y,z,a);
    Real Rag = 1./tau_g_bar * (mgEq(a) - mg(a)) + mPhig(a) + gn(x,y,z,a);
    mf(a) = mf(a) + Raf;
    mg(a) = mg(a) + Rag;  // ?????? 3/2 times larger?
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
  MultiFab& rho_eq, MultiFab& phi_eq, MultiFab& rhot_eq) {

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
  ParallelFor(fold, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    #ifdef AMReX_USE_CUDA
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x + 
                        threadIdx.y * blockDim.x * gridDim.x + 
                        threadIdx.z * blockDim.x * gridDim.x * gridDim.y;
    if (thread_id == 0) {
        printf("Block size (x, y, z): (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
        printf("Grid size (x, y, z): (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
        printf("Total threads: %d\n", gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z);
    }
    #endif
    collide_stream(x, y, z,
      fOld[nbx], gOld[nbx],
      fNew[nbx], gNew[nbx], h[nbx], fn[nbx], gn[nbx]);
  });
  /*
  Swap (MultiFab &dst, MultiFab &src, int srccomp, int dstcomp, int numcomp, int nghost)
  Swap from src to dst including nghost ghost cells. 
  */
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);

  // here all Multifab variables;
  LBM_hydrovars_modified(geom, fold, gold, hydrovsbar);
  //amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  thermal_noise_new(fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
  //amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  LBM_hydrovars(fold, gold, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
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
      const Real rho = 2.*C1;// + C2;
      const Real phi = 2.*C2;//(C1-C2)+0.01*RandomNormal(0., 1., engine);
      for (int i=0; i<nvel; i++) {
	      f(x,y,z,i) = w[i]*rho;
	      g(x,y,z,i) = w[i]*phi;
      }
    });
  }
  //Print() << "Call function 'LBM_hydrovars_modified' ...\n";
  LBM_hydrovars_modified(geom, mf, mg, hydrovsbar);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  //Print() << "Call function 'thermal_noise' ...\n";
  thermal_noise(fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  //Print() << "Call function 'LBM_hydrovars' ...\n";
  LBM_hydrovars(mf, mg, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
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
           MultiFab& rho_eq, MultiFab& phi_eq, MultiFab& rhot_eq) {

  mf.ParallelCopy(mf0); mg.ParallelCopy(mg0);
  mf.FillBoundary(geom.periodicity());
  mg.FillBoundary(geom.periodicity());
  //hydrovs.FillBoundary(geom.periodicity());
  //hydrovsbar.FillBoundary(geom.periodicity());

  LBM_hydrovars_modified(geom, mf, mg, hydrovsbar);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  thermal_noise(fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
  amrex::Gpu::synchronize();  AMREX_GPU_ERROR_CHECK();
  LBM_hydrovars(mf, mg, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
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
    const Real ry = y-box[1]/2;
#if (AMREX_SPACEDIM == 2)
    const Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    const Real rz = z-box[0]/2;
    const Real r2 = rx*rx+ry*ry+rz*rz;
#endif
    const Real r = sqrt(r2);

    const Real rho = 1.0;
    const Real phi0 = 0.5;
    const Real phi = phi0*tanh((R-r)/sqrt(kappa));

    // transform from free energy model (rho,phi) --> (rhof, rhog); droplet is fluid f
    const Real rhof = phi + 0.5;
    const Real rhog = 0.5 - phi;
    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rhof;
      g[nbx](x,y,z,i) = w[i]*rhog;
    }

  });
  LBM_hydrovars_modified(geom, mf, mg, hydrovsbar);
  thermal_noise(fnoisevs, gnoisevs, rho_eq, phi_eq, rhot_eq);
  //thermal_noise(fnoisevs, gnoisevs, hydrovsbar);
  LBM_hydrovars(mf, mg, hydrovs, hydrovsbar, fnoisevs, gnoisevs);
}

#endif
