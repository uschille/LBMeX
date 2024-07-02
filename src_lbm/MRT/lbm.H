#ifndef LBM_H_
#define LBM_H_

#include "d3q19.H"
#include "fluctuations.H"

// hard coded
AMREX_GPU_MANAGED Real density = 1.0;
AMREX_GPU_MANAGED Real tau = 1.0;

// user modifiable
AMREX_GPU_MANAGED Real temperature = 0.0;
AMREX_GPU_MANAGED Real kappa = 0.0;
AMREX_GPU_MANAGED Real Beta = 0.01;
AMREX_GPU_MANAGED Real rhov = 0.1;
AMREX_GPU_MANAGED Real rhol = 1.1;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure(Real rho) {

  // Real p0 = rho*T; // Eq. (40) in Swift et al.
  Real square_term = (rho - rhol)*(rho - rhov);
  Real p0 = Beta*square_term*(2*rho*(2*rho - rhol - rhov) - square_term); // eq 2 and para after in Gross 1010
  // note that gradient terms are included in fequilibrium

  return p0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect gradient(int x, int y, int z, const Array4<Real>& field) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  const Real system_cs2 = 2.0*Beta*pow(rhol - rhov, 2);
  for (int i=0; i<ncomp; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    for (int dir=0; dir<3; dir++) {
      gradient[dir] += w[i]/system_cs2*field(xp,yp,zp,0)*c[i][dir];
    }
  }
  return gradient;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real laplacian(int x, int y, int z, const Array4<Real>& field) {
  Real laplacian = 0.0;
  const Real system_cs2 = 2.0*Beta*pow(rhol - rhov, 2);
  for (int i=0; i<ncomp; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    laplacian += 2.*w[i]/system_cs2*(field(xp,yp,zp,0)-field(x,y,z,0));
  }
  return laplacian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> mfequilibrium(const Real& rho,
			      const Array1D<RealVect,0,1>& gradients,
			      const Array1D<Real,0,1>& laplacians,
			      const RealVect& u) {
  Array1D<Real,0,ncomp> mEq;

  const Real u2 = u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
  const RealVect Drho = gradients(0);
  const Real D2rho = laplacians(0);
  const Real pbulk = pressure(rho);

  Real G[3][3], traceG=0.; // Eq. (39) in Swift et al.
  G[0][0] = kappa*Drho[0]*Drho[0]; traceG  = G[0][0];
  G[1][1] = kappa*Drho[1]*Drho[1]; traceG += G[1][1];
  G[2][2] = kappa*Drho[2]*Drho[2]; traceG += G[2][2];
  G[0][1] = kappa*Drho[0]*Drho[1];
  G[1][2] = kappa*Drho[1]*Drho[2];
  G[2][0] = kappa*Drho[2]*Drho[0];

  mEq( 0) = rho;

  mEq( 1) = rho*u[0];
  mEq( 2) = rho*u[1];
  mEq( 3) = rho*u[2];

  mEq( 4) = 3.*(pbulk - kappa*rho*D2rho) + rho*u2 - 0.5*traceG - rho;
  mEq( 5) = rho*(3.*u[0]*u[0]-u2) + (3.*G[0][0]-traceG);
  mEq( 6) = rho*(u[1]*u[1]-u[2]*u[2]) + (G[1][1]-G[2][2]);
  mEq( 7) = rho*u[0]*u[1] + G[0][1];
  mEq( 8) = rho*u[1]*u[2] + G[1][2];
  mEq( 9) = rho*u[2]*u[0] + G[2][0];

  mEq(10) = 0.;
  mEq(11) = 0.;
  mEq(12) = 0.;
  mEq(13) = 0.;
  mEq(14) = 0.;
  mEq(15) = 0.;

  mEq(16) = 1.5*traceG - 3.*(pbulk - kappa*rho*D2rho) + rho;
  mEq(17) = 2.*(G[1][1]+G[2][2]-2.*G[0][0]);
  mEq(18) = 2.*(G[2][2]-G[1][1]);

  return mEq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> mfequilibrium(int x, int y, int z, const Array4<Real>& m) {
  const Real rho = m(x,y,z,0);
  const Array1D<RealVect,0,1> gradients = {gradient(x,y,z,m)};
  const Array1D<Real,0,1> laplacians = {laplacian(x,y,z,m)};
  const RealVect j = { m(x,y,z,1), m(x,y,z,2), m(x,y,z,3) };
  return mfequilibrium(rho, gradients, laplacians, j/rho);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> fequilibrium(const Real& rho, const RealVect& u) {
  Array1D<Real,0,ncomp> fEq;
  for (int i=0; i<ncomp; ++i) {
    Real uc = u.dotProduct(c[i]);
    fEq(i) = w[i] * rho * ( 1. + uc/cs2 + (uc*uc - u.dotProduct(u)*cs2)/(2.*cs2*cs2) );
  }
  return fEq;
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> relax_moments(int x, int y, int z, const Array4<Real>& m) {
  const Real gamma = 1. - 1./tau;
  const Array1D<Real,0,ncomp> mEq = mfequilibrium(x, y, z, m);
  Array1D<Real,0,ncomp> mom;
  for (int i=0; i<ncomp; ++i) {
    mom(i) = m(x,y,z,i) = mEq(i) + gamma * (m(x,y,z,i) - mEq(i));
  }
  return mom;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> thermalize_moments(int x, int y, int z, const Array4<Real>& m, RandomEngine const& engine) {
  const Real gamma = 1. - 1./tau;
  const Real rho = m(x,y,z,0);
  const Real phi = sqrt(rho*temperature/cs2*(1.-gamma*gamma));
  Array1D<Real,0,ncomp> r = {}; // {} to value initialize r
  for (int i=4; i<ncomp; ++i) {
    r(i) = sqrt(b[i])*phi*RandomNormal(0., 1., engine);
    m(x,y,z,i) += r(i);
  }
  return r;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,ncomp> thermalize_moments(int x, int y, int z, const Array4<Real>& m, const Array4<Real>& xi) {
  Array1D<Real,0,ncomp> r = {};
  for (int i=ncons; i<ncomp; ++i) {
    m(x,y,z,i) += xi(x,y,z,i);;
  }
  return r;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z, const Array4<Real>& f, const Array4<Real>& msite, const Array4<Real>& noise, const Array4<Real>& hsite, const Array4<Real>& hEqsite, RandomEngine const& engine) {
  const Real gamma = 1. - 1./tau;
  Array1D<Real,0,ncomp> m, h, hEq, r;
  // collision step
  m = moments(x, y, z, f, msite);
  relax_moments(x, y, z, msite);
#ifdef OLD_NOISE
  r = thermalize_moments(x, y, z, msite, engine);
#else
  r = thermalize_moments(x, y, z, msite, noise);
#endif
  // record hydrodynamic variables
  h = hydrovars(m);
  r = hydrovars(r);
  
  const Array1D<RealVect,0,1> gradients = {gradient(x,y,z,msite)};
  const Array1D<Real,0,1> laplacians = {laplacian(x,y,z,msite)};
  hEq = hydrovars(mfequilibrium(h(0), gradients, laplacians, { h(1), h(2), h(3) }));

  for (int i=0; i<=AMREX_SPACEDIM; ++i) hsite(x,y,z,i) = h(i);
  for (int i=AMREX_SPACEDIM+1; i<(AMREX_SPACEDIM+1)*(AMREX_SPACEDIM+2)/2; ++i) {
    // note that h is pre-collision
    hEqsite(x,y,z,i) = hEq(i);
    hsite(x,y,z,i) = hEq(i) + 0.5*(1.+gamma)*(h(i)-hEq(i)) + 0.5*r(i);
  }
  populations(x, y, z, f, msite);
}

// stream and collide (pull scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stream_collide(int x, int y, int z, const Array4<Real>& fOld, const Array4<Real>& fNew, const Array4<Real>& m, const Array4<Real>& noise, const Array4<Real>& h, const Array4<Real>& hEq, RandomEngine const& engine) {
  for (int i=0; i<ncomp; ++i) { // pull populations from neighbors
    int xm = x - c[i][0];
    int ym = y - c[i][1];
    int zm = z - c[i][2];
    fNew(x,y,z,i) = fOld(xm,ym,zm,i);
  }
  collide(x, y, z, fNew, m, noise, h, hEq, engine); // collide populations on current site
}

// collide and stream (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide_stream(int x, int y, int z, const Array4<Real>& fOld,  const Array4<Real>& fNew, const Array4<Real>& m, const Array4<Real>& noise, const Array4<Real>& h, const Array4<Real>& hEq, RandomEngine const& engine) {
  collide(x, y, z, fOld, m, noise, h, hEq, engine); // collide populations on current site
  for (int i=0; i<ncomp; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = fOld(x,y,z,i);
  }
}

// apply fluctuating lattice Boltzmann equation for whole lattice
inline void collide_stream(const Geometry& geom,
			   MultiFab& fold, MultiFab& fnew,
			   MultiFab& moments, MultiFab& noise,
			   MultiFab& hydro, MultiFab& hydroEq) {
  const int halo = 1;

  fold.FillBoundary(geom.periodicity());
  noise.FillBoundary(geom.periodicity());

  for (MFIter mfi(fold); mfi.isValid(); ++mfi) {
    const Box& valid_box = mfi.growntilebox(halo);
    const Array4<Real>& fOld = fold.array(mfi);
    const Array4<Real>& fNew = fnew.array(mfi);
    const Array4<Real>& mom = moments.array(mfi);
    const Array4<Real>& xi = noise.array(mfi);
    const Array4<Real>& hyd = hydro.array(mfi);
    const Array4<Real>& hydEq = hydroEq.array(mfi);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      collide_stream(x, y, z, fOld, fNew, mom, xi, hyd, hydEq, engine);
    });
  }

  std::swap(fold,fnew);

}

// // calculate hydrodynamic moments for whole lattice
// inline void hydrovars(MultiFab& f, MultiFab& m, MultiFab& hydrovs, MultiFab& hydroEq) {
//   auto const & fa = f.arrays();
//   auto const & ma = m.arrays();
//   auto const & ha = hydrovs.arrays();
//   auto const & haEq = hydroEq.arrays();
//   ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
//     Array1D<Real,0,ncomp> mv, hv, hEq;
//     mv = moments(x, y, z, fa[nbx], ma[nbx]);
//     hv = hydrovars(mv);
    
//     const Array1D<RealVect,0,1> gradients = {gradient(x,y,z,m)};
//     const Array1D<Real,0,1> laplacians = {laplacian(x,y,z,m)};
//     hEq = hydrovars(mfequilibrium(hv(0), gradients, laplacians, { hv(1), hv(2), hv(3) }));
//     for (int i=0; i<10; ++i) {
//       ha[nbx](x,y,z,i) = hv(i);
//       haEq[nbx](x,y,z,i) = hEq(i);
//     }
//   });
// }

// LB timestep for whole lattice
inline void LBM_timestep(const Geometry& geom,
			 MultiFab& fold, MultiFab& fnew,
			 MultiFab& moments, MultiFab& noise,
			 MultiFab& hydro, MultiFab& hydroEq) {

#ifndef OLD_NOISE
  generate_fluctuations(geom, moments, noise);
#endif

  collide_stream(geom, fold, fnew, moments, noise, hydro, hydroEq);

}

#endif
