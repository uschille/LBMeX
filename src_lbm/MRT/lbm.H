#ifndef LBM_H_
#define LBM_H_

#include "d3q19.H"
#include "fluctuations.H"

AMREX_GPU_MANAGED Real density = 1.0;
AMREX_GPU_MANAGED Real temperature = 0.0;
AMREX_GPU_MANAGED Real tau = 0.9;

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
  const Array1D<Real,0,ncomp> mEq = mequilibrium(x, y, z, m);
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
  hEq = hydrovars(mequilibrium(h(0), { h(1), h(2), h(3) }));
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

// calculate hydrodynamic moments for whole lattice
inline void hydrovars(MultiFab& f, MultiFab& m, MultiFab& hydrovs, MultiFab& hydroEq) {
  auto const & fa = f.arrays();
  auto const & ma = m.arrays();
  auto const & ha = hydrovs.arrays();
  auto const & haEq = hydroEq.arrays();
  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    Array1D<Real,0,ncomp> mv, hv, hEq;
    mv = moments(x, y, z, fa[nbx], ma[nbx]);
    hv = hydrovars(mv);
    hEq = hydrovars(mequilibrium(hv(0), { hv(1), hv(2), hv(3) }));
    for (int i=0; i<10; ++i) {
      ha[nbx](x,y,z,i) = hv(i);
      haEq[nbx](x,y,z,i) = hEq(i);
    }
  });
}

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
