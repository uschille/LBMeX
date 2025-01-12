#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"

AMREX_GPU_MANAGED Real G = 1.5;
AMREX_GPU_MANAGED Real rhoA = 2.3; // values reported by Belardinelli et al. [Phys. Rev. E 91, 023313 (2015)]
AMREX_GPU_MANAGED Real rhoB = 0.06;
AMREX_GPU_MANAGED Real kappa = 0.174;

AMREX_GPU_MANAGED Real temperature = 1e-5;

AMREX_GPU_CONSTANT Real lambda_d = 1.0;
AMREX_GPU_CONSTANT Real lambda_b = 1.0;
AMREX_GPU_CONSTANT Real lambda_s = 1.0;
AMREX_GPU_CONSTANT Real lambda_o = 1.0;
AMREX_GPU_CONSTANT Real lambda_e = 1.0;

AMREX_GPU_MANAGED const Array<Real,nvel>
lambda_f = { 0.,
  lambda_d, lambda_d, lambda_d,
  lambda_b, lambda_s, lambda_s, lambda_s, lambda_s, lambda_s,
  lambda_o, lambda_o, lambda_o, lambda_o, lambda_o,
  lambda_e, lambda_e, lambda_e
};

AMREX_GPU_MANAGED const Array<Real,nvel>
lambda_g = { 0.,
  lambda_d, lambda_d, lambda_d,
  lambda_b, lambda_s, lambda_s, lambda_s, lambda_s, lambda_s,
  lambda_o, lambda_o, lambda_o, lambda_o, lambda_o,
  lambda_e, lambda_e, lambda_e
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect gradient(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    for (int dir=0; dir<3; dir++) {
      gradient[dir] += w[i]/cs2*field(xp,yp,zp,icomp)*c[i][dir];
    }
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
  return laplacian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> force_moments(const Real& rho, const RealVect& v, const RealVect& a) {
  Array1D<Real,0,nvel> phi;

  /* density mode */
  phi(0) = 0.0;

  /* momentum modes */
  phi(1) = rho*a[0];
  phi(2) = rho*a[1];
  phi(3) = rho*a[2];

  /* stress modes */
  phi(4) = rho*(a[0]*v[0] + a[1]*v[1] + a[2]*v[2]);
  phi(5) = rho*(2.*a[0]*v[0] - a[1]*v[1] - a[2]*v[2]);
  phi(6) = rho*(a[1]*v[1] - a[2]*v[2]);
  phi(7) = 0.5*rho*(a[0]*v[1]+a[1]*v[0]);
  phi(8) = 0.5*rho*(a[1]*v[2]+a[2]*v[1]);
  phi(9) = 0.5*rho*(a[2]*v[0]+a[0]*v[2]);

  /* kinetic modes  */
  for (int i=10; i<nvel; ++i) phi(i) = 0.0;

  return phi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,2*nvel> thermal_noise(const Array<Real,2>& rho, const RandomEngine& engine) {
  Array1D<Real,0,2*nvel> xi = {};

  // TODO: need to use the reference densities [uschill 2025/01/12]

  // densities are conserved
  xi(0) = xi(nvel) = 0.0;

  // fluxes are anticorrelated (total momentum is conserved)
  for (int i=1; i<=AMREX_SPACEDIM; ++i ) {
    const Real pref = (2.*lambda_d-lambda_d*lambda_d)*temperature*rho[0]*rho[1]/(rho[0]+rho[1]);
    xi(i) = std::sqrt(pref) * RandomNormal(0., 1., engine);
    xi(i+nvel) = - xi(i);
  }

  // remaining modes
  for (int i=AMREX_SPACEDIM+1; i<nvel; ++i) {
    const Real pref = (2.*lambda_f[i]-lambda_f[i]*lambda_f[i])*b[i]*cs2*temperature*rho[0];
    const Real preg = (2.*lambda_g[i]-lambda_g[i]*lambda_g[i])*b[i]*cs2*temperature*rho[1];
    xi(i) = std::sqrt(pref) * RandomNormal(0., 1., engine);
    xi(i+nvel) = std::sqrt(preg) * RandomNormal(0., 1., engine);
  }

  return xi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z,
	     const Array4<Real>& f,
	     const Array4<Real>& g,
       const Array4<Real>& h,
       const RandomEngine& engine) {
  Array1D<Real,0,nvel> mf, mg, mfEq, mgEq;
  Array1D<Real,0,nvel> phif, phig;
  Array1D<Real,0,2*nvel> xi;
  Array<Real,2> rho;
  Array<RealVect,2> v, a;
  RealVect v_b, a_b;

  mf = moments(x,y,z,f);
  mg = moments(x,y,z,g);

  rho[0] = mf(0);
  rho[1] = mg(0);

  v[0] = RealVect(mf(1), mf(2), mf(3))/mf(0);
  v[1] = RealVect(mg(1), mg(2), mg(3))/mg(0);

  a[0] = -G*cs2*gradient(x,y,z,h,1);
  a[1] = -G*cs2*gradient(x,y,z,h,0);

  v_b = (rho[0]*v[0]+rho[1]*v[1])/(rho[0]+rho[1]);
  a_b = (rho[0]*a[0]+rho[1]*a[1])/(rho[0]+rho[1]);

  mfEq = equilibrium_moments(rho[0], v_b+0.5*a_b);
  mgEq = equilibrium_moments(rho[1], v_b+0.5*a_b);

  phif = force_moments(rho[0], v[0], a[0]);
  phig = force_moments(rho[1], v[1], a[1]);

  xi = thermal_noise(rho, engine);

  for (int i=0; i<nvel; ++i) {
    mf(i) = mf(i) - lambda_f[i] * (mf(i) - mfEq(i)) + (1.-0.5*lambda_f[i])*phif(i) + xi(i);
    mg(i) = mg(i) - lambda_g[i] * (mg(i) - mgEq(i)) + (1.-0.5*lambda_g[i])*phig(i) + xi(i+nvel);
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
		    const Array4<Real>& h,
        const RandomEngine& engine) {
  collide(x, y, z, fOld, gOld, h, engine); // collide populations on current site
  stream_push(x, y, z, fOld, gOld, fNew, gNew); // push populations to neighbors
}

// apply fluctuating lattice Boltzmann equation for whole lattice
inline void collide_stream(const Geometry& geom,
			   MultiFab& fold, MultiFab& gold,
			   MultiFab& fnew, MultiFab& gnew,
			   MultiFab& hydrovs) {
  const int halo = 1; // need to push from first halo layer into domain
  fold.FillBoundary(geom.periodicity());
  gold.FillBoundary(geom.periodicity());
  hydrovs.FillBoundary(geom.periodicity());
  for (MFIter mfi(fold); mfi.isValid(); ++mfi) {
    const Array4<Real>& fOld = fold.array(mfi);
    const Array4<Real>& gOld = gold.array(mfi);
    const Array4<Real>& fNew = fnew.array(mfi);
    const Array4<Real>& gNew = gnew.array(mfi);
    const Array4<Real>& h = hydrovs.array(mfi);
    const Box& valid_box = mfi.growntilebox(halo);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, const RandomEngine& engine) {
      collide_stream(x, y, z, fOld, gOld, fNew, gNew, h, engine);
    });
  }
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);
}

inline Vector<std::string> hydrovars_names(const int numVars) {
  Vector<std::string> var_names(numVars);
  std::string name;
  int cnt = 0;
  // rho, phi
  if (cnt<numVars) var_names[cnt++] = "rhoA";
  if (cnt<numVars) var_names[cnt++] = "rhoB";
  // u
  for (int d=0; d<AMREX_SPACEDIM && cnt<numVars; d++) {
    name = "uA";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // j
  for (int d=0; d<AMREX_SPACEDIM && cnt<numVars; d++) {
    name = "uB";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // remaining moments
  for (; cnt<nvel+AMREX_SPACEDIM+1 && cnt<numVars;) {
    name = "mf" + std::to_string(cnt-(AMREX_SPACEDIM+1));
    var_names[cnt++] = name;
  }
  for (; cnt<2*nvel && cnt<numVars;) {
    name = "mg"+std::to_string(cnt-nvel);
    var_names[cnt++] = name;
  }
  return var_names;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hydrovars(int x, int y, int z,
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& h) {
  const Array1D<Real,0,nvel> mf = moments(x,y,z,f);
  const Array1D<Real,0,nvel> mg = moments(x,y,z,g);
  h(x,y,z,0) = mf(0);
  h(x,y,z,1) = mg(0);
  for (int i=1; i<=AMREX_SPACEDIM; ++i) {     // momentum/velocity i=1..3
    h(x,y,z,i+1) = mf(i)/mf(0);               // u_f to be stored in 2..4
    h(x,y,z,i+1+AMREX_SPACEDIM) = mg(i)/mg(0);// u_g to be stored in 5..7
  }
  for (int i=1+AMREX_SPACEDIM; i<nvel; ++i) { // remaining moments i=4..18
    h(x,y,z,i+1+AMREX_SPACEDIM) = mf(i);      // f to be stored in 8..22
    h(x,y,z,i+nvel) = mg(i);                  // g to be stored in 23..37
  }
}

// calculate hydrodynamic moments for whole lattice
inline void hydrovars(MultiFab& mf, MultiFab& mg, MultiFab& hydrovs) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & h = hydrovs.arrays();
  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    hydrovars(x, y, z, f[nbx], g[nbx], h[nbx]);
  });
}

// LB timestep for whole lattice
inline void LBM_timestep(const Geometry& geom,
			 MultiFab& fold, MultiFab& gold,
			 MultiFab& fnew, MultiFab& gnew,
			 MultiFab& hydrovs) {
  collide_stream(geom, fold, gold, fnew, gnew, hydrovs);
  hydrovars(fold, gold, hydrovs);
}

// homogeneous mixture of C1 and C2 with random concentration fluctuations
inline void LBM_init_mixture(MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs) {
  const Real rho = 0.5*(rhoA+rhoB);
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Array4<Real>& f = mf.array(mfi);
    const Array4<Real>& g = mg.array(mfi);
    ParallelForRNG(mfi.validbox(), [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real noise = 0.01*RandomNormal(0., 1., engine);
      for (int i=0; i<nvel; i++) {
        f(x,y,z,i) = w[i]*(rho + noise);
        g(x,y,z,i) = w[i]*(rho - noise);
      }
    });
  }
  hydrovars(mf, mg, hydrovs);
}

// flat interface between C1 and C2
inline void LBM_init_flat_interface(const Geometry& geom,
				    MultiFab& mf,
				    MultiFab& mg,
				    MultiFab& hydrovs) {
  const IntVect box = geom.Domain().length();

  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real rx = x-box[0]/2;
    const Real ry = y-box[1]/2;
#if (AMREX_SPACEDIM == 2)
    const Real r2 = rx*rx+ry*ry;
#elif (AMREX_SPACEDIM == 3)
    const Real rz = z-box[0]/2;
    const Real r2 = rx*rx+ry*ry+rz*rz;
#endif

    const Real rho=1.0;
    Real phi;

    if (rx < 0.) {
      phi = 0.5;
    } else {
      phi = -0.5;
    }

    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*phi;
    }

  });
  hydrovars(mf, mg, hydrovs);
}

// binary droplet of C1 and C2
inline void LBM_init_droplet(const Real r,
			     const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs) {
  const IntVect box = geom.Domain().length();
  const Real R = r*box[0];

  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  GpuArray<Real,3> dx = geom.CellSizeArray();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real rx = x-box[0]/2;
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

    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*phi;
    }

  });
  hydrovars(mf, mg, hydrovs);
}

#endif
