#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"
// #include "rng_functions.H"

AMREX_GPU_MANAGED Real tau_r = 1.0; // What is used in Gross 2010 for all relaxation rates
// AMREX_GPU_MANAGED Real tau_p = 1.0;

AMREX_GPU_MANAGED int nx = 16;
AMREX_GPU_MANAGED Real kappa = 0.0;
AMREX_GPU_MANAGED Real Beta = 0.01;
AMREX_GPU_MANAGED Real temperature = 0.0;
AMREX_GPU_MANAGED Real rhov = 0.1;
AMREX_GPU_MANAGED Real rhol = 1.1;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure(Real rho, Real phi);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,1> chemical_potentials(const Array1D<Real,0,1>& fields,
				  const Array1D<RealVect,0,1>& gradients,
				  const Array1D<Real,0,1>& laplacians);

#include "LBM_fluctuations.H"

Real thermal_cs2(Real rho){
  Real square_term = (rho - rhol)*(rho - rhov);
  Real expr1 = Beta*square_term*(rhol + rhov - 6.0*rho);
  Real expr2 = Beta*(rho - rhol)*(2.0*rho*(-2.0*rho + rhol + rhov) + square_term);
  Real expr3 = Beta*(rho - rhov)*(2.0*rho*(-2.0*rho + rhol + rhov) + square_term);
  Real out = -expr1 - expr2 - expr3;
  return out;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect gradient(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  const Real system_cs2 = 2.0*Beta*pow(rhol - rhov, 2);
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
  const Real system_cs2 = 2.0*Beta*pow(rhol - rhov, 2);
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    laplacian += 2.*w[i]/cs2*(field(xp,yp,zp,icomp)-field(x,y,z,icomp));
  }
  return laplacian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,1> chemical_potentials(const Array1D<Real,0,1>& fields,
				  const Array1D<RealVect,0,1>& gradients,
				  const Array1D<Real,0,1>& laplacians) {
  
  const Real rho = fields(0);
  const Real D2rho = laplacians(0);

  Array1D<Real,0,1> mu = {0.0} ;
  // mu(0) = lambda/4.*(1.+phi2/rho2) + T/2*log((rho2-phi2)/4) - kappa*D2rho; //This is most definitely wrong for the multiphase case

  // Real square_term = (rho - rhol)*(rho - rhov);
  Real mu_0 = Beta*pow(rho - rhol, 2)*(2*rho - 2*rhov) + Beta*pow(rho - rhov, 2)*(2*rho - 2*rhol);
  mu(0) = mu_0 - kappa*D2rho;
  Print() << "Chemical Potential: " << mu(0) << "\n";

  return mu;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real pressure(Real rho) {

  // Real p0 = rho*T; // Eq. (40) in Swift et al.
  Real square_term = (rho - rhol)*(rho - rhov);
  Real p0 = Beta*square_term*(2*rho*(2*rho - rhol - rhov) - square_term); // eq 2 and para after in Gross 1010
  // note that gradient terms are included in fequilibrium
  // Print() << "Bulk pressure: " << p0 << "\n";

  return p0;
}


#if 0 // equilibrium populations not necessary for MRT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> fequilibrium(const Array1D<Real,0,1>& fields,
			      const Array1D<RealVect,0,1>& gradients,
			      const Array1D<Real,0,1>& laplacians,
			      const RealVect& u) {
  Array1D<Real,0,nvel> fEq;

  const Real system_cs2 = Beta*pow(rhol - rhov, 2);

  const Real rho = fields(0);
  const RealVect Drho = gradients(0);
  const Real D2rho = laplacians(0);
  const Real pbulk = pressure(rho);

  Real ruu[3][3];
  ruu[0][0] = rho*u[0]*u[0];
  ruu[1][1] = rho*u[1]*u[1];
  ruu[2][2] = rho*u[2]*u[2];
  ruu[0][1] = rho*u[0]*u[1];
  ruu[1][2] = rho*u[1]*u[2];
  ruu[2][0] = rho*u[2]*u[0];

  Real G[3][3]; // Eq. (39) in Swift et al.
  G[0][0] = kappa*Drho[0]*Drho[0];
  G[1][1] = kappa*Drho[1]*Drho[1];
  G[2][2] = kappa*Drho[2]*Drho[2];
  G[0][1] = kappa*Drho[0]*Drho[1];
  G[1][2] = kappa*Drho[1]*Drho[2];
  G[2][0] = kappa*Drho[2]*Drho[0];

  Real fi, sumf = 0.0;
  
  for (int i=1; i<nvel; ++i) {
    fi = w[i]/sysem_cs2 * pbulk;
    fi += w[i]/sysem_cs2 * rho * (u[0]*c[i][0]+u[1]*c[i][1]+u[2]*c[i][2]);
    fi += w[i]/(2*sysem_cs2**4)*((ruu[0][0]*c[i][0]+2.*ruu[0][1]*c[i][1])*c[i][0]
		     +(ruu[1][1]*c[i][1]+2.*ruu[1][2]*c[i][2])*c[i][1]
		     +(ruu[2][2]*c[i][2]+2.*ruu[2][0]*c[i][0])*c[i][2]);
    fi -= w[i]/(2*sysem_cs2**2)*(ruu[0][0]+ruu[1][1]+ruu[2][2]);
    // gradient terms
    fi -= w[i]/(sysem_cs2**2)*(kappa*rho*D2rho);
    fi += 3.*(wg[i][0][0]*G[0][0]+wg[i][1][1]*G[1][1]+wg[i][2][2]*G[2][2]
	     +wg[i][0][1]*G[0][1]+wg[i][1][2]*G[1][2]+wg[i][2][0]*G[2][0]);
    fEq(i) = fi;
    sumf += fi;
  }
  fEq(0) = rho - sumf;

  return fEq;
}
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,nvel> fequilibrium_moments(const Array1D<Real,0,1>& fields,
			      const Array1D<RealVect,0,1>& gradients,
			      const Array1D<Real,0,1>& laplacians,
			      const RealVect& u) {
  Array1D<Real,0,nvel> mEq;

  const Real rho = fields(0);
  const Real u2 = u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
  const RealVect Drho = gradients(0);

  const Real D2rho = laplacians(0);
  const Real pbulk = pressure(rho);

  // Print() << "drho/dx: " << Drho[0] << "\n";
  // Print() << "drho/dy: " << Drho[1] << "\n";
  // Print() << "drho/dz: " << Drho[2] << "\n";

  // Print() << "d2rho/dx: " << D2rho << "\n";

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
void relax_moments(int x, int y, int z,
		   Array1D<Real,0,nvel>& mf,
		   const Array4<Real>& h) {

  const Array1D<Real,0,1> fields = {h(x,y,z,0)};
  const RealVect u = {h(x,y,z,1), h(x,y,z,2), h(x,y,z,3)};

  const Array1D<RealVect,0,1> gradients = {gradient(x,y,z,h,0)};
  const Array1D<Real,0,1> laplacians = {laplacian(x,y,z,h,0)};
  // const Array1D<Real,0,1> mu = chemical_potentials(fields, gradients, laplacians);

  Array1D<Real,0,nvel> mfEq;
  mfEq = fequilibrium_moments(fields, gradients, laplacians, u);

  for (int i=0; i<nvel; ++i) {
    mf(i) = mf(i) - 1./tau_r * (mf(i) - mfEq(i));
  }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void thermalize_moments(Array1D<Real,0,nvel>& mf,
			RandomEngine const& engine) {
  const Real rho = mf(0);
  const Real gamma_r = 1. - 1./tau_r;
  const Real system_cs2 = Beta*pow(rhol - rhov, 2);
  const Real phi_r = sqrt(rho*temperature/system_cs2*(1.-gamma_r*gamma_r));
  Array<Real,nvel> r = {}; // {} to value initialize r
  for (int i=4; i<nvel; ++i) {
    mf(i) += sqrt(b[i])*phi_r*RandomNormal(0., 1., engine);
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void thermalize_moments(int x, int y, int z,
      Array1D<Real,0,nvel>& mf,
      const Array4<Real>& noise) {
  // diffusive moments 1,2,3 in noise at 5,6,7
  // nonconserved moments 4..18 in noise at 8..22 (f) and 23..37 (g)
  for (int i=ncons; i<nvel; ++i) {
    // Print() << noise(x,y,z,i) << " on index " << i << "\n";
    mf(i) += (1. - 0.5/tau_r) * noise(x,y,z,i); //might have to correct for time discretization here. Check in Gross paper once everything else isn't broken
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z,
	     const Array4<Real>& f,
	     const Array4<Real>& h,
       const Array4<Real>& noise) {
  Array1D<Real,0,nvel> mf;

  mf = moments(x,y,z,f);
  // Print() << "moments generated from f matrix\n";

  relax_moments(x,y,z,mf,h);
  // Print() << "Moments relaxed\n";

  thermalize_moments(x,y,z,mf,noise);

  populations(x,y,z,f,mf);
}

// stream from \vec{x} to \vec{x}+\vec{c}_i (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stream_push(int x, int y, int z,
		 const Array4<Real>& fOld,
		 const Array4<Real>& fNew) {
  for (int i=0; i<nvel; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = fOld(x,y,z,i);
  }
}

// collide and stream (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide_stream(int x, int y, int z,
		    const Array4<Real>& fOld,
		    const Array4<Real>& fNew,
		    const Array4<Real>& h,
        const Array4<Real>& noise,
		    RandomEngine const& rng) {
  collide(x, y, z, fOld, h, noise); // collide populations on current site
  // Print() << "Collision step complete\n";
  stream_push(x, y, z, fOld, fNew); // push populations to neighbors
  // Print() << "Streaming step complete\n";
}

// apply fluctuating lattice Boltzmann equation for whole lattice
inline void collide_stream(const Geometry& geom,
			   MultiFab& fold,
			   MultiFab& fnew,
			   MultiFab& hydrovs, MultiFab& noise) {
  const int halo = 1; // need to push from first halo layer into domain
  // Print() << "Entered collide stream function\n";
  fold.FillBoundary(geom.periodicity());
  noise.FillBoundary(geom.periodicity());
  hydrovs.FillBoundary(geom.periodicity());
  // Print() << "Periodic domain boundaries filled\n";
  for (MFIter mfi(fold); mfi.isValid(); ++mfi) {
    const Array4<Real>& fOld = fold.array(mfi);
    const Array4<Real>& fNew = fnew.array(mfi);
    const Array4<Real>& h = hydrovs.array(mfi);
    const Array4<Real>& xi = noise.array(mfi);
    const Box& valid_box = mfi.growntilebox(halo);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      // Print() << x << y << z << "\n";
      collide_stream(x, y, z, fOld, fNew, h, xi, engine);
    });
  }
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hydrovars(int x, int y, int z,
	       const Array4<Real>& f,
	       const Array4<Real>& h) {
  const Array1D<Real,0,nvel> mf = moments(x,y,z,f);
  h(x,y,z,0) = mf(0);
  for (int i=1; i<ncons; ++i) {
    h(x,y,z,i) = mf(i)/mf(0);
  }
  for (int i=ncons; i<nvel; ++i) {
    h(x,y,z,i) = mf(i);
  }
}

// calculate hydrodynamic moments for whole lattice
inline void hydrovars(MultiFab& mf, MultiFab& hydrovs) {
  auto const & f = mf.arrays();
  auto const & h = hydrovs.arrays();
  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    hydrovars(x, y, z, f[nbx], h[nbx]);
  });
}

// LB timestep for whole lattice
inline void LBM_timestep(const Geometry& geom,
			 MultiFab& fold,
			 MultiFab& fnew,
			 MultiFab& hydrovs, MultiFab& noise) {
  
  if (temperature > 0.0) {generate_fluctuations(geom, hydrovs, noise);}
  else {noise.setVal(0.);}
  // Print() << "Noise generated for moments\n";
  collide_stream(geom, fold, fnew, hydrovs, noise);
  // Print() << "Collide stream completed\n";
  hydrovars(fold, hydrovs);
}

// homogeneous mixture of C1 and C2 with random concentration fluctuations
inline void LBM_init_liquid(MultiFab& mf,
			     MultiFab& hydrovs) {
  // const Real C1 = 0.5*rhol + 0.5*rhov;
  const Real C1 = rhol;
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Box& valid_box = mfi.validbox();
    const Array4<Real>& f = mf.array(mfi);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real rho = C1;
      for (int i=0; i<nvel; i++) {
        f(x,y,z,i) = w[i]*rho;
      }
    });
  }
  hydrovars(mf, hydrovs);
}

// homogeneous mixture of C1 and C2 with random concentration fluctuations
inline void LBM_init_mixture(MultiFab& mf, MultiFab& hydrovs) {

  const Real perturbation = 0.00001;
  const Real C1 = 0.5*rhol + 0.5*rhov;

  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Box& valid_box = mfi.validbox();
    const Array4<Real>& f = mf.array(mfi);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      for (int i=0; i<nvel; i++) {
        const Real rho = C1 +  RandomNormal(0., perturbation, engine);
        f(x,y,z,i) = w[i]*rho;
      }
    });
  }
  hydrovars(mf, hydrovs);
}

// flat interface between C1 and C2
inline void LBM_init_flat_interface(const Geometry& geom,
				    MultiFab& mf,
				    MultiFab& hydrovs) {
  const IntVect box = geom.Domain().length();
  auto const & f = mf.arrays();
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

    for (int i=0; i<nvel; i++) {
      if (rx < 0.){f[nbx](x,y,z,i) = w[i]*rhol;}
      else{f[nbx](x,y,z,i) = w[i]*rhov;}
    }

  });
  hydrovars(mf, hydrovs);
}

// binary droplet of vapor in liquid
inline void LBM_init_droplet(const Real r,
			     const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& hydrovs) {

  const IntVect box = geom.Domain().length();
  const Real R = r*box[0];

  auto const & f = mf.arrays();
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
    const Real profile = tanh((R-r)/sqrt(kappa));

    for (int i=0; i<nvel; i++) {
      if (profile < 0.0){f[nbx](x,y,z,i) = w[i]*rhol;}
      else {f[nbx](x,y,z,i) = w[i]*rhov;}
    }

  });
  hydrovars(mf, hydrovs);
}

#endif
