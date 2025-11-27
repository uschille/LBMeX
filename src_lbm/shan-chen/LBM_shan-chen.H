#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"
#include "LBM_Debug.H"

const IntVect print_grid(0, 0, 0); // grid to print out

// default initial condition
std::string init_cond = "mixture";
const bool use_SC_pseudo = true; // false: use density field directly; true: use SC pseudopotentials;

AMREX_GPU_MANAGED Real G = 0.24; //1.5;
AMREX_GPU_MANAGED Real rho_hi = 0.7; //3.0; // values reported by Belardinelli et al. [Phys. Rev. E 91, 023313 (2015)]
AMREX_GPU_MANAGED Real rho_lo = 0.0;
AMREX_GPU_MANAGED Real sigma = 0.174; // interfacial tension
AMREX_GPU_MANAGED Real alpha = 1.; // interface width

AMREX_GPU_MANAGED Real temperature = 0.; //1e-5;

AMREX_GPU_MANAGED Real init_frac = 0.4; // fractional size of droplet or stripe

AMREX_GPU_CONSTANT Real const lambda_d = 1.0;
AMREX_GPU_CONSTANT Real const lambda_b = 1.0;
AMREX_GPU_CONSTANT Real const lambda_s = 1.0;
AMREX_GPU_CONSTANT Real const lambda_o = 1.0;
AMREX_GPU_CONSTANT Real const lambda_e = 1.0;

AMREX_GPU_CONSTANT Array<Real,nvel> const
lambda_f = { 0.,
  lambda_d, lambda_d, lambda_d,
  lambda_b, lambda_s, lambda_s, lambda_s, lambda_s, lambda_s,
  lambda_o, lambda_o, lambda_o, lambda_o, lambda_o, lambda_o,
  lambda_e, lambda_e, lambda_e
};

AMREX_GPU_CONSTANT Array<Real,nvel> const
lambda_g = { 0.,
  lambda_d, lambda_d, lambda_d,
  lambda_b, lambda_s, lambda_s, lambda_s, lambda_s, lambda_s,
  lambda_o, lambda_o, lambda_o, lambda_o, lambda_o, lambda_o,
  lambda_e, lambda_e, lambda_e
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
RealVect gradient(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    Real SC_pseudo_neighbor = use_SC_pseudo ? ::rho_hi*(1.-exp(-field(xp,yp,zp,icomp)/::rho_hi)) : field(xp,yp,zp,icomp);
    for (int dir=0; dir<3; dir++) {
      gradient[dir] += w[i]/cs2*SC_pseudo_neighbor*c[i][dir];
    }
  }
  return gradient;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real laplacian(int x, int y, int z, const Array4<Real>& field, int icomp) {
  Real laplacian = 0.0;
  Real SC_pseudo_center = use_SC_pseudo ? ::rho_hi*(1.-exp(-field(x,y,z,icomp)/::rho_hi)) : field(x,y,z,icomp);
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    Real SC_pseudo_neighbor = use_SC_pseudo ? ::rho_hi*(1.-exp(-field(xp,yp,zp,icomp)/::rho_hi)) : field(xp,yp,zp,icomp);
    laplacian += 2.*w[i]/cs2*(SC_pseudo_neighbor-SC_pseudo_center);
  }
  return laplacian;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Array<Array1D<Real,0,nvel>,2> thermal_noise(const Array<Real,2>& rho, const RandomEngine& engine, bool if_print_detail=false) {
  Array<Array1D<Real,0,nvel>,2> xi;
  Array1D<Real,0,nvel> & xif = xi[0];
  Array1D<Real,0,nvel> & xig = xi[1];

  // densities are conserved
  xif(0) = xig(0) = 0.0;

  // fluxes are anticorrelated (total momentum is conserved)
  for (int i=1; i<=AMREX_SPACEDIM; ++i ) {
    const Real pref = (2.-lambda_d)*lambda_d*temperature*rho[0]*rho[1]/(rho[0]+rho[1]);
    /*if(if_print_detail) {
      printf("pref for flux mode %d = %f\n and sqrt(pref) = %f\n", i, pref, std::sqrt(pref));
    }*/
    xif(i) = RandomNormal(0., std::sqrt(std::abs(pref)), engine);
    xig(i) = - xif(i);
  }

  // remaining modes
  for (int i=AMREX_SPACEDIM+1; i<nvel; ++i) {
    const Real pref = (2.-lambda_f[i])*lambda_f[i]*b[i]*temperature/cs2*rho[0];
    const Real preg = (2.-lambda_g[i])*lambda_g[i]*b[i]*temperature/cs2*rho[1];
    xif(i) = RandomNormal(0., std::sqrt(std::abs(pref)), engine);
    xig(i) = RandomNormal(0., std::sqrt(std::abs(preg)), engine);
  }

  if(if_print_detail) {
    printf("xi_f in thermal_noise() = ");
    for (int i=0; i<nvel; ++i) {
      printf("%f\t", xif(i));
    }
    printf("\n");
    printf("xi_g in thermal_noise() = ");
    for (int i=0; i<nvel; ++i) {
      printf("%f\t", xig(i));
    }
    printf("\n");
  }

  return xi;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Array<RealVect,2> transform_velocity(Array<Real,2> const& rho, Array<RealVect,2> const& v, Array<RealVect,2> const& a, Array<Array1D<Real,0,nvel>,2> const& xia,
  bool if_print_detail=false) {
  Array<RealVect, 2> vr;
  RealVect xi;

  // express v, v' in terms of v̄, v̄' [Belardinelli et al. Eq. (71)]
  // [uschill 2025/05/04: xi is the lattice noise, hence the prefactor (1-0.5*lambda_d) is absorbed]
  for (int i=0; i<2; ++i) {
    int j = (i+1)%2;
    for (int k=0; k<AMREX_SPACEDIM; ++k) xi[k] = xia[i](1+k);
    vr[i] = v[i] + 0.5*(a[i] - lambda_d*rho[j]/(rho[i]+rho[j])*(v[i]-v[j]+0.5*(a[i]-a[j])) + ((std::abs(rho[i]) > FLT_EPSILON) ? xi/rho[i] : RealVect(0.)));
  }
  if(if_print_detail) {
    printf("xi in transform_velocity() = %f\t%f\n", xi[0], xi[1]);
    printf("density in transform_velocity() = %f\t%f\n", rho[0], rho[1]);
    printf("velocity in transform_velocity() = vf=(%f,%f,%f),\tvg=(%f,%f,%f)\n", vr[0][0], vr[0][1], vr[0][2], vr[1][0], vr[1][1], vr[1][2]);
    printf("acceleration in transform_velocity() = af=(%f,%f,%f),\tag=(%f,%f,%f)\n", a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2]);
  }

  return vr;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Array<Array1D<Real,0,nvel>,2> force_moments(const Array<Real,2>& rhoa, Array<RealVect,2>& va, const Array<RealVect,2>& aa) {
  Array<Array1D<Real,0,nvel>,2> phia;

  for (int i=0; i<2; ++i) {
    Array1D<Real,0,nvel>& phi = phia[i];

    Real const& rho = rhoa[i];
    RealVect const& v = va[i];
    RealVect const& a = aa[i];

    /* density mode */
    phi(0) = 0.0;

    /* momentum modes */
    phi(1) = rho*a[0];
    phi(2) = rho*a[1];
    phi(3) = rho*a[2];

    /* stress modes [Belardinelli et al. Eq. (31)] */
    phi(4) = 2.*rho*(a[0]*v[0] + a[1]*v[1] + a[2]*v[2]);
    phi(5) = 2.*rho*(2.*a[0]*v[0] - a[1]*v[1] - a[2]*v[2]);
    phi(6) = 2.*rho*(a[1]*v[1] - a[2]*v[2]);
    phi(7) = rho*(a[0]*v[1]+a[1]*v[0]);
    phi(8) = rho*(a[1]*v[2]+a[2]*v[1]);
    phi(9) = rho*(a[2]*v[0]+a[0]*v[2]);

    /* kinetic modes */
    for (int i=10; i<nvel; ++i) phi(i) = 0.0;

  }

  return phia;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z,
	     const Array4<Real>& f,
	     const Array4<Real>& g,
       const Array4<Real>& h,
       const Array4<Real>& ref, // reference densities
       const RandomEngine& engine) {
  Array1D<Real,0,nvel> mf, mg, mfEq, mgEq;
  Array<Array1D<Real,0,nvel>,2> phi;
  Array<Array1D<Real,0,nvel>,2> xi;
  Array<Real,2> rho, rho_ref;
  Array<RealVect,2> v, a, vr;
  RealVect v_b, a_b;

  // reference densities
  rho_ref[0] = ref(x,y,z,0);
  rho_ref[1] = ref(x,y,z,1);

  // xi = thermal_noise(rho_ref, engine);
  Array1D<Real,0,nvel> const& xif = xi[0];
  Array1D<Real,0,nvel> const& xig = xi[1];

  mf = moments(x,y,z,f);
  mg = moments(x,y,z,g);

  rho[0] = mf(0);
  rho[1] = mg(0);

  if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]){
    xi = thermal_noise(rho, engine /*, true*/);  // use current densities for noise calculation
  }else{
    xi = thermal_noise(rho, engine);
  }

  v[0] = (mf(0) > FLT_EPSILON) ? RealVect(mf(1), mf(2), mf(3))/mf(0) : RealVect(0.);
  v[1] = (mg(0) > FLT_EPSILON) ? RealVect(mg(1), mg(2), mg(3))/mg(0) : RealVect(0.);

  Real SC_pseudo_rho = use_SC_pseudo ? ::rho_hi*(1.-exp(-rho[0]/::rho_hi)) : rho[0];
  Real SC_pseudo_phi = use_SC_pseudo ? ::rho_hi*(1.-exp(-rho[1]/::rho_hi)) : rho[1];
  a[0] = (std::abs(rho[0]) > FLT_EPSILON)? -G*cs2*SC_pseudo_rho*gradient(x,y,z,h,1)/rho[0] : RealVect(0.);
  a[1] = (std::abs(rho[1]) > FLT_EPSILON)? -G*cs2*SC_pseudo_phi*gradient(x,y,z,h,0)/rho[1] : RealVect(0.);

  if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]){
    vr = transform_velocity(rho, v, a, xi /*, true*/);
  }else{
    vr = transform_velocity(rho, v, a, xi);
  }

  v_b = (rho[0]*v[0]+rho[1]*v[1])/(rho[0]+rho[1]);
  a_b = (rho[0]*a[0]+rho[1]*a[1])/(rho[0]+rho[1]);

  mfEq = equilibrium_moments(rho[0], v_b+0.5*a_b);
  mgEq = equilibrium_moments(rho[1], v_b+0.5*a_b);

  /*if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]){
    //printf("fluid f,g velocity in collide() = \n");
    printf("xi before force_moments = %f\t%f\t%f\n", xi[0](0), xi[0](1), xi[0](2));
    phi = force_moments(rho, vr, a);
    printf("\n");
  }else{*/
    phi = force_moments(rho, vr, a);
  //}
  
  Array1D<Real,0,nvel> const& phif = phi[0];
  Array1D<Real,0,nvel> const& phig = phi[1];

  for (int i=0; i<nvel; ++i) {
    mf(i) = mf(i) - lambda_f[i] * (mf(i) - mfEq(i)) + (1.-0.5*lambda_f[i])*phif(i) + xif(i);
    mg(i) = mg(i) - lambda_g[i] * (mg(i) - mgEq(i)) + (1.-0.5*lambda_g[i])*phig(i) + xig(i);
  }

  populations(x,y,z,f,mf);
  populations(x,y,z,g,mg);

}

// stream from \vec{x} to \vec{x}+\vec{c}_i (push scheme)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
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
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void collide_stream(int x, int y, int z,
		    const Array4<Real>& fOld,  const Array4<Real>& gOld,
		    const Array4<Real>& fNew,  const Array4<Real>& gNew,
		    const Array4<Real>& h,
        const Array4<Real>& ref,
        const RandomEngine& engine) {
  collide(x, y, z, fOld, gOld, h, ref, engine); // collide populations on current site
  stream_push(x, y, z, fOld, gOld, fNew, gNew); // push populations to neighbors
}

// apply fluctuating lattice Boltzmann equation for whole lattice
AMREX_FORCE_INLINE
void collide_stream(const Geometry& geom,
			   MultiFab& fold, MultiFab& gold,
			   MultiFab& fnew, MultiFab& gnew,
			   MultiFab& hydrovs, MultiFab& refstate) {
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
    const Array4<Real>& ref = refstate.array(mfi);
    const Box& valid_box = mfi.growntilebox(halo);
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, const RandomEngine& engine) {
      collide_stream(x, y, z, fOld, gOld, fNew, gNew, h, ref, engine);
    });
  }
  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);
}

Vector<std::string> hydrovars_names(const int numVars) {
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

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void densities(int x, int y, int z,
         const Array4<Real>& f,
         const Array4<Real>& g,
         const Array4<Real>& h) {
  Real rho=0.0;
  Real phi=0.0;
  for (int i=0; i<nvel; ++i) {
    rho += f(x,y,z,i);
    phi += g(x,y,z,i);
  }
  h(x,y,z,0) = rho;
  h(x,y,z,1) = phi;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void hydrovars(int x, int y, int z,
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& h,
         const Array4<Real>& ref,
         const RandomEngine& engine) {
  const Array<Real,2> rho_ref = { ref(x,y,z,0), ref(x,y,z,1) };
  const Array1D<Real,0,nvel> mf = moments(x,y,z,f);
  const Array1D<Real,0,nvel> mg = moments(x,y,z,g);
  const Array<Array1D<Real,0,nvel>,2> xi = thermal_noise(rho_ref, engine);
  Array<Real,2> rho;
  Array<RealVect,2> v, a, vr;

  rho[0] = mf(0); 
  rho[1] = mg(0);

  v[0] = (mf(0) > FLT_EPSILON) ? RealVect(mf(1), mf(2), mf(3))/mf(0) : RealVect(0.);
  v[1] = (mg(0) > FLT_EPSILON) ? RealVect(mg(1), mg(2), mg(3))/mg(0) : RealVect(0.);

  a[0] = -G*cs2*gradient(x,y,z,h,1);
  a[1] = -G*cs2*gradient(x,y,z,h,0);

  vr = transform_velocity(rho, v, a, xi);

  h(x,y,z,0) = rho[0];
  h(x,y,z,1) = rho[1];
  for (int i=0; i<AMREX_SPACEDIM; ++i) {     // momentum/velocity i=1..3
    h(x,y,z,i+2) = vr[0][i];                  // u_f to be stored in 2..4
    h(x,y,z,i+2+AMREX_SPACEDIM) = vr[1][i];   // u_g to be stored in 5..7
  }
  // [uschill 2025/05/05: higher moments are the bare (untransformed) moments]
  for (int i=1+AMREX_SPACEDIM; i<nvel; ++i) { // remaining moments i=4..18
    h(x,y,z,i+1+AMREX_SPACEDIM) = mf(i);      // f to be stored in 8..22
    h(x,y,z,i+nvel) = mg(i);                  // g to be stored in 23..37
  }
}

// calculate hydrodynamic moments for whole lattice
AMREX_FORCE_INLINE
void hydrovars(const Geometry& geom, MultiFab& mf, MultiFab& mg, MultiFab& hydrovs, MultiFab& refstate) {
  const int halo=1;
  mf.FillBoundary(geom.periodicity());
  mg.FillBoundary(geom.periodicity());
  refstate.FillBoundary(geom.periodicity());
  for (MFIter mfi(hydrovs); mfi.isValid(); ++mfi) {
    auto const & f = mf.array(mfi);
    auto const & g = mg.array(mfi);
    auto const & h = hydrovs.array(mfi);
    auto const & ref = refstate.array(mfi);
    ParallelFor(mfi.growntilebox(halo), [=] AMREX_GPU_DEVICE(int x, int y, int z) {
      densities(x, y, z, f, g, h);
    });
    ParallelForRNG(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int x, int y, int z, const RandomEngine& engine) {
      hydrovars(x, y, z, f, g, h, ref, engine);
    });
  }
}

// LB timestep for whole lattice
AMREX_FORCE_INLINE
void LBM_timestep(const Geometry& geom,
			 MultiFab& fold, MultiFab& gold,
			 MultiFab& fnew, MultiFab& gnew,
			 MultiFab& hydrovs, MultiFab& refstate) {
  collide_stream(geom, fold, gold, fnew, gnew, hydrovs, refstate);
  hydrovars(geom, fold, gold, hydrovs, refstate);
}

// homogeneous mixture of C1 and C2 with random concentration fluctuations
void LBM_init_mixture(Geometry const& geom,
           MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs,
           MultiFab& refstate) {
  const Real rho = 0.5*(::rho_hi+::rho_lo);
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Array4<Real>& f = mf.array(mfi);
    const Array4<Real>& g = mg.array(mfi);
    const Array4<Real>& ref = refstate.array(mfi);
    ParallelForRNG(mfi.validbox(), [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real noise = 0.01*RandomNormal(0., 1., engine);
      for (int i=0; i<nvel; i++) {
        f(x,y,z,i) = w[i]*(rho + noise);
        g(x,y,z,i) = w[i]*(rho - noise);
      }
      ref(x,y,z,0) = rho;
      ref(x,y,z,1) = rho;
    });
  }
  hydrovars(geom, mf, mg, hydrovs, refstate);
}

// flat interface between C1 and C2
void LBM_init_stripe(const Real frac,
            const Geometry& geom,
				    MultiFab& mf,
				    MultiFab& mg,
				    MultiFab& hydrovs,
            MultiFab& refstate) {
  GpuArray<Real,3> dx = geom.CellSizeArray();
  const IntVect box = geom.Domain().length();
  const Real rho_t = ::rho_hi + ::rho_lo;
  const Real pos_lo = -0.5*frac;
  const Real pos_hi = 0.5*frac;
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & ref = refstate.arrays();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real pos = z-box[2]/2;
    const Real rho =  (::rho_hi-::rho_lo)*0.5*(std::tanh((pos-pos_lo)/std::sqrt(alpha)) + std::tanh((pos_hi-pos)/std::sqrt(alpha))) + ::rho_lo;
    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*(rho_t-rho);
    }
    ref[nbx](x,y,z,0) = rho;
    ref[nbx](x,y,z,1) = rho_t-rho;
  });
  hydrovars(geom, mf, mg, hydrovs, refstate);
}

// binary droplet of C1 and C2
void LBM_init_droplet(const Real dfrac,
			     const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs,
           MultiFab& refstate) {
  GpuArray<Real,3> dx = geom.CellSizeArray();
  const IntVect box = geom.Domain().length();
  const Real R = 0.5*dfrac*box[0];
  const Real rho_t = ::rho_hi + ::rho_lo;
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & ref = refstate.arrays();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real rx = x-box[0]/2;
    const Real ry = y-box[1]/2;
    const Real rz = z-box[2]/2;
    const Real r = std::sqrt(rx*rx+ry*ry+rz*rz);
    const Real rho = (::rho_hi-::rho_lo)*0.5*(1.+std::tanh((R-r)/std::sqrt(alpha))) + ::rho_lo;
    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*(rho_t-rho);
    }
    ref[nbx](x,y,z,0) = rho;
    ref[nbx](x,y,z,1) = rho_t-rho;
  });
  hydrovars(geom, mf, mg, hydrovs, refstate);
}

void LBM_init(const Geometry& geom, MultiFab& mf, MultiFab& mg, MultiFab& hydrovs, MultiFab& refstate) {
  if (init_cond == "mixture") LBM_init_mixture(geom, mf, mg, hydrovs, refstate);
  else if (init_cond == "stripe") LBM_init_stripe(init_frac, geom, mf, mg, hydrovs, refstate);
  else if (init_cond == "droplet") {
    LBM_init_droplet(init_frac, geom, mf, mg, hydrovs, refstate);
    Print() << "Initialized droplet with frac radius = " << init_frac/2.0 << "\n";
  } else {
    Print() << "ERROR: Unknown init_cond";
    ParallelDescriptor::Abort(-1, false);
  }
}

#endif
