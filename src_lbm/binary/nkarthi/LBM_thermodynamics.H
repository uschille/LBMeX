#ifndef LBM_THERMODYNAMICS_H_NK
#define LBM_THERMODYNAMICS_H_NK

#include <AMReX_GpuComplex.H>
#include <nkarthi/LBM_IO.H>
#include <nkarthi/LBM_d3q19.H>

namespace nkarthi {

const int ndof = 2*nvel;

struct bulk_free_energy {
    Real rho;  // Density
    Real phi;  // order_parameter

    // Overloaded constructor without k
    bulk_free_energy(Real density, Real order_parameter)
        : rho(density), phi(order_parameter) {}

    // Default constructor
    bulk_free_energy() : rho(1.), phi(0.) {}

    Real interface_width(){
      // const Real out = chi/4.*rho*(1. - pow(phi/rho, 2)) - rho*T + T*C1*log(C1) + T*C2*log(C2);
      // const Real out = sqrt((0.142*kappa)/((chi/2 - T) - 0.31*T*(phi*phi)));
      const Real out = sqrt(kappa/(chi/2 - T));
      return out;
    }

    Real calculate_f(){
      const Real C1 = (rho + phi)/2;
      const Real C2 = (rho - phi)/2;
      const Real out = chi/4.*rho*(1. - pow(phi/rho, 2)) - rho*T + T*C1*log(C1) + T*C2*log(C2);
      return out;
    }

    Real calculate_df_drho(){
      // const Real out = chi/4.*(1.+phi2/rho2) + T/2*log((rho2-phi2)/4);
      const Real out = 0.;
      return out;
    }

    Real calculate_df_dphi(){
      const Real out = -chi/2.*(phi/rho) + T/2.*log((1. + phi/rho)/(1. - phi/rho));
      return out;
    }
    
    Real calculate_dmur_drho(){
      const Real out = (-2.*T*pow(rho, 4) - chi*pow(phi, 4) + chi*pow(phi*rho, 2))/(2.*pow(rho, 3)*(pow(phi, 2) - pow(rho, 2)));
      return out;
    }

    Real calculate_dmup_drho(){
      const Real out = -T*phi/(rho*rho-phi*phi) + chi/2.*phi/(rho*rho);
      return out;
    }

    Real calculate_dmup_dphi(){
      // const Real out = (-2.*T*pow(rho, 2) - chi*pow(phi, 2) + chi*pow(rho, 2))/(2.*rho*(pow(phi, 2) - pow(rho, 2)));
    //   Real dfx = (-2.*T*pow(rho, 2) - chi*pow(x, 2) + chi*pow(rho, 2))/(2.*rho*(pow(x, 2) - pow(rho, 2)));
      const Real out = T*rho/(rho*rho-phi*phi) - chi/(2.*rho);
      return out;
    }

    Real bulk_pressure(){
      const Real out = rho*T;
      return out;
    }

    Real sound_speed_square(){
      const Real out = T;
      return T;
    }

    Real component_pressure(){
      const Real out = 0.;
      return out;
    }

};

// noise covariance matrix of the LB modes
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real,ndof*ndof> noise_covariance(const Real rho0, const Real phi0, const Real k2) {
    BL_PROFILE_VAR("noise_covariance()",noise_covariance);
    const uint Q = nvel;
    // Print() << "Discrete covariance matrix\n";
    const Real kT = temperature;

    bulk_free_energy curr_state(rho0, phi0);
    const Real cs2 = curr_state.sound_speed_square() + kappa*k2*rho0;
    const Real mu_rho = curr_state.calculate_dmup_drho();
    // const Real mu_rho = 0.;
    const Real mu_phi = curr_state.calculate_dmup_dphi() + k2*kappa;
    const Real p_phi = k2*kappa*phi0;
    
    // Print() << "rho:" << rho0 << " phi:" << phi0 << " k2:" << k2 << "\n";
    // Print() << "cs2: " << cs2 << "\n";
    // Print() << "mu_rho: " << mu_rho << "\n";
    // Print() << "mu_phi: " << mu_phi << "\n";
    // Print() << "p_phi: " << p_phi << "\n";

    // TODO: generalize to MRT
    const Real lambdaLB_r = -1./tau_r;
    const Real lambdaLB_p = -1./tau_p;

    // TODO: discretized needs check!
    Real lambda_r = -lambdaLB_r*(2+lambdaLB_r)/2;
    Real lambda_p = -lambdaLB_p*(2+lambdaLB_p)/2;
    Real lambda_rp = -lambdaLB_r*(2+lambdaLB_p)/2;
    Real lambda_pr = -lambdaLB_p*(2+lambdaLB_r)/2;
    // Real lambda_rp = -(lambdaLB_r + lambdaLB_p + lambdaLB_r*lambdaLB_p);
    // Real lambda_pr = lambda_rp;

    // Noise covariance of the modes in k-space
    GpuArray<Real,ndof*ndof> Xi = {};
    Xi.fill(0);

    // diagonal part
    Xi[(   5)*ndof+(   5)] = 2.*Gamma*kT/rho0*lambda_p;
    Xi[(   6)*ndof+(   6)] = 2.*Gamma*kT/rho0*lambda_p;
    Xi[(   7)*ndof+(   7)] = 2.*Gamma*kT/rho0*lambda_p;
    Xi[(   8)*ndof+(   8)] = 2.*kT*rho0*(5 - 9*cs2)*lambda_r;
    Xi[(   9)*ndof+(   9)] = 8.*kT*rho0*lambda_r;
    Xi[(  10)*ndof+(  10)] = (8.0/3.0)*kT*rho0*lambda_r;
    Xi[(  11)*ndof+(  11)] = (2.0/3.0)*kT*rho0*lambda_r;
    Xi[(  12)*ndof+(  12)] = (2.0/3.0)*kT*rho0*lambda_r;
    Xi[(  13)*ndof+(  13)] = (2.0/3.0)*kT*rho0*lambda_r;
    Xi[(  14)*ndof+(  14)] = 4.*kT*rho0*lambda_r;
    Xi[(  15)*ndof+(  15)] = 4.*kT*rho0*lambda_r;
    Xi[(  16)*ndof+(  16)] = 4.*kT*rho0*lambda_r;
    Xi[(  17)*ndof+(  17)] = (4.0/3.0)*kT*rho0*lambda_r;
    Xi[(  18)*ndof+(  18)] = (4.0/3.0)*kT*rho0*lambda_r;
    Xi[(Q+ 0)*ndof+(Q+ 0)] = (4.0/3.0)*kT*rho0*lambda_r;
    Xi[(Q+ 1)*ndof+(Q+ 1)] = 18.*kT*rho0*(1 - cs2)*lambda_r;
    Xi[(Q+ 2)*ndof+(Q+ 2)] = 8.*kT*rho0*lambda_r;
    Xi[(Q+ 3)*ndof+(Q+ 3)] = (8.0/3.0)*kT*rho0*lambda_r;
    Xi[(Q+ 4)*ndof+(Q+ 4)] = 2.*Gamma*kT/rho0*(-9*Gamma*mu_phi + 5)*lambda_p;
    Xi[(Q+ 5)*ndof+(Q+ 5)] = 8.*Gamma*kT/rho0*lambda_p;
    Xi[(Q+ 6)*ndof+(Q+ 6)] = (8.0/3.0)*Gamma*kT/rho0*lambda_p;
    Xi[(Q+ 7)*ndof+(Q+ 7)] = (2.0/3.0)*Gamma*kT/rho0*lambda_p;
    Xi[(Q+ 8)*ndof+(Q+ 8)] = (2.0/3.0)*Gamma*kT/rho0*lambda_p;
    Xi[(Q+ 9)*ndof+(Q+ 9)] = (2.0/3.0)*Gamma*kT/rho0*lambda_p;
    Xi[(Q+10)*ndof+(Q+10)] = 4.*Gamma*kT/rho0*lambda_p;
    Xi[(Q+11)*ndof+(Q+11)] = 4.*Gamma*kT/rho0*lambda_p;
    Xi[(Q+12)*ndof+(Q+12)] = 4.*Gamma*kT/rho0*lambda_p;
    Xi[(Q+13)*ndof+(Q+13)] = (4.0/3.0)*Gamma*kT/rho0*lambda_p;
    Xi[(Q+14)*ndof+(Q+14)] = (4.0/3.0)*Gamma*kT/rho0*lambda_p;
    Xi[(Q+15)*ndof+(Q+15)] = (4.0/3.0)*Gamma*kT/rho0*lambda_p;
    Xi[(Q+16)*ndof+(Q+16)] = 18.*Gamma*kT/rho0*(-Gamma*mu_phi + 1)*lambda_p;
    Xi[(Q+17)*ndof+(Q+17)] = 8.*Gamma*kT/rho0*lambda_p;
    Xi[(Q+18)*ndof+(Q+18)] = (8.0/3.0)*Gamma*kT/rho0*lambda_p;

    // rho-rho sector [0, 2..4, 8..(Q+3)]
    Xi[(   8)*ndof+(Q+ 1)] = 6.*kT*rho0*(3*cs2 - 1)*lambda_r;
    Xi[(Q+ 1)*ndof+(   8)] = 6.*kT*rho0*(3*cs2 - 1)*lambda_r;

    // phi-phi sector [1, 5..7, (Q+4)..(Q-1)]
    Xi[(Q+ 4)*ndof+(Q+16)] = 6.*Gamma*kT/rho0*(3*Gamma*mu_phi - 1)*lambda_p;
    Xi[(Q+16)*ndof+(Q+ 4)] = 6.*Gamma*kT/rho0*(3*Gamma*mu_phi - 1)*lambda_p;

    // rho-phi sector
    Xi[(   8)*ndof+(Q+ 4)] = -3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
    Xi[(Q+ 4)*ndof+(   8)] = -3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
    Xi[(   8)*ndof+(Q+16)] = 3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
    Xi[(Q+16)*ndof+(   8)] = 3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);

    Xi[(Q+ 1)*ndof+(Q+ 4)] = 3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
    Xi[(Q+ 4)*ndof+(Q+ 1)] = 3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
    Xi[(Q+ 1)*ndof+(Q+16)] = -3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);
    Xi[(Q+16)*ndof+(Q+ 1)] = -3.*kT*(Gamma*mu_phi*pow(rho0, 2)*(3*cs2 - 1)*mu_rho*lambda_pr + cs2*(3*Gamma*mu_phi - 1)*p_phi*lambda_rp)/(cs2*mu_phi*rho0);\

    Xi[(   0)*ndof+(Q+ 4)] = -3.*Gamma*kT*rho0*mu_rho/cs2*lambda_pr;
    Xi[(Q+ 4)*ndof+(   0)] = -3.*Gamma*kT*rho0*mu_rho/cs2*lambda_pr;
    Xi[(   0)*ndof+(Q+16)] = 3.*Gamma*kT*rho0*mu_rho/cs2*lambda_pr;
    Xi[(Q+16)*ndof+(   0)] = 3.*Gamma*kT*rho0*mu_rho/cs2*lambda_pr;

    Xi[(   1)*ndof+(Q+ 1)] = 3.*kT*p_phi/(mu_phi*rho0)*lambda_rp;
    Xi[(Q+ 1)*ndof+(   1)] = 3.*kT*p_phi/(mu_phi*rho0)*lambda_rp;
    Xi[(   1)*ndof+(   8)] = -3.*kT*p_phi/(mu_phi*rho0)*lambda_rp;
    Xi[(   8)*ndof+(   1)] = -3.*kT*p_phi/(mu_phi*rho0)*lambda_rp;
    Xi[(   2)*ndof+(   5)] = -phi0*kT*lambda_pr;
    Xi[(   3)*ndof+(   6)] = -phi0*kT*lambda_pr;
    Xi[(   4)*ndof+(   7)] = -phi0*kT*lambda_pr;
    Xi[(   5)*ndof+(   2)] = -phi0*kT*lambda_pr;
    Xi[(   6)*ndof+(   3)] = -phi0*kT*lambda_pr;
    Xi[(   7)*ndof+(   4)] = -phi0*kT*lambda_pr;

    // Print() << "Xi[23,8]:" << Xi[(Q+ 4)*ndof+(   8)] << " Xi[23,23]:" << Xi[(Q+ 4)*ndof+(Q+ 4)] << " rho:" << rho0 << " phi:" << phi0 << "\n";
    // Print() << "Xi[23,20]:" << Xi[(Q+ 4)*ndof+(Q+ 1)] << " Xi[23,23]:" << Xi[(Q+ 4)*ndof+(Q+ 4)] << " rho:" << rho0 << " phi:" << phi0 << "\n";
    // Print() << "Xi[23,0]:" << Xi[(Q+ 4)*ndof+(   0)] << " Xi[23,23]:" << Xi[(Q+ 4)*ndof+(Q+ 4)] << " rho:" << rho0 << " phi:" << phi0 << "\n";

  return Xi;
}
}
#endif