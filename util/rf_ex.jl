################ SETUP FOR RESONANCE FLUORESENCE ##################
# Same thing as Radiative Damping, but now there Hamiltonian is different
rf_EPS = 1e-5 # Tolerance for the distance respect to the Frobenious norm
rf_delta = 0.0 # Detuning. For comparison with the Analytic WTD this must be zero
rf_omega = 0.5 # Rabi Frequency
rf_gamma = 0.5
rf_H = rf_delta * sigma_z + rf_omega * sigma_x

rf_NLEVELS = 2
rf_NCHANNELS = 1

rf_psi0 = zeros(ComplexF64, 2)
rf_psi0[1] = 1 # Initial condition



rf_sys = System(rf_H, # Hamiltonian
    reshape(sqrt(rf_gamma) * sigma_m, rf_NLEVELS, rf_NLEVELS, rf_NCHANNELS), rf_NLEVELS, rf_NCHANNELS) #Jump Operators
rf_params = SimulParameters(rf_psi0,
    25.0, # Final time. Set very long so that all trajectories jump
    1, # seed
    10_000, # Number of trajectories
    75_000, # Number of samples in the finegrid
    3.0, # Multiplier to use in the fine grid
    1e-3 # Tolerance for passing Dark state test
)

# Distribution class for
