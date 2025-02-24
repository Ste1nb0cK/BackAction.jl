################ SETUP FOR QUBIT RADIATIVE DAMPING ##################

rd_EPS = 1e-5 # Tolerance for the distance respect to the Frobenious norm
rd_deltaomega = 1.3
rd_gamma = 3.0
rd_H = 0.5*rd_deltaomega * sigma_z
rd_L = sqrt(rd_gamma) * sigma_m
rd_J = rd_gamma * [[0,0] [0,1.0+0im]]
rd_He = [[-rd_deltaomega/2, 0.0] [0.0, 0.5*(rd_deltaomega - 1im*rd_gamma) ]]

rd_psi0 = zeros(ComplexF64, 2)
rd_psi0[2] = 1 # Initial condition

rd_sys = System(rd_H, # Hamiltonian
[sqrt(rd_gamma)*sigma_m]) #Jump Operators
rd_params = SimulParameters(rd_psi0,
    3.0, # Final time. Set very long so that all trajectories jump
    1, # seed
    100, # Number of trajectories
    50_000, # Number of samples in the finegrid
    10.5, # Multiplier to use in the fine grid
    1e-3 # Tolerance for passing Dark state test
)
