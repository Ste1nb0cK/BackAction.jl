############## SETUP FOR RESONANCE FLOURESCENE
rdt_gamma = 1.5
rdt_n = 1
const rdt_NLEVELS = 2
const rdt_NCHANNELS = 2
rdt_gamma1 = (rdt_n + 1) * rdt_gamma
rdt_gamma2 = (rdt_n) * rdt_gamma

Ls = Array{ComplexF64}(undef, rdt_NLEVELS, rdt_NLEVELS, rdt_NCHANNELS)
Ls[:, :, 1] = sqrt(rdt_gamma1) * BackAction.sigma_m
Ls[:, :, 2] = sqrt(rdt_gamma2) * BackAction.sigma_p
rdt_sys = System(zeros(ComplexF64, 2, 2), # Hamiltonian
    Ls, 2, 2) #Jump Operators
#### 2. Create the simulation parameters instance
rdt_psi0 = zeros(ComplexF64, 2)
rdt_psi0[2] = 1 # Initial condition
rdt_params = SimulParameters(rdt_psi0,
    5.0, # Final time. Set very long so that all trajectories jump
    10, # seed
    50, # Number of trajectories
    10_000, # Number of samples in the finegrid
    10.0, # Multiplier to use in the fine grid
    1e-3 # Tolerance for passing Dark state test
)
