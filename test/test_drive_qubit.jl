########################## INITIALIZATION
# Define the hamiltonian
delta = 1.43
gamma = 1.0
omega = 1.3
nbar = 0.2
H = 0.5*delta * BackAction.sigma_z + 0.5*omega*BackAction.sigma_x
# Define the jump operators
L1 = sqrt(gamma*(nbar+1))*BackAction.sigma_m
L2 = sqrt(gamma*(nbar))*BackAction.sigma_p
# Define the system
# As initial condition use a mixture of |+> and |-> states
plus = 0.5* [[1.0+0im, 1] [1,  1]]
minus = 0.5* [[1.0+0im, -1] [-1,  1]]
psi0 = 0.3*plus + 0.7*minus

sys = System(H, [L1, L2])
params = SimulParameters(psi0,
    25.0,
    1, # seed
    2500, # Number of trajectories
    25_000, # Number of samples in the finegrid
    4.0, # Multiplier to use in the fine grid
    1e-3 # Tolerance for passing Dark state test
)
################## Differential Equation of the Observables

function f_drivenqubit(t, r)
     [-0.5*gamma*(2*nbar+1)*r[1] - delta*r[2];
     delta*r[1] - 0.5*gamma*(2*nbar+1)*r[2] - omega*r[3];
     omega*r[2] - gamma*(2*nbar+1)*r[3] - gamma]
end

# Trajectory Sampling
sampled_trajectories = run_trajectories_gillipsie(sys, params);

@testset "Driven Qubit: Expectation Value Convergence" begin
    # Lindblad Evolution of Observables
    sigma = [BackAction.sigma_x, BackAction.sigma_y, BackAction.sigma_z]
    r0 = zeros(Float64, 3)
    for k in 1:3
        r0[k] = real(tr(sigma[k]*psi0))
    end
    tspan = (0.0, params.tf)
    ntimes = 1000
    t_given = collect(LinRange(0, params.tf, ntimes));

    # Analytical Solution

    # Obtain the states between jumps
    sample = zeros(ComplexF64, sys.NLEVELS, sys.NLEVELS, ntimes, params.ntraj)
    for n in 1:params.ntraj
        sample[:, :, :, n]  = BackAction.states_att(t_given, sampled_trajectories[n], sys,  params.psi0)
    end
    # Evaluate the observables
    r_sample = zeros(Float64, ntimes, 3, params.ntraj)

    for j in 1:params.ntraj
        for k in 1:3
            for tn in 1:ntimes
                r_sample[tn, k, j] = real( tr(sigma[k] * sample[:, :, tn, j]) )
            end
        end
    end
    # Average
    r_avg = dropdims(mean(r_sample, dims=3), dims=3);

    r_analytical = BackAction.rk4(f_drivenqubit, r0, tspan, ntimes)
    for k in 1:ntimes
        @test abs( r_analytical[1, k]- r_avg[k, 1]) < 0.05
        @test abs( r_analytical[2, k]- r_avg[k, 2]) < 0.05
        @test abs( r_analytical[3, k]- r_avg[k, 3]) < 0.05
    end
end
