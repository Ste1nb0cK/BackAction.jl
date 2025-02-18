# This test evaluates if the trace of the monitoring operator is working
# using BackAction
# Consider a radiative damping for the case of an N-level system.
# The logarithmic derivatives of the trajectory distribution
@testset "Trajectory Contributions" begin
    function analytical_contribution(t, traj::Trajectory, gamma::Float64, n)
        # Edge case: no click trajectory
        if isempty(traj)
            return -n*t
        end
        jump_times = cumsum([click.time for click in traj])
        m = 0 # number of jumps
        # Find which was the last jump that occured
        for t_jump in jump_times
            if t_jump > t
                break
            end
            m = m +1
        end
        if m == 0
            return -n*t
        end
        return m/gamma - sum(jump_times[1:m]) -(n-m)*t

    end

    ############# System Definition
    NLEVELS = 5
    # Construct the destruction operator
    a = zeros(ComplexF64, NLEVELS, NLEVELS)
    for k in 1:NLEVELS
        for m in 1:NLEVELS
            if k == m + 1
                a[m, k] = sqrt(k-1)
            end
        end
    end

    EPS = 1e-5 # Tolerance
    GAMMA = BackAction.rd_gamma

    H = 0.5*BackAction.rd_deltaomega*adjoint(a)*a
    L = sqrt(GAMMA)*a
    sys = System(H, [L]);

    PSI0 = zeros(ComplexF64, NLEVELS)
    PSI0[end] = 1 # Initial condition

    params = SimulParameters(PSI0,
                             3.0, # Final time. Set very long so that all trajectories jump
                             2, # seed
                             100, # Number of trajectories
                             50_000, # Number of samples in the finegrid
                             10.5, # Multiplier to use in the fine grid
                             1e-3 # Tolerance for passing Dark state test
                             );

    # Parametrization
    H_parametrized = (delta::Float64, gamma::Float64) -> (0.5*delta*adjoint(a)*a)::Matrix{ComplexF64}
    L_parametrized = (delta::Float64, gamma::Float64) -> (sqrt(gamma)*a)::Matrix{ComplexF64}
    He_parametrized = BackAction.getheff_parametrized(H_parametrized, [L_parametrized])
    trajectories = run_trajectories_gillipsie(sys, params)

    ntimes = 1000
    t_given = collect(LinRange(0, params.tf, ntimes));

    EPS = 0.1 # Difference tolerance for the test

    xi_sample = Array{ComplexF64}(undef, sys.NLEVELS, sys.NLEVELS, ntimes, params.ntraj)
    theta = [BackAction.rd_deltaomega, BackAction.rd_gamma]
    dtheta = [0.0, BackAction.rd_gamma/100]
    for n in 1:params.ntraj
        xi_sample[:, :, :, n] = monitoringoperator(t_given, sys, He_parametrized, [L_parametrized], trajectories[n], params.psi0,
                                                   theta, dtheta)
    end
    contribution_sample = Array{Float64}(undef, ntimes, params.ntraj)
    for k in 1:params.ntraj
        for tn in 1:ntimes
            contribution_sample[tn, k] = real(tr(xi_sample[:, :, tn, k]))
        end
    end
    # Check that the contribution from the monitoring operator coincides with the analytical one
    for k in 1:params.ntraj
        for t in 1:ntimes
            flag = abs(
                contribution_sample[t, k] - analytical_contribution(t_given[t],
                                                                    trajectories[k],
                                                                    BackAction.rd_gamma,
                                                                    sys.NLEVELS-1)) < EPS
            flag || @warn "Failure at trajectory=$k, t=$t"
            @test flag
        end
    end
end


function GetFI(delta::Float64, ntraj::Int64)
    # System and parameters
    H = (delta::Float64, omega::Float64, k::Float64) ->  delta*[[0, 0] [0, 1.0+0im]]  + 0.5*omega*BackAction.sigma_x
    L = (delta::Float64, omega::Float64, k::Float64) -> sqrt(k)*BackAction.sigma_m
    He = BackAction.getheff_parametrized(H, [L])
    omega = 1.0
    k = 0.5
    sys = System(H(delta, omega, k), # Hamiltonian
                 [L(delta, omega, k)]) #Jump Operators
    tf = 300.0
    params = SimulParameters(BackAction.rf_psi0,
                             tf, # Final time. Set very long so that all trajectories jump
                             1, # seed
                             ntraj, # Number of trajectories
                             30_000, # Number of samples in the finegrid
                             1.0, # Multiplier to use in the fine grid
                             1e-3 # Tolerance for passing Dark state test
                             )

    trajectories = run_trajectories_gillipsie(sys, params; progbar=false, psireset = params.psi0 );
    t_given = [tf];
    ntimes = size(t_given)[1]
    xi_sample = Array{ComplexF64}(undef, sys.NLEVELS, sys.NLEVELS, ntimes, params.ntraj)
    for n in 1:params.ntraj
        xi_sample[:, :, :, n] = monitoringoperator(t_given, sys, He, [L], trajectories[n],
                                                   params.psi0,[delta, omega, k], [delta/100, 0.0, 0.0])
    end
    fi_sample = Array{Float64}(undef, ntimes, params.ntraj)
    for n in 1:params.ntraj
        for k in 1:ntimes
            fi_sample[k, n] = real(tr(xi_sample[:, :, k, n]))^2
        end
    end
    return fi_sample./tf
end
# Data extracted with Trakcer from Gammelmark-Moler(PRL 2014). The factor is important to match the scale
delta_gammelmark = 0.2/0.478 * [0.146,0.262,0.383,0.511,0.648,0.806,0.965,1.118,1.287,1.438,1.596,
                                1.755,1.888, 2.057, 2.206, 2.381, 2.539, 2.682, 2.865, 3.013, 3.174, 3.334, 3.452]
fi_t_gammelmark = [0.115, 0.221, 0.338, 0.425, 0.507, 0.557, 0.573, 0.566, 0.532, 0.494,
                   0.448, 0.403, 0.363, 0.318, 0.281, 0.241, 0.210, 0.186, 0.156, 0.137, 0.115, 0.101, 0.09348]


@testset "Gammelmark-Molmer (2014): Fisher Information of Delta" begin
    ntraj = 1000
    fi_samples = Array{Float64}(undef, ntraj, size(delta_gammelmark)[1])
    @showprogress for delta_index in 1:size(delta_gammelmark)[1]
        fi_samples[:, delta_index] = GetFI(delta_gammelmark[delta_index], ntraj)
    end
    my_fi_t = dropdims(mean(fi_samples, dims=1), dims=1)
    # Check that all the differences to their data are wihtin tolerance
    for k in 1:size(delta_gammelmark)[1]
        @test abs(my_fi_t[k] - fi_t_gammelmark[k]) < 0.05
    end
end

