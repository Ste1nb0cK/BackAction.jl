# using BackAction
@testset verbose = true "Radiative Damping: Basic Operators" begin
    @test norm(BackAction.rd_sys.H - BackAction.rd_H) < BackAction.rd_EPS
    @test norm(BackAction.rd_sys.Ls[1] - BackAction.rd_L) < BackAction.rd_EPS
    @test norm(BackAction.rd_sys.Heff - BackAction.rd_He) < BackAction.rd_EPS
    @test norm(BackAction.rd_sys.J - BackAction.rd_J) < BackAction.rd_EPS
end;

########### WTD ######################################
# Data generation
sys = BackAction.rd_sys
params = SimulParameters(BackAction.rd_psi0,
    3.0, # Final time. Set very long so that all trajectories jump
    1, # seed
    10_000, # Number of trajectories
    50_000, # Number of samples in the finegrid
    10.5, # Multiplier to use in the fine grid
    1e-3 # Tolerance for passing Dark state test
)

trajectories = run_trajectories_gillipsie(sys, params)

@testset verbose = true "Radiative Damping: WTD (KS Test and fit)" begin
    rd_times = [trajectories[k][1].time for k in 1:BackAction.rd_params.ntraj if !isempty(trajectories[k])]
    rd_d = Distributions.Exponential(1 / BackAction.rd_gamma)
    ## Use a two sample Kolmogorov-Smirnov test, pvalue above 0.2 is accepted
    rd_pvalue = HypothesisTests.pvalue(
        HypothesisTests.ApproximateTwoSampleKSTest(rd_times, rand(rd_d, BackAction.rd_params.ntraj)))
    rd_p0WTD = 0.2 # Minimal pvalue for accepting the null hypothesis
    fit_par = Distributions.fit(Distributions.Exponential, rd_times).θ
    @test rd_pvalue > rd_p0WTD
    @test abs(fit_par - 1 / BackAction.rd_gamma) < 0.01
end

# Now from each trajectory, generate the states at the given times
# trajectories = BackAction.run_trajectories(sys, params)
@testset "Radiative Damping: Normalization and Convergence of Expectation Values" begin
    ntimes = 1000
    t = collect(LinRange(0, params.tf, ntimes))
    sample = Array{ComplexF64}(undef, sys.NLEVELS, ntimes, params.ntraj)
    for n in 1:params.ntraj
        sample[:, :, n] = BackAction.states_att(t, trajectories[n], sys, params.psi0)
    end
    # Check that all states are normalized
    global local_flag = true
    for n in 1:params.ntraj
        for k in 1:ntimes
            if (norm(sample[:, k, n]) - 1) > 0.01
                global local_flag = false
                break
            end
        end
    end
    # Check
    accepted_error = 0.1 # Accept a relative accumulated error of up to 10%
    # Obtain the observable on the sample
    x_sample = zeros(ComplexF64, ntimes, params.ntraj)
    for k in 1:params.ntraj
        for tn in 1:ntimes
            x_sample[tn, k] = dot(sample[:, tn, k], BackAction.sigma_z, sample[:, tn, k])
        end
    end
    x = real(dropdims(mean(x_sample, dims=2), dims=2))
    x_theo = 2 * exp.(-BackAction.rd_gamma .* t) .- 1
    error = sum(abs.((x - x_theo) ./ x_theo)) / (sum(abs.(x_theo)))
    @test local_flag
    @test error < accepted_error
end

######## Test the Fisher Information
@testset "Radiative Damping: Fisher Information in Time" begin
    # Parametrization stuff
    H_parametrized = (delta::Float64, gamma::Float64) -> (0.5 * delta * BackAction.sigma_z)::Matrix{ComplexF64}
    L_parametrized = (delta::Float64, gamma::Float64) -> (sqrt(gamma) * BackAction.sigma_m)::Matrix{ComplexF64}
    Heff_parametrized = BackAction.get_Heff(H_parametrized, [L_parametrized], sys.NLEVELS)

    ntimes = 100
    t_given = collect(LinRange(0, params.tf, ntimes))

    # Obtain Monitoring Operator
    xi_sample = Array{ComplexF64}(undef, sys.NLEVELS, sys.NLEVELS, ntimes, params.ntraj)
    for n in 1:params.ntraj
        xi_sample[:, :, :, n] = monitoringoperator(t_given, sys, Heff_parametrized, [L_parametrized], trajectories[n], params.psi0,
            [BackAction.rd_deltaomega, BackAction.rd_gamma], [0.0, BackAction.rd_gamma / 100])
    end

    # Calculate the sample Fisher Information
    fi_sample = Array{Float64}(undef, ntimes, params.ntraj)
    for n in 1:params.ntraj
        for k in 1:ntimes
            fi_sample[k, n] = real(tr(xi_sample[:, :, k, n]))^2
        end
    end
    # Fisher Information Average
    fi = dropdims(mean(fi_sample, dims=2), dims=2)

    # Check global error against analytical result
    f_analytical(t) = (1 - exp(-BackAction.rd_gamma * t)) / (BackAction.rd_gamma^2)
    fi_theo = f_analytical.(t_given)

    fi_min, fi_max = extrema(fi_theo)
    # Use as error measure the normalized MSRE
    accepted_error = 0.1
    error_global = sqrt(mean((fi_theo - fi) .^ 2)) / (fi_max - fi_min)
    @test error_global < accepted_error
end
