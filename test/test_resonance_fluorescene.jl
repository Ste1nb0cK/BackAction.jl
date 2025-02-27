########## Setup
sys = BackAction.rf_sys
params = BackAction.rf_params
r0 = [0.0; 0.0; -1.0] # Initial Condition
tspan = (0.0, params.tf)
ntimes = 1000
t_given = collect(LinRange(0, params.tf, ntimes));

psireset = zeros(ComplexF64, 2)
psireset[1] = 1

################## Average Simulation ################3
# Generate a set of trajectories and states

trajectories = BackAction.run_trajectories_gillipsie(sys, params; psireset=psireset, progbar=true)

@testset "Resonance Fluorescene: Expectation Value Convergence" begin
    ntimes = size(t_given)[1]
    sample = zeros(ComplexF64,  sys.NLEVELS, ntimes, params.ntraj) # states

    for n in 1:params.ntraj
        states = BackAction.states_att(t_given, trajectories[n], sys,  params.psi0)
        sample[:, :, n] = states[:, :]
    end
    # Obtain the value of the observables
    r_sample = zeros(Float64, ntimes, 3, params.ntraj)
    sigma = [BackAction.sigma_x, BackAction.sigma_y, BackAction.sigma_z]

    for j in 1:params.ntraj
        for k in 1:3
            for tn in 1:ntimes
                r_sample[tn, k, j] = dot(sample[:, tn, j], sigma[k], sample[:, tn, j])   # Drop the extra dimension
            end
        end
    end
    r_avg = dropdims(mean(r_sample, dims=3), dims=3) # Ensemble average


    # System of equations for RF (Source: Wiseman section 3.3.1)
    function f_resonancefluorescene(t, r)
        gamma = BackAction.rf_gamma
        delta = BackAction.rf_delta
        omega = BackAction.rf_omega
        return [-0.5*gamma*r[1] - 2*delta*r[2]; 2*delta*r[1] - 0.5*gamma*r[2] - 2*omega*r[3];
                2*omega*r[2] - gamma*(r[3] + 1)]
    end

    # Solution to the unconditional equation
    r_analytical = BackAction.rk4(f_resonancefluorescene, r0, tspan, ntimes)

    for k in 1:ntimes
        @test abs(r_analytical[1, k] - r_avg[k, 1]) < 0.1
        @test abs(r_analytical[2, k] - r_avg[k, 2]) < 0.1
        @test abs(r_analytical[3, k] - r_avg[k, 3]) < 0.1
    end
end

@testset verbose=true "WTD" begin
    # Exploiting that this is a renewal process, obtain a sample of waiting times
    tau_sample = Vector{Float64}()
    for traj in trajectories
        if !isempty(traj)
            for click in traj
                push!(tau_sample, click.time)
            end
        else
            continue
        end
    end

    # Define the analytical result for the WTD
    struct WTD_rf <: Distributions.ContinuousUnivariateDistribution
        omega::Float64
        gamma::Float64

    end

    function Distributions.support(d::WTD_rf)
        return Distributions.Interval(0, Inf)
    end

    function Distributions.pdf(d::WTD_rf, tau::Real)
        gamma = d.gamma
        omega = d.omega
        return (16*gamma*omega^2)*exp(-0.5*gamma*tau) * sin(0.25*tau*sqrt(16*omega^2-gamma^2))^2/(-gamma^2+16*omega^2)
    end

    function Distributions.cdf(d::WTD_rf, t::Real)
        gamma = d.gamma
        omega = d.omega
        pdf(tau) = (16*gamma*omega^2)*exp(-0.5*gamma*tau) * sin(0.25*tau*sqrt(16*omega^2-gamma^2))^2/(-gamma^2+16*omega^2)
        return quadgk(pdf, 0, t, rtol=1e-8)[1]
    end

    function Base.rand(rng::AbstractRNG, d::WTD_rf)
        # Use inversion sampling
        alpha = rand()
        t = 0
        dt = params.dt # This is a magic number, the point is that this matches the dt in rf_params
        while Distributions.cdf(d, t) < alpha
            t = dt + t
        end
        return t  # Return a sample
    end

    f = WTD_rf(BackAction.rf_gamma, BackAction.rf_omega) # Instance of the WTD
    @time begin
        f_sample = rand(f, 500) # Sample from the WTD
    end
    # Run a KS test
    @time begin
        ksresults = HypothesisTests.ApproximateTwoSampleKSTest(tau_sample, f_sample)
        p = HypothesisTests.pvalue(ksresults) #pvalue
    end
    rf_p0WTD = 0.2 # Minimal pvalue for accepting the null hypothesis
    print(p)
    @test p > rf_p0WTD
end
