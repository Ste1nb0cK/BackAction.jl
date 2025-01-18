import Distributions, HypothesisTests
using Test, QuTaM, Statistics, LinearAlgebra
@testset verbose=true "Basic Operators" begin
       @test norm(QuTaM.rd_sys.H - QuTaM.rd_H) < QuTaM.rd_EPS
       @test norm(QuTaM.rd_sys.Ls[1]- QuTaM.rd_L) < QuTaM.rd_EPS
       @test norm(QuTaM.rd_sys.Heff- QuTaM.rd_He) < QuTaM.rd_EPS
       @test norm(QuTaM.rd_sys.J - QuTaM.rd_J) < QuTaM.rd_EPS
    end;

########### WTD ######################################
# Data generation
@time begin
rd_data = run_trajectories(QuTaM.rd_sys, QuTaM.rd_params)
end
rd_times = [rd_data[k][1].time for k in 1:QuTaM.rd_params.ntraj if !isempty(rd_data[k])]
rd_d = Distributions.Exponential(1/QuTaM.rd_gamma)
## Use a two sample Kolmogorov-Smirnov test, pvalue above 0.2 is accepted
rd_pvalue = HypothesisTests.pvalue(
HypothesisTests.ApproximateTwoSampleKSTest(rd_times, rand(rd_d, QuTaM.rd_params.ntraj)))

@testset verbose=true "WTD (KS Test and fit)" begin
  rd_p0WTD = 0.2 # Minimal pvalue for accepting the null hypothesis
  fit_par = Distributions.fit(Distributions.Exponential, rd_times).θ
  @test rd_pvalue > rd_p0WTD
  @test abs(fit_par - 1/QuTaM.rd_gamma) < 0.01
end

sys = QuTaM.rd_sys
params = QuTaM.rd_params
# Now from each trajectory, generate the states at the given times
sample_clicks = QuTaM.run_trajectories(sys, params)
ntimes = 1000
t = collect(LinRange(0, params.tf, ntimes))
sample = Array{ComplexF64}(undef, sys.NLEVELS, ntimes, params.ntraj);
for n in 1:params.ntraj
    sample[:, :, n] = QuTaM.evaluate_at_t(t, sample_clicks[n], sys,  params.psi0)
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
        x_sample[tn, k] = dot(sample[:, tn, k], QuTaM.sigma_z * sample[:, tn, k])
    end
end
x = real(dropdims( mean(x_sample, dims=2), dims=2));
x_theo = 2*exp.(-QuTaM.rd_gamma.*t).-1
error = sum(abs.( (x - x_theo) ./ x_theo)) / (sum(abs.(x_theo)))

@test local_flag
@test error < accepted_error
