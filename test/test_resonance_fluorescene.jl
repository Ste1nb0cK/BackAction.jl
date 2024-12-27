using DifferentialEquations
sys = QuTaM.rf_sys
params = QuTaM.rf_params
function rf_de!(dr, r, p, t)
    gamma = QuTaM.rf_gamma
    delta = QuTaM.rf_delta
    omega = QuTaM.rf_omega
    dr[1] = -0.5*gamma*r[1] - delta*r[2]
    dr[2] = delta*r[1] - 0.5*gamma*r[2] - omega*r[3]
    dr[3] = omega*r[2] - gamma*(r[3] + 1)
end

r0 = [0.0; 0.0; 1.0] # Initial Condition
tspan = (0.0, params.tf)
t_given = collect(LinRange(0, params.tf, 1000))
prob = ODEProblem(rf_de!, r0, tspan)
sol = solve(prob, reltol = 1e-6, saveat = t_given);

# Steady State
gamma = QuTaM.rf_gamma
delta = QuTaM.rf_delta
omega = QuTaM.rf_omega
r_steady = 1/(gamma^2 + 2*omega^2+4*delta^2) * [-4*delta*omega; 2*omega*gamma;-gamma^2-4*delta^2 ]

# Basic Operators
L = sqrt(gamma)*[[0, 0] [1, 0]]
H = [[-delta, omega] [omega, delta]]
Heff = [[-delta, omega] [omega, delta-0.5im*gamma]]
J = gamma*[[0, 0] [0, 1]]
#################### Statistics
WTD_analytical(tau) =  (16*gamma*omega^2)*exp(-0.5*gamma*tau) * sin(0.25*tau*sqrt(16*omega^2-gamma^2))^2/(-gamma^2+16*omega^2)
# Average Simulation ################3
# Now from each trajectory, generate the states the given times
sample_clicks = QuTaM.run_trajectories(sys, params)
ntimes = size(t_given)[1]
sample = zeros(ComplexF64, ntimes, sys.NLEVELS, params.ntraj)
for n in 1:params.ntraj
    states = QuTaM.evaluate_at_t(t_given, sample_clicks[n], sys,  params.psi0)
    for j in 1:sys.NLEVELS
        for tn in 1:ntimes
            sample[tn, j, n] = states[tn, j]
        end
    end
end
# Get the jump times, we leverage that this is a renewal process
tau_sample = Vector{Float64}()
for traj in sample_clicks
    if !isempty(traj)
        for click in traj
            push!(tau_sample, click.time)
        end
    else
        continue
    end
end

@testset "Resonance Fluorescene: Basic Operators" begin
       @test norm(QuTaM.rf_sys.H - H) < QuTaM.rf_EPS
       @test norm(QuTaM.rf_sys.Ls[1]- L) < QuTaM.rf_EPS
       @test norm(QuTaM.rf_sys.Heff- Heff) < QuTaM.rf_EPS
       @test norm(QuTaM.rf_sys.J - J) < QuTaM.rf_EPS

end

@testset "Resonance Fluorescene: WTD" begin

end
