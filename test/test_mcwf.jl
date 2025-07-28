# Description:
# This test looks at the convergence of the MCWF method for resonance fluorescene. 
#
#    H = \Delta \ketbra{e}{e} + \Omega \sigma_x/2 
#    c = \sqrt{\gamma}\sigma_-
# 
#\dot{x} = -\Delta y - \frac{\gamma}{2}x, \\
#\dot{y} = -\Omega z + \Delta x - \frac{\gamma}{2}y, \\
#\dot{z} = +\Omega y - \gamma(z + 1),
#
# Stationary Solutions (Wiseman's book)
#
#\begin{pmatrix}
#x \\
#y \\
#z
#\end{pmatrix}_{\text{ss}}
#\begin{pmatrix}
#-4\Delta\Omega \\
#2\Omega\gamma \\
#-\gamma^2 - 4\Delta^2
#\end{pmatrix}
#\left( \gamma^2 + 2\Omega^2 + 4\Delta^2 \right)^{-1}.

using LinearAlgebra, BackAction, BackAction.MonteCarloWaveFunction, Statistics, Distributed, DifferentialEquations, Test
addprocs(4)
@everywhere using BackAction.MonteCarloWaveFunction


const NCHANNELS0::Int64 = 1
const NLEVELS::Int64 = 2
const OMEGA = 1.0
const GAMMA = 0.5

const tf = 2 * pi / OMEGA * 7
const tspan = (0.0, tf)
const ntimes = 100
const tlist = collect(LinRange(0.0, tf, ntimes))
const ntraj = 5000
const params_simul = BackAction.BackActionCoreStructs.SimulParameters([0.0 + 0im, 1.0],
    tf, # Final time. Set very long so that all trajectories jump
    1, # seed
    ntraj,
    # GILLIPSIE STUFF 
    75_000, 3.0, 1e-3);
# Parameter Information
const delta = 0.1
# Unravelings
const T = reshape([1.0 + 0.0im], 1, 1)
const alphaa = [0.0 + 0.0im]
const nchannels = length(alphaa)

const sigma = [BackAction.Utilities.sigma_x, BackAction.Utilities.sigma_y, BackAction.Utilities.sigma_z]

function get_lindbladsol(delta, tspan, u0)
    function rf_ode!(dr, r, p, t)
        dr[1] = -delta * r[2] - 0.5 * GAMMA * r[1]
        dr[2] = -OMEGA * r[3] + delta * r[1] - 0.5 * GAMMA * r[2]
        dr[3] = OMEGA * r[2] - GAMMA * (r[3] + 1)
    end
    prob = ODEProblem(rf_ode!, u0, tspan)
    return solve(prob)
end


## Stuff of the stationary solution
normalization_ss(delta) = 1 / (GAMMA^2 + 2 * OMEGA^2 + 4 * delta^2)
x_ss(delta) = normalization_ss(delta) * (-4 * delta * OMEGA)
y_ss(delta) = normalization_ss(delta) * (2 * GAMMA * OMEGA)
z_ss(delta) = normalization_ss(delta) * (-GAMMA^2 - 4 * delta^2)


u0 = [real(dot(params_simul.psi0, sigma[1], params_simul.psi0)),
    real(dot(params_simul.psi0, sigma[2], params_simul.psi0)),
    real(dot(params_simul.psi0, sigma[3], params_simul.psi0))]
sol_lindblad = get_lindbladsol(delta, tspan, u0)

H = 0.5 * sigma[1] * OMEGA + 0.5 * sigma[3] * delta
Ls = reshape(sqrt(GAMMA) * BackAction.Utilities.sigma_m, NLEVELS, NLEVELS, 1)
sys = BackAction.BackActionCoreStructs.System(H, Ls, NLEVELS, nchannels)

@time sim = get_sol_jumps(sys, params_simul, tspan, sigma, tlist)

########################## AVERAGE ##########################
teval = LinRange(tspan[1], tspan[end], ntimes)
# first index is the coordinate, second the time and third the sample number
psi = Vector{ComplexF64}(undef, 2)

tolerance_individualtrajectory_tracedistance = 0.01
@testset "Test MCWF" begin
    @testset "Correct Evolution, individual Trajectories" begin
        for i in 1:params_simul.ntraj
            @testset "Correct Evolution $i-th Trajectory " begin

                sol_sample_i = sim[i]
                affect!_sample_i = sol_sample_i.prob.kwargs[:callback].continuous_callbacks[1].affect!
                jumptimes_sample_i = affect!_sample_i.jump_times
                numberofjumps_sample_i = affect!_sample_i.jump_counter[] - 1 # The counter begins at 1

                innerproduct = 0.0
                expected_state = copy(params_simul.psi0)
                observed_state = Vector{ComplexF64}(undef, 2)

                expected_state .= exp(-1.0im * jumptimes_sample_i[1] * sys.Heff) * expected_state
                observed_state .= sol_sample_i(jumptimes_sample_i[1])

                innerproduct = dot(expected_state / norm(expected_state),
                    observed_state / norm(observed_state))
                @test sqrt(abs(1 - norm(innerproduct)^2)) < tolerance_individualtrajectory_tracedistance

                expected_state .= sys.Ls[:, :, 1] * expected_state
                expected_state .= expected_state / norm(expected_state)

                # check just before and just after the jump
                for k in 1:(numberofjumps_sample_i-1)
                    tau = jumptimes_sample_i[k+1] - jumptimes_sample_i[k]

                    expected_state .= exp(-1.0im * tau * sys.Heff) * expected_state
                    observed_state .= sol_sample_i(jumptimes_sample_i[k+1])
                    innerproduct = dot(expected_state / norm(expected_state), observed_state / norm(observed_state))

                    @test sqrt(abs(1 - norm(innerproduct)^2)) < tolerance_individualtrajectory_tracedistance

                    expected_state .= sys.Ls[:, :, 1] * expected_state
                    expected_state .= expected_state / norm(expected_state)
                    observed_state .= sol_sample_i(jumptimes_sample_i[k+1] + 1e-6)
                    innerproduct = dot(expected_state, observed_state / norm(observed_state))

                    @test sqrt(abs(1 - norm(innerproduct)^2)) < tolerance_individualtrajectory_tracedistance
                end
            end
        end
    end
    @testset "Correct Average Convergence" begin
        r_mean = average_expvals(sim)
        for k in 1:ntimes
            r_lindblad = sol_lindblad(tlist[k])
            difference_x = 0.5 * (r_lindblad[1] - r_mean[1, k])
            difference_y = 0.5 * (r_lindblad[2] - r_mean[2, k])
            difference_z = 0.5 * (r_lindblad[3] - r_mean[3, k])
            difference = difference_x * sigma[1] +
                         difference_y * sigma[2] +
                         difference_z * sigma[3]
            @test abs(0.5 * tr(sqrt(adjoint(difference) * difference))) < tolerance_individualtrajectory_tracedistance
        end
    end
end
