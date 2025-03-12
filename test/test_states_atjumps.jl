function identifystate_jump(s::Matrix{ComplexF64})
    excited = [[0, 0] [0, 1.0 + 0im]]
    ground = [[1.0 + 0im, 0] [0, 0]]
    if norm(ground - s) < 0.01
        return 1
    elseif norm(excited - s) < 0.01
        return 2
    else
        return 0
    end
end

function identifystate_jump(s::Vector{ComplexF64})
    s1 = [1.0 + 0im, 0]
    s2 = [0, 1.0 + 0im]
    if norm(s1 - s) < 0.01
        return 1
    elseif norm(s2 - s) < 0.01
        return 2
    else
        return 0
    end
end

function trajectory_to_vectors(traj::Trajectory{T2,T3}) where {T2<:Real,T3<:Int}
    njumps = length(traj)
    jumptimes = Vector{T2}(undef, njumps)
    labels = Vector{T3}(undef, njumps)
    for k in 1:njumps
        jumptimes[k] = traj[k].time
        labels[k] = traj[k].label
    end
    return jumptimes, labels
end


@testset "Pure States" begin
    ############### Sample some trajectories
    sys = BackAction.rdt_sys
    params = BackAction.rdt_params
    trajectories = BackAction.run_trajectories_gillipsie(sys, params)
    ############# For each trajectory, check that the states what what they should be
    for traj in trajectories
        states = BackAction.states_atjumps(traj, sys, params.psi0)
        counter = 1
        for click in traj
            @test identifystate_jump(states[:, counter]) == click.label
            counter = counter + 1
        end
    end
    ############## Edge case: no jumps
    traj = Trajectory{Float64,Int64}(undef, 0) #BackAction.sample_single_trajectory(sys, params, params.seed)
    states = BackAction.states_atjumps(traj, sys, params.psi0)
    @test isempty(states)
    ############## Method with vectors
    for traj in trajectories
        times, labels = trajectory_to_vectors(traj)
        states = BackAction.states_atjumps(times, labels, sys, params.psi0)
        counter = 1
        for click in traj
            @test identifystate_jump(states[:, counter]) == click.label
            counter = counter + 1
        end
    end

end
#
# @testset "Mixed States" begin
#     ############### Check sizes match for multiple jumps
#     sys = BackAction.rdt_sys
#     #### Same thing as params in rd_temperature. The difference is the initial state
#     rho0 = zeros(ComplexF64, 2, 2)
#     # Initial condition: Completely mixed state
#     rho0[2, 2] = 0.5 # Initial condition
#     rho0[1, 1] = 0.5
#     params = SimulParameters(rho0,
#         5.0, # Final time. Set very long so that all trajectories jump
#         10, # seed
#         50, # Number of trajectories
#         10_000, # Number of samples in the finegrid
#         10.0, # Multiplier to use in the fine grid
#         1e-3 # Tolerance for passing Dark state test
#     )
#     ################### Check the correct states are obtained
#     trajectories = BackAction.run_trajectories_gillipsie(sys, params)
#     for traj in trajectories
#         states = BackAction.states_atjumps(traj, sys, params.psi0)
#         counter = 1
#         for click in traj
#             @test identifystate_jump(states[:, :, counter]) == click.label
#             counter = counter + 1
#         end
#     end
#     ############## Edge case: no jumps
#     traj = Vector{DetectionClick}(undef, 0) #BackAction.sample_single_trajectory(sys, params, params.seed)
#     states = BackAction.states_atjumps(traj, sys, params.psi0)
#     @test isempty(states)
# end
