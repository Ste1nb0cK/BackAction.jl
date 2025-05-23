function identifystate(s::Vector{ComplexF64})
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

function identifystate(s::Matrix{ComplexF64})
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

function trajectory_to_vectors(traj::Trajectory)
    njumps = length(traj)
    jumptimes = Vector{Float64}(undef, njumps)
    labels = Vector{Int64}(undef, njumps)
    for k in 1:njumps
        jumptimes[k] = traj[k].time
        labels[k] = traj[k].label
    end
    return jumptimes, labels
end



sys = BackAction.rdt_sys
@testset "Pure States: Trajectory Method" begin
    params = BackAction.rdt_params
    # Edge case: empty t_given
    tgiven_empty = Vector{Float64}(undef, 0)
    traj_tgivenempty = [DetectionClick(0.5, 1)]
    states_empty = BackAction.states_att(tgiven_empty, traj_tgivenempty, sys, params.psi0)
    # Check that empty time vector returns empty vector
    @test isempty(states_empty)

    # Edge case: No-jump trajectory"
    traj_empty = Vector{BackAction.DetectionClick{Float64,Int64}}(undef, 0) # Empty trajectory
    tgiven_emptytraj = collect(LinRange(0, params.tf, 100))
    states_nojumps = BackAction.states_att(tgiven_emptytraj, traj_empty, sys, params.psi0)
    for k in 1:size(states_nojumps)[2]
        @test identifystate(states_nojumps[:, k]) == identifystate(params.psi0)
    end

    # Generic case
    trajectories = BackAction.run_trajectories_gillipsie(sys, params)
    # for traj in trajectories
    for traj in trajectories
        jumpdts = [click.time for click in traj]
        mindt = extrema(jumpdts)[1]
        jumptimes = cumsum(jumpdts)
        jumplabels = [click.label for click in traj]
        # times just after the jumps
        tgiven_a = jumptimes .+ mindt / 2
        # Verify that evaluation just after jump works,
        # and that there are as many states as jumps
        states_afterjumps = BackAction.states_att(tgiven_a, traj, sys, params.psi0)
        njumps = size(traj)[1]
        nstates = size(states_afterjumps)[2]
        @test nstates == njumps
        for k in 1:nstates
            @test identifystate(states_afterjumps[:, k]) == jumplabels[k]
            @test abs(norm(states_afterjumps[:, k]) - 1) < 0.01
        end
        # Verify that evaluation just before jump works,
        # and that there are as many states as jumps
        tgiven_b = jumptimes .- mindt / 2
        states_beforejumps = BackAction.states_att(tgiven_b, traj, sys, params.psi0)
        njumps = size(traj)[1]
        nstates = size(states_beforejumps)[2]
        @test nstates == njumps
        for k in 1:nstates
            @test identifystate(states_afterjumps[:, k]) == jumplabels[k]
            @test abs(norm(states_beforejumps[:, k]) - 1) < 0.01
        end

    end

end

@testset "Pure States: Vectors Method" begin
    params = BackAction.rdt_params
    # Edge case: empty t_given
    tgiven_empty = Vector{Float64}(undef, 0)
    # traj_tgivenempty = [DetectionClick(0.5, 1)]
    jumptimes_tgivenempty = [0.5]
    labels_tgivenempty = [1]
    states_empty = BackAction.states_att(tgiven_empty, jumptimes_tgivenempty, labels_tgivenempty, sys, params.psi0)
    # Check that empty time vector returns empty vector
    @test isempty(states_empty)

    # Edge case: No-jump trajectory"
    # traj_empty = Vector{BackAction.DetectionClick}(undef, 0) # Empty trajectory
    jumptimes_nojumps = Vector{Float64}(undef, 0)
    labels_nojumps = Vector{Int64}(undef, 0)
    tgiven_nojumps = collect(LinRange(0, params.tf, 100))
    states_nojumps = BackAction.states_att(tgiven_nojumps, jumptimes_nojumps, labels_nojumps, sys, params.psi0)
    for k in 1:length(tgiven_nojumps)
        @test identifystate(states_nojumps[:, k]) == identifystate(params.psi0)
    end

    # Generic case
    trajectories = BackAction.run_trajectories_gillipsie(sys, params)
    # for traj in trajectories
    for traj in trajectories
        jumpdts = [click.time for click in traj]
        mindt = extrema(jumpdts)[1]
        jumptimes = cumsum(jumpdts)
        labels = [click.label for click in traj]
        # times just after the jumps
        tgiven_a = jumptimes .+ mindt / 2
        # Verify that evaluation just after jump works,
        # and that there are as many states as jumps
        states_afterjumps = BackAction.states_att(tgiven_a, jumpdts, labels, sys, params.psi0)
        njumps = length(traj)
        nstates = size(states_afterjumps)[2]
        @test nstates == njumps
        for k in 1:nstates
            @test identifystate(states_afterjumps[:, k]) == labels[k]
            @test abs(norm(states_afterjumps[:, k]) - 1) < 0.01
        end
        # Verify that evaluation just before jump works,
        # and that there are as many states as jumps
        tgiven_b = jumptimes .- mindt / 2
        states_beforejumps = BackAction.states_att(tgiven_b, jumpdts, labels, sys, params.psi0)
        njumps = length(labels)
        nstates = size(states_beforejumps)[2]
        @test nstates == njumps
        for k in 1:nstates
            @test identifystate(states_afterjumps[:, k]) == labels[k]
            @test abs(norm(states_beforejumps[:, k]) - 1) < 0.01
        end

    end

end


#
# @testset "Mixed States" begin
#     rho0 = zeros(ComplexF64, 2, 2)
#     # Initial condition: Completely mixed state
#     rho0[2, 2] = 0.5 # Initial condition
#     rho0[1, 1] = 0.5
#     params = SimulParameters(rho0,
#         5.0, # Final time. Set very long so that all trajectories jump
#         10, # seed
#         1000, # Number of trajectories
#         10_000, # Number of samples in the finegrid
#         10.0, # Multiplier to use in the fine grid
#         1e-3 # Tolerance for passing Dark state test
#     )
#     # Edge case: empty t_given
#     tgiven_empty = Vector{Float64}(undef, 0)
#     traj_tgivenempty = [DetectionClick(0.5, 1)]
#     states_empty = BackAction.states_att(tgiven_empty, traj_tgivenempty, sys, params.psi0)
#     # Check that empty time vector returns empty vector
#     @test isempty(states_empty)
#     # Generic case
#     trajectories = BackAction.run_trajectories_gillipsie(sys, params)
#     for traj in trajectories
#         if isempty(traj)
#             continue
#         end
#         jumpdts = [click.time for click in traj]
#         mindt = extrema(jumpdts)[1]
#         jumptimes = cumsum(jumpdts)
#         jumplabels = [click.label for click in traj]
#         # times just after the jumps
#         tgiven_a = jumptimes .+ mindt / 2
#         # Verify that evaluation just after jump works,
#         # and that there are as many states as jumps
#         states_afterjumps = BackAction.states_att(tgiven_a, traj, sys, params.psi0)
#         njumps = size(traj)[1]
#         nstates = size(states_afterjumps)[3]
#         @test nstates == njumps
#         for k in 1:nstates
#             @test identifystate(states_afterjumps[:, :, k]) == jumplabels[k]
#             @test abs(real(tr(states_afterjumps[:, :, k])) - 1) < 0.01
#         end
#         # Verify that evaluation just before jump works,
#         # and that there are as many states as jumps
#         tgiven_b = jumptimes .- mindt / 2
#         states_beforejumps = BackAction.states_att(tgiven_b, traj, sys, params.psi0)
#         njumps = size(traj)[1]
#         nstates = size(states_beforejumps)[3]
#         @test nstates == njumps
#         for k in 1:nstates
#             @test identifystate(states_afterjumps[:, :, k]) == jumplabels[k]
#             @test abs(real(tr(states_beforejumps[:, :, k])) - 1) < 0.01
#         end
#     end
# end
