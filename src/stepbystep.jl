
const COL_TIMES_WHICH_INIT_SIZE = 200 # Number of maximum number of jumps that's initially expected to be stored
# Description: implementation of the usual MCW
struct LindbladJump{T1, #type of the list of jump operators
                    T2, #type of the effective hamiltonian
                    RNGType<:AbstractRNG, # type of the RNG
                    T3, # type of the random vector one uses to sample
                    WVType<:AbstractVector, # Weights Vector type
                    SVType<:AbstractVector, # State Vector type
                      # JTT<:AbstractVector,
    # JWT<:AbstractVector,
    # JTWIT,
    }
    Ls::T1 # Jump operators
    LLs::T1 # Products of the jump operators
    Heff::T2 # Effective Hamiltonian
    rng::RNGType # Random number generator
    r::T3 # Random number for sampling, this is inteded to be a Ref
    # Next stuff is for convenience in doing the jump update, here we basically preallocate memory for it
    weights::WVType
    cumsum::WVType
    cache_state::SVType
    # col_times::JTT
    # col_which::JWT
    # col_times_which_idx::JTWIT

end

# create a Lindblad jump with a new rng and new memory
function _similar_affect!(affect::LindbladJump, rng)
    r = Ref(rand(rng))
    cache_state = similar(affect.cache_state)
    weights = similar(affect.weights)
    cumsum = similar(affect.cumsum)
    # col_times = similar(affect.col_times)
    # col_which = similar(affect.col_which)
    # col_times_which_idx = Ref(1)
    return LindbladJump(affect.Ls,
        affect.LLs,
        affect.Heff,
        rng,
        r,
        weights,
        cumsum,
        cache_state,
        # col_times,
        # col_which,
        # col_times_which_idx
    )
end

# Function with which we will overload the type
function _lindblad_jump_affect!(integrator, Ls, LLs, Heff, rng, r, weights, cumsum, cache_state)
    # do the jump update
    ψ = integrator.u
    @inbounds for i in eachindex(weights)
        weights[i] = real(dot(ψ, LLs[i], ψ))
    end
    cumsum!(cumsum, weights)
    r[] = rand(rng) * sum(weights) # Multiply by the sum of weights because this is an unnormalized distribution
    collapse_idx = getindex(1:length(weights), findfirst(>(r[]), cumsum)) # inversion sampling
    mul!(cache_state, Ls[collapse_idx], ψ)
    normalize!(cache_state)
    copyto!(integrator.u, cache_state)
    #save jump information and prepare for new jump
    r[] = rand(rng)

    # idx = col_times_which_idx[]
    # @inbounds col_times[idx] = integrator.t
    # @inbounds col_which[idx] = collapse_idx
    # col_times_which_idx[] += 1
    # # in case the initially expected maximum number of jumps was surpassed, increase it
    # if col_times_which_idx[] > length(col_times)
    #     resize!(col_times, length(col_times) + COL_TIMES_WHICH_INIT_SIZE)
    #     resize!(col_which, length(col_which) + COL_TIMES_WHICH_INIT_SIZE)
    # end
    u_modified!(integrator, true) # Inform the solver that a discontinous change happened
    return nothing
end
# overload the LindbladJump type so we can call LindbladJump instances
function (f::LindbladJump)(integrator)
    _lindblad_jump_affect!(integrator, f.Ls, f.LLs, f.Heff, f.rng, f.r, f.weights, f.cumsum, f.cache_state )
end

function _create_callback(sys, params, t_eval, rng)
    rng = Random.default_rng()
    Random.seed!(rng, params.seed)
    _jump_affect! = LindbladJump(
            sys.Ls,
            sys.LLs,
            sys.Heff,
            rng,
            Ref(rand(rng)),
            similar(t_eval, sys.NCHANNELS),
            similar(t_eval, sys.NCHANNELS),
            similar(params.psi0)
        )

    function condition(u, t, integrator)
        real(dot(u, u)) - _jump_affect!.r[]
    end
    return ContinuousCallback(condition, _jump_affect!; save_positions=(true, false))
end

function _generate_trajectoryproblem_jumps(sys::System, params::SimulParameters, t_eval::AbstractVector; kwargs... )

    function f!(u, p, t)
        return -1im*sys.Heff*u
    end
    t0, tf = extrema(t_eval)
    tspan = (t0, tf)
    # create the LindbladJump that will hold the affect!
    rng = Random.default_rng()
    cb = _create_callback(sys, params, t_eval, rng)

    return ODEProblem(f!, params.psi0, tspan; callback = cb , saveat=t_eval, kwargs...)
end

function _prob_func_jumps(prob, i, repeat)
    # First, initialize a new RNG with the corresponding seed
    rng = Random.MersenneTwister()
    Random.seed!(rng, i)
    affect0! = prob.kwargs[:callback].affect!
    _jump_affect! = _similar_affect!(affect0!, rng)
    function condition(u, t, integrator)
        real(dot(u, u)) - _jump_affect!.r[]
    end
    cb = ContinuousCallback(condition, _jump_affect!)
    f = deepcopy(prob.f.f)
    return remake(prob, f = f, callback = cb)
end

#produce the ensemble problem
function _get_ensemble_problem_jumps(sys, params, t_eval; kwargs...)
    prob_sys = _generate_trajectoryproblem_jumps(sys, params, t_eval; kwargs...)
    return EnsembleProblem(prob_sys, prob_func = _prob_func_jumps)
end

# obtain the solutions to the jumps
function get_sol_jumps(sys, params, t_eval, alg=nothing, ensemblealg=EnsembleThreads(); kwargs...)
    # set the ensemble problem
    ensemble_prob = _get_ensemble_problem_jumps(sys, params, t_eval; kwargs...)
    return solve(ensemble_prob, alg, ensemblealg; trajectories=params.ntraj);
end

export get_sol_jumps
