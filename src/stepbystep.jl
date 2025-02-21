# Description: implementation of the usual MCW
struct LindbladJump{T1, #type of the list of jump operators
                    T2, #type of the effective hamiltonian
                    RNGType<:AbstractRNG, # type of the RNG
                    T3, # type of the random vector one uses to sample
                    WVType<:AbstractVector, # Weights Vector type
                    SVType<:AbstractVector, # State Vector type
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
end

# create a Lindblad jump with a new rng and new memory
function _similar_affect!(affect::LindbladJump, rng)
    r = Ref(rand(rng))
    cache_state = similar(affect.cache_state)
    weights = similar(affect.weights)
    cumsum = similar(affect.cumsum)

    return LindbladJump(affect.Ls,
        affect.LLs,
        affect.Heff,
        rng,
        r,
        weights,
        cumsum,
        cache_state
    )
end

# Function with which we will overload the type
function _lindblad_jump_affect!(integrator, Ls, LLs, Heff, rng, r, weights, cumsum, cache_state)
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

    r[] = rand(rng)

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
    return ContinuousCallback(condition, _jump_affect!)
end

function generate_trajectoryproblem(sys::System, params::SimulParameters, t_eval::AbstractVector; progbar::Bool, kwargs... )

    function f!(u, p, t)
        return -1im*sys.Heff*u
    end
    t0, tf = extrema(t_eval)
    tspan = (t0, tf)
    # create the LindbladJump that will hold the affect!
    rng = Random.default_rng()
    cb = _create_callback(sys, params, t_eval, rng)

    return ODEProblem(f!, params.psi0, tspan; callback = cb , kwargs...)
end
function prob_func(prob, i, repeat)
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

export generate_trajectoryproblem, prob_func
