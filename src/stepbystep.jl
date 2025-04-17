# AUTHOR: Nicolás Niño-Salas
# Date: 2025
# DESCRIPTION:
#  Implementation of the Monte Carlo Wavefunction method, heavily inspired by that
#  of the mcsolver in QuantumToolBox.jl. It relies very heavily in the
#  DifferentialEquations.jl library, particularly on the use of callbacks
#  and parallel ensemble solutions, see:
# https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/#Using-Callbacks
# https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/

# @docs " Number of maximum number of jumps that's initially expected to be stored"
const JUMP_TIMES_INIT_SIZE::Int64 = 200

# This avoids DifferentialEquations trying to deepcopy an anonymous function when 
# creating the new problems
# Overload the struct to use it as the ODEFunction in the problems
"""
```

    _LindbladJump(T1, T2, RNGType<:AbstractRNG, T3, WVType<:AbstractVector,
                  SVType<:AbstractVector)

```
Internal Parametric type to store information the integrator needs for sampling
random numbers and applying the jump update rule.
# Fields
 - `Ls::T1`: Jump operators.
 - `LLs::T1`: Products of the jump operators.
 - `Heff::T2`: Effective Hamiltonian.
 - `rng::RNGType`: Random number generator.
 - `r::T3`: Random number for sampling, this is inteded to be a Ref.
  Next stuff is for convenience in doing the jump update, here we basically preallocate memory for it.
 - `weights::WVType`:  dedicated vector for storing the weights used in the channel sampling.
 - `cumsum::WVType`:  dedicated vector for storing the cumulative sum of `weights`.
 - `cache_state::SVType`: auxiliary vector for storing the current state in the trajectory.


"""
struct _LindbladJump{T1<:Complex,
    T2<:Real, #type of the effective hamiltonian
    # RNGType<:Xoshiro, # type of the RNG
    # T3<:Ref{Float64}, # type of the random vector one uses to sample
    T4<:Int, # channel labels vector
    # JCT<:Ref{Int64}, # jump counter
}
    Ls::Array{T1,3}# Jump operators
    LLs::Array{T1,3} # Products of the jump operators
    Heff::Matrix{T1} # Effective Hamiltonian
    rng::Xoshiro # Random number generator
    r::Ref{Float64} # Random number for sampling, this is inteded to be a Ref
    # Next stuff is for convenience in doing the jump update, here we basically preallocate memory for it
    weights::Vector{T2}
    cumsum::Vector{T2}
    cache_state::Vector{T1}
    jump_times::Vector{T2}
    jump_channels::Vector{T4}
    jump_counter::Ref{Int64}

end

import Base.deepcopy_internal
function Base.deepcopy_internal(affect!::T, dict::IdDict) where {T<:BackAction._LindbladJump}
    return BackAction._LindbladJump(
        deepcopy_internal(getfield(affect!, :Ls), dict),
        deepcopy_internal(getfield(affect!, :LLs), dict),
        deepcopy_internal(getfield(affect!, :Heff), dict),
        deepcopy_internal(getfield(affect!, :rng), dict)::Xoshiro,
        Ref(1.0),
        deepcopy_internal(getfield(affect!, :weights), dict),
        deepcopy_internal(getfield(affect!, :cumsum), dict),
        deepcopy_internal(getfield(affect!, :cache_state), dict),
        deepcopy_internal(getfield(affect!, :jump_times), dict),
        deepcopy_internal(getfield(affect!, :jump_channels), dict),
        Ref(1)
    )
end



"""
```

_similar_affect!(affect::LindbladJump, rng)

```
Create a Lindblad jump with a new random number generator `rng` and new memory.
The sizes and types for `chache_state`, `weights` and `cumsum` are infered
from `affect`.
"""
function _similar_affect!(affect::_LindbladJump, rng)
    r = Ref(rand(rng))
    cache_state = similar(affect.cache_state)
    weights = similar(affect.weights)
    cumsum = similar(affect.cumsum)
    jump_times = similar(affect.jump_times)
    jump_channels = similar(affect.jump_channels)
    jump_counter = Ref(1)
    return _LindbladJump(affect.Ls,
        affect.LLs,
        affect.Heff,
        rng,
        r,
        weights,
        cumsum,
        cache_state,
        jump_times,
        jump_channels,
        jump_counter
    )
end

"""
```

_lindblad_jump_affect!(integrator, Ls, LLs, Heff, rng, r, weights, cumsum, cache_state)

```

Perform a jump update on `integrator`, this is supposed to be used as the `affect!`
 in a callback, for more information see the documentation for `callbacks` in DifferentialEquations.jl.

"""
function _lindblad_jump_affect!(integrator, Ls, LLs, Heff, rng, r, weights, cumsum, cache_state, jump_times, jump_channels, jump_counter)
    # do the jump update
    ψ = integrator.u
    @inbounds for i in eachindex(weights)
        weights[i] = real(dot(ψ, LLs[:, :, i], ψ))
    end
    cumsum!(cumsum, weights)
    r[] = rand(rng) * sum(weights) # Multiply by the sum of weights because this is an unnormalized distribution
    collapse_idx = getindex(1:length(weights), findfirst(>(r[]), cumsum)) # get the channel
    mul!(cache_state, Ls[:, :, collapse_idx], ψ)
    normalize!(cache_state)
    copyto!(integrator.u, cache_state)
    #save jump information and prepare for new jump
    r[] = rand(rng)

    idx = jump_counter[]
    @inbounds jump_times[idx] = integrator.t
    @inbounds jump_channels[idx] = collapse_idx
    jump_counter[] += 1
    # # in case the initially expected maximum number of jumps was surpassed, increase it
    if jump_counter[] > length(jump_times)
        resize!(jump_times, length(jump_times) + JUMP_TIMES_INIT_SIZE)
        resize!(jump_channels, length(jump_channels) + JUMP_TIMES_INIT_SIZE)
    end
    u_modified!(integrator, true) # Inform the solver that a discontinous change happened
    return nothing
end
# overload the LindbladJump type so we can call LindbladJump instances
"""
```
    (f:::_LindbladJump)(integrator)
```
Call overloading of `_LindbladJump`, does a jump update with  _lindblad_jump_affect!
according to the jump operators, state and rng in `f`. This is used used to perform the `affect!` of a callback.
For more information on `callbacks` see the documentation in DifferentialEquations.jl.
"""
function (f::_LindbladJump)(integrator)
    _lindblad_jump_affect!(integrator, f.Ls, f.LLs, f.Heff, f.rng, f.r, f.weights,
        f.cumsum, f.cache_state, f.jump_times, f.jump_channels, f.jump_counter)
end

struct HeffEvolution #{T1<:Complex}
    Heff::Matrix{ComplexF64}
end

function Base.deepcopy_internal(h::HeffEvolution, dict::IdDict) #where {T1<:Complex}
    return HeffEvolution(deepcopy_internal(h.Heff, dict))
end

function (Heff::HeffEvolution)(du::Vector{T1}, u::Vector{T1}, p, t) where {T1<:Complex}
    du .= -1im * getfield(Heff, :Heff) * u
end


struct JumpCondition
    r::Ref{Float64}
end

function Base.deepcopy_internal(c::JumpCondition, dict::IdDict)
    JumpCondition(Ref(1.0))
end

function (condition::JumpCondition)(u, t, integrator)
    real(dot(u, u)) - condition.r[]
end



"""
```

_create_callback(sys::System, params::SimulParameters, t_eval::AbstractVector, rng)

```
Create a callback to perform jump updates in the MCW method, appropiate for the given system.
`params` is used to provide the seed and the initial state, while `t_eval` for infering
the type of the `weights` and `cumsum`.
"""
function _create_callback(sys::System, params::SimulParameters, tspan::Tuple{T1,T1}, rng::Xoshiro) where {T1<:Real}
    ttype = eltype(tspan)
    Random.seed!(rng, params.seed)
    _jump_affect! = _LindbladJump(
        getfield(sys, :Ls),
        getfield(sys, :LLs),
        getfield(sys, :Heff),
        rng,
        Ref{Float64}(rand(rng)),
        Vector{ttype}(undef, sys.NCHANNELS),
        Vector{ttype}(undef, sys.NCHANNELS),
        similar(getfield(params, :psi0)),
        Vector{Float64}(undef, JUMP_TIMES_INIT_SIZE),
        Vector{Int64}(undef, JUMP_TIMES_INIT_SIZE),
        Ref{Int64}(1)
    )

    # function condition(u, t, integrator)
    #     real(dot(u, u)) - _jump_affect!.r[]
    # end
    return ContinuousCallback(JumpCondition(_jump_affect!.r[]), _jump_affect!; save_positions=(true, true))
end

"""
```

_generate_trajectoryproblem_jumps(sys::System, params::SimulParameters, t_eval::AbstractVector; kwargs... )

```
Generate a ODEProblem for the given `System`, corresponding to a single trajectory of the MCW method.
The initial condition and seed are passed via `params`, while `t_eval` sets the points at which the
solver should save the solution.
"""
function _generate_trajectoryproblem_jumps(sys::System, params::SimulParameters,
    tspan::Tuple{T2,T2}; kwargs...)::ODEProblem where {T2<:Real}
    f! = HeffEvolution(getfield(sys, :Heff))
    # # create the LindbladJump that will hold the affect!
    rng = Random.Xoshiro()
    cb = _create_callback(sys, params, tspan, rng)

    # return ODEProblem{true}(f!, params.psi0, tspan; callback=cb, saveat=t_eval, kwargs...)
    return ODEProblem{true}(f!, params.psi0, tspan; callback=cb, kwargs...)
end



# """
# ```
# _prob_func_jumps(prob, i, repeat)
# ```
# Function for creating new problems from a given trajectory problem. This is intended to be
# used in the initialization of an ensemble problem, see: https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/
# """
function _prob_func_jumps(prob, i, repeat)
    # First, initialize a new RNG with the corresponding seed
    rng = Random.Xoshiro()
    Random.seed!(rng, i)
    affect0! = prob.kwargs[:callback].affect!
    _jump_affect! = _similar_affect!(affect0!, rng)
    # function condition(u, t, integrator)
    #     real(dot(u, u)) - _jump_affect!.r[]
    # end
    # cb = ContinuousCallback(condition, _jump_affect!)
    cb = ContinuousCallback(JumpCondition(_jump_affect!.r), _jump_affect!)
    f = deepcopy(prob.f.f)
    return remake(prob; f=f, callback=cb)
end

"""
```
_get_ensemble_problem_jumps(sys, params, t_eval; kwargs...)
```
Initialize an ensemble ODEProblem from the given `System`. The initial
state for all the trajectories is assumed to be the same, and the seed is set
according to `params`.
"""
function _get_ensemble_problem_jumps(sys, params, tspan; kwargs...)
    prob_sys = _generate_trajectoryproblem_jumps(sys, params, tspan; kwargs...)
    return EnsembleProblem(prob_sys; prob_func=_prob_func_jumps, output_func=_output_func, safetycopy=false)
end


# Resize the vectors and make them have the waiting time instead of the global time
function _output_func(sol, i)
    idx = sol.prob.kwargs[:callback].affect!.jump_counter[]
    resize!(sol.prob.kwargs[:callback].affect!.jump_channels, idx - 1)
    resize!(sol.prob.kwargs[:callback].affect!.jump_times, idx - 1)
    return (sol, false)
end


"""
```
get_sol_jumps(sys, params, t_eval, alg=nothing, ensemblealg=ensemblethreads(); kwargs...)
```
Obtain an ensemble solution for the trajectories of the system `sys`, with the
seed and initial state passed in `params`, and the times at which the solver
will store the states are defined by `t_eval`. Additionally, you can choose the
algorithm for the solver via `alg` and `ensemblealg`, and  even pass any valid
`keyword argument` valid in `DifferentialEquations.jl` through `kwargs`
"""
function get_sol_jumps(sys::System, params::SimulParameters, tspan::Tuple{T,T}, alg=Tsit5(), ensemblealg=EnsembleThreads();
    kwargs...) where {T<:Real}
    # set the ensemble problem
    ensemble_prob = _get_ensemble_problem_jumps(sys, params, tspan; kwargs...)
    return solve(ensemble_prob, alg, ensemblealg, trajectories=params.ntraj)
end


export get_sol_jumps
