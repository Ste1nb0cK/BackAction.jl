# AUTHOR: Nicolás Niño-Salas
# Date: 2025
# DESCRIPTION:
#  Implementation of the Monte Carlo Wavefunction method, heavily inspired by that
#  of the mcsolver in QuantumToolBox.jl. It relies very heavily in the
#  DifferentialEquations.jl library, particularly on the use of callbacks
#  and parallel ensemble solutions, see:
# https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/#Using-Callbacks
# https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/




"""
```

_create_callback(sys::System, params::SimulParameters, t_eval::AbstractVector, rng)

```
Create a callback to perform jump updates in the MCW method, appropiate for the given system.
`params` is used to provide the seed and the initial state, while `t_eval` for infering
the type of the `weights` and `cumsum`.
"""
function _create_callback(sys::System, params::SimulParameters, tspan::Tuple{T1,T1}, rng::Xoshiro,
    e_ops::Vector{Matrix{T2}}, tlist::Vector{T1}) where {T1<:Real,T2<:Complex}
    jumpcb = _create_jumpcallback(sys, params, tspan, rng)
    savecb = _create_savecallback(e_ops, tlist)
    return CallbackSet(jumpcb, savecb)
end

#
"""
```

_generate_trajectoryproblem_jumps(sys::System, params::SimulParameters, t_eval::AbstractVector; kwargs... )

```
Generate a ODEProblem for the given `System`, corresponding to a single trajectory of the MCW method.
The initial condition and seed are passed via `params`, while `t_eval` sets the points at which the
solver should save the solution.
"""
function _generate_trajectoryproblem_jumps(sys::System, params::SimulParameters,
    tspan::Tuple{T2,T2}, e_ops::Vector{Matrix{T1}}, tlist::Vector{T2};
    kwargs...)::ODEProblem where {T1<:Complex,T2<:Real}
    f! = HeffEvolution(getfield(sys, :Heff))
    # # create the LindbladJump that will hold the affect!
    rng = Random.Xoshiro()
    cb = _create_callback(sys, params, tspan, rng, e_ops, tlist)

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
function _prob_func(prob, i, repeat, tlist)
    cb_jumps = _initialize_similarcb(i, prob.kwargs[:callback].continuous_callbacks[1].affect!)
    cb_save = _initialize_similarcb(prob.kwargs[:callback].discrete_callbacks[1].affect!.func, tlist)
    f = deepcopy(prob.f.f)

    return remake(prob; f=f, callback=CallbackSet(cb_jumps, cb_save))
end


"""
```
_get_ensemble_problem_jumps(sys, params, t_eval; kwargs...)
```
Initialize an ensemble ODEProblem from the given `System`. The initial
state for all the trajectories is assumed to be the same, and the seed is set
according to `params`.
"""

function _get_ensemble_problem_jumps(sys, params, tspan,
    e_ops, tlist; kwargs...)
    prob_sys = _generate_trajectoryproblem_jumps(sys, params, tspan, e_ops, tlist; kwargs...)
    return EnsembleProblem(prob_sys; prob_func=((prob, i, repeat) -> _prob_func(prob, i, repeat, tlist)),
        output_func=_output_func, safetycopy=false)
end


# Resize the vectors and make them have the waiting time instead of the global time
function _output_func(sol, i)
    idx = sol.prob.kwargs[:callback].continuous_callbacks[1].affect!.jump_counter[]
    resize!(sol.prob.kwargs[:callback].continuous_callbacks[1].affect!.jump_channels, idx - 1)
    resize!(sol.prob.kwargs[:callback].continuous_callbacks[1].affect!.jump_times, idx - 1)
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
function get_sol_jumps(sys::System, params::SimulParameters, tspan::Tuple{T,T},
    e_ops::Vector{Matrix{T2}}, tlist::Vector{T},
    alg=Tsit5(), ensemblealg=EnsembleDistributed();
    kwargs...) where {T<:Real,T2<:Complex}
    # set the ensemble problem
    ensemble_prob = _get_ensemble_problem_jumps(sys, params, tspan, e_ops, tlist; kwargs...)
    return solve(ensemble_prob, alg, ensemblealg, trajectories=params.ntraj)
end


export get_sol_jumps
