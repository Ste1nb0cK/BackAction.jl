export get_sol_jumps_monitoring, evaluate_and_fill_Ls!, evaluate_and_fill_Ls_dLs
function _create_callbacks_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, rng, e_ops, tlist)
    jumpcb = _create_jumpcallback_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, rng)
    savecb = _create_savecallback_monitoring(e_ops, tlist)
    return CallbackSet(jumpcb, savecb)
end

function _prob_func_monitoring(prob, i, repeat, tlist)
    cb_jumps = _initialize_similarcb(i, prob.kwargs[:callback].continuous_callbacks[1].affect!)
    cb_save = _initialize_similarcb(prob.kwargs[:callback].discrete_callbacks[1].affect!.func, tlist)
    f = deepcopy(prob.f.f)

    return remake(prob; f=f, callback=CallbackSet(cb_jumps, cb_save))
end




"""
```

_generate_trajectoryproblem_jumps(sys::System, params::SimulParameters, t_eval::AbstractVector; kwargs... )

```
Generate a ODEProblem for the given `System`, corresponding to a single trajectory of the MCW method.
The initial condition and seed are passed via `params`, while `t_eval` sets the points at which the
solver should save the solution.
"""
function _generate_trajectoryproblem_jumps_monitoring(dtheta::T2, dLs::Array{T1}, He_forward::Matrix{T1},
    He_backward::Matrix{T1}, sys::System, params::SimulParameters,
    tspan::Tuple{T2,T2}, e_ops, tlist; kwargs...)::ODEProblem where {T1<:Complex,T2<:Real}
    # function f!(du::Vector{ComplexF64}, u::Vector{ComplexF64}, p, t)::Vector{ComplexF64}
    #     du .= -1im * sys.Heff * u
    # end
    f! = HeffEvolution(getfield(sys, :Heff))
    # # create the LindbladJump that will hold the affect!
    rng = Random.Xoshiro()
    cb = _create_callbacks_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, rng, e_ops, tlist)
    # return ODEProblem{true}(f!, params.psi0, tspan; callback=cb, saveat=t_eval, kwargs...)
    return ODEProblem{true}(f!, params.psi0, tspan; callback=cb, kwargs...)
end


function _output_func_monitoring(sol, i)
    ##### Move the last phi to the final time
    jumpaffect! = sol.prob.kwargs[:callback].continuous_callbacks[1].affect!
    auxt = -1im * (sol.prob.tspan[end] - jumpaffect!.cache_jtime[])
    # Calculate \partia_\theta exp(-i*\tau*Heff) \psi with \psi at the last jump 
    mul!(jumpaffect!.cache_aux1,
        exp(auxt * jumpaffect!.Heff_forward), jumpaffect!.cache_state)
    mul!(jumpaffect!.cache_aux2,
        exp(auxt * jumpaffect!.Heff_backward), jumpaffect!.cache_state)
    jumpaffect!.cache_aux1 .-= jumpaffect!.cache_aux2
    jumpaffect!.cache_aux1 .*= 1 / jumpaffect!.dtheta
    # Get exp(-i(T-tN)He)\phi
    mul!(jumpaffect!.cache_aux2,
        exp(auxt * jumpaffect!.Heff), jumpaffect!.cache_phi)
    # Add and normalize
    jumpaffect!.cache_phi .= jumpaffect!.cache_aux1 +
                             jumpaffect!.cache_aux2
    lmul!(1 / norm(sol.u[end]), jumpaffect!.cache_phi)
    return (sol, false)
end





"""
```
_get_ensemble_problem_jumps(sys, params, t_eval; kwargs...)
```
Initialize an ensemble ODEProblem from the given `System`. The initial
state for all the trajectories is assumed to be the same, and the seed is set
according to `params`.
"""
function _get_ensemble_problem_jumps_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, e_ops, tlist; kwargs...)
    prob_sys = _generate_trajectoryproblem_jumps_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, e_ops, tlist; kwargs...)
    return EnsembleProblem(prob_sys; prob_func=((prob, i, repeat) -> _prob_func_monitoring(prob, i, repeat, tlist)),
        output_func=_output_func_monitoring, safetycopy=false)
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
function get_sol_jumps_monitoring(dtheta::T2, dLs::Array{T1}, He_forward::Matrix{T1},
    He_backward::Matrix{T1}, sys::System, params::SimulParameters,
    tspan::Tuple{T2,T2}, e_ops, tlist; alg=Tsit5(),
    ensemblealg=EnsembleDistributed(), kwargs...) where {T1<:Complex,T2<:Real}
    # set the ensemble problem
    ensemble_prob = _get_ensemble_problem_jumps_monitoring(dtheta, dLs, He_forward, He_backward,
        sys, params, tspan, e_ops, tlist; kwargs...)
    return solve(ensemble_prob, alg, ensemblealg, trajectories=params.ntraj)
end


