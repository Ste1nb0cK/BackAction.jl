export get_sol_jumps_monitoring, evaluate_and_fill_Ls!, evaluate_and_fill_Ls_dLs
# addprocs()
function evaluate_and_fill_Ls!(Ls::Vector{TJ}, theta::T2,
    Ls_store::Array{T1}, nchannels::T3) where {T1<:Complex,T2<:Real,T3<:Int,TJ<:Function}
    for k in nchannels
        Ls_store[:, :, k] .= Ls[k](theta)
    end
end

function evaluate_and_fill_Ls_dLs(Ls::Vector{TJ}, theta::T2,
    Ls_store::Array{T1}, dLs_store::Array{T1}, nchannels::T3) where {T1<:Complex,T2<:Real,T3<:Int,TJ<:Function}

    for k in 1:nchannels
        Ls_store[:, :, k] .= Ls[k](theta)
        ForwardDiff.derivative!(view(dLs_store, :, :, k), Ls[k], theta)
    end
end

function _create_callbacks_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, rng)
    jumpcb = _create_jumpcallback_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, rng)
    return jumpcb
end

function _prob_func_monitoring(prob, i, repeat)
    # First, initialize a new RNG with the corresponding seed
    # rng = Random.Xoshiro()
    # Random.seed!(rng, i)
    # affect0! = prob.kwargs[:callback].affect!
    # _jump_affect! = _similar_affect!(affect0!, rng)
    # cb = ContinuousCallback(JumpCondition(_jump_affect!.r), _jump_affect!)
    cb_jumps = _initialize_similarcb(i, prob.kwargs[:callback].affect!) #.continuous_callbacks[1].affect!)
    # cb_save = _initialize_similarcb(prob.kwargs[:callback].discrete_callbacks[1].affect!.func, tlist)
    f = deepcopy(prob.f.f)

    return remake(prob; f=f, callback=cb_jumps)# callback=CallbackSet(cb_jumps, cb_save))
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
    tspan::Tuple{T2,T2}; kwargs...)::ODEProblem where {T1<:Complex,T2<:Real}
    # function f!(du::Vector{ComplexF64}, u::Vector{ComplexF64}, p, t)::Vector{ComplexF64}
    #     du .= -1im * sys.Heff * u
    # end
    f! = HeffEvolution(getfield(sys, :Heff))
    # # create the LindbladJump that will hold the affect!
    rng = Random.Xoshiro()
    cb = _create_callbacks_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, rng)
    # return ODEProblem{true}(f!, params.psi0, tspan; callback=cb, saveat=t_eval, kwargs...)
    return ODEProblem{true}(f!, params.psi0, tspan; callback=cb, kwargs...)
end


function _output_func_monitoring(sol, i)
    ##### Move the last phi to the final time
    auxt = -1im * (sol.prob.tspan[end] - sol.prob.kwargs[:callback].affect!.cache_jtime[])
    # Calculate \partia_\theta exp(-i*\tau*Heff) \psi with \psi at the last jump 
    mul!(sol.prob.kwargs[:callback].affect!.cache_aux1,
        exp(auxt * sol.prob.kwargs[:callback].affect!.Heff_forward), sol.prob.kwargs[:callback].affect!.cache_state)
    mul!(sol.prob.kwargs[:callback].affect!.cache_aux2,
        exp(auxt * sol.prob.kwargs[:callback].affect!.Heff_backward), sol.prob.kwargs[:callback].affect!.cache_state)
    sol.prob.kwargs[:callback].affect!.cache_aux1 .-= sol.prob.kwargs[:callback].affect!.cache_aux2
    sol.prob.kwargs[:callback].affect!.cache_aux1 .*= 1 / sol.prob.kwargs[:callback].affect!.dtheta
    # Get exp(-i(T-tN)He)\phi
    mul!(sol.prob.kwargs[:callback].affect!.cache_aux2,
        exp(auxt * sol.prob.kwargs[:callback].affect!.Heff), sol.prob.kwargs[:callback].affect!.cache_phi)
    # Add and normalize
    sol.prob.kwargs[:callback].affect!.cache_phi .= sol.prob.kwargs[:callback].affect!.cache_aux1 +
                                                    sol.prob.kwargs[:callback].affect!.cache_aux2
    lmul!(1 / norm(sol.u[end]), sol.prob.kwargs[:callback].affect!.cache_phi)
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
function _get_ensemble_problem_jumps_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan; kwargs...)
    prob_sys = _generate_trajectoryproblem_jumps_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan; kwargs...)
    return EnsembleProblem(prob_sys; prob_func=_prob_func_monitoring, output_func=_output_func_monitoring, safetycopy=false)
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
    tspan::Tuple{T2,T2}, alg=Tsit5(),
    ensemblealg=EnsembleDistributed(); kwargs...) where {T1<:Complex,T2<:Real}
    # set the ensemble problem
    ensemble_prob = _get_ensemble_problem_jumps_monitoring(dtheta, dLs, He_forward, He_backward,
        sys, params, tspan; kwargs...)
    return solve(ensemble_prob, alg, ensemblealg, trajectories=params.ntraj)
end


