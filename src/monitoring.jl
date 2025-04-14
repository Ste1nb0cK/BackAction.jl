export get_sol_jumps_monitoring, evaluate_and_fill_Ls!, evaluate_and_fill_Ls_dLs
### Functions for passing from one unraveling to the next one
function isometric_mixing_i(Ls::Vector{TJ}, Ti::Vector{T1}, nlevels::T3) where {T1<:Complex,TJ<:Function,T3<:Int}
    f = let Ls = Ls, Ti = Ti, nlevels = nlevels
        (x...) -> begin
            # Lsnew_i = zeros(eltype(Ti), nlevels, nlevels)
            Lsnew_i = zero(Ti[1] * Ls[1](x...))
            for j in eachindex(Ls)
                Lsnew_i .+= Ti[j] * Ls[j](x...)
            end
            return Lsnew_i
        end
    end
    return f
end

function isometric_mixing(Ls::Vector{TJ}, Ti::Matrix{T1}, nlevels::T3, nchannels::T3) where {T1<:Complex,TJ<:Function,T3<:Int}
    return [isometric_mixing_i(Ls, Ti[i, :], nlevels) for i in 1:nchannels]
end

function add_cfield_i(L::TJ, alpha_i::T1, nlevels::T3) where {T1<:Complex,T3<:Int,TJ<:Function}
    f = let L = L, alpha_i = alpha_i, nlevels = nlevels
        (x...) -> L(x...) + alpha_i * Matrix{T1}(I, nlevels, nlevels)
    end
    return f
end

function add_cfields(Ls::Vector{TJ}, alpha::Vector{T1}, nlevels::T3) where {T1<:Complex,T3<:Int,TJ<:Function}
    # nchannels0 = length(Ls)
    return [add_cfield_i(Ls[i], alpha[i], nlevels) for i in eachindex(Ls)]
end

function get_LL(L::TJ) where {TJ<:Function}
    f = let L = L
        (x...) -> adjoint(L(x...)) * L(x...)
    end
    return f
end

function get_J(Ls::Vector{TJ}, nlevels::T3) where {TJ<:Function,T3<:Int}
    f = let Ls = Ls, nlevels = nlevels
        (x...) -> begin
            J = zeros(ComplexF64, nlevels, nlevels)
            for i in eachindex(Ls)
                J .+= get_LL(Ls[i])(x...)
            end
            return J
        end
    end
    return f
end

function get_Heff(H::TH, Ls::Vector{TJ}, nlevels::T3) where {TH<:Function,TJ<:Function,T3<:Int}
    f = let H = H, nlevels = nlevels
        (x...) -> H(x...) - 0.5im * get_J(Ls, nlevels)(x...)
    end
    return f
end


function get_cfield_hamiltonian_correctionterm_i(L::TJ, alpha_i::T1) where {TJ<:Function,T1<:Complex}
    f = let alpha_i = alpha_i, L = L
        (x...) -> conj(alpha_i) * L(x...) - alpha_i * adjoint(L(x...))
    end
    return f
end

function get_cfield_hamiltonian_correctionterm(Ls::Vector{TJ}, alpha::Vector{T1}, nlevels::T3) where {TJ<:Function,T1<:Complex,T3<:Int}
    f = let Ls = Ls, alpha = alpha, nlevels = nlevels
        (x...) -> begin
            aux = zeros(ComplexF64, nlevels, nlevels)
            # nchannels = length(Ls)
            for i in eachindex(Ls)
                aux .+= get_cfield_hamiltonian_correctionterm_i(Ls[i], alpha[i])(x...)
            end
            return aux
        end
    end
    return f
end


function add_cfield_hamiltonian_correctionterm(H::TH, Ls::Vector{TJ}, alpha::Vector{T1}, nlevels::T3) where {TH<:Function,TJ<:Function,T1<:Complex,T3<:Int}
    f = let H = H, Ls = Ls, alpha = alpha, nlevels = nlevels
        (x...) -> H(x...) - 0.5im * get_cfield_hamiltonian_correctionterm(Ls, alpha, nlevels)(x...)
    end
    return f
end

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

################### Integration stuff
"""
```

    _LinbladJump_Monitoring(T1, T2, RNGType<:AbstractRNG, T3, WVType<:AbstractVector,
                  SVType<:AbstractVector, TH<:Function )

```
Internal Parametric type to store information the integrator needs for sampling
random numbers and applying the jump update rule. Also, this one applies the 
update rule for the monitoring operator.
# Fields
 - `theta::T2`: Value of the parameter
 - `Ls::T1`: Jump operators.
 - `dLs::T1`: Derivatives of the jump operators at the value of the parameters
 - `LLs::T1`: Products of the jump operators.
 - `Heff::T2`: Effective Hamiltonian.
 - `He_par::TH`: Parametrization of the Effective Hamiltonian
 - `rng::RNGType`: Random number generator.
 - `r::T3`: Random number for sampling, this is inteded to be a Ref.
  Next stuff is for convenience in doing the jump update, here we basically preallocate memory for it.
 - `weights::WVType`:  dedicated vector for storing the weights used in the channel sampling.
 - `cumsum::WVType`:  dedicated vector for storing the cumulative sum of `weights`.
 - `cache_state::SVType`: auxiliary vector for storing the current state in the trajectory.



"""
struct _LindbladJump_Monitoring{T1<:Complex,
    T2<:Real, #type of the effective hamiltonian
    # RNGType<:Xoshiro, # type of the RNG
    # T3<:Ref{Float64}, # type of the random vector one uses to sample
    T4<:Int, # channel labels vector
    # JCT<:Ref{Int64}, # jump counter
}
    dtheta::T2 #Parameter
    Ls::Array{T1,3}# Jump operators
    dLs::Array{T1,3} # Derivatives of the jump operators
    LLs::Array{T1,3} # Products of the jump operators
    Heff::Matrix{T1} # Effective Hamiltonian
    Heff_forward::Matrix{T1}
    Heff_backward::Matrix{T1}
    rng::Xoshiro # Random number generator
    r::Ref{Float64} # Random number for sampling, this is inteded to be a Ref
    # Next stuff is for convenience in doing the jump update, here we basically preallocate memory for it
    weights::Vector{T2}
    cumsum::Vector{T2}
    cache_state::Vector{T1}
    cache_phi::Vector{T1}
    cache_aux1::Vector{T1}
    cache_aux2::Vector{T1}
    jump_times::Vector{T2}
    jump_channels::Vector{T4}
    jump_counter::Ref{Int64}

end


function _similar_affect!(affect::_LindbladJump_Monitoring, rng)
    r = Ref(rand(rng))
    # cache_phi = similar(affect.cache_state)
    weights = similar(affect.weights)
    cumsum = similar(affect.cumsum)
    jump_times = similar(affect.jump_times)
    jump_channels = similar(affect.jump_channels)
    jump_counter = Ref(1)
    cache_aux1 = similar(affect.cache_aux1)
    cache_aux2 = similar(affect.cache_aux1)
    cache_state = copy(affect.cache_state)
    cache_phi = zero(affect.cache_phi)#similar(affect.cache_phi)
    return _LindbladJump_Monitoring(affect.dtheta,
        affect.Ls,
        affect.dLs,
        affect.LLs,
        affect.Heff,
        affect.Heff_forward,
        affect.Heff_backward,
        rng,
        r,
        weights,
        cumsum,
        cache_state,
        cache_phi,
        cache_aux1,
        cache_aux2,
        jump_times,
        jump_channels,
        jump_counter
    )
end

function (f::_LindbladJump_Monitoring)(integrator)
    # _lindblad_jump_monitoring_affect!(integrator, f.theta, f.Ls, f.dLs, f.LLs, f.Heff, f.He_par, f.rng, f.r, f.weights,
    # f.cumsum, f.cache_state, f.cache_phi, f.jump_times, f.jump_channels, f.jump_counter)
    _lindblad_jump_monitoring_affect!(integrator, f.dtheta, f.Ls, f.dLs, f.LLs, f.Heff, f.Heff_forward,
        f.Heff_backward, f.rng, f.r, f.weights,
        f.cumsum, f.cache_state, f.cache_phi, f.cache_aux1, f.cache_aux2, f.jump_times, f.jump_channels, f.jump_counter)

end

# function _lindblad_jump_monitoring_affect!(integrator, theta, Ls, dLs, LLs, Heff, He_par, rng, r, weights, cumsum, cache_state, cache_phi, jump_times, jump_channels, jump_counter)
function _lindblad_jump_monitoring_affect!(integrator, dtheta, Ls, dLs, LLs, Heff, Heff_forward, Heff_backward, rng, r, weights, cumsum, cache_state, cache_phi, cache_aux1, cache_aux2, jump_times, jump_channels, jump_counter)
    # Obtain the channel jump
    ψ = integrator.u # This is exp(-i\tau H_e)\psi_n
    @inbounds for i in eachindex(weights)
        weights[i] = real(dot(ψ, LLs[:, :, i], ψ))
    end
    cumsum!(cumsum, weights)
    r[] = rand(rng) * sum(weights) # Multiply by the sum of weights because this is an unnormalized distribution
    collapse_idx = getindex(1:length(weights), findfirst(>(r[]), cumsum)) # get the channel

    ####### PHI UPDATE without rescaling
    idx = jump_counter[]
    auxt = idx > 1 ? -1im * (integrator.t - jump_times[idx-1]) : -1im * integrator.t
    central_exp = exp(auxt * Heff)
    # Obtain  \partial_\theta exp(-i\tau*H_eff(\theta))*\psi , store in aux1
    mul!(cache_aux1, exp(auxt * Heff_forward), cache_state)
    mul!(cache_aux2, exp(auxt * Heff_backward), cache_state)
    cache_aux1 .-= cache_aux2
    cache_aux1 .*= 1 / dtheta
    # Obtain \partial_\theta exp(-i\tau*H_eff(\theta))*\psi + exp(-i\tau*H_eff(theta))*\phi, store where the derivative was
    mul!(cache_aux1, central_exp, cache_phi, 1.0, 1.0)
    # Multiply by the jump operator and store in phi_cache
    mul!(cache_phi, view(Ls, :, :, 1), cache_aux1)
    # Prepare the last term
    mul!(cache_aux2, view(dLs, :, :, collapse_idx), ψ)
    # Now put everything together and store in cache_phi
    cache_phi .+= cache_aux2
    ###### STATE UPDATE without normalization
    mul!(cache_state, view(Ls, :, :, collapse_idx), ψ)
    ##### NORMALIZATION
    # Normalize phi
    normalization = norm(cache_state)
    lmul!(1 / normalization, cache_phi)
    # Normalize the after jump state
    lmul!(1 / normalization, cache_state)
    copyto!(integrator.u, cache_state)
    #save jump information and prepare for new jump
    @inbounds jump_times[idx] = integrator.t
    @inbounds jump_channels[idx] = collapse_idx
    jump_counter[] += 1
    r[] = rand(rng)

    # # in case the initially expected maximum number of jumps was surpassed, increase it
    if jump_counter[] > length(jump_times)
        resize!(jump_times, length(jump_times) + JUMP_TIMES_INIT_SIZE)
        resize!(jump_channels, length(jump_channels) + JUMP_TIMES_INIT_SIZE)
    end
    u_modified!(integrator, true) # Inform the solver that a discontinous change happened
    return nothing
end

function _create_callback_monitoring(dtheta::T2, dLs::Array{T1},
    Heff_forward::Matrix{T1}, Heff_backward::Matrix{T1}, sys::System, params::SimulParameters, tspan::Tuple{T2,T2},
    rng::Xoshiro) where {T1<:Complex,T2<:Real}
    ttype = eltype(tspan)
    Random.seed!(rng, params.seed)
    _jump_affect! = _LindbladJump_Monitoring(
        dtheta,
        getfield(sys, :Ls),
        dLs,
        getfield(sys, :LLs),
        getfield(sys, :Heff),
        Heff_forward,
        Heff_backward,
        rng,
        Ref{Float64}(rand(rng)),
        Vector{ttype}(undef, sys.NCHANNELS),
        Vector{ttype}(undef, sys.NCHANNELS),
        params.psi0,
        zeros(Complex{ttype}, sys.NLEVELS),
        zeros(Complex{ttype}, sys.NLEVELS),
        zeros(Complex{ttype}, sys.NLEVELS),
        zeros(ttype, JUMP_TIMES_INIT_SIZE),
        Vector{Int64}(undef, JUMP_TIMES_INIT_SIZE),
        Ref{Int64}(1)
    )

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
function _generate_trajectoryproblem_jumps_monitoring(dtheta::T2, dLs::Array{T1}, He_forward::Matrix{T1},
    He_backward::Matrix{T1}, sys::System, params::SimulParameters,
    tspan::Tuple{T2,T2}; kwargs...)::ODEProblem where {T1<:Complex,T2<:Real}
    # function f!(du::Vector{ComplexF64}, u::Vector{ComplexF64}, p, t)::Vector{ComplexF64}
    #     du .= -1im * sys.Heff * u
    # end
    f! = HeffEvolution(getfield(sys, :Heff))
    # # create the LindbladJump that will hold the affect!
    rng = Random.Xoshiro()
    cb = _create_callback_monitoring(dtheta, dLs, He_forward, He_backward, sys, params, tspan, rng)
    # return ODEProblem{true}(f!, params.psi0, tspan; callback=cb, saveat=t_eval, kwargs...)
    return ODEProblem{true}(f!, params.psi0, tspan; callback=cb, kwargs...)
end


function _output_func_monitoring(sol, i)
    idx = sol.prob.kwargs[:callback].affect!.jump_counter[]
    resize!(sol.prob.kwargs[:callback].affect!.jump_channels, idx - 1)
    resize!(sol.prob.kwargs[:callback].affect!.jump_times, idx - 1)
    if idx - 1 == 0
        return (sol, false)
    end
    ##### Move the last phi to the final time
    auxt = -1im * (sol.prob.tspan[end] - sol.prob.kwargs[:callback].affect!.jump_times[end])
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
    return EnsembleProblem(prob_sys; prob_func=_prob_func_jumps, output_func=_output_func_monitoring, safetycopy=false)
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
    ensemblealg=EnsembleThreads(); kwargs...) where {T1<:Complex,T2<:Real}
    # set the ensemble problem
    ensemble_prob = _get_ensemble_problem_jumps_monitoring(dtheta, dLs, He_forward, He_backward,
        sys, params, tspan; kwargs...)
    return solve(ensemble_prob, alg, ensemblealg, trajectories=params.ntraj)
end


