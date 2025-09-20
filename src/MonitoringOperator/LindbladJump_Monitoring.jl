
struct _LindbladJump_Monitoring{T1<:Complex,
    T2<:Real,
    T4<:Int,
    F<:Function
}
    theta0::T2 #Parameter
    Ls::Array{T1,3}# Jump operators
    dLs::Array{T1,3} # Derivatives of the jump operators
    LLs::Array{T1,3} # Products of the jump operators
    Heff::Matrix{T1} # Effective Hamiltonian
    Heff_par::F
    # Heff_forward::Matrix{T1}
    # Heff_backward::Matrix{T1}
    rng::Xoshiro # Random number generator
    r::Ref{T2} # Random number for sampling, this is inteded to be a Ref
    # Next stuff is for convenience in doing the jump update, here we basically preallocate memory for it
    weights::Vector{T2}
    cumsum::Vector{T2}
    cache_state::Vector{T1}
    cache_phi::Vector{T1}
    cache_aux1::Vector{T1}
    cache_aux2::Vector{T1}
    # jump_times::Vector{T2}
    cache_jtime::Ref{T2}
    # jump_channels::Vector{T4}
    cache_channel::Ref{T4}

end


function _similar_affect!(affect::_LindbladJump_Monitoring, rng)
    r = Ref(rand(rng))
    # cache_phi = similar(affect.cache_state)
    weights = similar(affect.weights)
    cumsum = similar(affect.cumsum)
    # jump_times = similar(affect.jump_times)
    # jump_channels = similar(affect.jump_channels)
    cache_jtime = Ref(0.0)
    cache_channel = Ref(1)
    cache_aux1 = similar(affect.cache_aux1)
    cache_aux2 = similar(affect.cache_aux1)
    cache_state = copy(affect.cache_state)
    cache_phi = zero(affect.cache_phi)#similar(affect.cache_phi)
    return _LindbladJump_Monitoring(affect.theta0,
        affect.Ls,
        affect.dLs,
        affect.LLs,
        affect.Heff,
        affect.Heff_par,
        # affect.Heff_forward,
        # affect.Heff_backward,
        rng,
        r,
        weights,
        cumsum,
        cache_state,
        cache_phi,
        cache_aux1,
        cache_aux2,
        cache_jtime,
        cache_channel,
    )
end

function (f::_LindbladJump_Monitoring)(integrator)
    # _lindblad_jump_monitoring_affect!(integrator, f.theta, f.Ls, f.dLs, f.LLs, f.Heff, f.He_par, f.rng, f.r, f.weights,
    # f.cumsum, f.cache_state, f.cache_phi, f.jump_times, f.jump_channels, f.jump_counter)
    _lindblad_jump_monitoring_affect!(integrator, f.theta0, f.Ls, f.dLs, f.LLs, f.Heff, f.Heff_par,
        f.rng, f.r, f.weights,
        f.cumsum, f.cache_state, f.cache_phi, f.cache_aux1, f.cache_aux2, f.cache_jtime, f.cache_channel)

end

function _lindblad_jump_monitoring_affect!(integrator, theta0, Ls, dLs, LLs, Heff, Heff_par, rng, r, weights,
    cumsum, cache_state, cache_phi, cache_aux1, cache_aux2, cache_jtime, cache_channel)
    # Obtain the channel jump
    ψ = integrator.u # This is exp(-i\tau H_e)\psi_n
    @inbounds for i in eachindex(weights)
        weights[i] = real(dot(ψ, LLs[:, :, i], ψ))
    end
    cumsum!(cumsum, weights)
    r[] = rand(rng) * sum(weights) # Multiply by the sum of weights because this is an unnormalized distribution
    cache_channel = getindex(1:length(weights), findfirst(>(r[]), cumsum)) # get the channel

    ####### PHI UPDATE without rescaling
    auxt = -1im * (integrator.t - cache_jtime[])
    cache_aux1 .= ForwardDiff.derivative(theta -> expv(auxt, Heff_par(theta), cache_state), theta0) +
                  expv(auxt, Heff, cache_phi)
    mul!(cache_phi, view(Ls, :, :, cache_channel), cache_aux1)
    mul!(cache_aux2, view(dLs, :, :, cache_channel), ψ)
    cache_phi .+= cache_aux2

    ###### STATE UPDATE without normalization
    mul!(cache_state, view(Ls, :, :, cache_channel), ψ)
    ##### NORMALIZATION
    # Normalize phi
    normalization = norm(cache_state)
    lmul!(1 / normalization, cache_phi)
    # Normalize the after jump state
    lmul!(1 / normalization, cache_state)
    copyto!(integrator.u, cache_state)
    #save jump information and prepare for new jump
    @inbounds cache_jtime[] = integrator.t
    r[] = rand(rng)

    # # in case the initially expected maximum number of jumps was surpassed, increase it
    u_modified!(integrator, true) # Inform the solver that a discontinous change happened
    return nothing
end


