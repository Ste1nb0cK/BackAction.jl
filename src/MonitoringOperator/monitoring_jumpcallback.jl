
################### Integration stuff
function _initialize_similarcb(i, affect!::_LindbladJump_Monitoring)::ContinuousCallback
    rng = Random.Xoshiro()
    Random.seed!(rng, i)
    _jump_affect! = _similar_affect!(affect!, rng)
    cb = ContinuousCallback(JumpCondition(_jump_affect!.r), _jump_affect!)
    return cb
end


# function _lindblad_jump_monitoring_affect!(integrator, theta, Ls, dLs, LLs, Heff, He_par, rng, r, weights, cumsum, cache_state, cache_phi, jump_times, jump_channels, jump_counter)
function _create_jumpcallback_monitoring(dtheta::T2, dLs::Array{T1},
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
        Ref(0.0),
        Ref(1),
    )

    return ContinuousCallback(JumpCondition(_jump_affect!.r[]), _jump_affect!; save_positions=(true, true))
end


