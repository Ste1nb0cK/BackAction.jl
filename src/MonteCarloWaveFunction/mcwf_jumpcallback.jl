function _initialize_similarcb(i, affect!::_LindbladJump)::ContinuousCallback
    rng = Random.Xoshiro()
    Random.seed!(rng, i)
    _jump_affect! = _similar_affect!(affect!, rng)
    cb = ContinuousCallback(JumpCondition(_jump_affect!.r), _jump_affect!)
    return cb
end


function _create_jumpcallback(sys::System, params::SimulParameters, tspan::Tuple{T1,T1}, rng::Xoshiro) where {T1<:Real}

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
    return ContinuousCallback(JumpCondition(_jump_affect!.r[]), _jump_affect!; save_positions=(true, true))
end

