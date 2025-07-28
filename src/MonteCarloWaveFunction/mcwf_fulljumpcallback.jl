struct _FullJump{TPJ<:Function,TJU<:_LindbladJump}
    prejump::TPJ
    lindblad_jump::TJU
end

function _similar_affect!(affect::_FullJump, prejump_similar::F, rng) where {F<:Function}
    lindblad_jump = _similar_affect!(affect.lindblad_jump, rng)
    prejump = prejump_similar(affect.prejump, rng)
    return _FullJump(prejump, lindblad_jump)
end

function _fullaffect!(integrator,
    prejump::TPJ, lindblad_jump::_LindbladJump) where {TPJ<:Function}
    prejump(integrator)
    lindblad_jump(integrator)
end

function (f::_FullJump)(integrator)
    _fullaffect!(integrator, f.prejump, f.lindblad_jump)
end

function _create_fulljumpcallback(sys::System, params::SimulParameters, tspan::Tuple{T1,T1}, rng::Xoshiro,
    prejump_constructor::PJC) where {T1<:Real,PJC<:Function}

    Random.seed!(rng, params.seed)
    ttype = eltype(tspan)
    lindbladjump_affect! = _LindbladJump(
        getfield(sys, :Ls),
        getfield(sys, :LLs),
        getfield(sys, :Heff),
        rng,
        Ref{Float64}(rand(rng)),
        Vector{ttype}(undef, sys.NCHANNELS),
        Vector{ttype}(undef, sys.NCHANNELS),
        getfield(params, :psi0),
        Vector{Float64}(undef, JUMP_TIMES_INIT_SIZE),
        Vector{Int64}(undef, JUMP_TIMES_INIT_SIZE),
        Ref{Int64}(1)
    )

    prejumpaffect! = prejump_constructor(sys, params, tspan, rng)
    jumpaffect! = _FullJump(prejumpaffect!, lindbladjump_affect!)
    return ContinuousCallback(JumpCondition(lindbladjump_affect!.r[]), jumpaffect!; save_positions=(true, true))
end

function _initialize_similarcb(i, affect!::_FullJump, prejump_similar)::ContinuousCallback
    rng = Random.Xoshiro()
    Random.seed!(rng, i)
    _jump_affect! = _similar_affect!(affect!, prejump_similar, rng)
    return ContinuousCallback(JumpCondition(_jump_affect!.lindblad_jump.r[]), _jump_affect!)
end

