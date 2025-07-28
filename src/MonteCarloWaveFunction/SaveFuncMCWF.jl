struct SaveFuncMCWF{T<:Complex} <: Function
    e_ops::Vector{Matrix{T}}
    counter::Ref{Int64}
    expvals::Array{T}
end

(f::SaveFuncMCWF)(u, t, integrator) = _save_func_mcwf(u, integrator, f.e_ops, f.counter, f.expvals)

function _save_func_mcwf(u, integrator, e_ops, counter, expvals)
    cache_state = integrator.opts.callback.continuous_callbacks[1].affect!.cache_state

    copyto!(cache_state, u)
    normalize!(cache_state)
    ψ = cache_state
    _expect = op -> dot(ψ, op, ψ)
    @. expvals[:, counter[]] = _expect(e_ops)
    counter[] += 1

    u_modified!(integrator, false)
    return nothing
end


