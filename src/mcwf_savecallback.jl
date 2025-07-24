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

function _initialize_similarcb(savefunc::SaveFuncMCWF, tlist::Vector{T1}) where {T1<:Real}
    expvals = Array{ComplexF64}(undef, length(savefunc.e_ops), length(tlist))
    save_func = SaveFuncMCWF(savefunc.e_ops, Ref(1), expvals)
    return FunctionCallingCallback(save_func, funcat=tlist)
end

function average_expvals(solsample)
    affect!_1 = solsample[1].prob.kwargs[:callback].discrete_callbacks[1].affect!
    ftype = eltype(affect!_1.func.expvals)
    nops = length(affect!_1.func.e_ops)
    ntimes = size(affect!_1.func.expvals)[2]
    r = zeros(ftype, nops, ntimes)
    for sol in solsample
        expvals = sol.prob.kwargs[:callback].discrete_callbacks[1].affect!.func.expvals
        r .+= expvals
    end
    return r ./ length(solsample)
end

export average_expvals
