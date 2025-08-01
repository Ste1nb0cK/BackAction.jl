function _create_savecallback_monitoring(e_ops, tlist)
    expvals = Array{ComplexF64}(undef, length(e_ops), length(tlist))
    save_func = SaveFuncMonitoring(e_ops, Ref(1), expvals)
    return FunctionCallingCallback(save_func, funcat=tlist)
end

function _initialize_similarcb(savefunc::SaveFuncMonitoring, tlist::Vector{T1}) where {T1<:Real}
    expvals = Array{ComplexF64}(undef, length(savefunc.e_ops), length(tlist))
    save_func = SaveFuncMonitoring(savefunc.e_ops, Ref(1), expvals)
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
