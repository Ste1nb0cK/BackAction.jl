function _create_savecallback_monitoring(tlist::Vector{T2}, nlevels::T3) where {T2<:Real,T3<:Int}
    # cache_aux1 = Vector{Complex{T2}}(undef, nlevels)
    cache_phi = zeros(Complex{T2}, nlevels)
    inner = Vector{Complex{T2}}(undef, length(tlist))
    save_func = SaveFuncMonitoring(cache_phi, Ref(1), inner)
    return FunctionCallingCallback(save_func, funcat=tlist)
end

function _initialize_similarcb(savefunc::SaveFuncMonitoring, tlist::Vector{T1}) where {T1<:Real}
    # cache_aux1 = similar(savefunc.cache_aux1)
    inner = similar(savefunc.inner)
    cache_phi = zeros(Complex{T1}, length(savefunc.cache_phi))
    save_func = SaveFuncMonitoring(cache_phi, Ref(1), inner)
    return FunctionCallingCallback(save_func, funcat=tlist)
end

