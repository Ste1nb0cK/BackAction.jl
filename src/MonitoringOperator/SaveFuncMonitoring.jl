struct SaveFuncMonitoring{T<:Complex} <: Function
    # cache_aux1::Vector{T}
    cache_phi::Vector{T}
    counter::Ref{Int64}
    inner::Vector{T}
end

(f::SaveFuncMonitoring)(u, t, integrator) = _save_func_monitoring(u, integrator, f.cache_phi, f.counter, f.inner)

function _save_func_monitoring(u, integrator, cache_phi, counter, inner)
    affect! = integrator.opts.callback.continuous_callbacks[1].affect!
    cache_state = affect!.cache_state
    auxt = -1im * (integrator.t - affect!.cache_jtime[])
    cache_phi .= ForwardDiff.derivative(theta -> expv(auxt, affect!.Heff_par(theta), affect!.cache_state), affect!.theta0) +
                 expv(auxt, affect!.Heff, affect!.cache_phi)

    normalization = 1 / norm(integrator.u)
    lmul!(normalization, cache_phi)
    inner[counter[]] = normalization * dot(cache_phi, integrator.u)
    counter[] += 1

    u_modified!(integrator, false)
    return nothing
end


