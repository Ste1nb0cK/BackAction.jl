### Functions for passing from one unraveling to the next one
#TODO: Add isometric mixing
#
#
struct ParametricUnravelingJumpOperator{F<:Function,T1<:Complex} <: Function
    L0::F
    cfield::T1
end

function (f::ParametricUnravelingJumpOperator)(theta...)
    return f.L0(theta...) + f.cfield * I
end

struct ParametricUnravelingHamiltonian{PJO<:ParametricUnravelingJumpOperator,F<:Function} <: Function
    H0_par::F
    Ls_par::Vector{PJO}
end

function adjointdifference(L::PJO, theta...) where {PJO<:ParametricUnravelingJumpOperator}
    return conj(L.cfield) * L.L0(theta...) - L.cfield * adjoint(L.L0(theta...))
end

function productwithadjoint(L::PJO, theta...) where {PJO<:ParametricUnravelingJumpOperator}
    return adjoint(L(theta...)) * L(theta...)
end

function (f::ParametricUnravelingHamiltonian)(theta...)
    return f.H0_par(theta...) - 0.5im * sum(L -> adjointdifference(L, theta...), f.Ls_par)
end

struct ParametricUnravelingEffectiveHamiltonian{PH<:ParametricUnravelingHamiltonian} <: Function
    H_par::PH
end

function (f::ParametricUnravelingEffectiveHamiltonian)(theta...)
    return f.H_par(theta...) - 0.5im * sum(L -> productwithadjoint(L, theta...), f.H_par.Ls_par)
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

function obtain_parametric_unraveling_operators(L0s::Union{Vector{Function},Vector{FL}}, H0::F, alpha::Vector{T1}) where {T1<:Complex,F<:Function,FL<:Function}
    Ls_par = [ParametricUnravelingJumpOperator(L0s[k], alpha[k]) for k in eachindex(alpha)]
    H_par = ParametricUnravelingHamiltonian(H0, Ls_par)
    return Ls_par, H_par, ParametricUnravelingEffectiveHamiltonian(H_par)
end
