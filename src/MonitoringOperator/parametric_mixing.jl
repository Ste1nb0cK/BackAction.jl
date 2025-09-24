### Functions for passing from one unraveling to the next one

struct ParametricUnravelingJumpOperator{F<:Function,T1<:Complex,T2<:Real} <: Function
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


function obtain_parametric_unraveling_operators(L0::F1, H0::F2, T::Matrix{T1}, alpha::Vector{T1}, nlevels::T3) where {T1<:Complex,T3<:Int,F1<:Function,F2<:Function}
    # First do the unitary mixing 
    Ls_mixed = isometric_mixing([L0], T, nlevels, size(T)[1])
    H = add_cfield_hamiltonian_correctionterm(H0, Ls_mixed, alpha, nlevels)
    Ls_mixed_cfield = add_cfields(Ls_mixed, alpha, nlevels)
    He = get_Heff(H, Ls_mixed_cfield, nlevels)
    return Ls_mixed_cfield, H, He
end
