### Functions for passing from one unraveling to the next one

struct ParametricUnravelingJumpOperator{F<:Function,T1<:Complex,T2<:Real} <: Function
    L0::F
    phase::T2
    cfield::T1
end

function (f::ParametricUnravelingJumpOperator)(theta...)
    return exp(1.0im * f.phase) * f.L0(theta...) + f.cfield * I
end

struct ParametricUnravelingHamiltonian{PJO<:ParametricUnravelingJumpOperator,F<:Function} <: Function
    H0_par::F
    Ls_par::Vector{PJO}
end

function adjointdifference(L::PJO, theta...) where {PJO<:ParametricUnravelingJumpOperator}
    return conj(L.cfield) * exp(1.0im * L.phase) * L.L0(theta...) - L.cfield * exp(-1.0im * L.phase) * adjoint(L.L0(theta...))
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

function isometric_mixing_i(Ls::Vector{TJ}, Ti::Vector{T1}, nlevels::T3) where {T1<:Complex,TJ<:Function,T3<:Int}
    f = let Ls = Ls, Ti = Ti, nlevels = nlevels
        (x...) -> begin
            # Lsnew_i = zeros(eltype(Ti), nlevels, nlevels)
            Lsnew_i = zero(Ti[1] * Ls[1](x...))
            for j in eachindex(Ls)
                Lsnew_i .+= Ti[j] * Ls[j](x...)
            end
            return Lsnew_i
        end
    end
    return f
end

function isometric_mixing(Ls::Vector{TJ}, Ti::Matrix{T1}, nlevels::T3, nchannels::T3) where {T1<:Complex,TJ<:Function,T3<:Int}
    return [isometric_mixing_i(Ls, Ti[i, :], nlevels) for i in 1:nchannels]
end

function add_cfield_i(L::TJ, alpha_i::T1, nlevels::T3) where {T1<:Complex,T3<:Int,TJ<:Function}
    f = let L = L, alpha_i = alpha_i, nlevels = nlevels
        (x...) -> L(x...) + alpha_i * Matrix{T1}(I, nlevels, nlevels)
    end
    return f
end

function add_cfields(Ls::Vector{TJ}, alpha::Vector{T1}, nlevels::T3) where {T1<:Complex,T3<:Int,TJ<:Function}
    # nchannels0 = length(Ls)
    return [add_cfield_i(Ls[i], alpha[i], nlevels) for i in eachindex(Ls)]
end

function get_LL(L::TJ) where {TJ<:Function}
    f = let L = L
        (x...) -> adjoint(L(x...)) * L(x...)
    end
    return f
end

function get_J(Ls::Vector{TJ}, nlevels::T3) where {TJ<:Function,T3<:Int}
    f = let Ls = Ls, nlevels = nlevels
        (x...) -> begin
            J = zeros(ComplexF64, nlevels, nlevels)
            for i in eachindex(Ls)
                J .+= get_LL(Ls[i])(x...)
            end
            return J
        end
    end
    return f
end

function get_Heff(H::TH, Ls::Vector{TJ}, nlevels::T3) where {TH<:Function,TJ<:Function,T3<:Int}
    f = let H = H, nlevels = nlevels
        (x...) -> H(x...) - 0.5im * get_J(Ls, nlevels)(x...)
    end
    return f
end


function get_cfield_hamiltonian_correctionterm_i(L::TJ, alpha_i::T1) where {TJ<:Function,T1<:Complex}
    f = let alpha_i = alpha_i, L = L
        (x...) -> conj(alpha_i) * L(x...) - alpha_i * adjoint(L(x...))
    end
    return f
end

function get_cfield_hamiltonian_correctionterm(Ls::Vector{TJ}, alpha::Vector{T1}, nlevels::T3) where {TJ<:Function,T1<:Complex,T3<:Int}
    f = let Ls = Ls, alpha = alpha, nlevels = nlevels
        (x...) -> begin
            aux = zeros(ComplexF64, nlevels, nlevels)
            # nchannels = length(Ls)
            for i in eachindex(Ls)
                aux .+= get_cfield_hamiltonian_correctionterm_i(Ls[i], alpha[i])(x...)
            end
            return aux
        end
    end
    return f
end


function add_cfield_hamiltonian_correctionterm(H::TH, Ls::Vector{TJ}, alpha::Vector{T1}, nlevels::T3) where {TH<:Function,TJ<:Function,T1<:Complex,T3<:Int}
    f = let H = H, Ls = Ls, alpha = alpha, nlevels = nlevels
        (x...) -> H(x...) - 0.5im * get_cfield_hamiltonian_correctionterm(Ls, alpha, nlevels)(x...)
    end
    return f
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
