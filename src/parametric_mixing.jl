### Functions for passing from one unraveling to the next one
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


