export HeffEvolution
struct HeffEvolution #{T1<:Complex}
    Heff::Matrix{ComplexF64}
end

function Base.deepcopy_internal(h::HeffEvolution, dict::IdDict) #where {T1<:Complex}
    return HeffEvolution(deepcopy_internal(h.Heff, dict))
end

function (Heff::HeffEvolution)(du::Vector{T1}, u::Vector{T1}, p, t) where {T1<:Complex}
    du .= -1im * getfield(Heff, :Heff) * u
end


