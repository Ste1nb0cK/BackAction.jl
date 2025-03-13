export  System, SimulParameters, DetectionClick, Trajectory

################# SYSTEM #######################################################
"""

    System(
    NLEVELS::Int64, NCHANNELS::Int64, H::Matrix{ComplexF64}
    Ls::Vector{Matrix{ComplexF64}}, J::Matrix{ComplexF64},
    Heff::Matrix{ComplexF64})

A `struct` that characterizes the dynamics via specification of
 the jump and hamiltonian operators.

# Fields
- `NLEVELS::Int64`: Number of levels of the system
- `NCHANNELS::Int64`: Number of jump channels
- `H::Matrix{ComplexF64}`: Hamiltonian
- `Ls::Vector{Matrix{ComplexF64}}`: List of jump operators
- `J::Matrix{ComplexF64}`: Sum of all the ``L_k^{*}L_k``
- `Heff::Matrix{ComplexF64}`: Effective Hamiltonian

# Constructor
To create an instance it's enough to provide the hamiltonian and the jump operators in a vector.
`System(H::Matrix{ComplexF64}, Ls::Vector{Matrix{ComplexF64}})`

"""
struct System{T1<:Complex}
    NLEVELS::Int64 # Number of levels of the system
    NCHANNELS::Int64 # Number of jump channels
    H::Matrix{T1} # Hamiltonian
    Ls::Vector{Matrix{T1}} # List of jump operators
    LLs::Vector{Matrix{T1}} # List of L^\daggerL
    J::Matrix{T1} # Sum of Jump operators
    Heff::Matrix{T1} # Effective Hamiltonian

    @doc "
         Inner Constructor of `System` struct.
         # Arguments:
         `H::Matrix{ComplexF64}`
         `Ls::Vector{Matrix{ComplexF64}}`
"
   function System(H::Matrix{T1}, Ls::Vector{Matrix{T1}}) where T1<:Complex
        NLEVELS = size(H)[1]
        NCHANNELS = size(Ls)[1] # Number of jump channels
        J = zeros(T1, NLEVELS, NLEVELS)
        LLs = Vector{Matrix{T1}}(undef, NCHANNELS)
        for k in 1:NCHANNELS
            product = adjoint(Ls[k])*Ls[k]
            J = J + product
            LLs[k] = product
        end
       He = H - 0.5im*J
       new{T1}(NLEVELS, NCHANNELS, H, Ls, LLs, J, He)
    end
    "
        Constructor that allows specifying the alphas and the T's.
        The dimension of T must be Nxk, where k=lenght(Ls) but N might be arbitrary.
        It is expected for T to be unitary
    "
    function  System(H::Matrix{T1}, Ls::Vector{Matrix{T1}},
                     T::Matrix{T1}, alphas::Vector{T1}) where T1<:Complex
        NLEVELS = size(H)[1]
        NCHANNELS = size(T)[1] # Number of jump channels
        k = length(Ls) # number of original jump operators
        # To set th
        J = zeros(T1, NLEVELS, NLEVELS)
        H_ = copy(H)
        # unitary mixing
        Lprimes = [sum(T[i, j] * Ls[j] for j in 1:k) for i in 1:NCHANNELS]
        # Now add the fields and update the hamiltonian
        for i in 1:NCHANNELS
            Lprimes[i] = Lprimes[i] + alphas[i]*I
            H_ = H_ - 0.5im*(conj(alphas[i])*Lprimes[i] -alphas[i]*adjoint(Lprimes[i])  )
        end

        #  set the effective hamiltonian
        LLs = Vector{Matrix{T1}}(undef, NCHANNELS)
        J = zeros(T1, NLEVELS, NLEVELS)
        for k in 1:NCHANNELS
            product = adjoint(Lprimes[k])*Lprimes[k]
            J = J + product
            LLs[k] = product
        end
        He = H_ - 0.5im*J
        new{T1}(NLEVELS, NCHANNELS, H_, Lprimes, LLs, J, He)
    end

end
Base.show(io::IO, s::System) = print(io,
    "System(NLEVELS=$(s.NLEVELS)\nNCHANNELS=$(s.NCHANNELS)\nH=$(s.H)\nLs=$(s.Ls)\nJ=$(s.J))\nHeff=$(s.Heff))")
function Base.deepcopy(sys::System{T}) where {T<:ComplexF64}
    return System(
        deepcopy(sys.H),
        [deepcopy(L) for L in sys.Ls]  # Ensures elements are fully copied
    )
end


################ Data Point ################
"""

    DetectionClick(time::Float64, label::Int64)
`Inmutable struct` that represents the clicks by the time waited to see the click and the
label of the channel in which it occured.

# Fields
- `time::Float64`: Waiting time
- `label::Int64`: Label of the channel of the click
 """
struct DetectionClick
    time::Float64
    label::Int64
end

@doc "Alias for `Vector{DetectionClick}`"
const Trajectory = Vector{DetectionClick}
################# SIMULATION PARAMETERS ########################################
"""

    SimulParameters(
        psi0::Array{ComplexF64}, nsamples::Int64, seed::Int64,
                ntraj::Int64, multiplier::Float64, tf::Float64,
                dt::Float64, eps::Float64)


A `struct` containing all the necessary information for running the
the simulation.

# Fields
- `psi0::Array{ComplexF64}`: Initial state, mixed or pure.
- `nsamples::Int64`: Number of samples in the finegrid
- `seed::Int64`: seed
- `ntraj::Int64`: Number of trajectories
- `multiplier::Float64`: Multiplier to use in the fine grid
- `tf::Float64`: Final time
- `dt::Float64`: time step for the finegrid
- `eps::Float64`: Tolerance for passing WTD normalziation

# Constructor
To create an instance it's enough to provide initial state, final time, seed and
number of trajectories. Unless given nsamples, multiplier and eps use default values.
`SimulParameters(psi0::Vector{ComplexF64}, tf::Float64,
        s::Int64, ntraj::Int64, nsamples::Int64=10000, m::Float64=10.0,
                             eps::Float64=1e-3)`
# About the multiplier
For the Gillipsie algorithm to work it's key to have a grid that's capable of
resolving the statistical details of the WTD, this grid is taken in the interval
`(0, tf*multiplier)`.
"""
struct SimulParameters{T1<:Complex, T2<:Real, T3<:Int}
    psi0::Vector{T1}
    nsamples::T3 # Number of samples in the finegrid
    seed::T3 # seed
    ntraj::T3 # Number of trajectories
    multiplier::T2 # Multiplier to use in the fine grid
    tf::T2 # Final time
    dt::T2 # time step for the finegrid
    eps::T2 # Tolerance for passing WTD normalziation
    @doc "Inner constructor of `SimulParameters` SimulParameters(psi0::Vector{ComplexF64}, tf::Float64,
        s::Int64, ntraj::Int64, nsamples::Int64=10000, m::Float64=10.0,
                             eps::Float64=1e-3)"
    function SimulParameters(psi0::Union{Vector{T1}, Matrix{T1}}, tf::T2,
                             s::T3, ntraj::T3, nsamples::T3=10000, m::T2=10.0,
                             eps::T2=1e-3) where {T1<:Complex, T2<:Real, T3<:Int}
        deltat = m*tf/nsamples
        new{T1, T2, T3}(psi0, nsamples, s, ntraj, m, tf, deltat, eps)
    end
end
Base.show(io::IO, s::SimulParameters) = print(io,
"SimulParameters(psi0=$(s.psi0)\nnsamples=$(s.nsamples)\nseed=$(s.seed)\nntraj=$(s.ntraj))\nmultiplier=$(s.multiplier)\ntf=$(s.tf)\ndt=$(s.dt)\neps=$(s.eps))")


