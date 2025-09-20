export System, SimulParameters, DetectionClick, Trajectory

################# SYSTEM #######################################################
"""

    System(
    NLEVELS::T3, NCHANNELS::T3, H::Matrix{T1}
    Ls::Array{T1,3}, J::Matrix{T1},
    Heff::Matrix{T1}) where T1<:Complex,T3<:Int

A `struct` that characterizes the dynamics via specification of
 the jump and hamiltonian operators.

# Fields
- `NLEVELS::T3`: Number of levels of the system
- `NCHANNELS::T3`: Number of jump channels
- `H::Matrix{T1}`: Hamiltonian
- `Ls::Array{T1,3}`: List of jump operators
- `LLs::Array{T1,3}`: List of the products ``L_k^{*}L_k`` of the jump operators
- `J::Matrix{T1}`: Sum of all the ``L_k^{*}L_k``
- `Heff::Matrix{T1}`: Effective Hamiltonian

# Constructors
To create an instance it's enough to provide the hamiltonian, and the jump operators in an array where 
`Ls[:, :, i]` is the i-th jump operator.
`System(H::Matrix{T1}, Ls::Array{T1,3} where T1<:Complex`

"""
struct System{T1<:Complex,T3<:Int}
    NLEVELS::T3 # Number of levels of the system
    NCHANNELS::T3 # Number of jump channels
    H::Matrix{T1} # Hamiltonian
    Ls::Array{T1,3} # array of jump operators
    LLs::Array{T1,3} # List of L^\daggerL
    J::Matrix{T1} # Sum of Jump operators
    Heff::Matrix{T1} # Effective Hamiltonian

    function System(H::Matrix{T1}, Ls::Array{T1}, nlevels::T3, nchannels::T3) where {T1<:Complex,T3<:Int}
        J = zeros(T1, nlevels, nlevels)
        tmp = zeros(T1, nlevels, nlevels)
        LLs = zeros(T1, nlevels, nlevels, nchannels)
        for k in 1:nchannels
            mul!(tmp, adjoint(Ls[:, :, k]), Ls[:, :, k])
            LLs[:, :, k] .= tmp
            J .+= tmp
        end
        new{T1,T3}(nlevels, nchannels, H, Ls, LLs, J, H - 0.5im * J)
    end
    "   Construct a 'System' struct with specified cfields and isometric mixing. 
        The constructor first performs the isometric mixing of the provided jump
        operators according to 'T' and then adds the cfields from 'alphas'. 
    "
    function System(H::Matrix{T1}, Ls::Array{T1},
        T::Matrix{T1}, alphas::Vector{T1}) where {T1<:Complex}
        NLEVELS = size(H, 1)
        NCHANNELS = size(T, 1) # Number of jump channels
        nchannels0 = size(Ls, 3) # number of original jump operators
        # To set th
        J = zeros(T1, NLEVELS, NLEVELS)
        # unitary mixing
        Lprimes = zeros(T1, NLEVELS, NLEVELS, NCHANNELS)
        for i in 1:NCHANNELS
            for j in 1:nchannels0
                Lprimes[:, :, i] .+= T[i, j] * Ls[:, :, j]
            end
        end
        # Now update the hamiltonian and add the fields
        for i in 1:NCHANNELS
            H .+= -0.5im * (conj(alphas[i]) * Lprimes[:, :, i] - alphas[i] * adjoint(Lprimes[:, :, i]))
            Lprimes[:, :, i] = Lprimes[:, :, i] + alphas[i] * I
        end
        #  set the effective hamiltonian
        LLs = Array{T1}(undef, NLEVELS, NLEVELS, NCHANNELS)
        J = zeros(T1, NLEVELS, NLEVELS)
        for k in 1:NCHANNELS
            LLs[:, :, k] .= adjoint(Lprimes[:, :, k]) * Lprimes[:, :, k]
            J .+= LLs[:, :, k]
        end
        new{T1,typeof(NLEVELS)}(NLEVELS, NCHANNELS, H, Lprimes, LLs, J, H - 0.5im * J)
    end

end
Base.show(io::IO, s::System) = print(io,
    "System(NLEVELS=$(s.NLEVELS)\nNCHANNELS=$(s.NCHANNELS)\nH=$(s.H)\nLs=$(s.Ls)\nJ=$(s.J))\nHeff=$(s.Heff))")
# Ensures elements are fully copied
import Base.deepcopy
function Base.deepcopy(sys::S) where {S<:System}
    return System(
        deepcopy(getfield(sys, :H)),
        deepcopy(getfield(sys, :Ls)),
        deepcopy(getfield(sys, :NLEVELS)),
        deepcopy(getfield(sys, :NCHANNELS))
    )
end


################ Data Point ################
"""

DetectionClick(time::T1, label::T2){T1<:Real, T2<:Int}
`Inmutable struct` that represents the clicks by the time waited to see the click and the
label of the channel in which it occured.

# Fields
- `time::T2`: Waiting time
- `label::T3`: Label of the channel of the click
 """
struct DetectionClick{T2<:Real,T3<:Int}
    time::T2
    label::T3
end

@doc "Alias for `Vector{DetectionClick}`"
const Trajectory{T2,T3} = Vector{DetectionClick{T2,T3}}
################# SIMULATION PARAMETERS ########################################
"""

    SimulParameters(
        psi0::Array{ComplexF64}, nsamples::Int64, seed::Int64,
                ntraj::Int64, multiplier::Float64, tf::Float64,
                dt::Float64, eps::Float64)


A `struct` containing all the necessary information for running the
the simulation.

# Fields
- `psi0::Array{T1}`: Initial state, mixed or pure.
- `nsamples::T3`: Number of samples in the finegrid
- `seed::T3`: seed
- `ntraj::T3`: Number of trajectories
- `multiplier::T2`: Multiplier to use in the fine grid
- `tf::T2`: Final time
- `dt::T2`: time step for the finegrid
- `eps::T2`: Tolerance for passing WTD normalziation
where `T1<:Complex, T2<:Real, T3<:Int`

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
struct SimulParameters{T1<:Complex,T2<:Real,T3<:Int}
    psi0::Vector{T1}
    nsamples::T3 # Number of samples in the finegrid
    seed::T3 # seed
    ntraj::T3 # Number of trajectories
    multiplier::T2 # Multiplier to use in the fine grid
    tf::T2 # Final time
    dt::T2 # time step for the finegrid
    eps::T2 # Tolerance for passing WTD normalziation
    @doc "Inner constructor of `SimulParameters` SimulParameters(psi0::Vector{T1}, tf::T2,
        s::T3, ntraj::T3, nsamples::T3=10000, m::T2=10.0,
                             eps::T2=1e-3)"
    function SimulParameters(psi0::Vector{T1}, tf::T2,
        s::T3, ntraj::T3, nsamples::T3=10000, m::T2=10.0,
        eps::T2=1e-3) where {T1<:Complex,T2<:Real,T3<:Int}
        deltat = m * tf / nsamples
        new{T1,T2,T3}(psi0, nsamples, s, ntraj, m, tf, deltat, eps)
    end
end
Base.show(io::IO, s::SimulParameters) = print(io,
    "SimulParameters(psi0=$(s.psi0)\nnsamples=$(s.nsamples)\nseed=$(s.seed)\nntraj=$(s.ntraj))\nmultiplier=$(s.multiplier)\ntf=$(s.tf)\ndt=$(s.dt)\neps=$(s.eps))")

