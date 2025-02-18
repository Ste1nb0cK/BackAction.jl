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
- `Cfields::Vector{ComplexF64}`: Coherent fields
- `T::Matrix{ComplexF64}`: Unitary mixing matrix

# Constructor
To create an instance it's enough to provide the hamiltonian and the jump operators in a vector.
`System(H::Matrix{ComplexF64}, Ls::Vector{Matrix{ComplexF64}})`

"""
struct System
    NLEVELS::Int64 # Number of levels of the system
    NCHANNELS::Int64 # Number of jump channels
    H::Matrix{ComplexF64} # Hamiltonian
    Ls::Vector{Matrix{ComplexF64}} # List of jump operators
    LLs::Vector{Matrix{ComplexF64}} # List of L^\daggerL
    J::Matrix{ComplexF64} # Sum of Jump operators
    Heff::Matrix{ComplexF64} # Effective Hamiltonian
    @doc "
         Inner Constructor of `System` struct.
         # Arguments:
         `H::Matrix{ComplexF64}`
         `Ls::Vector{Matrix{ComplexF64}}`
"
   function System(H::Matrix{ComplexF64}, Ls::Vector{Matrix{ComplexF64}})
        NLEVELS = size(H)[1]
        NCHANNELS = size(Ls)[1] # Number of jump channels
        J = zeros(ComplexF64, NLEVELS, NLEVELS)
        LLs = Vector{Matrix{ComplexF64}}(undef, NCHANNELS)
        for k in 1:NCHANNELS
            product = adjoint(Ls[k])*Ls[k]
            J = J + product
            LLs[k] = product
        end
        CurvyLs = Vector{Function}(undef, NCHANNELS)
       He = H - 0.5im*J
       new(NLEVELS, NCHANNELS, H, Ls, LLs, J, He)
    end
end
Base.show(io::IO, s::System) = print(io,
    "System(NLEVELS=$(s.NLEVELS)\nNCHANNELS=$(s.NCHANNELS)\nH=$(s.H)\nLs=$(s.Ls)\nJ=$(s.J))\nHeff=$(s.Heff))")


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
struct SimulParameters
    psi0::Array{ComplexF64}
    nsamples::Int64 # Number of samples in the finegrid
    seed::Int64 # seed
    ntraj::Int64 # Number of trajectories
    multiplier::Float64 # Multiplier to use in the fine grid
    tf::Float64 # Final time
    dt::Float64 # time step for the finegrid
    eps::Float64 # Tolerance for passing WTD normalziation
    @doc "Inner constructor of `SimulParameters` SimulParameters(psi0::Vector{ComplexF64}, tf::Float64,
        s::Int64, ntraj::Int64, nsamples::Int64=10000, m::Float64=10.0,
                             eps::Float64=1e-3)"
    function SimulParameters(psi0::Array{ComplexF64}, tf::Float64,
        s::Int64, ntraj::Int64, nsamples::Int64=10000, m::Float64=10.0,
                             eps::Float64=1e-3)
        deltat = m*tf/nsamples
        new(psi0, nsamples, s, ntraj, m, tf, deltat, eps)
    end
end
Base.show(io::IO, s::SimulParameters) = print(io,
"SimulParameters(psi0=$(s.psi0)\nnsamples=$(s.nsamples)\nseed=$(s.seed)\nntraj=$(s.ntraj))\nmultiplier=$(s.multiplier)\ntf=$(s.tf)\ndt=$(s.dt)\neps=$(s.eps))")


