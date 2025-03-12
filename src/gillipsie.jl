# DESCRIPTION: functions for the Quantum Gillipsie Method
############# Precomputation ######################
"""

```
setVs!(sys::System, nsamples::Int64, ts::Vector{Float64}, Vs::Array{ComplexF64})
````

Calculate the matrix exponentials ``\\mathrm{exp}(-iH_e t_s)`` for each ``t_s`` in the vector `ts`,
where ``H_e`` is the effective hamiltonian of the system `sys`, and the results are written
in `Vs`, which is an array of dimensions `(sys.NLEVELS, sys.NLEVELS, nsamples)`. To access
the exponential corresponding to `ts[k]` you would do `Vs[:, ;, k]`.
"""
function setVs!(sys::System{T1,T3}, nsamples::T3, ts::Vector{T2}, Vs::Array{T1}) where {T1<:Complex,T2<:Real,T3<:Int}
    tmp = copy(sys.Heff)
    @inbounds @simd for k in 1:nsamples
        Vs[:, :, k] = exp(-1im * ts[k] * sys.Heff)
    end
end


"""

```
setQs!(sys::System, nsamples::Int64,ts::Vector{Float64}, Qs::Array{ComplexF64}, Vs::Array{ComplexF64})
```

Calculate the matrix producs ``VJV^\\dagger`` for each ``V``
in  `Vs`, where  ``J=\\sum_k L_{k}^\\dagger L_k`` is  `sys.J`, and the results are written in `Qs` which is an array of dimensions
 `(sys.NLEVELS, sys.NLEVELS, nsamples)`. To access the product corresponding to `ts[k]` you would do `Qs[:, ;, k]`.
"""
function setQs!(sys::System{T1,T3}, nsamples::T3,
    ts::Vector{T2}, Qs::Array{T1}, Vs::Array{T1}) where {T1<:Complex,T2<:Real,T3<:Int}
    @inbounds @simd for k in 1:nsamples
        Qs[:, :, k] = Vs[:, :, k] * sys.J * adjoint(Vs[:, :, k])
    end
end


"""

```
precompute!(sys::System, nsamples::Int64, ts::Vector{Float64}, Qs::Array{ComplexF64}, Vs::Array{ComplexF64})
```

Precompute the ``Q(t_s)`` and ``V(t_s)`` necessary for running the *Quantum Gillipsie Algorithm*
 [radaelli2024gillespie](@cite) with the time grid `ts`. The result is written in `Qs` and `Vs`.
Under the hood, this simply calls `setVs!` and `setQs!`.
"""
function precompute!(sys::System{T1,T3}, nsamples::T3,
    ts::Vector{T2}, Qs::Array{T1}, Vs::Array{T1}) where {T1<:Complex,T2<:Real,T3<:Int}

    setVs!(sys, nsamples, ts, Vs)
    setQs!(sys, nsamples, ts, Qs, Vs)
end
######## Evolution Stage
"""

```
gillipsiestep_returntau!(sys::System, params::SimulParameters, W::Vector{Float64},
                        P::Vector{Float64}, Vs::Array{ComplexF64}, ts::Vector{Float64},
                        t::Float64, psi::VecOrMat{ComplexF64}, traj::Trajectory )

```
Do a step of the Gillipsie algorithm, updating the state and the weights, and returning the
obtained jump time. In this version the time jump sampling is done by calling `StatsBase`.
"""
function gillipsiestep_returntau!(sys::System{T1,T3}, params::SimulParameters{T1,T2,T3}, W::Vector{T2},
    P::Vector{T2}, Vs::Array{T1}, ts::Vector{T2},
    t::T2, psi::VecOrMat{T1}, traj::Trajectory{T2,T3}) where {T1<:Complex,T2<:Real,T3<:Int}
    #Sample jump time and  move state to pre-jump state
    tau_index = StatsBase.sample(1:params.nsamples, StatsBase.weights(W))
    prejumpupdate!(Vs[:, :, tau_index], psi)
    # Sample jump channel
    calculatechannelweights!(P, psi, sys)
    channel = StatsBase.sample(1:sys.NCHANNELS, StatsBase.weights(P))
    # State update
    postjumpupdate!(sys.Ls[channel], psi)
    tau = ts[tau_index]
    push!(traj, DetectionClick(tau, channel))
    return tau

end


"""

```
gillipsiestep_returntau!(sys::System, params::SimulParameters, W::Vector{Float64},
                        P::Vector{Float64}, Qs::Array{ComplexF64}, Vs::Array{ComplexF64},
 ts::Vector{Float64},
                        t::Float64, psi::VecOrMat{ComplexF64}, traj::Trajectory )

```

Do a step of the Gillipsie algorithm, updating the state and the weights, and returning the
obtained jump time. In this version the time is extracted using inversion sampling instead of
calling `StatsBase`.
"""
function gillipsiestep_returntau!(sys::System{T1,T3}, params::SimulParameters{T1,T2,T3}, W::Vector{T2},
    P::Vector{T2}, Vs::Array{T1}, ts::Vector{T2},
    t::T2, psi::VecOrMat{T1}, traj::Trajectory{T2,T3}, Qs::Array{T1}) where {T1<:Complex,T2<:Real,T3<:Int}
    tau_index = sampletauindex!(W, Qs, psi, params)
    # in case the last index was at the last index, return already to avoid errors with dark states
    if tau_index == params.nsamples
        # push!(traj, DetectionClick(ts[tau_index], channel))
        return tau_index
    end
    prejumpupdate!(Vs[:, :, tau_index], psi)
    # Sample jump channel
    calculatechannelweights!(P, psi, sys)
    channel = StatsBase.sample(1:sys.NCHANNELS, StatsBase.weights(P))
    # State update
    postjumpupdate!(sys.Ls[channel], psi)
    tau = ts[tau_index]
    push!(traj, DetectionClick(tau, channel))
    return tau

end

############# Single Trajectory Routine ######################
"""
```
run_singletrajectory_gillipsie(sys::System, params::SimulParameters,
    W::Vector{Float64}, P::Vector{Float64}, ts::Vector{Float64},
    Qs::Array{ComplexF64}, Vs::Array{ComplexF64}; seed::Int64 = 1)
```

Sample a jump trajectory for the system `sys` using the *Quantum Gillipsie Algorithm* [radaelli2024gillespie](@cite).

# Positional Arguments
- `sys::System`: the system from which the trajectory is obtained.
- `params::SimulParameters`:  specifies the number of points
                             in the grid, the initial state and the tolerance for the dark state test.
- `W::Vector{Float64}`: to store the probabilities of the WTDs used at each step
- `P::Vector{Float64}`: to store the probabilites of jumps to each channel used at each step
- `ts::Vector{Float64}`: the fine grid used to sample from the WTD
- `Qs::Array{ComplexF64}`: the precomputed matrices from which the WTD weights are calculated
- `Vs::Array{ComplexF64}`:  the precomputed exponentials that evolve the state from jump to jump.

# Keyword Arguments
- `seed::Int64 = 1`: the seed of the sample. It does not need to coincide with that in `params`

# Returns
- `traj::Trajectory`: vector with the obtained detection clicks.
"""
function run_singletrajectory_gillipsie(sys::System{T1,T3}, params::SimulParameters{T1,T2,T3},
    W::Vector{T2}, P::Vector{T2}, ts::Vector{T2},
    Qs::Array{T1}, Vs::Array{T1}; seed::T3=1) where {T1<:Complex,T2<:Real,T3<:Int}
    Random.seed!(seed)
    channel = 0
    traj = Vector{DetectionClick{T2,T3}}()
    psi = copy(params.psi0)
    t::Float64 = 0
    channel = 0
    # Run the trajectory
    # calculatewtdweights!(W, Qs, psi, params)
    while t < params.tf
        t = t + gillipsiestep_returntau!(sys, params, W, P, Vs, ts, t, psi, traj, Qs)
    end
    return traj
end


"""
```
run_singletrajectory_gillipsie_renewal(sys::System, params::SimulParameters,
    W::Vector{Float64}, W0::Vector{Float64}, P::Vector{Float64}, ts::Vector{Float64},
    Qs::Array{ComplexF64}, Vs::Array{ComplexF64}, psireset::VecOrMat{ComplexF64}; seed::Int64 = 1)
```

Same as `run_singletrajectory_gillipsie` but uses `psireset` to optimize the jump time sampling
by exploiting the process is renewal. Additionally, `W0` must be provided to sample the
first jump from the initial state, which may not coincide with `psireset`.
"""
function run_singletrajectory_gillipsie_renewal(sys::System{T1,T3}, params::SimulParameters{T1,T2,T3},
    W::Vector{T2}, W0::Vector{T2}, P::Vector{T2}, ts::Vector{T2},
    Qs::Array{T1}, Vs::Array{T1}, psireset::VecOrMat{T1};
    seed::T3=1) where {T1<:Complex,T2<:Real,T3<:Int}
    Random.seed!(seed)
    channel = 0
    traj = Vector{DetectionClick{T2,T3}}()
    psi = copy(params.psi0)
    t::T2 = 0
    channel = 0
    # For the first jump use the WTD of the initial state
    t = gillipsiestep_returntau!(sys, params, W0, P, Vs, ts, t, psi, traj)
    # For the rest use the WTD of psireset
    while t < params.tf
        t = t + gillipsiestep_returntau!(sys, params, W, P, Vs, ts, t, psireset, traj)
    end
    return traj
end
