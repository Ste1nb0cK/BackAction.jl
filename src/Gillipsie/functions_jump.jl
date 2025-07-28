# Description: generic functions used by the jump trajectory methods

export states_att, states_atjumps

"""

```
fixlastindex(array::Array{T1}, k::T3)
```

Return a `SubArray` of `array`, defined by fixing the last index to `k`.
# Example
```jldoctest
using BackAction
arr = [[1+1.0im, 2] [3, 4]]
BackAction.fixlastindex(arr, 2)
# output
2-element view(::Matrix{T1}, :, 2) with eltype T1:
 3.0 + 0.0im
 4.0 + 0.0im
```

"""
function fixlastindex(array::Array{T1}, k::T3) where {T1<:Complex,T3<:Int}
    indices = ntuple(d -> d == ndims(array) ? k : Colon(), ndims(array))
    # Add a singleton of dimension 1
    return view(array, indices...)
end


"""

```
calculatewtdweights!(W::Array{T2}, Qs::Array{T1}, psi::Vector{T1},
                                        params::SimulParameters{T1, T2, T3})
```

Calculate the discretized *Waiting Time Distribution* for a pure state ``|\\psi\\rangle`` i.e.
``\\langle\\psi|Q(t_s)\\psi\\rangle``, and writes it at `W`. This is done using `LinearAlgebra`'s `dot`,
and usually is the thing in which `run_singletrajectory` spends most of the time since `params.nsamples`
is typically in the thousands.
"""
function calculatewtdweights!(W::Array{T2}, Qs::Array{T1}, psi::Vector{T1},
    params::SimulParameters{T1,T2,T3}) where {T1<:Complex,T2<:Real,T3<:Int}
    @inbounds @simd for k in 1:params.nsamples
        W[k] = real(dot(psi, Qs[:, :, k], psi)) # dot product without storing A*x. THIS IS THE KEY FOR SPEED
    end
end


"""

```
calculatewtdweights!(W::Array{T2}, Qs::Array{T1}, psi::Matrix{T1},
                                         params::SimulParameters{T1, T2, T3})
```

Calculate the discretized *Waiting Time Distribution* for a mixed state ``\\psi`` i.e.
``\\mathrm{Tr}(Q(t_s)\\psi)``, and writes it at `W`. This is done using `LinearAlgebra`'s `tr`,
and usually is the thing in which `run_singletrajectory` spends most of the time since `params.nsamples`
is typically in the thousands.
"""
function calculatewtdweights!(W::Array{T2}, Qs::Array{T1}, psi::Matrix{T1},
    params::SimulParameters{T1,T2,T3}) where {T1<:Complex,T2<:Real,T3<:Int}
    @inbounds @simd for k in 1:params.nsamples
        W[k] = real(tr(Qs[:, :, k] * psi))
    end
end


"""

```
calculatechannelweights!(P::Vector{T2}, psi::Vector{T1}, sys::System{T1, T3})
```

Calculate the probabilities for a pure state ``|\\psi\\rangle`` to jump to any of the given channels i.e.
``\\langle\\psi| L^\\dagger L|\\psi\\rangle`` for each jump operator ``L``, and writes it at `P`.
 This is done using the square of `LinearAlgebra`'s `norm`.
"""
function calculatechannelweights!(P::Vector{T2}, psi::Vector{T1},
    sys::System{T1,T3}) where {T1<:Complex,T2<:Real,T3<:Int}

    aux_P = real(dot(psi, sys.J * psi))
    @inbounds @simd for k in 1:sys.NCHANNELS
        P[k] = norm(sys.Ls[k] * psi)^2
    end
    P .= P / aux_P
end


"""

```
calculatechannelweights!(P::Vector{T2}, psi::Matrix{T1}, sys::System{T1, T3})
```

Calculate the probabilities for a mixed state ``\\psi`` to jump to any of the given channels i.e.
``\\mathrm{Tr}(L^\\dagger L\\psi)`` for each jump operator ``L``, and writes it at `P`.
 This is done using of `LinearAlgebra`'s `tr`.
"""
function calculatechannelweights!(P::Vector{T2}, psi::Matrix{T1},
    sys::System{T1,T3}) where {T1<:Complex,T2<:Real,T3<:Int}

    aux_P = real(dot(psi, sys.J * psi))
    @inbounds @simd for k in 1:sys.NCHANNELS
        P[k] = real(tr(sys.LLs[k] * psi))
    end
    P .= P / aux_P
end


"""

```
prejumpupdate!(V::Matrix{T1}, psi::Vector{T1}; normalize=false)
```

Do the pure state transformation ``|\\psi\\rangle\\to V|\\psi\\rangle`` modifying `psi` ,
if `normalize=true` it also normalizes the final state.
"""
function prejumpupdate!(V::Matrix{T1}, psi::Vector{T1}; normalize=false) where {T1<:Complex}
    psi .= V * psi
    if normalize
        psi .= psi / norm(psi)
    end
end


"""

```
prejumpupdate!(psi::Vector{T1}, V::Matrix{T1},
               psi0::Union{Vector{T1}, SubArray{T1}}; normalize=false)
```
Do the pure state transformation ``|\\psi_0\\rangle\\to V|\\psi_0\\rangle`` and store the
result in `psi`, if `normalize=true` it also normalizes the final state.
"""
function prejumpupdate!(psi::Vector{T1}, V::Matrix{T1},
    psi0::Union{Vector{T1},SubArray{T1}}; normalize=false) where {T1<:Complex}
    psi .= V * psi0
    if normalize
        psi .= psi / norm(psi)
    end
end


"""

```
prejumpupdate!(psi::Matrix{T1}, V::Matrix{T1},
               psi0::Union{Vector{T1}, SubArray{T1}}; normalize=false)
```
Do the mixed state transformation ``\\psi_0\\to V\\psi_0 V^\\dagger`` and store the
result in `psi`, if `normalize=true` it also normalizes the final state.
"""
function prejumpupdate!(psi::Matrix{T1}, V::Matrix{T1},
    psi0::Union{Matrix{T1},SubArray{T1}}; normalize=false) where {T1<:Complex}
    psi .= V * psi0 * adjoint(V)
    if normalize
        psi .= psi / tr(psi)
    end
end


"""

```
prejumpupdate!(V::Matrix{T1}, psi::Vector{T1}; normalize=false)
```

Do the mixed state transformation ``\\psi\\to V\\psi V^\\dagger`` modifying `psi` ,
if `normalize=true` it also normalizes the final state.
"""
function prejumpupdate!(V::Matrix{T1}, psi::Matrix{T1}; normalize=false) where {T1<:Complex}
    psi .= V * psi * adjoint(V)
    if normalize
        psi .= psi / tr(psi)
    end
end


"""

```
postjumpupdate!(L::Matrix{T1}, psi::Vector{T1}; normalize=true)
```

Do the pure state transformation ``|\\psi\\rangle\\to L|\\psi\\rangle`` modifying `psi` ,
if `normalize=true` it also normalizes the final state.
"""
function postjumpupdate!(L::Matrix{T1}, psi::Vector{T1}; normalize=true) where {T1<:Complex}
    psi .= L * psi # State without normalization
    if normalize
        psi .= psi / norm(psi)
    end
end


"""

```
postjumpupdate!(L::Matrix{T1}, psi::Vector{T1}; normalize=true)
```

Do the pure mixed state transformation ``\\psi\\to L\\psi L^\\dagger`` modifying `psi` ,
if `normalize=true` it also normalizes the final state.
"""
function postjumpupdate!(L::Matrix{T1}, psi::Matrix{T1}; normalize=true) where {T1<:Complex}
    psi .= L * psi * adjoint(L) # State without normalization
    if normalize
        psi .= psi / tr(psi)
    end
end

"""

```
samplejumptime!(W::Vector{T2}, Qs::Array{T1}, psi::VecOrMat{T1})
```

Sample a jump time index from the state `psi` (pure or mixed), modfying `W` to write on it.
The technique is inversion sampling
"""
function sampletauindex!(W::Vector{T2}, Qs::Array{T1}, psi::Vector{T1},
    params::SimulParameters{T1,T2,T3}) where {T1<:Complex,T2<:Real,T3<:Int}

    # First, sample a random number and divide by dt to avoid multiplying by dt the weights
    alpha = rand() / params.dt
    u = 0.0
    # now sum until alpha is exceded
    # println(psi)
    tau_index = 1
    while u < alpha && tau_index < params.nsamples
        u = u + real(dot(psi, Qs[:, :, tau_index], psi))
        tau_index = tau_index + 1
    end
    return tau_index
end

"""

```
samplejumptime!(W::Vector{T2}, Qs::Array{T1}, psi::VecOrMat{T1})
```

Sample a jump time index from the state `psi` (pure or mixed), modfying `W` to write on it.
The technique is inversion sampling
"""
function sampletauindex!(W::Vector{T2}, Qs::Array{T1}, psi::Matrix{T1},
    params::SimulParameters{T1,T2,T3}) where {T1<:Complex,T2<:Real,T3<:Int}
    # First, sample a random number and divide by dt to avoid multiplying by dt the weights
    alpha = rand() / params.dt
    u = 0.0
    # now sum until alpha is exceded
    tau_index = 1
    while u < alpha && tau_index < params.nsamples
        u = u + real(tr(Qs[:, :, tau_index] * psi))
        tau_index = tau_index + 1
    end
    return tau_index
end





"""
```
writestate!(states::array{complexf64}, psi::union{vector{complexf64},
                                        matrix{complexf64}}, counter::int64)
```
Writes `psi` in `states` at the subarray with the last index fixed at `counter`.
"""
function writestate!(states::Array{T1},
    psi::Union{Vector{T1},Matrix{T1}}, counter::T3) where {T1<:Complex,T3<:Int}
    fixlastindex(states, counter) .= psi
end


"""
```
states_atjumps(traj::Trajectory{T2, T3}, sys::System{T1, T3}, psi0::Union{Vector{T1},
               Matrix{T1}}; normalize::Bool=true)
```
Obtain the states at jumps of the trajectory given the initial state `psi0`, they
are (un)normalized if `normalize` is `true`(`false`). The return
is an `Array` of dimensions `(sys.NLEVELS, njumps)` if the initial state was pure,
and `(sys.NLEVELS, sys.NLEVELS, njumps)` if it was mixed; `njumps` is the number of
jumps in the trajectory. You would access the state at the k-th jump with something
like  `states_atjumps(traj, sys, psi0)[:, k]`.

In case `isempty(traj)=true` the returned array is also empty.

"""
function states_atjumps(traj::Vector{DetectionClick{T2,T3}}, sys::System{T1,T3},
    psi0::Union{Vector{T1},Matrix{T1}}; normalize::Bool=true) where {T1<:Complex,T2<:Real,T3<:Int}
    njumps = size(traj)[1]
    if isa(psi0, Vector{T1})
        states = Array{T1}(undef, sys.NLEVELS, njumps)
    elseif isa(psi0, Matrix{T1})
        states = Array{T1}(undef, sys.NLEVELS, sys.NLEVELS, njumps)
    end
    psi = copy(psi0)
    jump_counter = 1
    for click in traj
        prejumpupdate!(exp(-1im * (click.time) * sys.Heff), psi)
        postjumpupdate!(sys.Ls[click.label], psi; normalize=normalize)
        writestate!(states, psi, jump_counter)
        jump_counter = jump_counter + 1
    end
    return states
end

# This one works with the times and labels as vectors
function states_atjumps(jumptimes::Vector{T2}, labels::Vector{T3}, sys::System{T1,T3},
    psi0::Vector{T1}; normalize::Bool=true) where {T1<:Complex,T2<:Real,T3<:Int}
    # T3 = eltype(psi0)
    njumps = length(jumptimes)
    states = Array{T1}(undef, sys.NLEVELS, njumps)
    psi = copy(psi0)
    jump_counter = 1
    for k in 1:njumps
        prejumpupdate!(exp(-1im * (jumptimes[k]) * sys.Heff), psi)
        postjumpupdate!(sys.Ls[labels[k]], psi; normalize=normalize)
        writestate!(states, psi, jump_counter)
        jump_counter = jump_counter + 1
    end
    return states
end





"""
```
states_att(t_given::Vector{T2}, traj::Trajectory{T2, T3}, sys::System{T1, T3},
                       psi0::Union{Vector{T1}, Matrix{T1}};
                       normalize::Bool=true)
```
Provided the initial state  `psi0` obtain the states at the times in `t_given` on the trajectory,
they are (un)normalized if `normalize` is `true`(`false`).
 The return is an `Array` of dimensions `(sys.NLEVELS, ntimes)` if the initial state was pure
and `(sys.NLEVELS, sys.NLEVELS, ntimes)` if it was mixed; `ntimes` is the number of
times in `t_given`. In case `isempty(t_given)=true` the returned array is also empty.

"""
function states_att(t_given::Vector{T2}, traj::Trajectory{T2,T3}, sys::System{T1,T3},
    psi0::Union{Vector{T1},Matrix{T1}};
    normalize::Bool=true) where {T1<:Complex,T2<:Real,T3<:Int}
    # Special case: if the time array is empty, return an empty array
    if isempty(t_given)
        return Array{T1}(undef, 0, 0) # empty 2 dimensional array
    end
    psi = copy(psi0)
    ntimes = size(t_given)[1]
    jump_states = states_atjumps(traj, sys, psi0; normalize=normalize)
    njumps = size(traj)[1]
    t_ = 0
    counter = 1
    counter_c = 1
    # states = Array{T1}(undef, sys.NLEVELS, ntimes)
    if isa(psi0, Vector{T1})
        states = Array{T1}(undef, sys.NLEVELS, ntimes)
    elseif isa(psi0, Matrix{T1})
        states = Array{T1}(undef, sys.NLEVELS, sys.NLEVELS, ntimes)
    end
    # Edge case: if the trajectory is empty, evaluate exponentials and return
    if isempty(traj)
        while counter <= ntimes
            prejumpupdate!(psi, exp(-1im * (t_given[counter]) * sys.Heff), psi0;
                normalize=normalize)
            # fixlastindex(states, counter)
            writestate!(states, psi, counter)
            counter = counter + 1
            if counter > ntimes
                break
            end
        end
        return states
    end
    # All the states before the first jump can be handled like this:
    while (t_given[counter] < traj[counter_c].time) && (counter <= ntimes)
        prejumpupdate!(psi, exp(-1im * (t_given[counter]) * sys.Heff), psi0;
            normalize=normalize)
        writestate!(states, psi, counter)
        counter = counter + 1
        if counter > ntimes
            break
        end
    end
    t_ = t_ + traj[counter_c].time
    counter_c = counter_c + 1
    while (counter_c <= njumps) && (counter <= ntimes)
        timeclick = traj[counter_c].time
        while (t_ < t_given[counter] < t_ + timeclick) && (counter <= ntimes)
            prejumpupdate!(psi, exp(-1im * (t_given[counter] - t_) * sys.Heff),
                fixlastindex(jump_states, counter_c - 1); normalize=normalize)
            writestate!(states, psi, counter)
            counter = counter + 1
            if counter > ntimes
                break
            end
        end
        t_ = t_ + timeclick
        counter_c = counter_c + 1
    end

    while counter <= ntimes
        prejumpupdate!(psi, exp(-1im * (t_given[counter] - t_) * sys.Heff),
            fixlastindex(jump_states, njumps); normalize=normalize)
        writestate!(states, psi, counter)
        counter = counter + 1
    end
    return states
end

function states_att(t_given::Vector{T2}, jumptimes::Vector{T2}, labels::Vector{T3},
    sys::System{T1,T3}, psi0::Vector{T1}; normalize::Bool=true) where {T1<:Complex,T2<:Real,T3<:Int}
    # Special case: if the time array is empty, return an empty array
    if isempty(t_given)
        return Array{T1}(undef, 0, 0) # empty 2 dimensional array
    end
    psi = copy(psi0)
    ntimes = length(t_given)
    jump_states = states_atjumps(jumptimes, labels, sys, psi0; normalize=normalize)
    njumps = length(jumptimes)
    t_ = 0
    counter = 1
    counter_c = 1
    # states = Array{T1}(undef, sys.NLEVELS, ntimes)
    states = Array{T1}(undef, sys.NLEVELS, ntimes)
    # Edge case: if the trajectory is empty, evaluate exponentials and return
    if isempty(labels)
        while counter <= ntimes
            prejumpupdate!(psi, exp(-1im * (t_given[counter]) * sys.Heff), psi0;
                normalize=normalize)
            # fixlastindex(states, counter)
            writestate!(states, psi, counter)
            counter = counter + 1
            if counter > ntimes
                break
            end
        end
        return states
    end
    # All the states before the first jump can be handled like this:
    while (t_given[counter] < jumptimes[counter_c]) && (counter <= ntimes)
        prejumpupdate!(psi, exp(-1im * (t_given[counter]) * sys.Heff), psi0;
            normalize=normalize)
        writestate!(states, psi, counter)
        counter = counter + 1
        if counter > ntimes
            break
        end
    end
    t_ = t_ + jumptimes[counter_c]
    counter_c = counter_c + 1
    while (counter_c <= njumps) && (counter <= ntimes)
        timeclick = jumptimes[counter_c]
        while (t_ < t_given[counter] < t_ + timeclick) && (counter <= ntimes)
            prejumpupdate!(psi, exp(-1im * (t_given[counter] - t_) * sys.Heff),
                fixlastindex(jump_states, counter_c - 1); normalize=normalize)
            writestate!(states, psi, counter)
            counter = counter + 1
            if counter > ntimes
                break
            end
        end
        t_ = t_ + timeclick
        counter_c = counter_c + 1
    end

    while counter <= ntimes
        prejumpupdate!(psi, exp(-1im * (t_given[counter] - t_) * sys.Heff),
            fixlastindex(jump_states, njumps); normalize=normalize)
        writestate!(states, psi, counter)
        counter = counter + 1
    end
    return states
end


