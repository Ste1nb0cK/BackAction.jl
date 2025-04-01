export monitoringoperator

struct SumProductJumpOperators{TypeParametricJumps<:Function,T3<:Int}
    Ls::Vector{TypeParametricJumps}
    nchannels::T3
    nlevels::T3
    function SumProductJumpOperators(Ls::Vector{T}, nlevels::T3) where {T<:Function,T3<:Int}
        new{T,T3}(Ls, length(Ls), nlevels)
    end
end

function (J::SumProductJumpOperators)(theta...)
    aux = zeros(ComplexF64, J.nlevels, J.nlevels)
    for k in 1:J.nchannels
        aux = aux + adjoint(J.Ls[k](theta...)) * J.Ls[k](theta...)
    end
    return aux
end

struct ParametricEffectiveHamiltonian{TypeParametricHamiltonian<:Function,
    TypeParametricJumps<:Function}
    H::TypeParametricHamiltonian
    J::SumProductJumpOperators{TypeParametricJumps,Int64}
    function ParametricEffectiveHamiltonian(H::TH, Ls::Vector{TJ}, nlevels::T3) where {TH<:Function,TJ<:Function,T3<:Int}
        new{TH,TJ}(H, SumProductJumpOperators(Ls, nlevels))
    end
end



function (He::ParametricEffectiveHamiltonian)(theta...)
    return He.H(theta...) - 0.5im * He.J(theta...)
end
"""


```
getheff_parametrized(H_par::Function, Ls_par)
```

From the function `H_par`, which is assumed to be the parametrization of the hamiltonian, and
an array of functions `Ls` which are assumed to parametrize the jump operators, return a function
that's the  parametrization of the effective Hamiltonian.

!!! note "About the Arguments"
    The calculation of monitoring operator assumes that the arguments of `H_par` and the functions in
    `Ls` are the same and in the same order.

"""
function getheff_parametrized(H_par::Tf, Ls_par::Vector{Tf2})::Function where {Tf<:Function,Tf2<:Function}
    # here theta is expected to be a vector
    return (theta...) -> begin # Get an arbitrary number of arguments
        # The ... "splatts" the vector so it is passed as a tuple to the function
        LLs_par = [adjoint(L_par(theta...)) * L_par(theta...) for L_par in Ls_par]
        return (H_par(theta...) - 0.5im * sum(LLs_par))::Matrix{ComplexF64}
    end
end


"""

```
expheff_derivative(Heff_par::Function, tau::Float64, theta::Vector{Float64}, dtheta::Vector{Float64})
```

Calculate the derivative  ``\\partial_{i}e^{-i\\tau H_e(\\theta)}``, where ``i`` is the ``i-th`` component
of the vector ``\\theta``.

# Arguments
- `Heff_par::Function`: the parametrization of the effective hamiltonian
- `tau::Float64`: the time in the exponential
- `theta::Vector{Float64}`: vector with the values of the parameters at which the derivative is calculated
- `dtheta::Vector{Float64}`: the displacement vector used to calculate the derivative, if you want the derivative
                             respect to the ``i-th`` parameter `dtheta` must have zero entries except for `dtheta[i]`.

!!! note "Derivative order "
    The derivative is calculate using the five-point stencil rule.

!!! todo "TODO: add an example"
    Preferably one in which  ``\\partial_i H_e`` commutes with ``H_e``, those are easier.
"""
function expheff_derivative(Heff_par::Tf, tau::T2, theta::Vector{T2}, dtheta::Vector{T2}) where {T2<:Real,Tf<:Function}
    aux1 = -1im * tau
    norm_dtheta = norm(dtheta)
    return (-exp(aux1 * Heff_par((theta + 2 * dtheta)...)::ComplexF64)
            +
            8 * exp(aux1 * Heff_par((theta + 1 * dtheta)...)::ComplexF64)
            -
            8 * exp(aux1 * Heff_par((theta - 1 * dtheta)...)::ComplexF64)
            +
            exp(-1im * tau * Heff_par((theta - 2 * dtheta)...)::ComplexF64)) / (12 * norm_dtheta)
end


"""

```
jumpoperators_derivatives(Ls_par, theta::Vector{Float64}, dtheta::Vector{Float64})
```

Calculate the derivatives of the list of jump operators.

# Arguments
- `Ls_par`: array with the functions that represent the parametrization of the jump operators
- `theta::Vector{Float64}`: vector with the values of the parameters at which the derivative is calculated
- `dtheta::Vector{Float64}`: the displacement vector used to calculate the derivative, if you want the derivative
                             respect to the ``i-th`` parameter `dtheta` must have zero entries except for `dtheta[i]`.

!!! note "Derivative order"
    The derivative is calculate using the five-point stencil rule.

"""
function jumpoperators_derivatives(Ls_par, theta::Vector{T2},
    dtheta::Vector{T2}) where {T2<:Real}
    nchannels = length(Ls_par)
    nlevels = size(Ls_par[1](theta...))[1]
    dLs = Array{ComplexF64}(undef, nlevels, nlevels, nchannels)
    norm_dtheta = norm(dtheta)
    for k in 1:nchannels
        # f1 = Ls_par[k]((theta + 2 * dtheta)...)
        # f2 = Ls_par[k]((theta + dtheta)...)
        # f3 = Ls_par[k]((theta - dtheta)...)
        # f4 = Ls_par[k]((theta - 2 * dtheta)...)
        dLs[:, :, k] = (
            -Ls_par[k]((theta + 2 * dtheta)...)
            +
            8 * Ls_par[k]((theta + dtheta)...)
            -
            8 * Ls_par[k]((theta - dtheta)...)
            +
            Ls_par[k]((theta - 2 * dtheta)...)
        ) / (12 * norm_dtheta)
    end
    return dLs
end


"""

```
writederivative!(dpsi::SubArray{ComplexF64, 1}, L::Matrix{ComplexF64},
                          dL::SubArray{ComplexF64, 2},
                          V::Matrix{ComplexF64}, dV::Matrix{ComplexF64},
                          psi0::Vector{ComplexF64})
```

Writes the derivative of ``|\\psi\\rangle = L(\\theta)V(\\theta)|\\psi_0\\rangle ``
at ``\\theta`` in the subarray `dpsi` respect to the ``i-th`` component of ``theta``,
following the same logic as `expheff_derivative`. The derivatives of ``V`` and ``L`` at
must be provided via `dL` and `dV`.

!!! note "Initial State dependency"
    This is intended to be used when ``|\\psi_0\\rangle`` doesn't have dependeny on ``\\theta``.
"""
function writederivative!(dpsi::SubArray{T1,1},
    L::Matrix{T1},
    dL::SubArray{T1,2},
    V::Matrix{T1}, dV::Matrix{T1},
    psi0::Vector{T1}) where {T1<:Complex}
    dpsi .= (dL * V + L * dV) * psi0
end

"""

```
writederivative!(dpsi::SubArray{ComplexF64, 1},
                 L::Matrix{ComplexF64},
                 dL::SubArray{ComplexF64, 2},
                 V::Matrix{ComplexF64}, dV::Matrix{ComplexF64},
                 psi0::SubArray{ComplexF64, 1},
                 dpsi0::SubArray{ComplexF64, 1})
```

Writes the derivative of ``|\\psi\\rangle = L(\\theta)V(\\theta)|\\psi_0\\rangle ``
at ``\\theta`` in the subarray `dpsi` respect to the ``i-th`` component of ``theta``,
following the same logic as `expheff_derivative`. The derivatives of ``V`` and ``L`` at
must be provided via `dL` and `dV`, and also that of ``|\\psi(0)\\rangle`` as `dpsi0`.


"""
function writederivative!(dpsi::SubArray{T1,1},
    L::Matrix{T1},
    dL::SubArray{T1,2},
    V::Matrix{T1}, dV::Matrix{T1},
    psi0::SubArray{T1,1},#Union{Vector{ComplexF64}, SubArray{ComplexF64}},
    dpsi0::SubArray{T1,1}) where {T1<:Complex}
    dpsi .= (dL * V + L * dV) * psi0 + L * V * dpsi0
end

"""

```
derivatives_atjumps(sys::System, Heff_par::Function, Ls_par, traj::Trajectory, psi0::Vector{ComplexF64}, theta::Vector{Float64},
                            dtheta::Vector{Float64})

```

Given a trajectory `traj` calculate all the ``\\partial_i|\\psi_n(\\theta)\\rangle`` where ``|\\psi_n(\\theta)\\rangle``
is the state just after the ``n-th`` jump in the trajectory. They are returned as an array of dimensions
`(size(psi0)[1], size(traj))` so to access the derivative at the ``k-th`` jump you would do
`derivatives_atjumps(sys, Heff_par, Ls_par, traj, psi0, theta, dtheta)[:, k]`.

# Arguments
- `sys::System`: the function to which the trajectory corresponds
- `Heff_par::Function`: the function that gives the parametrization of the effective hamiltonian
- `Ls_par`: the array with the functions that give the parametrizatio of the jump operators
- `theta::Vector{Float64}`: vector with the values of the parameters at which the derivative is calculated
- `dtheta::Vector{Float64}`: the displacement vector used to calculate the derivative, if you want the derivative
                             respect to the ``i-th`` parameter `dtheta` must have zero entries except for `dtheta[i]`.

!!! note "Derivative order "
    The derivative is calculated using the five-point stencil rule.
"""
function derivatives_atjumps(sys::System{T1,T3}, Heff_par::Function, Ls_par, traj::Trajectory{T2,T3}, psi0::Vector{T1},
    theta::Vector{T2}, dtheta::Vector{T2}) where {T1<:Complex,T2<:Real,T3<:Int}
    # 0. Special Case: if the trajectory is empty, return an empty array
    if isempty(traj)
        return Array{ComplexF64}(undef, 0, 0)
    end
    # 1. Get the derivatives of L
    dLs = jumpoperators_derivatives(Ls_par, theta, dtheta)
    # 2.1 Setup
    njumps = size(traj)[1]
    dpsis = zeros(T1, sys.NLEVELS, njumps)
    # 2.2 Set up the first jump
    click = traj[1]
    label = click.label
    tau = click.time
    writederivative!(fixlastindex(dpsis, 1),
        sys.Ls[label], fixlastindex(dLs, label),
        exp(-1im * tau * sys.Heff),
        expheff_derivative(Heff_par, tau, theta, dtheta), psi0)
    # In case there are no more jumps, return
    if njumps == 1
        return dpsis
    end
    # 3. Go over the rest of the jumps
    psitildes = states_atjumps(traj, sys, psi0; normalize=false)
    for k in 2:njumps
        click = traj[k]
        label = click.label
        tau = click.time
        # Calculate the derivative
        writederivative!(fixlastindex(dpsis, k),
            sys.Ls[label], fixlastindex(dLs, label),
            exp(-1im * tau * sys.Heff),
            expheff_derivative(Heff_par, tau, theta, dtheta),
            fixlastindex(psitildes, k - 1), fixlastindex(dpsis, k - 1))
    end
    return dpsis

end

function derivatives_atjumps(sys::System{T1,T3}, Heff_par::Function, Ls_par,
    jumptimes::Vector{T2}, labels::Vector{T3}, psi0::Vector{T1},
    theta::Vector{T2},
    dtheta::Vector{T2}) where {T1<:Complex,T2<:Real,T3<:Int}
    # 0. Special Case: if the trajectory is empty, return an empty array
    if isempty(labels)
        return Array{ComplexF64}(undef, 0, 0)
    end
    # 1. Get the derivatives of L
    dLs = jumpoperators_derivatives(Ls_par, theta, dtheta)
    # 2.1 Setup
    njumps = length(labels)
    dpsis = zeros(T1, sys.NLEVELS, njumps)
    # 2.2 Set up the first jump
    label = labels[1]
    tau = jumptimes[1]
    writederivative!(fixlastindex(dpsis, 1),
        sys.Ls[label], fixlastindex(dLs, label),
        exp(-1im * tau * sys.Heff),
        expheff_derivative(Heff_par, tau, theta, dtheta), psi0)
    # In case there are no more jumps, return
    if njumps == 1
        return dpsis
    end
    # 3. Go over the rest of the jumps
    psitildes = states_atjumps(jumptimes, labels, sys, psi0; normalize=false)
    for k in 2:njumps
        # Calculate the derivative
        label = labels[k]
        tau = jumptimes[k]
        writederivative!(fixlastindex(dpsis, k),
            sys.Ls[label], fixlastindex(dLs, label),
            exp(-1im * tau * sys.Heff),
            expheff_derivative(Heff_par, tau, theta, dtheta),
            fixlastindex(psitildes, k - 1), fixlastindex(dpsis, k - 1))
    end
    return dpsis

end


"""

```
writexi!(xi::SubArray{ComplexF64, 2}, dV::Matrix{ComplexF64},
         psi::SubArray{ComplexF64, 1}, psi0::Vector{ComplexF64})
```
Calculate the monitoring operator when ``|\\psi(\\theta)\\rangle = V|\\psi_0\\rangle`` and ``|\\psi_0\\rangle`` doesn't
depend on ``\\theta``, and write it at the `SubArray` `xi`. This is, calculate
``d|\\psi(\\theta)\\rangle = dV|\\psi_0\\rangle``.
"""
function writexi!(xi::SubArray{T1,2}, dV::Matrix{T1},
    psi::SubArray{T1,1}, psi0::Vector{T1}) where {T1<:Complex}
    xi .= ((dV * psi0) .* adjoint(psi) + psi .* adjoint(dV * psi0)) / dot(psi, psi)
end

# Method for when xi is a matrix at which we want to write at  and 
# ``|\\psi(\\theta)\\rangle = V|\\psi_0\\rangle`` and ``|\\psi_0\\rangle`` doesn't
# depend on ``\\theta``
function writexi!(xi::Matrix{T1}, dV::Matrix{T1},
    psi::SubArray{T1,1}, psi0::Vector{T1}) where {T1<:Complex}
    xi .= ((dV * psi0) .* adjoint(psi) + psi .* adjoint(dV * psi0)) / dot(psi, psi)
end


"""

```
writexi!(xi::SubArray{ComplexF64, 2}, V::Matrix{ComplexF64}, dV::Matrix{ComplexF64},
          psijump::SubArray{ComplexF64, 1}, dpsijump::SubArray{ComplexF64, 1},
         psi::SubArray{ComplexF64, 1})
```

Calculate the monitoring operator when ``|\\psi(\\theta)\\rangle = V|\\psi_N\\rangle`` and ``|\\psi_N\\rangle`` is
a state that depends on ``\\theta``, the result is written at the `SubArray` `xi`. This is, calculate
``d|\\psi(\\theta)\\rangle = dV|\\psi_N\\rangle + Vd|\\psi_N\\rangle ``.
"""
function writexi!(xi::SubArray{T1,2}, V::Matrix{T1}, dV::Matrix{T1},
    psijump::SubArray{T1,1}, dpsijump::SubArray{T1,1},
    psi::SubArray{T1,1}) where {T1<:Complex}
    xi .= ((dV * psijump + V * dpsijump) .* adjoint(psi) + psi .* adjoint(dV * psijump + V * dpsijump)) / dot(psi, psi)
end

# Method for when xi is a matrix
function writexi!(xi::Matrix{T1}, V::Matrix{T1}, dV::Matrix{T1},
    psijump::SubArray{T1,1}, dpsijump::SubArray{T1,1},
    psi::SubArray{T1,1}) where {T1<:Complex}
    xi .= ((dV * psijump + V * dpsijump) .* adjoint(psi) + psi .* adjoint(dV * psijump + V * dpsijump)) / dot(psi, psi)
end


"""

```
monitoringoperator(t_given::Vector{Float64},
    sys::System, Heff_par::Function, Ls_par, traj::Trajectory, psi0::Vector{ComplexF64}, theta::Vector{Float64},
                            dtheta::Vector{Float64})
```
Given a trajectory `traj` and an initial state ``|\\psi_0\\rangle``, calculate the monitoring state at the
times in `t_given`. The result is returned in an array of dimensions `(sys.NLEVELS, sys.NLEVELS, size(t_given)[1])`,
so to access it at the time `t` you would do
`monitoringoperator(t_given, sys, Heff_par, Ls_par, traj, psi0, theta, dtheta)[:, :, t]`.
"""
function monitoringoperator(t_given::Vector{T2},
    sys::System{T1,T3}, Heff_par::Function, Ls_par, traj::Trajectory{T2,T3}, psi0::Vector{T1}, theta::Vector{T2},
    dtheta::Vector{T2}) where {T1<:Complex,T2<:Real,T3<:Int}

    # Special case: if the time array is empty, return an empty array
    if isempty(t_given)
        return Array{T1}(undef, 0, 0, 0)
    end
    psi = states_att(t_given, traj, sys, psi0; normalize=false)
    ntimes = size(t_given)[1]
    njumps = size(traj)[1]
    t_ = 0
    counter = 1
    counter_c = 1
    xis = ntimes > 1 ? Array{T1}(undef, sys.NLEVELS, sys.NLEVELS, ntimes) : Matrix{T1}(undef, sys.NLEVELS, sys.NLEVELS)
    # Edge case
    if isempty(traj)
        while counter <= ntimes
            writexi!(fixlastindex(xis, counter),
                expheff_derivative(Heff_par, t_given[counter], theta, dtheta),
                fixlastindex(psi, counter), psi0)

            counter = counter + 1
            if counter > ntimes
                break
            end
        end
        return xis
    end
    # Evaluations before first jump
    while (t_given[counter] < traj[counter_c].time) && (counter <= ntimes)
        writexi!(fixlastindex(xis, counter),
            expheff_derivative(Heff_par, t_given[counter], theta, dtheta),
            fixlastindex(psi, counter), psi0)
        counter = counter + 1
        if counter > ntimes
            break
        end
    end
    dpsijumps = derivatives_atjumps(sys, Heff_par, Ls_par, traj, psi0, theta, dtheta)
    psijumps = states_atjumps(traj, sys, psi0; normalize=false)
    t_ = t_ + traj[counter_c].time
    counter_c = counter_c + 1
    # Evaluation after first jump
    while (counter_c <= njumps) && (counter <= ntimes)
        timeclick = traj[counter_c].time
        while (t_ < t_given[counter] < t_ + timeclick) && (counter <= ntimes)
            writexi!(fixlastindex(xis, counter), exp(-1im * (t_given[counter] - t_) * sys.Heff),
                expheff_derivative(Heff_par, t_given[counter] - t_, theta, dtheta),
                fixlastindex(psijumps, counter_c - 1), fixlastindex(dpsijumps, counter_c - 1),
                fixlastindex(psi, counter))
            counter = counter + 1
            if counter > ntimes
                break
            end
        end
        t_ = t_ + timeclick
        counter_c = counter_c + 1
    end

    while counter <= ntimes
        writexi!(fixlastindex(xis, counter), exp(-1im * (t_given[counter] - t_) * sys.Heff),
            expheff_derivative(Heff_par, t_given[counter] - t_, theta, dtheta),
            fixlastindex(psijumps, njumps), fixlastindex(dpsijumps, njumps),
            fixlastindex(psi, counter))

        counter = counter + 1
    end
    return xis
end

function monitoringoperator(t_given::Vector{T2},
    sys::System{T1,T3}, Heff_par::Function, Ls_par, jumptimes::Vector{T2}, labels::Vector{T3}, psi0::Vector{T1}, theta::Vector{T2},
    dtheta::Vector{T2}) where {T1<:Complex,T2<:Real,T3<:Int}

    # Special case: if the time array is empty, return an empty array
    if isempty(t_given)
        return Array{T1}(undef, 0, 0, 0)
    end
    psi = states_att(t_given, jumptimes, labels, sys, psi0; normalize=false)
    ntimes = length(t_given)
    njumps = length(labels)
    t_ = 0
    counter = 1
    counter_c = 1
    xis = ntimes > 1 ? Array{T1}(undef, sys.NLEVELS, sys.NLEVELS, ntimes) : Matrix{T1}(undef, sys.NLEVELS, sys.NLEVELS)
    # Edge case
    if isempty(labels)
        while counter <= ntimes
            ntimes > 1 ? writexi!(fixlastindex(xis, counter),
                expheff_derivative(Heff_par, t_given[counter], theta, dtheta),
                fixlastindex(psi, counter), psi0) :
            writexi!(xis,
                expheff_derivative(Heff_par, t_given[counter], theta, dtheta),
                fixlastindex(psi, counter), psi0)

            counter = counter + 1
            if counter > ntimes
                break
            end
        end
        return xis
    end
    # Evaluations before first jump
    while (t_given[counter] < jumptimes[counter_c]) && (counter <= ntimes)
        ntimes > 1 ? writexi!(fixlastindex(xis, counter),
            expheff_derivative(Heff_par, t_given[counter], theta, dtheta),
            fixlastindex(psi, counter), psi0) :
        writexi!(xis,
            expheff_derivative(Heff_par, t_given[counter], theta, dtheta),
            fixlastindex(psi, counter), psi0)
        counter = counter + 1
        if counter > ntimes
            break
        end
    end
    dpsijumps = derivatives_atjumps(sys, Heff_par, Ls_par, jumptimes, labels, psi0, theta, dtheta)
    psijumps = states_atjumps(jumptimes, labels, sys, psi0; normalize=false)
    t_ = t_ + jumptimes[counter_c]
    counter_c = counter_c + 1
    # Evaluation after first jump
    while (counter_c <= njumps) && (counter <= ntimes)
        timeclick = jumptimes[counter_c]
        while (t_ < t_given[counter] < t_ + timeclick) && (counter <= ntimes)
            ntimes > 1 ? writexi!(fixlastindex(xis, counter), exp(-1im * (t_given[counter] - t_) * sys.Heff),
                expheff_derivative(Heff_par, t_given[counter] - t_, theta, dtheta),
                fixlastindex(psijumps, counter_c - 1), fixlastindex(dpsijumps, counter_c - 1),
                fixlastindex(psi, counter)) :
            writexi!(xis, exp(-1im * (t_given[counter] - t_) * sys.Heff),
                expheff_derivative(Heff_par, t_given[counter] - t_, theta, dtheta),
                fixlastindex(psijumps, counter_c - 1), fixlastindex(dpsijumps, counter_c - 1),
                fixlastindex(psi, counter))
            counter = counter + 1
            if counter > ntimes
                break
            end
        end
        t_ = t_ + timeclick
        counter_c = counter_c + 1
    end

    while counter <= ntimes
        ntimes > 1 ? writexi!(fixlastindex(xis, counter), exp(-1im * (t_given[counter] - t_) * sys.Heff),
            expheff_derivative(Heff_par, t_given[counter] - t_, theta, dtheta),
            fixlastindex(psijumps, njumps), fixlastindex(dpsijumps, njumps),
            fixlastindex(psi, counter)) :
        writexi!(xis, exp(-1im * (t_given[counter] - t_) * sys.Heff),
            expheff_derivative(Heff_par, t_given[counter] - t_, theta, dtheta),
            fixlastindex(psijumps, njumps), fixlastindex(dpsijumps, njumps),
            fixlastindex(psi, counter))


        counter = counter + 1
    end
    return xis
end

