"""

```
rk4(f, y0, tspan, nsteps)
```
# Arguments
- `f::Function`: function ``f(t, y)`` defininf the system of ODEs
- `y0::Vector{Float}`: inital condition
- `tspan::Tuple{Float64}`: initial and final time
- `nsteps::Int`: number of steps.

# Returns
An array `Array{typeof(y0[1])}` with dimensions `(ndims(y0), nsteps)`
with the values of ``y`` at each step.

"""
function rk4(f, y0, tspan, nsteps)
    t0, tf = tspan
    h = (tf-t0)/nsteps
    t_values = collect(LinRange(0, tf, nsteps))
    vectordim = length(y0)
    y_values = Array{typeof(y0[1])}(undef, vectordim, nsteps)
    y_values[:, 1] = y0

    for i in 1:nsteps-1
        t = t_values[i]
        y = y_values[:, i]

        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)

        y_values[:, i+1] .= y + (k1 + 2k2 + 2k3 + k4) / 6
    end

    return y_values
end
