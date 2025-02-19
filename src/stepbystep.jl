# Description: implementation of the usual MCW

function rk4step(rho::VecOrMat{ComplexF64}, Heff::Matrix{ComplexF64}, dt::Float64)
end
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)

        y_values[:, i+1] .= y + (k1 + 2k2 + 2k3 + k4) / 6

function run_single_stepbystep()
end
