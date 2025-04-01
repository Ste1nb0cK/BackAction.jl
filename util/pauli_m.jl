# This is the convention in Wiseman's
@doc "Pauli Matrix ``\\sigma_x = |0\\rangle\\langle 1| + |1\\rangle\\langle 0|``"
const sigma_x::Matrix{ComplexF64} = [[0, 1.0 + 0im] [1, 0]]
@doc "Pauli Matrix ``\\sigma_y = i|0\\rangle\\langle 1| - i|1\\rangle\\langle 0|``"
const sigma_y::Matrix{ComplexF64} = [[0, -1im] [1im, 0]]
@doc "Pauli Matrix ``\\sigma_z = |0\\rangle\\langle 0| - |1\\rangle\\langle 1|``"
sigma_z = [[-1.0 + 0im, 0] [0, 1]]

@doc "Lowering operator ``\\sigma_- = |0\\rangle\\langle 1|``"
const sigma_m::Matrix{ComplexF64} = [[0.0 + 0im, 0] [1, 0]]
@doc "Raising operator ``\\sigma_+ = |1\\rangle\\langle 0|``"
const sigma_p::Matrix{ComplexF64} = [[0.0 + 0im, 1] [0, 0]]
