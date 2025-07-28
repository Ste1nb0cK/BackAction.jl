module BackAction

module SharedDependencies
using Reexport
@reexport using LinearAlgebra
@reexport using Statistics
@reexport using Random
end

using Distributed
include("structs.jl")
include("Gillipsie/Gillipsie.jl")
include("MonteCarloWaveFunction/MonteCarloWaveFunction.jl")
include("Utilities/Utilities.jl")
#
# module MonitoringOperator
# using ..BackAction: System, SimulParameters
# using DifferentialEquations
# using DiffEqCallbacks
# using ForwardDiff
# include("./MonitoringOperator/parametric_mixing.jl")
# include("./MonitoringOperator/monitoring.jl")
# end
#
end
