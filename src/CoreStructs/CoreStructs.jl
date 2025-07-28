module CoreStructs
using ..SharedDependencies
import Base.deepcopy_internal
include("SystemSimulParameters.jl")
include("HeffEvolution.jl")
include("JumpCondition.jl")
end

