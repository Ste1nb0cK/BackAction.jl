module Gillipsie
using ..BackActionCoreStructs
using ProgressMeter
using Base.Threads
import StatsBase
include("functions_jump.jl")
include("gillipsie.jl")
include("run_trajectories.jl")
end


