export JumpCondition
struct JumpCondition
    r::Ref{Float64}
end

function Base.deepcopy_internal(c::JumpCondition, dict::IdDict)
    JumpCondition(Ref(1.0))
end

function (condition::JumpCondition)(u, t, integrator)
    real(dot(u, u)) - condition.r[]
end


