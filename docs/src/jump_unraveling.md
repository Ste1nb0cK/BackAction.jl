# Exported functions

## Running trajectories
```@docs
run_trajectories_gillipsie(sys::System, params::SimulParameters; progbar::Bool = true,
                          psireset::VecOrMat{ComplexF64}=zeros(ComplexF64, 0))
```

## State Evaluation
```@docs
states_atjumps(traj::Trajectory, sys::System,
                      psi0::Union{Vector{ComplexF64}, Matrix{ComplexF64}}; normalize::Bool=true)
```

```@docs
states_att(t_given::Vector{Float64}, traj::Trajectory, sys::System,
psi0::Union{Vector{ComplexF64}, Matrix{ComplexF64}};normalize::Bool=true)
```
## Internal functions
### Utilities
```@docs
BackAction.fixlastindex
```
### Precomputation
```@docs
BackAction.setVs!(sys::System, nsamples::Int64, ts::Vector{Float64}, Vs::Array{ComplexF64})
```

```@docs
BackAction.setQs!(sys::System, nsamples::Int64, ts::Vector{Float64}, Qs::Array{ComplexF64}, Vs::Array{ComplexF64})
```

```@docs
BackAction.precompute!(sys::System, nsamples::Int64, ts::Vector{Float64}, Qs::Array{ComplexF64}, Vs::Array{ComplexF64})
```
### Click Sampling
```@docs
BackAction.calculatewtdweights!(W::Array{Float64}, Qs::Array{ComplexF64}, psi::Vector{ComplexF64}, params::SimulParameters)
```

```@docs
BackAction.calculatewtdweights!(W::Array{Float64}, Qs::Array{ComplexF64}, psi::Matrix{ComplexF64}, params::SimulParameters)
```

```@docs
BackAction.calculatechannelweights!(P::Vector{Float64}, psi::Vector{ComplexF64}, sys::System)
```
```@docs
BackAction.calculatechannelweights!(P::Vector{Float64}, psi::Matrix{ComplexF64}, sys::System)
```

```@docs
BackAction.sampletauindex!(W::Vector{Float64}, Qs::Array{ComplexF64}, psi::Vector{ComplexF64},
                         params::SimulParameters)
```

```@docs
BackAction.sampletauindex!(W::Vector{Float64}, Qs::Array{ComplexF64}, psi::Matrix{ComplexF64},
                         params::SimulParameters)
```

### State Updates
```@docs
BackAction.prejumpupdate!(V::Matrix{ComplexF64}, psi::Vector{ComplexF64}; normalize=false)
```

```@docs
BackAction.prejumpupdate!(psi::Vector{ComplexF64}, V::Matrix{ComplexF64},
                       psi0::Union{Vector{ComplexF64}, SubArray{ComplexF64}}; normalize=false)
```

```@docs
BackAction.prejumpupdate!(psi::Matrix{ComplexF64}, V::Matrix{ComplexF64},
               psi0::Union{Matrix{ComplexF64}, SubArray{ComplexF64}}; normalize=false)
```

```@docs
BackAction.prejumpupdate!(V::Matrix{ComplexF64}, psi::Matrix{ComplexF64}; normalize=false)

```

```@docs
BackAction.postjumpupdate!(L::Matrix{ComplexF64}, psi::Vector{ComplexF64}; normalize=true)
```

```@docs
BackAction.postjumpupdate!(L::Matrix{ComplexF64}, psi::Matrix{ComplexF64}; normalize=true)
```

### Trajectory Evaluation
```@docs
BackAction.run_singletrajectory_gillipsie(sys::System, params::SimulParameters, W::Vector{Float64}, P::Vector{Float64}, ts::Vector{Float64}, Qs::Array{ComplexF64}, Vs::Array{ComplexF64}; seed::Int64 = 1, isrenewal=false)
```


```@docs
BackAction.run_singletrajectory_gillipsie_renewal(sys::System, params::SimulParameters, W::Vector{Float64}, W0::Vector{Float64}, P::Vector{Float64}, ts::Vector{Float64},Qs::Array{ComplexF64}, Vs::Array{ComplexF64}, psireset::VecOrMat{ComplexF64}; seed::Int64 = 1)
```


```@docs
BackAction.gillipsiestep_returntau!(sys::System, params::SimulParameters, W::Vector{Float64},P::Vector{Float64}, Vs::Array{ComplexF64}, ts::Vector{Float64},t::Float64, psi::VecOrMat{ComplexF64}, traj::Trajectory )
```

```@docs
BackAction.gillipsiestep_returntau!(sys::System, params::SimulParameters, W::Vector{Float64},
                        P::Vector{Float64}, Vs::Array{ComplexF64}, ts::Vector{Float64},
                        t::Float64, psi::VecOrMat{ComplexF64}, traj::Trajectory, Qs::Array{ComplexF64}  )
```

```@docs
BackAction.writestate!(states::Array{ComplexF64}, psi::Union{Vector{ComplexF64}, Matrix{ComplexF64}}, counter::Int64)
```
    
