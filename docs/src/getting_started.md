## Installation
For the moment the library is not available through Pkg, so you will need to clone the repository 

```console
$ git clone https://github.com/Ste1nb0cK/BackAction.jl
```

and build it with `] develop path_to_clonedrepo` in the Julia REPL, this automatically adds it to your
current project. 

```julia
import Pkg; Pkg.develop(path="./BackAction.jl")
```

To check the installation do

```julia
Pkg.status
```

and you should see `BackAction` in the output e.g.

```console
  [32f8aca8] BackAction v0.1.0 `..`
  [e30172f5] Documenter v1.8.0
  [daee34ce] DocumenterCitations v1.3.5
  [b964fa9f] LaTeXStrings v1.4.0
  [91a5bcdd] Plots v1.40.9
  [10745b16] Statistics v1.10.0
  [8dfed614] Test

```


## Example: Jump Unraveling of a Driven Qubit
To explain how the library works  we'll consider a driven qubit in a thermal enviroment, 
whose density matrix follows the Lindblad equation [wiseman2009quantum](@cite):

$\dot{\rho}=-i[H,\rho]+\gamma(n+1)\mathcal{D}[\sigma^-]\rho+\gamma n\mathcal{D}[\sigma^+]\rho$

where $\sigma^- (\sigma^+)$ is the lowering (raising) atomic operator and

$H=\frac{1}{2}\Delta\sigma_z+\frac{1}{2}\Omega\sigma_x$
$\mathcal{D}[\sigma^\pm]\rho = \sigma^\pm\rho\sigma^\mp-\frac{1}{2}\{\sigma^\mp\sigma^\pm, \rho\}.$
To it we associate the *Stochastic Jump Unraveling*:

$d\rho=-i[H,\rho]dt+\left(\frac{\sigma^-\rho\sigma^+}{\mathrm{Tr}{(\sigma^-\rho\sigma^+)}}-\rho\right)dN_1
+ \left(\frac{\sigma^+\rho\sigma^-}{\mathrm{Tr}{(\sigma^+\rho\sigma^-)}}-\rho\right)dN_2$

whose ensemble average returns the Lindblad equation. Our objective, and the main functionality 
of the library, is to sample trajectories of this type of equations and obtain from them relevant physical quantities.

## Basic Definitions
The basic way to use the library is to first define the system and 
the conditions of the simulation via the structs `System` and `SimulParameters` (see [System and SimulParameters](@ref basic_structs)). 

### Defining the System
`System` is the struct intended to store all the relevant information of the system of interest. 
```@example workflow1
using BackAction
delta = 1.43
gamma = 1.0
omega = 1.3
nbar = 0.2
# Note: the library already has a dedicated variables for the Pauli matrices.
H = 0.5*delta * BackAction.sigma_z + 0.5*omega*BackAction.sigma_x
# Define the jump operators
L1 = sqrt(gamma*(nbar+1))*BackAction.sigma_m
L2 = sqrt(gamma*(nbar))*BackAction.sigma_p
sys = System(H, [L1, L2])
print(sys)
```

As you can see to create a `System` type one only needs the hamiltonian and the jump operators, the internal
constructor takes care of infering all the derived quantities.

### Defining the Simulation
All the information related to the details of generating a sample is organized with the struct `SimulParameters`.
The initial state might be mixed (`Matrix`) or pure (`Vector`), the library takes care of handling each case 
efficiently.

```@example workflow1
 # As initial condition use a mixture of |+> and |-> states
plus = 0.5* [[1.0+0im, 1] [1,  1]]
minus = 0.5* [[1.0+0im, -1] [-1,  1]]
psi0 = 0.3*plus + 0.7*minus

params = SimulParameters(psi0,
    25.0, # Final time
    1, # seed
    3, # Number of trajectories
    25_000, # Number of samples in the finegrid
    4.0, # Multiplier to use in the fine grid
    1e-3 # Tolerance for passing Dark state test
)
print(params)
```
For the moment the only available method of solution is the /Quantum Gillipsie Algorithm/
presented in [radaelli2024gillespie](@cite). The multiplier is used to generate the finegrid from which one samples the waiting times. Basically that finegrid is a discretization of `(dt, params.multiplier * params.tf)` with step size `params.dt`. A small but important difference with the [original implementation](https://github.com/marcoradaelli/GillespieQuantumJumps)is that here in each iteration the normalization of the WTD is checked; in case of undernormalization the current state is declared dark and the sampling stops. Sample quality strongly depends on the size of `dt`, but in contrast to other methods it's not necessary for it to be effectevely infinitesimal, small enough to resolve the structure of the WTD will suffice.

The main bottleneck is in the calculation of the WTD at each jump, this means that the factors that
affect the time it takes to sample a single trajectory are mainly the number of samples and total
number of jumps that happen in it.


### ClickDetection and Trajectory
The data about a click is modeled by the struct `DetectionClick` [(more info)](@ref DetectionClick), this in practice works as a glorified tuple. Similarly, the library has an alias for `Vector{DetectionClick}` called `Trajectory`. 

## Sampling trajectories
To obtain a sample of trajectories one uses `run_trajectories`, this returns a vector of trajectories as the sample :

```@example workflow1
data = run_trajectories_gillipsie(sys, params)
for k in 1:3
    println(data[k])
end
```

The first element shown of a `DetectionClick` is the time it took since the last jump to see this click, 
and the channel in which it was detected.

We can access the clicks in a trajectory doing:

```@example workflow1
 traj = data[1] # Trajectory
 click = traj[1] # first click in the trajectory
 print(click.time,"\n") # time we waited to see the jump 
 print(click.label, "\n") # Channel in which the jump occured.
```

## Obtaining the states and evaluating observables
To obtain the states of a trajectory at given times the function `states_att` is used, it evaluates them 
from a given initial state and returns an array. Below we use to obtain a sample of the expectation values
of the Pauli matrices

```@example workflow1
using LinearAlgebra, Plots, LaTeXStrings
ntimes = 1000 
tgiven = collect(LinRange(0, params.tf, ntimes));
statetrajectory = states_att(tgiven, traj, sys,  params.psi0)

# Obtain sigma_z expectation value over the trajectory
r_trajectory = zeros(ComplexF64, ntimes, 3)
for tn in 1:ntimes
   r_trajectory[tn, 1] = tr(BackAction.sigma_x*statetrajectory[:,:, tn]) 
   r_trajectory[tn, 2] = tr(BackAction.sigma_y*statetrajectory[:,:, tn]) 
   r_trajectory[tn, 3] = tr(BackAction.sigma_z*statetrajectory[:,:, tn]) 
end 

plot(tgiven, real.(r_trajectory[:, 1]), xlabel=L"t", ylabel=L"\langle\sigma\rangle", color="red", 
    line=:dash, label=L"\langle\sigma_x\rangle")
plot!(tgiven, real.(r_trajectory[:, 2]), color="blue", 
    line=:dash, label=L"\langle\sigma_y\rangle")
plot!(tgiven, real.(r_trajectory[:, 3]), color="green", 
    line=:dash, label=L"\langle\sigma_z\rangle")

```
## Multithreading
The library natively supports multithreading, all you need to do is run with the desired number of threads specified in the enviroment variable `JULI_NUM_THREADS` i.e. do something like `export JULIA_NUM_THREADS=4`.

That would be it!

## More Examples
You can find more elaboted examples in the Jupyter Notebooks available at the [repository](https://github.com/Ste1nb0cK/BackAction.jl/tree/5ed9dd8e60f16a799882b3e3a22aec53fe7428b2/notebooks).
