+++
date = 2026-01-16T10:00:00+08:00
title = "Population Method"
description = "An in-depth look at the Population Method" 
tags = ["Programming", "Population Method","Mathematical Modeling"]
math = true
[cover]
    image = "/picture/population_cover.png"
    alt = "Population Method Diagram"
    caption = "An overview of various population methods in optimization"
    relative = false
+++

This post presents a variety of population methods that involve optimization using a collection of design points, called individuals.

Having a large number of individuals distributed throughtout the design space can help the algorithm to explore the design space more effectively.

Many population methods are stochastica in nature, and it is generally easy to parallelize the computation.

## Population Iteration

population iteratively improve a population of m designs 

<div>
$$
x^{(1)}, x^{(2)}, \dots, x^{(m)}
$$
</div>

The population at a particular iteration is represented as a generation.

The algorithms are designed so that the individuals in the population converge to one or more local minima over multiple generations.

The various methods discussed in this post differ in how they generate a new generation from the current generation.

Population mehtods begin with an initial population, which is often generated randomly.The initial population should be spread over the design space to encourage exploration.

```julia
abstract type PopulationMethod end
function population_method(M::PopulationMethod, f, desings, k_max)
    population = init!(M,f,designs)
    for k in 1:k_max
        population = iterate!(M, f, population)
    end
    return population
end
```

The following are there distributions used to generate the initial population:
- Uniform Distribution
- Normal Distribution
- Cauchy Distribution

![Population Initialization](/picture/population_method01.png)

## Genetic Algorithm
The genetic algorithm is inspired by the process of natural selection in biological evolution.

The design point associated with each individual is represented as a chromosome. At each generation, the chromosomes of the fitter individuals are passed on to the next generation through selection, crossover, and mutation operations.

```julia
struct GeneticAlgorithm <: PopulationMethod
    S # SelectionMethod
    C # CrossoverMethod
    U # MutationMethod
end

init!(M::GeneticAlgorithm, f, designs) = designs

function step!(M::GeneticAlgorithm, f, population)
    S, C, U = M.S, M.C, M.U
    parents = select(S, f.(population))
    children = [crossover(C,population[p[1]],population[p[2]])
    for p in parents]
    return [mutate(U, c) for c in children]
end
```

### Selection

Selection chooses individuals from the current population to serve as parents for the next generation. Common selection methods include rank-based selection, tournament selection, and roulette wheel selection.

rank-based selection sorts individuals based on their fitness and assigns selection probabilities accordingly.
![Rank-based Selection](/picture/population_method02.png)

tournament selection randomly selects a subset of individuals and chooses the fittest among them as parents.
![Tournament Selection](/picture/population_method03.png)

roulette wheel selection assigns selection probabilities proportional to fitness, allowing fitter individuals a higher chance of being selected.
![Roulette Wheel Selection](/picture/population_method05.png)

```julia
abstract type SelectionMethod end

# Pick pairs randomly from top k parents
struct TruncationSelection <: SelectionMethod
    k # top k to keep
end

function select(t::TruncationSelection, y)
    p = sortperm(y)
    return [p[rand(1:t.k, 2)] for i in y]
end

# Pick parents by choosing best among random subsets
struct TournamentSelection <: SelectionMethod
    k # top k to keep
end

function select(t::TournamentSelection, y)
    getparent() = begin
        p = randperm(length(y))
        p[argmin(y[p[1:t.k]])]
    end
    return [[getparent(), getparent()] for i in y]
end

# Sample parents proportionately to fitness
struct RouletteWheelSelection <: SelectionMethod end

function select(::RouletteWheelSelection, y)
    y = maximum(y) .- y
    cat = Categorical(normalize(y, 1))
    return [rand(cat, 2) for i in y]
end
```

### Crossover

Crossover combines the chromosomes of parents to form children. As with selection, there are several crossover schemes

1. In single-point crossover, a random crossover point is selected, and the segments of the parents' chromosomes are swapped to create children.
![Single-point Crossover](/picture/population_method06.png)
2. In two-point crossover, two crossover points are selected, and the segments between these points are exchanged.
![Two-point Crossover](/picture/population_method07.png)
3. In uniform crossover, each gene is independently chosen from one of the parents with equal probability.
![Uniform Crossover](/picture/population_method08.png)
```julia
abstract type CrossoverMethod end

struct SinglePointCrossover <: CrossoverMethod end

function crossover(::SinglePointCrossover, a, b)
    i = rand(eachindex(a))
    return [a[1:i]; b[i+1:end]]
end

struct TwoPointCrossover <: CrossoverMethod end

function crossover(::TwoPointCrossover, a, b)
    n = length(a)
    i, j = rand(1:n, 2)
    if i > j
        (i,j) = (j,i)
    end
    return [a[1:i]; b[i+1:j]; a[j+1:n]]
end

struct UniformCrossover <: CrossoverMethod
    p # crossover probability
end

function crossover(U::UniformCrossover, a, b)
    return [rand() > U.p ? u : v for (u,v) in zip(a,b)]
end

struct InterpolationCrossover <: CrossoverMethod
    λ # interpolant
end

crossover(C::InterpolationCrossover, a, b) = (1-C.λ)*a + C.λ*b
```

### Mutation
Mutation introduces random changes to individuals to maintain genetic diversity within the population. Common mutation method is zero-mean Gaussian distribution.
```julia
abstract type MutationMethod end

struct DistributionMutation <: MutationMethod
    λ # mutation rate
    D # mutation distribution
end

function mutate(M::DistributionMutation, child)
    return [rand() < M.λ ? v + rand(M.D) : v for v in child]
end

GaussianMutation(σ) = DistributionMutation(1.0, Normal(0,σ))
```

Each gene in the chromosome typically has a small probability λ of being changed. For a chromosome with m genes, this mutation rate is typically set to λ = 1/m, yielding an average of one mutation per child chromosome.

![Gaussian Mutation](/picture/population_method09.png)

## Differential Evolution
Differential Evolution (DE) is a population-based optimization algorithm that utilizes vector differences for perturbing the population members.

For each individual x:
1. Select three distinct individuals a, b, and c from the population.
2. Generate a trial vector by adding the weighted difference between b and c to a:
   $$ 
   v = a + F \cdot (b - c) 
   $$
   where F is a scaling factor typically in the range [0, 2].
3. Evaluate the fitness of the trial vector v. If v has a better fitness than x, replace x with v in the next generation; otherwise, retain x.

```julia
mutable struct DifferentialEvolution <: PopulationMethod
    p # crossover probability
    w # differential weight
end

init!(M::DifferentialEvolution, f, designs) = designs

function step!(M::DifferentialEvolution, f, population)
    p, w = M.p, M.w
    n, m = length(population[1]), length(population)
    for x in population
        a, b, c = sample(population, 3, replace=false)
        z = a + w*(b-c)
        x′ = crossover(UniformCrossover(p), x, z)
        if f(x′) < f(x)
            x .= x′
        end
    end
    return population
end
```
![Differential Evolution](/picture/population_method10.png)

## Particle Swarm Optimization
Particle swarm optimization introduces momentum to accelerate convergence toward minima. Each individual, or particle, in the population keeps track of its current position, velocity, and the best position it has seen so far. Momentum allows an individual to accumulate speed in a favorable direction, independent of local perturbations.

At each iteration, each individual is accelerated toward both the best position it has seen and the best position found thus far by any individual. The acceleration is weighted by a random term.

```julia
mutable struct Particle
    x # position
    v # velocity
    x_best # best design thus far
end

mutable struct ParticleSwarm <: PopulationMethod
    w # inertia
    c1 # first momentum coefficient
    c2 # second momentum coefficient
    V # initial particle velocity distribution
    best # best overall design thus far, and its value
end

function init!(M::ParticleSwarm, f, designs)
    population = [Particle(x,rand(M.V),copy(x)) for x in designs]
    best = (x=copy(population[1].x), y=Inf)
    for P in population
        y = f(P.x)
        if y < best.y; best = (x=P.x, y=y); end
    end
    M.best = best
    return population
end

function step!(M::ParticleSwarm, f, population)
    w, c1, c2, best = M.w, M.c1, M.c2, M.best
    n = length(best.x)
    for P in population
        r1, r2 = rand(n), rand(n)
        P.x += P.v
        P.v = w*P.v + c1*r1.*(P.x_best - P.x) +
              c2*r2.*(best.x - P.x)
        y = f(P.x)
        if y < best.y; best = (x=copy(P.x), y=y); end
        if y < f(P.x_best); P.x_best .= P.x; end
    end
    M.best = best
    return population
end
```
![Particle Swarm Optimization](/picture/population_method11.png)

## Firefly Algorithm
The firefly algorithm was inspired by the manner in which fireflies flash their lights to attract mates of the same species. In the firefly algorithm, each individual in the population is a firefly and can flash to attract other fireflies.

At each iteration, all fireflies are moved toward all more attractive fireflies. A firefly's attraction is proportional to its performance.

```julia
struct Firefly <: PopulationMethod
    α # walk step size
    β # source intensity
    brightness # intensity function
end

init!(M::Firefly, f, designs) = designs

function step!(M::Firefly, f, population)
    α, β, brightness = M.α, M.β, M.brightness
    m = length(population[1])
    N = MvNormal(I(m))
    for a in population, b in population
        if f(b) < f(a)
            r = norm(b-a)
            a .+= β*brightness(r)*(b-a) + α*rand(N)
        end
    end
    return population
end
```
![Firefly Algorithm](/picture/population_method12.png)

## Cuckoo Search
Cuckoo search is another nature-inspired algorithm named after the cuckoo bird, which engages in a form of brood parasitism. Cuckoos lay their eggs in the nests of other birds.

In cuckoo search, each nest represents a design point. New design points can be produced using Lévy flights from nests, which are random walks with step-lengths from a heavy-tailed distribution (typically a Cauchy distribution).

```julia
mutable struct CuckooSearch <: PopulationMethod
    p_s # search fraction
    p_a # nest abandonment fraction
    C # flight distribution
end

function init!(M::CuckooSearch, f, designs)
    return [(x=x, y=f(x)) for x in designs]
end

function step!(M::CuckooSearch, f, population)
    p_s, p_a, C = M.p_s, M.p_a, M.C
    m, n = length(population), length(population[1].x)
    m_search = round(Int, m*p_s)
    m_abandon = round(Int, m*p_a)
    for i in 1:m_search
        j, k = rand(1:m), rand(1:m)
        x = population[j].x + rand(C,n)
        y = f(x)
        if y < population[k].y
            population[k] = (x=x, y=y)
        end
    end
    p = sortperm(population, by=nest->nest.y, rev=true)
    for i in 1:m_abandon
        j = rand(1:m-m_abandon)+m_abandon
        x′ = population[p[j]].x + rand(C,n)
        population[p[i]] = (x=x′, y=f(x′))
    end
    return population
end
```
![Cuckoo Search](/picture/population_method13.png)

## Hybrid Methods
Many population methods perform well in global search, being able to avoid local minima and finding the best regions of the design space. Unfortunately, these methods do not perform as well in local search in comparison to descent methods.

Several hybrid methods have been developed to extend population methods with descent-based features to improve their performance in local search.

*   In **Lamarckian learning**, the population method is extended with a local search method that locally improves each individual. The original individual and its objective function value are replaced by the individual’s optimized counterpart.
*   In **Baldwinian learning**, the same local search method is applied to each individual, but the results are used only to update the individual’s objective function value. Individuals are not replaced but are merely associated with optimized objective function values.

## Summary

Population methods are powerful optimization techniques that leverage a collection of design points to explore the design space effectively. By utilizing mechanisms inspired by natural processes, such as genetic algorithms, differential evolution, and particle swarm optimization, these methods can navigate complex landscapes and avoid local minima. Hybrid approaches that combine population methods with local search techniques further enhance their performance, making them versatile tools for solving a wide range of optimization problems.