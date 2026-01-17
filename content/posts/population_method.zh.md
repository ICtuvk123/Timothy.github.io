+++
date = 2026-01-16T10:00:00+08:00
title = "种群方法"
description = "深入探讨种群方法" 
tags = ["编程", "种群方法", "数学建模"]
math = true
[cover]
    image = "/picture/population_cover.png"
    alt = "Population Method Diagram"
    caption = "优化中各种种群方法的概述"
    relative = false
+++

本文介绍各种涉及使用一组设计点（称为个体）进行优化的种群方法。

拥有分布在整个设计空间中的大量个体有助于算法更有效地探索设计空间。

许多种群方法本质上是随机的，并且通常很容易并行化计算。

## 种群迭代

种群方法迭代地改进 m 个设计的种群

<div>
$$
x^{(1)}, x^{(2)}, \dots, x^{(m)}
$$
</div>

特定迭代中的种群被称为一代。

算法的设计使得种群中的个体在多代后收敛到一个或多个局部最小值。

本文讨论的各种方法在如何从当前一代生成新一代方面有所不同。

种群方法从初始种群开始，通常是随机生成的。初始种群应分布在设计空间中以鼓励探索。

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

以下是用于生成初始种群的分布：
- 均匀分布 (Uniform Distribution)
- 正态分布 (Normal Distribution)
- 柯西分布 (Cauchy Distribution)

![Population Initialization](/picture/population_method01.png)

## 遗传算法
遗传算法受到生物进化中自然选择过程的启发。

与每个个体关联的设计点表示为染色体。在每一代中，适应度较高的个体的染色体通过选择、交叉和变异操作传递给下一代。

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

### 选择

选择操作从当前种群中选择个体作为下一代的父母。常见的选择方法包括基于排名的选择、锦标赛选择和轮盘赌选择。

基于排名的选择根据个体的适应度对其进行排序，并相应地分配选择概率。
![Rank-based Selection](/picture/population_method02.png)

锦标赛选择随机选择一个个体子集，并选择其中适应度最高的作为父母。
![Tournament Selection](/picture/population_method03.png)

轮盘赌选择分配与适应度成比例的选择概率，使适应度较高的个体有更高的被选中机会。
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

### 交叉

交叉结合父母的染色体以形成后代。与选择一样，有几种交叉方案

1. 在单点交叉中，选择一个随机交叉点，并交换父母染色体的片段以创建后代。
![Single-point Crossover](/picture/population_method06.png)
2. 在两点交叉中，选择两个交叉点，并交换这些点之间的片段。
![Two-point Crossover](/picture/population_method07.png)
3. 在均匀交叉中，每个基因以相等的概率独立地从父母之一中选择。
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

### 变异
变异引入随机变化以保持种群内的遗传多样性。常见的变异方法是零均值高斯分布。
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

染色体中的每个基因通常有很小的概率 λ 发生改变。对于具有 m 个基因的染色体，此变异率通常设置为 λ = 1/m，平均每个子染色体产生一个变异。

![Gaussian Mutation](/picture/population_method09.png)

## 差分进化
差分进化 (DE) 是一种基于种群的优化算法，利用向量差分来扰动种群成员。

对于每个个体 x：
1. 从种群中选择三个不同的个体 a、b 和 c。
2. 通过将 b 和 c 之间的加权差添加到 a 来生成试验向量：
   $$ 
   v = a + F \cdot (b - c) 
   $$
   其中 F 是缩放因子，通常在 [0, 2] 范围内。
3. 评估试验向量 v 的适应度。如果 v 的适应度优于 x，则在下一代中用 v 替换 x；否则，保留 x。

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

## 粒子群优化
粒子群优化引入动量以加速向最小值的收敛。种群中的每个个体（或粒子）都会跟踪其当前位置、速度以及迄今为止看到的最佳位置。动量允许个体在有利方向上积累速度，而不受局部扰动的影响。

在每次迭代中，每个个体都会加速向其看到的最佳位置以及任何个体迄今为止找到的最佳位置移动。加速度由随机项加权。

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

## 萤火虫算法
萤火虫算法的灵感来自于萤火虫闪烁光芒以吸引同种配偶的方式。在萤火虫算法中，种群中的每个个体都是一只萤火虫，可以通过闪光来吸引其他萤火虫。

在每次迭代中，所有萤火虫都向所有更有吸引力的萤火虫移动。萤火虫的吸引力与其表现成正比。

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

## 布谷鸟搜索
布谷鸟搜索是另一种受自然启发的算法，以布谷鸟命名，布谷鸟从事一种巢寄生行为。布谷鸟将蛋产在其他鸟类的巢中。

在布谷鸟搜索中，每个巢代表一个设计点。可以使用来自巢的莱维飞行（Lévy flights）产生新的设计点，这是步长服从重尾分布（通常是柯西分布）的随机游走。

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

## 混合方法
许多种群方法在全局搜索中表现良好，能够避免局部最小值并找到设计空间的最佳区域。不幸的是，与下降方法相比，这些方法在局部搜索中表现不佳。

已经开发了几种混合方法，将种群方法与基于下降的特征相结合，以提高其在局部搜索中的性能。

*   在 **拉马克学习 (Lamarckian learning)** 中，种群方法扩展了局部搜索方法，该方法局部改进每个个体。原始个体及其目标函数值被个体的优化对应物所取代。
*   在 **鲍德温学习 (Baldwinian learning)** 中，相同的局部搜索方法应用于每个个体，但结果仅用于更新个体的目标函数值。个体不会被替换，而只是与优化的目标函数值相关联。

## 总结

种群方法是强大的优化技术，利用一组设计点来有效地探索设计空间。通过利用受自然过程（如遗传算法、差分进化和粒子群优化）启发的机制，这些方法可以导航复杂的景观并避免局部最小值。结合种群方法和局部搜索技术的混合方法进一步增强了它们的性能，使它们成为解决各种优化问题的通用工具。