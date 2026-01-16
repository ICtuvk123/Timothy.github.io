+++
date = 2026-01-16T10:00:00+08:00
draft = false
tags = ['reinforcement learning', 'AI', 'machine learning']
title = 'Introduction to Reinforcement Learning'
math = true
[cover]
    image = "/picture/reinforcement_learning01.png"
    alt = "Reinforcement Learning Diagram"
    caption = "An overview of reinforcement learning concepts"
    relative = false
+++

## Introduction
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. Unlike supervised learning, where the model learns from a labeled dataset, RL relies on feedback from the environment in the form of rewards or penalties.

## Policy and Action
The agent maintains an internal state $z_t$, which it passes to its policy $\pi$ to choose an action $a_t = \pi(z_t).$


## Goal
The goal of the agent is to choose a policy π so as to maximize the sum of expected rewards:


<div>
$$
\begin{aligned}
V^\pi(s_0) = \mathbb{E}_{a_0, s_1, a_1, \dots, a_T, s_T \sim p(\cdot \mid s_0, \pi)} \left[ \sum_{t=0}^T R(s_t, a_t) \right]
\end{aligned}
$$
</div>

where $s_0$ is the initial state, $p(\cdot|s_0, \pi)$ is the distribution over trajectories induced by the policy π starting from state $s_0$, and $R(s_t, a_t)$ is the reward received after taking action $a_t$ in state $s_t$.
and the expectation is wrt:    

<div>
$$
\begin{aligned}
p(a_0, s_1, a_1, \dots, a_T, s_T \mid s_0, \pi) 
&= \pi(a_0 \mid s_0) p_{env}(o_1 \mid a_0) \vartheta(s_1 = U(s_0, a_0, o_1)) \\
&= \prod_{t=1}^T \pi(a_t \mid s_t) p_{env}(o_{t+1} \mid a_t) \vartheta(s_{t+1} = U(s_t, a_t, o_{t+1}))
\end{aligned}
$$
</div>

where $p_{env}(o_{t+1} \mid a_t)$ is the environment's observation model, and $U(s_t, a_t, o_{t+1})$ is the state update function based on the current state, action taken, and observation received.

## Episodic vs continual tasks
If the agent can potentially interact with the environment forever, we call it a continual task.
Alternatively, we say the agent is in an episodic task if its interaction terminates once the system enters a terminal state or absorbing state.

We define the return for a state at time t to be the sum of expected rewards obtained going forwards,
where each reward is multiplied by a discount factor $\gamma$:

<div>
$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \\
&= \sum_{k=0}^\infty \gamma^k R_{t+k+1} \\
&= R_{t} + \gamma G_{t+1}
\end{aligned}
$$
</div>



## Overview of the architecture
![overview of the architecture](/picture/agent_and_environment.png)

The agent and environment interact at each time step t as follows:
1. The agent has a internal state $z_t$ that summarizes its history of interaction with the environment up to time t.
2. The agent selects an action $a_t$ based on its policy $\pi(z_t)$, which means that $a_t \sim \pi(z_t)$.
3. Then it predicts the next state $z_{t+1|t}$ via the predict function P: $z_{t+1|t} = P(z_t, a_t)$ and optionally predicts the resulting observation $\hat{o}_{t+1|t} = O(z_{t+1|t})$.
4. The environment has hidden state $w_t$, which is not directly observable by the agent.
5. The environment transitions to a new state $w_{t+1}$ according to its dynamics: $w_{t+1} \sim M(w_t, a_t)$.
6. The environment generates an observation $o_{t+1} \sim O(w_{t+1})$ which is sent back to the agent.
7.