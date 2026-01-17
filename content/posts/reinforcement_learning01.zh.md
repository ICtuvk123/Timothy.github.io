+++
date = 2026-01-16T10:00:00+08:00
draft = false
tags = ['强化学习', '人工智能', '机器学习']
title = '强化学习简介'
math = true
[cover]
    image = "/picture/reinforcement_learning01.png"
    alt = "Reinforcement Learning Diagram"
    caption = "强化学习概念概览"
    relative = false
+++

## 简介
强化学习 (RL) 是一种机器学习类型，其中智能体通过在环境中采取行动来学习做出决策，以最大化某种累积奖励的概念。与从标记数据集学习模型的监督学习不同，RL 依赖于来自环境的奖励或惩罚形式的反馈。

## 策略与行动
智能体维护一个内部状态 $z_t$，并将其传递给策略 $\pi$ 以选择行动 $a_t = \pi(z_t)$。


## 目标
智能体的目标是选择一个策略 π，以最大化预期奖励的总和：


<div>
$$
\begin{aligned}
V^\pi(s_0) = \mathbb{E}_{a_0, s_1, a_1, \dots, a_T, s_T \sim p(\cdot \mid s_0, \pi)} \left[ \sum_{t=0}^T R(s_t, a_t) \right]
\end{aligned}
$$
</div>

其中 $s_0$ 是初始状态，$p(\cdot|s_0, \pi)$ 是策略 π 从状态 $s_0$ 开始诱导的轨迹分布，$R(s_t, a_t)$ 是在状态 $s_t$ 采取行动 $a_t$ 后收到的奖励。
期望是关于：

<div>
$$
\begin{aligned}
p(a_0, s_1, a_1, \dots, a_T, s_T \mid s_0, \pi) 
&= \pi(a_0 \mid s_0) p_{env}(o_1 \mid a_0) \vartheta(s_1 = U(s_0, a_0, o_1)) \\
&= \prod_{t=1}^T \pi(a_t \mid s_t) p_{env}(o_{t+1} \mid a_t) \vartheta(s_{t+1} = U(s_t, a_t, o_{t+1}))
\end{aligned}
$$
</div>

其中 $p_{env}(o_{t+1} \mid a_t)$ 是环境的观测模型，$U(s_t, a_t, o_{t+1})$ 是基于当前状态、采取的行动和收到的观测的状态更新函数。

## 周期性任务与持续性任务
如果智能体可能永远与环境交互，我们称之为持续性任务。
或者，如果智能体的交互在系统进入终止状态或吸收状态后终止，我们说智能体处于周期性任务中。

我们将时间 t 的状态回报定义为未来获得的预期奖励的总和，其中每个奖励乘以折扣因子 $\gamma$：

<div>
$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \\
&= \sum_{k=0}^\infty \gamma^k R_{t+k+1} \\
&= R_{t} + \gamma G_{t+1}
\end{aligned}
$$
</div>



## 架构概览
![overview of the architecture](/picture/agent_and_environment.png)

智能体和环境在每个时间步 t 交互如下：
1. 智能体有一个内部状态 $z_t$，总结了直到时间 t 与环境交互的历史。
2. 智能体根据其策略 $\pi(z_t)$ 选择一个行动 $a_t$，这意味着 $a_t \sim \pi(z_t)$。
3. 然后它通过预测函数 P 预测下一个状态 $z_{t+1|t}$：$z_{t+1|t} = P(z_t, a_t)$ 并可选地预测结果观测 $\hat{o}_{t+1|t} = O(z_{t+1|t})$。
4. 环境具有隐藏状态 $w_t$，智能体无法直接观察到。
5. 环境根据其动力学转换到新状态 $w_{t+1}$：$w_{t+1} \sim M(w_t, a_t)$。
6. 环境生成观测 $o_{t+1} \sim O(w_{t+1})$，并将其发送回智能体。
7.