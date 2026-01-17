+++
date = '2026-01-17T02:53:59+08:00'
draft = false
math = true
tags = ['geometric deep learning', 'machine learning']
title = '几何深度学习简介'
[cover]
    image = "/geometric_deeplearning/geocover.png"
    relative = false
+++

## 几何深度学习的演变

几千年来，几何学一直是人类知识的基本组成部分。古希腊人通过欧几里得的《几何原本》将几何学研究形式化。欧几里得的垄断地位在十九世纪结束了，罗巴切夫斯基、波尔约、高斯和黎曼等人分别构建了非欧几里得几何。

到了那个世纪末，这些研究已经分化成不同的领域，数学家和哲学家们争论这些几何的有效性和相互关系，以及“唯一真实几何”的本质。

年轻的数学家菲利克斯·克莱因指出了摆脱这种困境的出路。克莱因提议将几何学作为不变量的研究，即在某类变换（称为几何的对称性）下保持不变的性质。这种方法通过表明当时已知的各种几何可以通过选择适当的对称变换来定义（使用群论语言形式化），从而创造了清晰度。


## 深度学习简介
深度学习的本质建立在两个简单的算法原则之上：第一，表示或特征学习的概念，即适应性的、通常是分层的特征捕捉每个任务的适当规律性概念；第二，通过局部梯度下降进行学习，通常实现为反向传播。

## 几何深度学习的目标
虽然在高维空间学习通用函数是一个棘手的估计问题，但大多数任务并不是通用的，且带有源于物理世界底层低维性和结构的基本预定义规律。

这种“几何统一”的努力具有双重目的：一方面，它提供了一个通用的数学框架来研究最成功的神经网络架构，如 CNN、RNN、GNN 和 Transformer。另一方面，它提供了一个建设性的程序，将先验物理知识融入神经架构，并为构建尚未发明的未来架构提供了原则性的方法。

## 高维学习
监督机器学习，在其最简单的形式化中，考虑一组 N 个观测值 $D = \{(x_i, y_i)\}_{i=1}^N$，这些观测值是从定义在 $\mathcal{X} \times \mathcal{Y}$ 上的底层数据分布 P 中独立同分布 (i.i.d.) 抽取的。

让我们进一步假设标签 y 是由未知函数 f 生成的，使得 $y_i = f(x_i)$，并且学习问题简化为使用参数化函数类 $F = \{f_\theta \in \Theta\}$ 来估计函数 f。

现代深度学习系统通常在所谓的插值机制中运行，其中估计的 $\widetilde{f}$ $\in F$ 满足 $\widetilde{f}(x_i) = f(x_i)$ 对于所有 $i = 1, . . . , N$。


学习算法的性能是根据从 $P$ 中抽取的样本的预期性能来衡量的，使用损失函数$L(.,.)$：
$$\mathcal{R} = \mathbb{E}_{(x,y) \sim P}[L(\widetilde{f}(x), f(x))]$$

这样构建出来的函数几乎可以逼近一切的函数(通用逼近定理)，但这并不意味着我们不需要用归纳偏置来约束学习问题。给定一个具有通用逼近能力的函数类$\mathcal{F}$,我们可以定义一个复杂度函数:
$$c: \mathcal{F} \rightarrow \mathbb{R}$$,将我们的插值问题定义为：
$$\widetilde{f} = \arg\min_{g \in \mathcal{F}} c(g) \quad s.t. \quad \widetilde{f}(x_i) = f(x_i), \quad i = 1, . . . , N$$

![通用逼近定理示意图](/geometric_deeplearning/geometric01.png)

所以，我们不仅希望函数拟合度好，同时也希望函数的复杂度低。这样的复杂度函数，我们把它称为函数的范数。
这样定义的好处是$\mathcal{F}$变成了Banach空间(配备有范数的完备的向量空间)，我们可以使用泛函分析中的工具来研究它们的性质。

## 几何先验
为了克服高维数据的“维度灾难”，我们必须利用数据的物理特性（对称性和尺度分离）；虽然数据的几何形状（Domain）可能很复杂，但在上面定义的信号（Signals）却构成了一个拥有良好数学性质的希尔伯特空间，让我们依然可以用线性代数和泛函分析的工具来处理它们。

例如，在图像处理中，数据通常被表示为定义在二维欧几里得空间 $\mathbb{R}^2$ 上的信号（像素值）。这种空间具有平移和旋转对称性，这意味着如果我们平移或旋转图像，其内容和结构保持不变。卷积神经网络（CNN）正是利用了这种平移对称性，通过共享权重来捕捉局部特征，从而提高了学习效率和泛化能力。


## 对称性与不变性
在机器学习中，对称性指的是数据或任务在某些变换下保持不变的性质。这些变换可以是平移、旋转、缩放等。
例如，在图像分类任务中，如果我们对图像进行平移或旋转，图像的类别通常不会改变。这种不变性是我们希望模型能够捕捉和利用的。

### 对称群
对称性可以通过群论来形式化。一个群是一个集合，配备有一个二元运算，满足封闭性、结合性、单位元和逆元等性质。

对称操作构成一个群，称为对称群，证明如下：

1. **封闭性**：如果 $g_1$ 和 $g_2$ 是对称操作，那么它们的组合 $g_1 \circ g_2$ 也是一个对称操作，因为连续应用两个对称变换仍然保持数据的不变性。
2. **结合性**：如果 $g_1$ 和 $g_2$ 是对称操作，那么对于任何 $g_3$，都有 $(g_1 \circ g_2) \circ g_3 = g_1 \circ (g_2 \circ g_3)$，因为变换的顺序不会影响最终结果。   
3. **单位元**：存在一个单位操作 $e$，使得对于任何对称操作 $g$，都有 $e \circ g = g \circ e = g$。这个单位操作对应于不进行任何变换。
4. **逆元**：对于每个对称操作 $g$，存在一个逆操作 $g^{-1}$，使得 $g \circ g^{-1} = g^{-1} \circ g = e$。这个逆操作对应于撤销变换。

注意：$g \circ h$ 表示先应用变换 $g$，再应用变换 $h$。

下面以等边三角行为例说明对称群的概念：
![等边三角形的对称群](/geometric_deeplearning/geometric02.png)

在这个例子中，等边三角形的对称群包含六个元素：三个旋转操作（0度、120度、240度）和三个反射操作（通过每个顶点的垂直线）。这些操作满足群的四个性质，因此构成一个群，称为 $D_3$，即三角形的二面体群。
数学上，我们可以把这种操作看做映射：
$$g: \mathcal{X} \rightarrow \mathcal{X}$$
其中 $\mathcal{X}$ 是数据空间。对于一个对称群 $G$中的每个元素 $g \in G$，我们都有一个对应的映射 $g$。
例如，$D_3$同样可以表示为矩阵群：
$$
D_3 = \left\\{ 
  \begin{pmatrix} 1 & 2 & 3 \\\\ 1 & 2 & 3 \end{pmatrix}, 
  \begin{pmatrix} 1 & 2 & 3 \\\\ 2 & 3 & 1 \end{pmatrix}, 
  \begin{pmatrix} 1 & 2 & 3 \\\\ 3 & 1 & 2 \end{pmatrix}, 
  \begin{pmatrix} 1 & 2 & 3 \\\\ 1 & 3 & 2 \end{pmatrix}, 
  \begin{pmatrix} 1 & 2 & 3 \\\\ 2 & 1 & 3 \end{pmatrix}, 
  \begin{pmatrix} 1 & 2 & 3 \\\\ 3 & 2 & 1 \end{pmatrix} 
\right\\}
$$

## 参考资料
1. Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., & Vandergheynst, P. (2017). Geometric deep learning: going beyond Euclidean data. *IEEE Signal Processing Magazine*, 34(4), 18-42.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT press.
3. Cohen, T. S., & Welling, M. (2016). Group equivariant convolutional networks. In *International conference on machine learning* (pp. 2990-2999). PMLR.
4. Kondor, R., & Trivedi, S. (2018). On the generalization of equivariance and convolution in neural networks to the action of compact groups. In *International conference on machine learning* (pp. 2747-2755). PMLR.
5. Wood, T., & Shawe-Taylor, J. (1996). Representation theory and invariant neural networks. *Neural computation*, 8(5), 1003-1013.
6. Mallat, S. (2016). Understanding deep convolutional networks. *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 374(2065), 20150203.
7. Lee, J. M. (2013). *Introduction to smooth manifolds*. Springer Science & Business Media.
