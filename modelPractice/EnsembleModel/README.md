### 集成学习

#### 概念

**同质：**同种类型的个体学习器，如若干决策树

**异质：**不同类型的个体学习器，如若干决策树+BP神经网络

**基学习器：**同质集成中的个体学习器，如决策树

#### PAC学习

PAC可学习理论，即模型在短时间内利用少量样本能够找到一个假设h'，使其满足
$$
P(E(h') \leq \epsilon) \geq 1-\delta \ \ 0<\epsilon,\delta < 1
$$

#### Hoeffding's theory

假设每个基学习器的错误率为ϵ，且相互之间独立：
$$
P(h_i(x) \neq f(x)) = \epsilon
$$
假设采用简单投票法（少数服从多数）的方法决定集成学习器的结果，则集成器的错误率为：
$$
P(H(x) \neq f(x)) = \sum_k^{\lfloor T/2 \rfloor}(\complement_{T}^{k}(1-\epsilon)^k\epsilon^{T-k})
$$
根据Hoeffding不等式，上式子的上界为：
$$
P(H(x) \neq f(x))  \leq exp(-\frac1{2}T(1-2\epsilon)^2)
$$
当T不断增大，上界逼近于0。

根据个体学习器的生成方式，可分为两类，Boosting方式与Bagging方法。

#### VC维理论

[Boosting]()
[Bagging]()