#### 感知机（Perceptron）

$$
f(x) = sign( wx + b)\\
y = \{-1,+1\}
$$

以错误驱动的思想，使用误分类点更新参数w和b，不断逼近正确的分类超平面的过程。

为了更新参数w和b，需要一个可导的损失函数，所以采用误分类点到超平面总距离，作为优化到目标：
$$
\frac{-\sum_{x\in{M}}yi(w_ix_i+b)}{||w||}\propto{L(w,b)}\\
（M是误分类点集合）
$$

- 原始形式：

  1. w，b设置初值，w_0,b_0

  2. 训练集中选取数据（x_i,y_i）

  3. 如果y_i(wx_i+b) <= 0

     w <= w + ηyixi

     b <= b + ηyi

  4. 转至2，直至无误分类点

  该算法的收敛性由Novikoff定理给出。

  该定理证明：

  ![IMG_0047](/Users/jiayi/Documents/IMG_0047.jpg)

- 对偶形式

  对偶形式的感知机模型是对原始形式执行速度的优化，不同于原始形式，它将在某点上因为误分类而进行更新的次数记录下来，在最后统一更新参数。

  对偶形式由原始形式的更新参数方式推出。

  1. α = (α1,α2,α3,...αΝ)^Τ ,αi 代表在点(xi,yi)因为误分类而进行更新的次数，初值为0 学习率为η

     b = 0

  2. 训练集中选取数据（x_i,y_i）

  3. 如果 
     $$
     y_i(\sum_{j=1}^N{α_jy_jx_j} x_i + b) \leq 0
     $$
     αi <= αi + η  	b <= b + ηyi

  4. 转至2直至没有误分类数据

  步骤3中的向量内积，可以使用Gram矩阵提前算出：
  $$
  G = [x_i·x_j]_{N*N}
  $$
  可以看出，借助Gram对称矩阵，降低了计算成本，原先的循环计算转化为可复用的内积，效率大大提升。

  参考链接：

  [感知机算法](https://www.cnblogs.com/yifanrensheng/p/12354924.html)

  [Gram矩阵定义](https://blog.csdn.net/wangyang20170901/article/details/79037867/)

  [Novikoff收敛性定理证明](https://blog.csdn.net/iwangzhengchao/article/details/54486473?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)

#### 