#### 高斯混合模型（Gaussian Mixture Model，GMM）

高斯混合模型是指具有如下形式的概率分布模型：
$$
P(y|\theta) = \sum_{k=1}^K\alpha_k\phi(y|\theta_k)\\
(\sum_{k=1}^k\alpha_k = 1;\theta_k = （\mu_k,\sigma_k）)
$$
相比于单高斯模型，GMM可以拟合更大地，更好地拟合假设空间的样本。

下图是单高斯模型和混合高斯模型的对比，可以看出，通过调整各高斯模型的权重，混合高斯模型可以拟合更加复杂多变的曲线

![截屏2020-03-07下午7.17.33](/Users/jiayi/Library/Application Support/typora-user-images/截屏2020-03-07下午7.17.33.png)

在使用混合高斯模型时，通常我们只有可以观测到的数据，比如分类类别等，但**反映观测数据y来自第k个分类模型等数据是未知的**。这时候我们可以使用EM算法来解决这个问题。

![截屏2020-03-07下午8.21.07](/Users/jiayi/Library/Application Support/typora-user-images/截屏2020-03-07下午8.21.07.png)

以上图的例子，假设用2类二维高斯分布模型拟合该分布，初始化α为1/2，μ与σ赋予一个初始值，使用EM算法求解模型参数。样本数据点属于哪个高斯分布是隐变量。E-step对数据点属于哪类分布做出估计，M-step优化参数，不断迭代，直到参数(α,μ,σ)收敛。

[详解EM算法与混合高斯模型](https://blog.csdn.net/lin_limin/article/details/81048411)

[二维高斯分布的参数分析](https://blog.csdn.net/lin_limin/article/details/81024228)

[协方差矩阵](http://pinkyjie.com/2010/08/31/covariance/)

### 