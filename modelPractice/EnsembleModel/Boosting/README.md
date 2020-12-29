#### Boosting

Boosting方法中个体学习器之间存在强相互依赖关系，基本流程大概为：

- 根据初始训练集训练出一个基学习器
- 根据学习器表现，对样本集分布进行调整，使得在上一步学习器错误预测的样本收获更多关注
- 根据调整后的训练集，训练下一个学习器
- 重复2、3步，直到得到T个学习器

Boosting方法之间的区别，是调整样本，也就是第三步存在差异。

##### AdaBoosting

AdaBoosting 最初是在PCA学习理论下提出的，后面被扩展至Gradient Boosting，形成了一个更普适的框架。

Adaptive Boosting可以看作Gradient Boosting 的一个特例，Adaptive是指样本分布的Adaptive。

AdaBoosting是一个模型为加法模型、损失函数为指数损失函数、学习算法为前向分布算法的算法

**样本调整方式**：通过提高在前一轮弱分类器中被错误分类的样本的权值，削弱正确分类样本的权值

**分类器组合方式**：通过加权多数表决方式，提高误差率小的分类器的权重。
$$
H(x) = \sum_t^T\alpha_th_t(x)
$$
AdaBoosting是特殊的前向分布算法，采用基本分类器与指数损失函数。

**前向分布算法**：（李航，统计学习方法）
$$
\begin{align*}
& 输入T = \{(x_1,y_1),(x_2,y_2),(x_3,y_3),...,(x_N,y_N)\};损失函数L(y,f(x));基函数集\{b(x;\gamma)\} \\
& 输出：加法模型f(x) = \sum_i^M \beta_i f_i(x) \\

& (1) 训练f_0(x) \\
&(2) 损失函数最小化，对接下来的1-M：\\
&(\beta_m，\gamma_m) = argmin_{\beta,\gamma}\sum_{i}^{N}L(y_i,f_{m-1}(x_i) +\beta b(x;\gamma)) \\
&得到下一步的参数\beta_m,\gamma_m. \\
&更新 f_m(x) = f_{m-1}(x_i) +\beta_m b(x;\gamma_m) \\
&(3) 得到最后的加法模型f_M(x)
\end{align*}
$$
*最大熵原理：当我们需要对一个随机事件的概率分布进行预测时，我们的预测应当满足全部已知的条件，而对未知的情况不要做任何主观假设。*

**指数损失函数**(Exponential Loss)：
$$
exploss = \frac{1}{N}\sum_i^Nexp(-y_i*f(x_i))
$$
AdaBoosting计算步骤：

1. 根据均分布的训练集，得到h0(x)

2. 计算模型当前错误率epsilon，如果大于随机预测错误率，则整个算法停止计算。

3. 最小化指数损失函数:
   $$
   \begin{align*}
   L & = \frac{1}{N}\sum_i^Nexp(-y_i*\alpha_th_t(x_i)) \\
   & = E(e^{-\alpha_t}(y_i=h_t(x_i) + e^{\alpha_t}(y_i \neq h_t(x_i)))) \\
   & = e^{-\alpha_t}(1-\epsilon_t) + e^{\alpha_t} \epsilon_t \\
   对\alpha_t进行求导： \\
   \frac{\partial L}{\partial \alpha_t} 
   & = - e^{-\alpha_t}(1-\epsilon_t) + e^{\alpha_t} \epsilon_t \\
   得到更新值 \alpha_t = \frac{1}{2} ln(\frac{1-\epsilon_t}{\epsilon_t})
   \end{align*}
   $$

4. 调整样本分布：
   $$
   w_{t+1,i} = \frac{w_{t,i}}{Z_t} exp(-\alpha_ty_ih_t(x_i)),i=1,2,...,N \\
   Z_t = \sum_{i=1}^{N}w_{t,i}exp(-\alpha_ty_ih_t(x_i)) 即总和，用于规范化。
   $$

5. 

6. 集合所有分类器，得到最终的模型

AdaBoosting中，由于指数损失函数对噪声数据太敏感，因此该算法模型在噪声较多的数据集上表现不佳。

##### Gradient Boosting

Gradient Boosting是一个模型为加法模型、学习算法为前向分布算法的算法。

针对AdaBoosting对问题，后来提出一个更加灵活的框架方法，即梯度提升。AdaBoosting可以看作GB的一个特例，GB的整体框架即通过梯度下降方法，求解加性模型。

*（上面对AdaBoosting是通过一种更加直觉型的方式去理解解释的，其实它也可以从梯度方面去进行阐述。）*

GB整体的思想，其实是一种**贪婪搜索算法**，即使用启发式算法，在每一步做出局部最优选择，以找到一个全局最优解。学习方法是在每一步去拟合上一步学习到的模型与真实值之间的残差。

**算法框架：**

1. 学习器最终表达式：

$$
H(x) = \sum_t^T\alpha_th_t(x)
$$

2. 损失函数一般形式：

   D是当前t下样本分布，Ht是当前t的总学习器
   $$
   L(H_t|D) = \frac{1}{N}\sum_i^Nerr(H_t(x_i),y_i) = E_{X\in D}[err(H_t(x),f(x))]
   $$

3. 根据前向分布算法：

   alpha是当前t下基学习器h(x)权重
   $$
   H_t(x) = H_{t-1}(x) + \alpha_th_t(x)
   $$

   可以看到，当前步是在前人做出的工作H{t-1}的基础上，把剩下的部分进行弥补。

   应该如何求得权重与当前步骤的模型呢？


   $$
   L(H_t|D) = E_{X\in D}[err(H_{t-1}(x) + \alpha_th_t(x),f(x))] \\
   对损失函数，在H_{t-1}处一阶泰勒展开：\\
   L(H_t|D) =  E_{X\in D}[err(H_{t-1}(x),f(x)) + \frac{\partial err}{\partial H_t(x)}|_{H_t(x) = H_{t-1}(x)}(H_t(x) - H_{t-1}(x))] \\
   $$
   可以看到损失函数和当前基学习器ht(x)的关系，为了降低Loss，err(Ht-1,f)为常数constant，则需要对后面部分进行优化。

   为了降低Loss，因此肯定右侧选择损失函数的负梯度方向。
   $$
   求解h_t(x),\alpha_t: (第一部分为常数，只需要考虑第二部分，首先忽略\alpha_t)\\
   h_t(x) = argmin E(\frac{\partial err}{\partial H_t(x)}|_{H_t(x) = H_{t-1}(x)}h_t(x)) \\
   得到h_t(x)后，求解\alpha_t \\
   \alpha_t = argmin E(err(H_{t-1}(x) + \alpha_th_t(x),f(x)))
   $$

**GB For Regression:**
$$
h_t(x) = argmin E(2(y-f(x)h_t(x)) \\
为了控制h_t(x)不要取得太大，增加h_t(x)^2的惩罚项，并进行组合，得到下式子 \\
= argmin E((h_t(x) - (f(x) - y))^2 + constant)
$$
可以看到，选用平方损失函数时是通过残差rediual来求当前最佳的基学习器。其他损失函数则不然，只有使用平方损失函数时，负梯度 = 残差。

之后，求解alpha
$$
\alpha_t = argmin E((\alpha_th_t(x) - (f(x) - y))^2 + constant)
$$
**回归问题：**

Square loss
$$
L = (y - f(x))^2 \\
\frac{\partial L}{\partial f(x)} = -2(y - f(x)) = residual
$$
Absolute loss
$$
L = |y-f(x)|
$$
Huber loss
$$
L = \left\{
\begin{array}{**lr**}
\frac{1}{2}(y-f(x))^2, \ for|y-f(x)| \le \delta, \\
\delta|y-f(x)|-\frac{1}{2}\delta^2,  \ otherwise
\end{array}
\right.
$$

##### GBDT

*Gradient Boosting Decision Tree*

**CART：**GBDT中，采用的基学习器为CART，具体介绍可见前文，在CART中，回归使用平方损失，而分类采用基尼指数。

在GBDT中，分类与回归都是用回归树，如果采用分类树，每轮之间负梯度计算得到的值其实是没有意义的，如二分类A-B。

通过对Loss进行梯度下降优化后，我们可以得到ht(x)，但**如何在决策树中表示呢？**

考虑一个决策树，输入一个x，x最终会被分到一个叶子结点，因此可以通过遍历所有的叶子结点，找到该样本点x。

那么，x与y之间的映射关系，我们可以通过 $w_{q(x_i)}$ 进行表示$w$表示该叶结点表示的权重值，而$q(x_i)$则表示遍历所有叶结点后该x所处的叶结点编号。

因此上述GB最终的公式化简为：
$$
h_t(x) = argmin E(g_iw_{q(x_i)})\\
g_i表示err的一阶导
$$
使用叶子结点集合，进一步简化：
$$
Loss = \sum_t^T((\sum_{i \in I_t}g_i)w_t)\\
$$
GBDT目标是寻找一个CART树，并且，通过该树计算得到的Loss是最小的。

那么可以：

(1) 枚举所有可能的树结构

(2) 计算Loss，分数越小，结构越好

(3) 找到最佳树结构，预测新值

但是枚举所有的树结构是NP的，因此需要通过启发式策略来处理：

level-wise：XGBoost

leaf-wise：LightGBM

##### XGBoost

XGBoost算法核心思想与GBDT是一致的，但在实现上更加高效，稳定。

XGBoost的加法模型**不需要计算基模型的权重**：
$$
H_t(x) = \sum_i^kh_t(x)
$$


相比于GBDT，XGBoost算法优化的目标函数**使用二阶导以及添加正则化项**：
$$
\begin{align}
L(H_t(x),y) &= \sum_i^Nerr(H_t(x_i),y_i) \\
& \xlongequal{H_t(x) - H_{t-1}(x) = h_t(x)}\sum_i^Nerr(H_{t-1}(x_i) + h_t(x_i),y_i) \\
\end{align}
$$
Loss func 加上**正则化项**，就是最终的目标优化函数：

*正则化项是为了抑制过拟合，降低模型复杂度*
$$
Obj = L(H_t(x),y) + \sum_t^{k}\Omega(H_t) \\
\Omega(H_t)  = \gamma T_t + \frac{1}{2}\lambda\sum_{j=1}^T\omega_j^2 \\
其中，\gamma,\lambda是事先给定的超参数，T_t表示叶子结点数目，\omega_j表示对应叶子结点的节点权重
$$
正则化项的计算实例图：

![](https://pic1.zhimg.com/80/v2-8fbf510ce664cfbb2c67a3cf5c9d6f18_1440w.jpg)

对目标函数，在Ht-1处，进行二阶Taylor展开：
$$
\begin{align}
Obj &= L(H_t(x),y) + \sum_t^{k}\Omega(H_t) \\
&= \sum_i^N(err(H_{t-1}(x_i),y_i)+\frac{\partial err}{\partial H_t(x)}|_{H_t(x) = H_{t-1}(x)}h_t(x)+\frac{1}{2}\frac{\partial^2 err}{\partial H_t(x)}|_{H_t(x) = H_{t-1}(x)}h_t^2(x)) +  \sum_t^{k}\Omega(H_t)  \\
& gi = \frac{\partial err}{\partial H_t(x)}|_{H_t(x) = H_{t-1}},
si = \frac{\partial^2 err}{\partial H_t(x)}|_{H_t(x) = H_{t-1}(x)} \\
& = \sum_i^N(err(H_{t-1}(x_i),y_i)+g_ih_t(x)+\frac{1}{2}s_ih_t^2(x)) +  \sum_t^{k}\Omega(H_t)
\end{align}
$$
去除式子中的常数项，因为对优化没有影响：
$$
= \sum_i^N(g_ih_t(x)+\frac{1}{2}s_ih_t^2(x)) +  \sum_t^{k}\Omega(H_t) （1）
$$
因此，只需要求的损失函数的一阶导、二阶导，之后对目标函数Obj，求其极值即可得到ht(x)。

考虑$h_t(x)$在决策树中的表示：
$$
\begin{align}
&= \sum_i^N(g_iw_{q(x)}+\frac{1}{2}s_iw^2_{q(x)}) +   \gamma T_t + \sum_t^T( \frac{1}{2}\lambda\sum_{j=1}^T\omega_j^2 )\\
&\xlongequal{考虑化简为叶结点}\sum_t^T((\sum_{i\in I_t}g_i)w_t+\frac{1}{2}(\sum_{i\in I_t}s_i)w^2_t  + \frac{1}{2}\lambda\omega_t^2 )+ \gamma T_t \\
& G_t = \sum_{i\in I_t}g_i, H_t = \sum_{i\in I_t}s_i \\
& = \sum_t^T(G_tw_t+\frac{1}{2}H_tw^2_t  + \frac{1}{2}\lambda\omega_t^2 )+ \gamma T_t \\
&\xlongequal {G_t,H_t已知，，对Obj求w一阶导为0}  w_t = - \frac{G_t}{H_t+\lambda} \\
\end{align}
$$
构建决策树算法：

XGBoost采用二叉树，所有结点在开始都处于一个结点，之后逐层进行二分裂。

因此，算法的目标就是选择最优的特征下的最优分割点。

与原本的C4.5、CART不同，XGBoost利用了推导出的Obj。
$$
Obj_1 = - \frac{1}{2}[\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] + \gamma \\
二分裂后的Obj值: \\
Obj_2 = - \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda}]+2\gamma \\
收益值：\\
Gain = Obj_1 - Obj_2 =\frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda}  - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma
$$
**收益值算法**确定之后，划分算法步骤为：

- 对所有特征按升序进行排列，对每个间隔，尝试作为分割点，计算其收益值。（这一块可以并行，可以cache）
- 选择收益最大的，作为分割点。
- 递归执行，直到满足特定条件如高度限制、收益阈值等为止。

该算法的缺点是需要将数据全部读入内存，而且，如果对G、H进行cache时，数据量更大。

XGBoost提出一种优化思路：

根据一定策略选出一些分割点进行遍历。作者采用的是Weight Quantile Sketch，也就是采用了一种对loss影响权重的等值划分算法。

那么需要解决两个问题：

- 对loss的影响权重怎么计算
- 划分的间隔怎么设定合理

1. 对（1）式进行化简

$$
\begin{align*}
& = \sum_i^N(g_ih_t(x)+\frac{1}{2}s_ih_t^2(x)) +  \sum_t^{k}\Omega(H_t) \\
& = \sum_i^N(\frac{1}{2s_i}\times2 \times s_ig_ih_t(x)+\frac{1}{2}s_ih_t^2(x)) +  \sum_t^{k}\Omega(H_t) \\
& = \sum_i^N\frac{s_i}{2}(\frac{2 \times g_ih_t(x)}{s_i}+h_t^2(x)) + \sum_t^{k}\Omega(H_t)  \\
& = \sum_i^N\frac{s_i}{2}(\frac{2 \times g_ih_t(x)}{s_i} + h_t^2(x)+\frac{g_i^2}{s_i^2} - \frac{g_i^2}{s_i^2}) + \sum_t^{k}\Omega(H_t) \\
& = \sum_i^N\frac{s_i}{2}((\frac{g_i}{s_i} + h_t(x))^2 - \frac{g_i^2}{s_i^2}) + \sum_t^{k}\Omega(H_t) \\
& = \sum_i^N\frac{s_i}{2}((h_t(x) - (-\frac{g_i}{s_i}))^2) + \sum_t^{k}\Omega(H_t) + constant
\end{align*}
$$

可以看出，s_i就是权重值。

2. 划分间隔

   超参数设定，首先计算总权重，之后，寻找满足条件的切分点即可。

   ![](https://img-blog.csdnimg.cn/20200407172358195.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)

这样，就得到了每轮的决策树模型。

其次，XGBoost提出了**Shrinkage的超参数**，用于约束每棵树的作用，**防止过拟合**，使得学习过程更加**平滑**。

XGBoost对缺失值的处理：**稀疏感知算法**

将缺失值节点带入左结点计算下，再带入右结点计算下，选择增益高的作为最终的分裂方案。

##### LightGBM

Light Gradient Boosting Machine。

LGBM对XGBoost的优化，主要体现在运行速度与空间利用率熵的提升。直方图算法、单边梯度抽样算法（GOSS）、互斥特征捆绑算法（EFB）。

1. **直方图算法**

   直方图算法针对的是XGBoost中对分裂点数量的优化。

   - 对连续浮点特征离散化为k个整数，k个bin；
   - 遍历特征，将处于bin区间的样本进行梯度累加和个数统计；
   - 根据离散值，遍历寻找最佳的分割点；

   ![](https://pic4.zhimg.com/80/v2-7edb458940edd9c83563eba7427b07f3_1440w.jpg)

   通过该种方法，减少了分裂点个数，同时只需要保存特征离散后的值，内存消耗也降低了。

   除此之外，在节点进行分裂时，可以通过**直方图作差加速**，只需要根据父节点的直方图和兄弟结点的直方图，就可以得到另一个叶结点的直方图。

   因此，每次只需要找到最佳的特征及其分割点，其余特征不需要重新计算，可以做差得到。

   ![](https://pic2.zhimg.com/80/v2-457cdfa909f3ba14ed312dcf44269339_1440w.jpg)

2. **单边梯度抽样算法（GOSS）**

   GOSS从减少样本数量角度出发，减少权重小的样本。但不同于Ada，样本是没有权重的。在LGBM中，使用梯度作为权重，保留梯度大的样本，大梯度对降低损失效果更重要；对梯度小的样本，采取随机抽样，同时对梯度小的样本引入一个常数进行平衡。

   - 对要进行分裂的特征，从大到小进行排序；
   - 选择前a%的样本，后b%的样本，计算增益时，后b%乘上$\frac{1-a}{b}$放大梯度小的样本；

   ![](https://mmbiz.qpic.cn/mmbiz_png/FIzOEib8VQUqpnqWXRp4qZnaAMjQLj38PU17PwXVXFGhTClRyrfiaX6wnm2iclicPcg5fFavAiaia2CKuqQjfAGWywYw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

3. **互斥特征捆绑算法（EFB）**

   高维度的特征通常是稀疏的。LGBM使用冲突比率衡量两个特征之间的不互斥程度。

   将冲突比率小于给定阈值的两个特征，进行捆绑，降低了特征的数量。

   可以通过以下的方案进行特征的捆绑。

   ![](https://pic4.zhimg.com/80/v2-0ea30952f393d3f6a84268b565c877c7_1440w.jpg)

   但是上述特征数量过多时，效率比较低，将特征按照非零值个数进行排序，因为更多的非零值的特征会导致更多的冲突，所以跳过了上面的第一步，直接排序然后第三步分簇。

   因为需要将特征从合并特征中分离出来，因此需要加一个offset。

   ![](https://pic3.zhimg.com/80/v2-e0d6e4774309ed962ed3b919f480a91a_1440w.jpg)

除此之外，与XGBoost每层每层的生成顺序不同，LGBM每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。不过这样可能增加过拟合的风险，可以通过设定max_depth来控制树的最大高度。

![](https://pic3.zhimg.com/80/v2-22bd8eb2b98d4571756ed06c4a920206_1440w.jpg)

LGBM其他细节暂时不做深究...之后再补

[白话机器学习](https://zhuanlan.zhihu.com/p/149522630)
