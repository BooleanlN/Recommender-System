### 对数模型
#### LR算法

线性回归用于回归，当需要做对是分类任务时，可以引入一个函数，将linear model的是输出与类别预测值练习起来。

**Logistic Function**：a smooth、monotonic、**sigmoid function** of s
$$
\theta(x) = \frac{e^s}{1 + e^s} = \frac{1}{1+e^{-s}}
$$
**Logistic Regression：**use logistic function and linear model to approximate target function.
$$
h(x) = \frac{1}{1+e^{-w^T*x}}
$$
事件发生机率odds的定义：
$$
odds = \frac{P(Y=1|x)}{1 - P(Y=1|x)} = \frac{\pi(x)}{1-\pi(x)} \\
$$
对odds取自然对数，可得到回归式：
$$
logit(odds) = ln(odds) = w^Tx+b
$$
这也是该模型被称作**对数几率回归**的原因。

对于模型参数的预测，我们可以采用极大似然估计。假设存在训练集合$T= {(x_1,y_1),(x_2,y_2),...,(x_n,y_n)},\ y\in\{0,1\}
$$
\begin{align*}
argmax \ likelihood(h) &= \sum_i^N[ln{\pi(x_i)}  + (1-y_i)ln(1-{\pi(x_i)})] \\
&= \sum_i^N[y_iln{\frac{\pi(x_i)}{1-{\pi(x_i)}}}  + ln(1-{\pi(x_i)})] \\
&= \sum_i^N[ln{(w^Tx_i)} - ln(1+{e^{w^Tx_i}})] \\
&\pi(x) = \frac{1}{1 + e^{-w^T*x}}
\end{align*}
$$

求解该最优化问题：

1. 牛顿迭代法

   **Hasse矩阵：**

2. 拟牛顿法：

   

3. 梯度下降法

   对likelihood，求其极大值，得到w的估计值，其中$\alpha$是超参数学习率
   $$
   w_{i+1} = w_i + \alpha * \frac{\partial \ {likelihood}}{\partial w}
   = w_i + \alpha *  \sum_i^N(y_i -\frac{1}{1+e^{-w^Tx_i}})*x_i
   $$

   ```python
   def batchGD(self,epochs):
           """梯度下降"""
           for epoch in range(epochs):
               print(">>>>第{}轮训练开始>>>\n".format(str(epoch)))
               loss = np.zeros(self.X.shape[1])
               for index,x in enumerate(self.X):
                   loss =loss + (self.y[index] - self.sigmoid(x)) * x
               self.weight = self.weight + self.learning_rate * loss
               #self.weight = self.weight + self.learning_rate * np.sum((self.y - np.apply_along_axis(self.sigmoid,1,self.X))*self.X.T)
               print(">>>>当前accuracy：{}>>>>>>\n".format(self.loss()))
   ```

4. 随机梯度下降法
   $$
   \begin{align*}
   & for \ j \ of  \ m: \\
   &w_{i+1} = w_i + \alpha * \frac{\partial \ {likelihood}}{\partial w}
   = w_i + \alpha *  (y_j -\frac{1}{1+e^{-w^Tx_j}})*x_j
   \end{align*}
   $$

   ```python
   def stochasticGD(self,epochs):
           """随机梯度下降"""
           for epoch in range(epochs):
               print(">>>>第{}轮训练开始>>>\n".format(str(epoch)))
               for index,x in enumerate(self.X): 
       #                 print(self.learning_rate * (self.y[index] - self.sigmoid(x)))
                   if not self.with_regular:
                       self.weight = self.weight + self.learning_rate * (self.y[index] - self.sigmoid(x)) * x
                   else:
                       self.weight = self.weight + self.learning_rate * (self.y[index] - self.sigmoid(x)) * x
               print(">>>>当前accuracy：{}>>>>>>\n".format(self.loss()))
   ```

   

5. 小批量梯度下降
   $$
   \begin{align*}
   & for \ j \ of  \ \{ batchsize,batchsize,batchsize\}: \\
   & \ \ \ for \ batch \ batchs: \\
   & \ \ \ \ \ w_{i+1} = w_i + \alpha * \frac{\partial \ {likelihood}}{\partial w}
   = w_i + \alpha * \frac{1}{batchSize} \sum_j^{batch}(y_j -\frac{1}{1+e^{-w^Tx_j}})*x_j
   \end{align*}
   $$

   ```python
   def miniBatchGD(self,epochs,batch_size=20):
           """小批量样本梯度下降"""
           for epoch in range(epochs):
               start = 0
               print(">>>>第{}轮训练开始>>>\n".format(str(epoch)))
               for start in range(0,self.X.shape[0],batch_size):
                   loss = np.zeros(self.X.shape[1])
                   for index,x in enumerate(self.X[start:start+batch_size]):
                       loss =loss + (self.y[index] - self.sigmoid(x)) * x
                   self.weight = self.weight + self.learning_rate *(1/batch_size) * loss
               print(">>>>当前accuracy：{}>>>>>>\n".format(self.loss()))
   ```

**Multi-nominal LR 多项logistic回归**

上述LR只能解决而分类问题，如果需要进行多分类时，一般有两种方案：

1. 拆分为多个二分类问题
2. 对LR模型进行改造

**多分类拆分：**

n个类别

- OvO(one vs one)，将多个分类，两两配对，使用$n(n-1)/2$个分类器，最后采用结果次数最多的作为答案。

  ![](https://static001.infoq.cn/resource/image/26/f7/264a3d3129351a6fbc7dafc8969a2ef7.png)

- OvM(one vs many)，将多个分类，一个类别作为正例，其余类别作为反例，产生n个分类器。如果结果只有一个正例，则将其作为最终结果，否则，选择置信度最大的作为结果。

  ![](https://static001.infoq.cn/resource/image/1e/5a/1e6c7da0f88649b7491d96f7bd2f295a.png)

- MvM(many vs many)，每次将若干类作为正类，若干其他类作为反类，构造如ECOC等编码（纠错输出码，Error Correcting Output Codes），之后对N个类别进行M次划分，得到M个分类器，将M个分类器得到的结果组合成一个编码，将这个编码与各类别编码做比较，返回其中距离最小的类别作为最终预测结果。

**改造：**

多项逻辑斯蒂回归模型
$$
P(Y=k|x) = \frac{exp(w_k*x)}{1 + \sum_{k=1}^{K-1}exp(w_k * x)}, k=1,2,...,K-1 \\
P(Y=K|x) = \frac{1}{1 + \sum_{k=1}^{K-1}exp(w_k * x)}, k=1,2,...,K-1
$$

#### Softmax回归

softmax回归是LR模型在多分类的推广。

**Softmax Function：**
$$
softmax(x) = \frac{e^i}{\sum_{k=1}^ce^k}
$$

Softmax回归假设函数为：
$$
h_\theta(x^i) = [p(y^{(i)} = 1|x^{(i)};\theta),p(y^{(i)} = 2|x^{(i)};\theta),...,p(y^{(i)} = k|x^{(i)};\theta)]^T = \frac{1}{\sum_i^Ke^{\theta_j^Tx(i)}}[e^{\theta_1^Tx(i)},e^{\theta_2^Tx(i)},...,e^{\theta_k^Tx(i)}]^T
$$
与LR类似，其优化目标函数为：
$$
J(\theta) = -\frac{1}{m}[\sum_i^m\sum_j^K\{y^{(i)} = j\}log(softmax(\theta_j^Tx^{(i)}))]
$$


#### 最大熵模型

最大熵原理：该原理认为，学习概率模型时，在所有可能的概率模型（分布）中，熵最大的模型是最好的模型。最大熵原理给出了最优模型选择的一个准则。



#### 