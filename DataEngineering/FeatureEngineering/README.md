### 特征选择（sklearn.feature_selection）

#### filter 过滤式

1. 方差

   [VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold)方法通过删除某些样本特征方差值低于给定threshold的特征，实现特征的选择，

   threshold缺省值为0，意思是去掉在所有样本中无变化的特征。

   该方法对模型提升效果一般，一般用来处理一些几乎用不到的特征。

   [方法效果](https://www.cnblogs.com/zhange000/articles/10750489.html)

2. 单变量特征选择

   该方法通过计算特征的统计指标，对每一个特征计算其重要度，然后根据transform方法，保留指定数目/比例的特征。

   - [`SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest)保留得分最高的k个特征
   - [`SelectPercentile`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile)保留前百分比例的特征
   - [`GenericUnivariateSelect`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect) 可以设置不同的策略来进行单变量特征选择

   常用的统计指标有：

   - 分类问题：卡方(chi2)、f_classif、mutual_info_classif（互信息）
   - 回归问题：person(皮尔森相关系数)、f_regression、mutual_info_regression、信息熵

#### wrapper 包裹式

1. 递归特征消除（Recursive feature elimination）

   使用特征具有权重的模型，如决策树、对数回归，首先，预测模型在原数据上进行训练，训练得到每个特征的权重，然后将权重绝对值最小的若干特征从特征集中剔除，循环递归，直至特征数量等于所需的特征数量。

2. 带交叉验证的RFE（Recursive feature elimination with crossvalidation）
   使用交叉验证方法，选择最佳数量个特征

#### embeddeded 嵌入式

1. SelectFromModel

   使用拥有coef_或者`feature_importances_`属性的模型，训练模型，保留coef/`feature_importances值大于预设的threshold值的特征。

   说白了就是用这些模型预训练，根据训练结果，选择特征。

   - 基于L1的特征选择
   - 基于树的特征选择

2. 将特征选择加入pipeline

```python
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)
```

参考链接：

[机器学习：特征选择（feature selection）](https://blog.csdn.net/qq_33876194/article/details/88403394)

[特征选择](https://www.cnblogs.com/bjwu/p/9103002.html)

[[特征选择](https://www.cnblogs.com/stevenlk/p/6543628.html)]

[Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)

##### 