#### 数据清洗

##### 缺失值处理

1. 删除缺失值的行数据

2. 归纳填充

   [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer.fit_transform)

   指定填充策略，使用数据的中值，均值，最多值等进行填充

#### 数据泄漏 data leakage

- target leakage

  时间序列未考虑清楚，未来的特征被用于计算当前模型。

  如：

  | 是否得肺炎 | 年龄 | 体重 | 性别 | 是否服用抗生素 |
  | ---------- | ---- | ---- | ---- | -------------- |
  | False      | 65   | 100  | 男   | False          |
  | True       | 72   | 130  | 女   | True           |
  | True       | 58   | 100  | 男   | True           |

  服用抗生素应该在得肺炎之后，任何未服抗生素的病人都未得肺炎，这对模型在实际落地有很大的影响。

- Train-test leakage

  对数据进行预处理时，使用了整个数据集（训练集+验证集）来计算的，如缺失值处理等

- 解决方法：

  使用Pipeline、剔除不合适的特征