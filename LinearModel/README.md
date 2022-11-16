# 线性模型

## 1.基本形式

$$
f(x)=w_{1}x_{1}+w_{2}x_{2}+\dots+w_{d}x_{d}+b
$$

一般向量形式：
$$
f(x)=w^{T}x+b
$$
$w$ 项直观解释了各个属性在预测的重要性，如：$f(x)=0.2x_{色泽}+0.5x_{根蒂}+0.3x_{敲声}+b$

## 2.线性回归

既：给定数据集D，求 $w,b$。

#### 2.1数据预处理

- 属性间存在“序”关系

  将其转换为连续值；如身高属性，["高"，"中等"，"矮"] → [1, 0.5, 0]

- 属性间不存在“序”关系

  通常转换为 k 维向量；如瓜类属性，["南瓜"，"西瓜"，"黄瓜"] →[(1,0,0), (0,1,0), (0,0,1)]

#### 2.2求解方法

最小化均方误差（既最小二乘法）

### 2.3多元线性回归

当数据集有多个属性时，此时需要 **多元线性回归**。

设数据集 $D$ 表示为：
$$
\bold{X} = \begin{pmatrix}
 x_{11} & x_{12} & \dots& x_{1d} & 1\\
 x_{21} & x_{22} & \dots& x_{2d} & 1\\
\vdots  & \vdots & \ddots& \vdots & \vdots\\
 x_{m1} & x_{m2} & \dots& x_{md} & 1
\end{pmatrix}
=\begin{pmatrix}
 x^{T}_{1} & 1\\
 x^{T}_{2}  & 1\\
\vdots   & \vdots\\
 x_{m}^{T} & 1
\end{pmatrix}
$$
标记 $\bold{y}$ 表示为:
$$
\bold{y}=(y_{1};y_{2};\dots;y_{m})
$$
设 $\hat{\boldsymbol{w}}=(\boldsymbol{w};b)=(w_1;...;w_d;b)\in\mathbb{R}^{(d+1)\times 1},\hat{\boldsymbol{x}}_i=(x_{i1};...;x_{id};1)\in\mathbb{R}^{(d+1)\times 1}$

则：
$$
\hat{\boldsymbol{w}}^{*}=\underset{\hat{\boldsymbol{w}}}{\arg \min }(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})^{\mathrm{T}}(\boldsymbol{y}-\mathbf{X} \hat{\boldsymbol{w}})
$$

> 公式详解：https://datawhalechina.github.io/pumpkin-book/#/chapter3/chapter3?id=_39

当 $X^{T}X$是满秩矩阵（Full-rank matrix）或正定矩阵（Positive definite matrix）时
$$
\hat{\boldsymbol{w}}^{*}=(X^{T}X)^{-1}X^{T}y
$$
当 $X^{T}X$ 不是满秩矩阵时，可解出多个 $\hat{w}$  。常用的办法是**引入正则化项**。

## 3.对数几率回归

也称对率回归、逻辑回归，本质是分类方法
$$
y=\frac{1}{1+e^{-z}}
$$

## 4.线性判定分析

最大化类间均值，最小化类内协方差

LDA欲最大化目标：$S_{b}$ 与 $S_{w}$ 的 **广义瑞利商**
$$
J=\frac{w^{T}S_{b}w}{w^{T}S_{w}w}
$$
这一章不理解。。

## 5.多分类学习

基本思路：“拆解法”

### 5.1拆解策略

#### 5.1.1一对一

![image-20221116201945369](https://www.wangwangyz.site/%E4%B8%AA%E4%BA%BA%E5%9B%BE%E5%BA%8A/image-20221116201945369.png)

若给定数据集 D 有 N 个类别

OvO 将这N个类别两两配对，将得到 N（N-1）个分类器，结果经过投票法而得。

#### 5.1.2一对剩余

![image-20221116202953919](https://www.wangwangyz.site/%E4%B8%AA%E4%BA%BA%E5%9B%BE%E5%BA%8A/image-20221116202953919.png)

OvR 是每次将一个类别作为正例，剩余别的类别作为反例。得到 N 个分类器。

若分类结果仅有一个正例，则对应其类别为最终预测结果；若分类结果有多个正例，通常选择置信度最大的类别作为最终预测结果。

#### 5.1.3多对多

MvM是若干个类作为正例，若干个类作为反例。常用方法：ECOC（纠错输出码，Error Correcting Output Codes）

> DAG(Directed Acyclic Graph)拆分法

## 6. 类别不平衡问题

类别不平衡是指分类任务中不同类别的训练样例数目差别很大的情况。

由 $y=w^{T}x+b$ 代表预测为正例的概率, 几率 $\frac{y}{1-y}$令 $m^{+}$ 表示正例数目；$m^{-}$ 表示反例数目，则观测几率为 $\frac{m^{+}}{m^{-}}$。既：
$$
若\frac{y}{1-y}>\frac{m^{+}}{m^{-}},则预测为正例
$$


### 6.1再缩放

$$
\frac{y^{'}}{1-y^{'}}=\frac{y}{1-y}×\frac{m^{-}}{m^{+}}
$$

### 6.2 无偏采样

秉承着“训练集是真实样本总体的无偏采样”理念

#### 6.2.1 欠采样

去除训练集内的一些反例，使得正、反例数目相近。

代表算法：SMOTE

#### 6.2.2 过采样

增加正例到训练集中，使得正、反例数目相近。

代表算法：EasyEnsemble

#### 6.2.3 阈值移动

也叫**调整权重**，直接基于原训练集进行训练，但在使用训练好的分类器进行预测时，通过[再缩放](# 再缩放)式子重新计算阈值。