[TOC]

# 深度学习基础

**线性回归**和 **softmax 回归** 是两种单层神经网络。

对于**过拟合现象**，采用**权重衰减**和**丢弃法**应对。

## &sect;1 线性回归

线性回归，就是能够用一个直线较为精确地描述数据之间的关系。

这样当出现新的数据的时候，就能够预测出一个简单的值。

线性回归输出的是一个连续值。

### &sect;1.1 线性回归的基本要素

以预测房屋价格为例，假定价格（元）只取决于面积（平方米）和房龄（年）。

**1、模型**

设面积为 ![](http://latex.codecogs.com/gif.latex?x_1)，房龄为 ![](http://latex.codecogs.com/gif.latex?x_2)，售出价格为 ![](http://latex.codecogs.com/gif.latex?y)。建立由输入 ![](http://latex.codecogs.com/gif.latex?x_1) 和 ![](http://latex.codecogs.com/gif.latex?x_2) 计算输出 ![](http://latex.codecogs.com/gif.latex?y) 的表达式，就是**模型**（model）。

![](http://latex.codecogs.com/gif.latex?\hat{y}=x_1w_1+x_2w_2+b)

其中 ![](http://latex.codecogs.com/gif.latex?w_1) 和 ![](http://latex.codecogs.com/gif.latex?w_2) 是权重（weight），![](http://latex.codecogs.com/gif.latex?b) 是偏差（bias），且均为标量。它们是线性回归模型的参数。模型输出 ![](http://latex.codecogs.com/gif.latex?\hat{y}) 是对真实价格 ![](http://latex.codecogs.com/gif.latex?y) 的预测或估计。

**2、模型训练**

通过数据确定模型参数值，使模型在数据上的误差尽量小。该过程叫做模型训练（model training）。

**3、训练数据**

用于训练模型的真实数据集叫做**训练集**（training set），一栋房屋的数据叫做一个**样本**（sample），其真实售价叫做**标签**（label），用于预测标签的两个因素叫做**特征**（feature）。

假设样本数为 ![](http://latex.codecogs.com/gif.latex?n)，索引为 ![](http://latex.codecogs.com/gif.latex?i) 的样本的特征为 ![](http://latex.codecogs.com/gif.latex?x_1^{(i)}) 和 ![](http://latex.codecogs.com/gif.latex?x_2^{\(i\)})，标签为 ![](http://latex.codecogs.com/gif.latex?y^{\(i\)})，则线性回归表达式为

![](http://latex.codecogs.com/gif.latex?\hat{y}^{(i)}=x_1^{(i)}w_1+x_2^{(i)}w_2+b)

 **4、损失函数**

模型训练中，预测价格和真实价格之间的误差称为**损失函数**（loss function）。

评估索引为 ![](http://latex.codecogs.com/gif.latex?i) 的样本误差，采用平方损失（square loss）。

<img src="http://latex.codecogs.com/gif.latex?l^{(i)}(w_1,w_2,b)=\frac{1}{2}(\hat{y}^{(i)}-y^i)^2"/>

所有样本误差的平均值衡量模型预测的质量，即

<img src="http://latex.codecogs.com/gif.latex?l(w_1,w_2,b)=\frac{1}{n}\sum_{i=1}^nl^{(i)}(w_1,w_2,b)=\frac{1}{n}\sum_{i=1}^n\frac{1}{2}(x_1^{(i)}w_1+x_2^{(i)}w_2+b-y^{(i)})^2"/>

在模型训练中，我希望找到一组模型参数，记为 <img src="http://latex.codecogs.com/gif.latex?w_1^*"/>，<img src="http://latex.codecogs.com/gif.latex?w_2^*"/>，<img src="http://latex.codecogs.com/gif.latex?b^*"/>，来使训练样本平均损失最小：

<img src="http://latex.codecogs.com/gif.latex?w_1^*,w_2^*,b^*=\mathop{\rm{argmin}}\limits_{w_1,w_2,b}\;l(w_1,w_2,b)"/>

**5、优化算法**

误差最小化问题的解可以直接用公式表达的，叫做**解析解**（analytical solution）。

只能通过优化算法有限次迭代降低损失函数的，叫做**数值解**（numerical solution）。

**小批量随机梯度下降**（min-batch stochastic gradient descent）常用在数值解的优化算法中。

> 先取一组模型参数的初始值，如随机选取；接下来对参数进行多次迭代，使每次迭代都可能降低损失函数值。每次迭代，随机均匀选取固定数目的训练数据组成**小批量**（min-batch）B，求小批量数据样本平均损失对模型参数的导数（梯度），此梯度乘以一个正数作为迭代的减小量。

迭代如下：

<img src="http://latex.codecogs.com/gif.latex?w_1=w_1-\frac{\eta}{|B|}\sum_{i\in{B}}\frac{\partial{l^{(i)}(w_1,w_2,b)}}{\partial w_1}"/>

<img src="http://latex.codecogs.com/gif.latex?w_2=w_2-\frac{\eta}{|B|}\sum_{i\in{B}}\frac{\partial l^{(i)}(w_1,w_2,b)}{\partial w_2}=w_2-\frac{\eta}{|B|}\sum_{i\in{B}}x_2^{(i)}\left(x_1^{(i)}w_1+x_2^{(i)}w_2+b-y^{(i)} \right)"/>

<img src="http://latex.codecogs.com/gif.latex?b=b-\frac{\eta}{|B|}\sum_{i\in{B}}\frac{\partial l^{(i)}(w_1,w_2,b)}{\partial b}=b-\frac{\eta}{|B|}\sum_{i\in{B}}\left(x_1^{(i)}w_1+x_2^{(i)}w_2+b-y^{(i)} \right)"/>

其中，<img src='http://latex.codecogs.com/gif.latex?|B|'/> 代表小批量的**样本个数**（批量大小，batch_size），<img src='http://latex.codecogs.com/gif.latex?\eta'/> 是**学习率**（learning rate）并取正数。

批量大小和学习率是人为设定的，并不是模型训练出来的，因此叫做**超参数**（hyperparamter）。通常所说的**调参**就是调节超参数。

**6、模型预测**

设模型参数 <img src="http://latex.codecogs.com/gif.latex?w_1, w_2, b"/> 优化后的值分别为 <img src="http://latex.codecogs.com/gif.latex?w_1^*, w_2^*, b^*"/>，

