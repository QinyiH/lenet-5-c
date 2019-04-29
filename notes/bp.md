# 反向传播

## ReLU求导

## SoftMax

softmax用于多分类过程中，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解，从而来进行多分类

$$S_i=\frac{e^i}{\sum_je^j}$$

<img src=http://upload-images.jianshu.io/upload_images/5236230-12cd299a8d571d1e.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240>

### softmax求导

当我们对分类的Loss进行改进的时候，我们要通过梯度下降，每次优化一个step大小的梯度，这个时候就要**求Loss对每个权重矩阵的偏导，然后应用链式法则。**

#### 交叉熵(cross entropy)

$$loss=-\sum _i y_ilna_i$$

y为真实值，a为输出值。

$$\Rightarrow loss=y_jlna_j$$

$$a_i-y_i$$

## 全连接层权重更新

$$[weights\; diff\;]_{i\times  j}=[data_{in}]_{i\times batch} *[top\;diff]_{batch*\times j}$$

$$
\frac{\partial a_{i}}{\partial b_{i}}=1
$$

<img src="https://img-blog.csdn.net/20180731151033235?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzI1MTA0NA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/0">