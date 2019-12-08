softmax函数：
$$
S_{i}=\frac{e^{a_{i}}}{\sum_{j} e^{a_{j}}}
$$

## **softmax求导**

$$
\frac{\partial S_{i}}{\partial a_{j}}=\frac{\frac{\partial e^{a_{i}}}{\partial a_{j}} \cdot \Sigma-\frac{\partial \Sigma}{\partial a_{j}} \cdot e^{a_{i}}}{\sum^{2}}
$$

当$i = j$时：
$$
\frac{\partial S_{i}}{\partial a_{j}}=\frac{e^{a_{i}} \cdot \Sigma-e^{a_{j}} e^{a_{i}}}{\Sigma^{2}} = \frac{e^{a_{i}}}{\Sigma} \cdot \frac{\Sigma-e^{a_{j}}}{\Sigma}=S_{i} \cdot\left(1-S_{j}\right)
$$


当$i \neq j$时：
$$
\frac{\partial S_{i}}{\partial a_{j}}=-\frac{e^{\alpha_{j}} \cdot e^{a_{i}}}{\Sigma^{2}}=-S_{i} \cdot S_{j}
$$

## **交叉熵损失函数**

$$
L=-\sum y_{i} \log S_{i}
$$

$$
\frac{\partial L}{\partial S_{i}}=-y_{i} \cdot \frac{1}{S_{i}}
$$

对softmax的输入求导：

$$
\frac{\partial L}{\partial a_{i}}=\sum_{j} \frac{\partial L}{\partial S_{j}} \cdot \frac{\partial S_{j}}{\partial a_{i}}=\frac{\partial L}{\partial S_{i}} \cdot \frac{\partial S_{i}}{\partial a_{i}}+\sum_{j \neq i} \frac{\partial L}{\partial S_{j}} \cdot \frac{\partial S_{j}}{\partial a_{i}}
\\ =-y_{i} \cdot \frac{1}{S_{i}} \cdot S_{i}\left(1-S_{i}\right)+\sum_{j \neq i} \frac{-y_{j}}{S_{j}} \cdot(-1) S_{i} S_{j}
\\ =-y_{i}\left(1-S_{i}\right)+\sum_{j \neq i} y_{j} \cdot S_{i}
\\ =-y_{i}+y_{i} S_{i}+\sum_{j \neq i} y_{j} \cdot S_{i}
\\ =S_{i}-y_{i}
$$