# 请问faster rcnn和ssd 中为什么用smooth l1 loss，和l2有什么区别？

作者：知乎用户

链接：https://www.zhihu.com/question/58200555/answer/621174180

来源：知乎

著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

为了从两个方面限制梯度：

1. 当预测框与 ground truth 差别过大时，梯度值不至于过大；
2. 当预测框与 ground truth 差别很小时，梯度值足够小。

考察如下几种损失函数，其中 ![x](https://www.zhihu.com/equation?tex=x) 为预测框与 groud truth 之间 elementwise 的差异：

![L_2(x)=x^2\tag{1}](https://www.zhihu.com/equation?tex=L_2%28x%29%3Dx%5E2%5Ctag%7B1%7D)

![L_1(x)=|x|\tag{2}](https://www.zhihu.com/equation?tex=L_1%28x%29%3D%7Cx%7C%5Ctag%7B2%7D)

![\mathrm{smooth}_{L_1}(x)= \begin{cases} 0.5x^2& \text{if } |x|<1\\ |x|-0.5& \text{otherwise} \end{cases}\tag{3}](https://www.zhihu.com/equation?tex=%5Cmathrm%7Bsmooth%7D_%7BL_1%7D%28x%29%3D+%5Cbegin%7Bcases%7D+0.5x%5E2%26+%5Ctext%7Bif+%7D+%7Cx%7C%3C1%5C%5C+%7Cx%7C-0.5%26+%5Ctext%7Botherwise%7D+%5Cend%7Bcases%7D%5Ctag%7B3%7D)

损失函数对 ![x](https://www.zhihu.com/equation?tex=x) 的导数分别为：

![\frac{\mathrm{d}L_2(x)}{\mathrm{d}x}=2x\tag{4}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cmathrm%7Bd%7DL_2%28x%29%7D%7B%5Cmathrm%7Bd%7Dx%7D%3D2x%5Ctag%7B4%7D)

![\frac{\mathrm{d}L_1(x)}{\mathrm{d}x}=\begin{cases} &1 &\text{if } x\geq0 \\ &-1 &\text{otherwise}   \end{cases}\tag{5}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cmathrm%7Bd%7DL_1%28x%29%7D%7B%5Cmathrm%7Bd%7Dx%7D%3D%5Cbegin%7Bcases%7D+%261+%26%5Ctext%7Bif+%7D+x%5Cgeq0+%5C%5C+%26-1+%26%5Ctext%7Botherwise%7D+++%5Cend%7Bcases%7D%5Ctag%7B5%7D)

![\frac{\mathrm{d}\mathrm{\ smooth}_{L_1}}{\mathrm{d}x}=\begin{cases} &x &\text{if}\ |x|<1\\ &\pm1 &\text{otherwise}\tag{6}  \end{cases}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cmathrm%7Bd%7D%5Cmathrm%7B%5C+smooth%7D_%7BL_1%7D%7D%7B%5Cmathrm%7Bd%7Dx%7D%3D%5Cbegin%7Bcases%7D+%26x+%26%5Ctext%7Bif%7D%5C+%7Cx%7C%3C1%5C%5C+%26%5Cpm1+%26%5Ctext%7Botherwise%7D%5Ctag%7B6%7D++%5Cend%7Bcases%7D)

观察 (4)，当 ![x](https://www.zhihu.com/equation?tex=x) 增大时 ![L_2](https://www.zhihu.com/equation?tex=L_2) 损失对 ![x](https://www.zhihu.com/equation?tex=x) 的导数也增大。这就导致训练初期，预测值与 groud truth 差异过于大时，损失函数对预测值的梯度十分大，训练不稳定。

根据方程 (5)，![L_1](https://www.zhihu.com/equation?tex=L_1) 对 ![x](https://www.zhihu.com/equation?tex=x) 的导数为常数。这就导致训练后期，预测值与 ground truth 差异很小时， ![L_1](https://www.zhihu.com/equation?tex=L_1) 损失对预测值的导数的绝对值仍然为 1，而 learning rate 如果不变，损失函数将在稳定值附近波动，难以继续收敛以达到更高精度。

最后观察 (6)， ![\mathrm{smooth}_{L_1}](https://www.zhihu.com/equation?tex=%5Cmathrm%7Bsmooth%7D_%7BL_1%7D) 在 ![x](https://www.zhihu.com/equation?tex=x) 较小时，对 ![x](https://www.zhihu.com/equation?tex=x) 的梯度也会变小，而在 ![x](https://www.zhihu.com/equation?tex=x) 很大时，对 ![x](https://www.zhihu.com/equation?tex=x) 的梯度的绝对值达到上限 1，也不会太大以至于破坏网络参数。 ![\mathrm{smooth}_{L_1}](https://www.zhihu.com/equation?tex=%5Cmathrm%7Bsmooth%7D_%7BL_1%7D) 完美地避开了 ![L_1](https://www.zhihu.com/equation?tex=L_1) 和 ![L_2](https://www.zhihu.com/equation?tex=L_2) 损失的缺陷。其函数图像如下：

![img](https://pic4.zhimg.com/50/v2-dccf66172c8e4440f34cebaeb7018a29_hd.jpg)![img](https://pic4.zhimg.com/80/v2-dccf66172c8e4440f34cebaeb7018a29_hd.jpg)

由图中可以看出，它在远离坐标原点处，图像和 ![L_1](https://www.zhihu.com/equation?tex=L_1) loss 很接近，而在坐标原点附近，转折十分平滑，不像 ![L_1](https://www.zhihu.com/equation?tex=L_1) loss 有个尖角，因此叫做 smooth ![L_1](https://www.zhihu.com/equation?tex=L_1) loss。

[编辑于 2019-03-13]()

