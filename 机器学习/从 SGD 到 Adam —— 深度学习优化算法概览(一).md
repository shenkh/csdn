# [从 SGD 到 Adam —— 深度学习优化算法概览(一)](https://zhuanlan.zhihu.com/p/32626442)

## **楔子**

前些日在写计算数学课的期末读书报告，我选择的主题是「分析深度学习中的各个优化算法」。在此前的工作中，自己通常就是无脑「Adam 大法好」，而对算法本身的内涵不知所以然。一直希望能抽时间系统的过一遍优化算法的发展历程，直观了解各个算法的长处和短处。这次正好借着作业的机会，补一补课。

本文主要借鉴了 

[@Juliuszh](https://www.zhihu.com/people/9f65e9890272246d013f336374f0586a)

 的文章[1]思路，使用一个 general 的框架来描述各个梯度下降变种算法。实际上，本文可以视作对[1]的重述，在此基础上，对原文描述不够详尽的部分做了一定补充，并修正了其中许多错误的表述和公式。

另一主要参考文章是 Sebastian Ruder 的综述[2]。该文十分有名，大概是深度学习优化算法综述中质量最好的一篇了。建议大家可以直接阅读原文。本文许多结论和插图引自该综述。

对优化算法进行分析和比较的文章已有太多，本文实在只能算得上是重复造轮，旨在个人学习和总结。**希望对优化算法有深入了解的同学可以直接查阅文末的参考文献。**

## **引言**

最优化问题是计算数学中最为重要的研究方向之一。而在深度学习领域，优化算法的选择也是一个模型的重中之重。即使在数据集和模型架构完全相同的情况下，采用不同的优化算法，也很可能导致截然不同的训练效果。

梯度下降是目前神经网络中使用最为广泛的优化算法之一。为了弥补朴素梯度下降的种种缺陷，研究者们发明了一系列变种算法，从最初的 SGD (随机梯度下降) 逐步演进到 NAdam。然而，许多学术界最为前沿的文章中，都并没有一味使用 Adam/NAdam 等公认“好用”的自适应算法，很多甚至还选择了最为初级的 SGD 或者 SGD with Momentum 等。

本文旨在梳理深度学习优化算法的发展历程，并在一个更加概括的框架之下，对优化算法做出分析和对比。

## **Gradient Descent**

梯度下降是指，在给定待优化的模型参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%5Cin+%5Cmathbb%7BR%7D%5Ed) 和目标函数 ![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29) 后，算法通过沿梯度 ![[公式]](https://www.zhihu.com/equation?tex=%5Cnabla_%5Ctheta+J%28%5Ctheta%29) 的相反方向更新 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 来最小化 ![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29) 。学习率 ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta) 决定了每一时刻的更新步长。对于每一个时刻 ![[公式]](https://www.zhihu.com/equation?tex=t) ，我们可以用下述步骤描述梯度下降的流程：

(1) 计算目标函数关于参数的梯度

![[公式]](https://www.zhihu.com/equation?tex=g_t+%3D+%5Cnabla_%5Ctheta+J%28%5Ctheta%29)

(2) 根据历史梯度计算一阶和二阶动量

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cphi%28g_1%2C+g_2%2C+%5Ccdots%2C+g_t%29)

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+%5Cpsi%28g_1%2C+g_2%2C+%5Ccdots%2C+g_t%29)

(3) 更新模型参数

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_t+-+%5Cfrac%7B1%7D%7B%5Csqrt%7Bv_t+%2B+%5Cepsilon%7D%7D+m_t)

其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 为平滑项，防止分母为零，通常取 1e-8。

## **Gradient Descent 和其算法变种**

根据以上框架，我们来分析和比较梯度下降的各变种算法。

## Vanilla SGD

朴素 SGD (Stochastic Gradient Descent) 最为简单，没有动量的概念，即

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Ceta+g_t)

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+I%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon+%3D+0)

这时，更新步骤就是最简单的

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bi%2B1%7D%3D+%5Ctheta_t+-+%5Ceta+g_t)

SGD 的缺点在于收敛速度慢，可能在鞍点处震荡。并且，如何合理的选择学习率是 SGD 的一大难点。

## Momentum

SGD 在遇到沟壑时容易陷入震荡。为此，可以为其引入动量 Momentum[3]，加速 SGD 在正确方向的下降并抑制震荡。

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cgamma+m_%7Bt-1%7D+%2B+%5Ceta+g_t)

SGD-M 在原步长之上，增加了与上一时刻步长相关的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma+m_%7Bt-1%7D) ，![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 通常取 0.9 左右。这意味着参数更新方向不仅由当前的梯度决定，也与此前累积的下降方向有关。这使得参数中那些梯度方向变化不大的维度可以加速更新，并减少梯度方向变化较大的维度上的更新幅度。由此产生了加速收敛和减小震荡的效果。

![img](https://pic4.zhimg.com/80/v2-2476080e4cdfd489ae64ae3ceeafe48b_hd.jpg)图 1(a): SGD

![img](https://pic4.zhimg.com/80/v2-b9388fd6e465d82687680f9d16edcd2b_hd.jpg)图 1(b): SGD with momentum

从图 1 中可以看出，引入动量有效的加速了梯度下降收敛过程。

## Nesterov Accelerated Gradient

![img](https://pic4.zhimg.com/80/v2-fecd469405501ad82788f068985b25cb_hd.jpg)图 2: Nesterov update

更进一步的，人们希望下降的过程更加智能：算法能够在目标函数有增高趋势之前，减缓更新速率。

NAG 即是为此而设计的，其在 SGD-M 的基础上进一步改进了步骤 1 中的梯度计算公式：

![[公式]](https://www.zhihu.com/equation?tex=g_t+%3D+%5Cnabla_%5Ctheta+J%28%5Ctheta+-+%5Cgamma+m_%7Bt-1%7D%29)

参考图 2，SGD-M 的步长计算了当前梯度（短蓝向量）和动量项 （长蓝向量）。然而，既然已经利用了动量项来更新 ，那不妨先计算出下一时刻 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的近似位置 （棕向量），并根据该未来位置计算梯度（红向量），然后使用和 SGD-M 中相同的方式计算步长（绿向量）。这种计算梯度的方式可以使算法更好的「预测未来」，提前调整更新速率。

## Adagrad

SGD、SGD-M 和 NAG 均是以相同的学习率去更新 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的各个分量。而深度学习模型中往往涉及大量的参数，不同参数的更新频率往往有所区别。对于更新不频繁的参数（典型例子：更新 word embedding 中的低频词），我们希望单次步长更大，多学习一些知识；对于更新频繁的参数，我们则希望步长较小，使得学习到的参数更稳定，不至于被单个样本影响太多。

Adagrad[4] 算法即可达到此效果。其引入了二阶动量：

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+%5Ctext%7Bdiag%7D%28%5Csum_%7Bi%3D1%7D%5Et+g_%7Bi%2C1%7D%5E2%2C+%5Csum_%7Bi%3D1%7D%5Et+g_%7Bi%2C2%7D%5E2%2C+%5Ccdots%2C+%5Csum_%7Bi%3D1%7D%5Et+g_%7Bi%2Cd%7D%5E2%29)

其中， ![[公式]](https://www.zhihu.com/equation?tex=v_t+%5Cin+%5Cmathbb%7BR%7D%5E%7Bd%5Ctimes+d%7D) 是对角矩阵，其元素 ![[公式]](https://www.zhihu.com/equation?tex=v_%7Bt%2C+ii%7D) 为参数第 ![[公式]](https://www.zhihu.com/equation?tex=i) 维从初始时刻到时刻 ![[公式]](https://www.zhihu.com/equation?tex=t) 的梯度平方和。

此时，可以这样理解：学习率等效为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta+%2F+%5Csqrt%7Bv_t+%2B+%5Cepsilon%7D) 。对于此前频繁更新过的参数，其二阶动量的对应分量较大，学习率就较小。这一方法在稀疏数据的场景下表现很好。

## RMSprop

在 Adagrad 中， ![[公式]](https://www.zhihu.com/equation?tex=v_t) 是单调递增的，使得学习率逐渐递减至 0，可能导致训练过程提前结束。为了改进这一缺点，可以考虑在计算二阶动量时不累积全部历史梯度，而只关注最近某一时间窗口内的下降梯度。根据此思想有了 RMSprop[5]。记 ![[公式]](https://www.zhihu.com/equation?tex=g_t+%5Codot+g_t) 为 ![[公式]](https://www.zhihu.com/equation?tex=g_t%5E2) ，有

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+%5Cgamma+v_%7Bt-1%7D+%2B+%281-%5Cgamma%29+%5Ccdot+%5Ctext%7Bdiag%7D%28g_t%5E2%29)

其二阶动量采用指数移动平均公式计算，这样即可避免二阶动量持续累积的问题。和 SGD-M 中的参数类似，![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 通常取 0.9 左右。

## Adadelta

待补充

## Adam

Adam[6] 可以认为是 RMSprop 和 Momentum 的结合。和 RMSprop 对二阶动量使用指数移动平均类似，Adam 中对一阶动量也是用指数移动平均计算。

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Ceta%5B+%5Cbeta_1+m_%7Bt-1%7D+%2B+%281+-+%5Cbeta_1%29g_t+%5D)

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+%5Cbeta_2+v_%7Bt-1%7D+%2B+%281-%5Cbeta_2%29+%5Ccdot+%5Ctext%7Bdiag%7D%28g_t%5E2%29)

其中，初值

![[公式]](https://www.zhihu.com/equation?tex=m_0+%3D+0)

![[公式]](https://www.zhihu.com/equation?tex=v_0+%3D+0)

注意到，在迭代初始阶段，![[公式]](https://www.zhihu.com/equation?tex=m_t) 和 ![[公式]](https://www.zhihu.com/equation?tex=v_t) 有一个向初值的偏移（过多的偏向了 0）。因此，可以对一阶和二阶动量做偏置校正 (bias correction)，

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bm%7D_t+%3D+%5Cfrac%7Bm_t%7D%7B1-%5Cbeta_1%5Et%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bv%7D_t+%3D+%5Cfrac%7Bv_t%7D%7B1-%5Cbeta_2%5Et%7D)

再进行更新，

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_t+-+%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D+%2B+%5Cepsilon+%7D+%5Chat%7Bm%7D_t)

可以保证迭代较为平稳。

## NAdam

NAdam[7] 在 Adam 之上融合了 NAG 的思想。

首先回顾 NAG 的公式，

![[公式]](https://www.zhihu.com/equation?tex=g_t+%3D+%5Cnabla_%5Ctheta+J%28%5Ctheta_t+-+%5Cgamma+m_%7Bt-1%7D%29)

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cgamma+m_%7Bt-1%7D+%2B+%5Ceta+g_t)

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_t+-+m_t)

NAG 的核心在于，计算梯度时使用了「未来位置」![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_t+-+%5Cgamma+m_%7Bt-1%7D)。NAdam 中提出了一种公式变形的思路[7]，大意可以这样理解：只要能在梯度计算中考虑到「未来因素」，即能达到 Nesterov 的效果；既然如此，那么在计算梯度时，可以仍然使用原始公式 ![[公式]](https://www.zhihu.com/equation?tex=g_t+%3D+%5Cnabla_%5Ctheta+J%28%5Ctheta_t%29) ，但在前一次迭代计算 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_t) 时，就使用了未来时刻的动量，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_t+%3D+%5Ctheta_%7Bt-1%7D+-+m_t) ，那么理论上所达到的效果是类似的。

这时，公式修改为，

![[公式]](https://www.zhihu.com/equation?tex=g_t+%3D+%5Cnabla_%5Ctheta+J%28%5Ctheta_t%29)

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cgamma+m_%7Bt-1%7D+%2B+%5Ceta+g_t)

![[公式]](https://www.zhihu.com/equation?tex=%5Cbar%7Bm%7D_t+%3D+%5Cgamma+m_t+%2B+%5Ceta+g_t)

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_t+-+%5Cbar%7Bm%7D_t)

理论上，下一刻的动量为 ![[公式]](https://www.zhihu.com/equation?tex=m_%7Bt%2B1%7D+%3D+%5Cgamma+m_t+%2B+%5Ceta+g_%7Bt%2B1%7D)，在假定连续两次的梯度变化不大的情况下，即 ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bt%2B1%7D+%5Capprox+g_t)，有 ![[公式]](https://www.zhihu.com/equation?tex=m_%7Bt%2B1%7D+%5Capprox+%5Cgamma+m_t+%2B+%5Ceta+g_t+%5Cequiv+%5Cbar%7Bm%7D_t)。此时，即可用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbar%7Bm%7D_t) 近似表示未来动量加入到 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的迭代式中。

类似的，在 Adam 可以加入 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbar%7Bm%7D_t+%5Cleftarrow+%5Chat%7Bm%7D_t) 的变形，将 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bm%7D_t) 展开有

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bm%7D_t+%3D+%5Cfrac%7Bm_t%7D%7B1-%5Cbeta_1%5Et%7D+%3D+%5Ceta%5B+%5Cfrac%7B%5Cbeta_1+m_%7Bt-1%7D%7D%7B1-%5Cbeta_1%5Et%7D+%2B+%5Cfrac%7B%281+-+%5Cbeta_1%29g_t%7D%7B1-%5Cbeta_1%5Et%7D+%5D)

引入

![[公式]](https://www.zhihu.com/equation?tex=%5Cbar%7Bm%7D_t+%3D+%5Ceta%5B+%5Cfrac%7B%5Cbeta_1+m_%7Bt%7D%7D%7B1-%5Cbeta_1%5E%7Bt%2B1%7D%7D+%2B+%5Cfrac%7B%281+-+%5Cbeta_1%29g_t%7D%7B1-%5Cbeta_1%5Et%7D+%5D)

再进行更新，

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_t+-+%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D+%2B+%5Cepsilon+%7D+%5Cbar%7Bm%7D_t)

即可在 Adam 中引入 Nesterov 加速效果。

## **可视化分析**

![img](https://pic1.zhimg.com/v2-5d5166a3d3712e7c03af74b1ccacbeac_b.jpg)

图 3: SGD optimization on loss surface contours

![img](https://pic1.zhimg.com/v2-4a3b4a39ab8e5c556359147b882b4788_b.gif)

图 4: SGD optimization on saddle point

图 3 和图 4 两张动图直观的展现了不同算法的性能。(Image credit: [Alec Radford](https://link.zhihu.com/?target=https%3A//twitter.com/alecrad))

图 3 中，我们可以看到不同算法在损失面等高线图中的学习过程，它们均同同一点出发，但沿着不同路径达到最小值点。其中 Adagrad、Adadelta、RMSprop 从最开始就找到了正确的方向并快速收敛；SGD 找到了正确方向但收敛速度很慢；SGD-M 和 NAG 最初都偏离了航道，但也能最终纠正到正确方向，SGD-M 偏离的惯性比 NAG 更大。

图 4 展现了不同算法在鞍点处的表现。这里，SGD、SGD-M、NAG 都受到了鞍点的严重影响，尽管后两者最终还是逃离了鞍点；而 Adagrad、RMSprop、Adadelta 都很快找到了正确的方向。

关于两图的讨论，也可参考[2]和[8]。

可以看到，几种自适应算法在这些场景下都展现了更好的性能。

## **讨论、选择策略**

读书报告中的讨论内容较为杂乱，该部分待整理完毕后再行发布。

## **References**

[1] [Adam那么棒，为什么还对SGD念念不忘 (1) —— 一个框架看懂优化算法](https://zhuanlan.zhihu.com/p/32230623)

[2] [An overview of gradient descent optimization algorithms](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1609.04747)

[3] [On the momentum term in gradient descent learning algorithms](https://link.zhihu.com/?target=https%3A//dl.acm.org/citation.cfm%3Fid%3D307376)

[4] [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://link.zhihu.com/?target=http%3A//www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

[5] [CSC321 Neural Networks for Machine Learning - Lecture 6a](https://link.zhihu.com/?target=http%3A//www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

[6] [Adam: A Method for Stochastic Optimization](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1412.6980)

[7] [Incorporating Nesterov Momentum into Adam](https://link.zhihu.com/?target=http%3A//cs229.stanford.edu/proj2015/054_report.pdf)

[8] [CS231n Convolutional Neural Networks for Visual Recognition](https://link.zhihu.com/?target=http%3A//cs231n.github.io/neural-networks-3/)