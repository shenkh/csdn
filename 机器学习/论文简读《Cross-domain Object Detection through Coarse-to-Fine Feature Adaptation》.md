![在这里插入图片描述](https://img-blog.csdnimg.cn/20210307112425612.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dybGxlcnk=,size_16,color_FFFFFF,t_70)[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Cross-domain_Object_Detection_through_Coarse-to-Fine_Feature_Adaptation_CVPR_2020_paper.pdf)

**思想**：强调前景的重要性，在对齐时提高判别器对前景部分的权重（RPN）；除了类别不可知的对齐，引入细类别的对齐（Prototype）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021030720505686.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dybGxlcnk=,size_16,color_FFFFFF,t_70)

## Attentionbased Region Transfer (ART)

首先对于层$l$的对抗损失如下：
$$
% \begin{equation}

    \begin{aligned}
        \mathcal{L}_{ADV}^{l} = \min_{\theta_{G_l}} \max_{\theta_{D_l}} \mathbb{E}_{x_s \in \mathcal{D}_S} \log (D_l(G_l(x_s))) \\

        + \mathbb{E}_{x_t \in \mathcal{D}_T} \log (1-D_l(G_l(x_t)))

    \end{aligned}

%\end{equation}
$$

直观上RPN的feaure map，（$F_{RPN}$，对共享层做了3×3卷积，> the output feature map of the convolutional layer in the RPN module）包含了前景的位置信息

> the RPN in Faster R-CNN serves as the attention to tell the detection model where to look

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210307172428384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dybGxlcnk=,size_16,color_FFFFFF,t_70)

因此作者将$F_{RPN}$每个位置按照通道维度取均值并$sigmoid$，得到$M(x)$，$M(x)$的均值$T(x)$作为判断是否有目标的阈值，将小于阈值对应的权重点置0。

$$
M(x) = S(\frac{1}{C}\sum_{C}\left\lvert F_{RPN}^{C}(x)\right\rvert )
$$

$$
T(x) = \frac{1}{HW}\sum_{H, W}M(x)^{H, W}
$$

$$
A(x) = I(M(x) > T(x)) \otimes M(x)
$$

将上采样的权重图$U_l$叠加到原对抗损失上（按像素），实现每个点的对抗损失Attention。

$$
\mathcal{L}_{ART} = \sum_{l, h, w}(1+U_{l}(A(x)^{h, w})) \cdot \mathcal{L}_{ADV}^{l, h, w}
$$

作者在不同尺度上采用了以上策略。

## Prototype-based Semantic Alignment (PSA)

类别不可知的对齐对于检测任务可能会造成**negative transfer**，因此作者借鉴few shot learning的方式维护了源域和目标域的对应类别的prototypes：

$$
P_k^S = \frac{1}{\left\lvert GT_k\right\rvert} \sum_{r \in GT_k} F(r)
$$

$$
P_k^T = \frac{1}{\left\lvert RoI_k\right\rvert} \sum_{r \in RoI_k} F(r)
$$

$F(r)$指代的是在RoI head中经过了两次全连接，用于多分类的那个特征

> $F(r)$ denotes the feature of foreground region r after the second fully-connected (FC) layer in the RoI head.

因为目标域没有物体的GT，因此使用的是RoI。

由于每个batch，所有的目标往往不会同时出现，因此这里维护了各个类别全局的prototypes($GP_k$)，其中$GP_k^0$

> based on the pretrained model from the labeled source domain.

然后根据当前的prototype与全局的prototype的余弦相似性更新这个全局值。

$$
\alpha = sim(P_k^{(i)}, GP_k^{(i)})
$$

$$
GP_k^{(i)} = \alpha P_k^{(i)} + (1- \alpha) GP_k^{(i)}
$$

这里并未采用对齐的方式来约束$GP$，而是采用最小化的方式来缩小源域和目标域对应的prototype值：

$$
\mathcal{L}_{PSA} = \sum_k \left\lVert GP_k^{S(i)}, GP_k^{T(i)}\right\rVert ^2
$$
