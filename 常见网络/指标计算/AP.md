## AP相关的基础概念

要讲清楚AP，我们先看一下Accuracy，Precision，Recall，IOU，Recall@TopK，Precision@TopK。介绍完这几个基础概念之后，我们会介绍AP的基本计算思想，即PR曲线下的面积，之后会介绍不同数据集PASCAL VOC，COCO，ImageNet计算AP的方式。

### Accuracy：准确率

这个计算比较简单，准确率=预测正确的样本数/所有样本数，即预测正确的样本比例（包括预测正确的正样本和预测正确的负样本，不过在目标检测领域，没有预测正确的负样本这一说法，所以目标检测里面没有用Accuracy的）。计算公式如下：
$$
Accuracy=\frac{TruePositive+TrueNegative}{AllSamples}
$$

### Precision：查准率

Precision的计算则很容易和Accuracy弄混，Precision表示某一类样本预测有多准，比如说，模型预测出有100个病患，可是实际上这里面只有80个是病患，那么Precision就是80%。顾名思义，即查准的几率为80%。注意：**Precision针对的是某一类样本，如果没有说明类别，那么Precision是毫无意义的（有些地方不说明类别，直接说Precision，是因为二分类问题通常说的Precision都是正样本的Precision）。**Accuracy则针对的是所有类别的样本，这一点很多人并没有去区分。

![Precision=\frac{TruePositives}{AllDetections}\\](http://www.zhihu.com/equation?tex=Precision%3D%5Cfrac%7BTruePositives%7D%7BAllDetections%7D%5C%5C)

### Recall：召回率

Recall和Precision一样，脱离类别是没有意义的。说道Recall，一定指的是某个类别的Recall。Recall表示某一类样本，预测正确的与所有Ground Truth的比例。比如说，某批数据有100个人为患者，可是模型只预测出80个人为患者，那么被召回医院就诊的比例就是80%，即Recall是80%。仔细和Precision的计算方法对比就可以发现，Recall和Precision计算中的分子都是一样的，但是**Recall计算的时候，分母是Ground Truth中某一类样本的数量，而Precision计算的时候，是预测出来的某一类样本数**。自然地，可以得到如下有关Recall的计算公式：

![Recall=\frac{TruePositives}{All Ground Truths}\\](http://www.zhihu.com/equation?tex=Recall%3D%5Cfrac%7BTruePositives%7D%7BAll+Ground+Truths%7D%5C%5C)

### IOU

目标检测里面我们如何判定一个目标检测正确与否呢？第一，我们需要看CNN给出的标签是不是对的，第二，我们需要看预测的框和真实框的重叠比例有多少。这个重叠比例就是IOU：

![img](https://pic1.zhimg.com/v2-d577c80eb89d2cf0ca1fd2337b34468b_b.jpg)

如果重叠比例IOU>Threshold，且类别正确，那么就认为检测正确，否则错误。有了这个规则，我们对于目标检测问题，就可以计算Precision，Recall了，进一步也可以计算mAP。

### Recall@TopK，Precision@TopK

说到Recall的时候，有时候会提及TopK的Recall和Precision。下面是一个小例子，假设有一个目标检测数据集，这个数据集中7张图片。下面是检测结果：

![img](https://pic1.zhimg.com/v2-d0da48377904222bf06769d61893f870_b.jpg)

上面七张图每张图片的检测结果均已标出，绿色是Ground Truth，红色是检测到的对象（总共24个，A~Y）。对上面的所有检测结果按照confidence排名统计出一个表如下，（ ![IOU\ge0.5](http://www.zhihu.com/equation?tex=IOU%5Cge0.5) 则被判定为正样本）：

![img](https://pic3.zhimg.com/v2-bbe9a91407f6658cb128d98a8c44f6c4_b.jpg)

第三列表示预测的置信度(Confidence)排名，第四列表示当前目标框预测正确与否，倒数第一，第二列实际上是TopK的Precision和Recall。前文说过，Precision表示预测出来的正样本正确的比例有多少。那么Top1时，预测了一个正样本，且预测正确，所以Precision=1.0。同理可以算出Recall=0.0666，即总共15个正样本，却只预测出来1个，召回比例就是0.0666。同理TopK的Precision和Recall均可以计算。

## AP

显然，上述例子中，随着K的增大，Recall和Precision会一增一减。那么以Recall为横轴，Precision为纵轴，就可以画出一条PR曲线如下（实际计算时，当Precision下降到一定程度时，后面就直接默认为0，不算了。）：

![img](https://pic1.zhimg.com/v2-733fd447cb68935f6192451101f0f24a_b.jpg)

上图PR曲线下的面积就定义为AP，即：

![AP=\int_0^1 p(r)dr\\](http://www.zhihu.com/equation?tex=AP%3D%5Cint_0%5E1+p%28r%29dr%5C%5C)

上面就是AP的基本思想，实际计算过程中，PASCAL VOC，COCO比赛在上述基础上都有不同的调整策略。

## Interpolated AP（PASCAL VOC 2008的评测指标）

在PASCAL VOC 2008中，在计算AP之前会对上述曲线进行平滑，平滑方法为，对每一个Precision值，使用其右边最大的Precision值替代。具体示意图如下：

![img](https://pic1.zhimg.com/v2-5fbd8eda126a0b1ab50d899961f2cfb2_b.jpg)

通过平滑策略，上面蓝色的PR曲线就变成了红色的虚线了。平滑的好处在于，平滑后的曲线单调递减，不会出现摇摆的情况。这样的话，随着Recall的增大，Precision逐渐降低，这才是符合逻辑的。实际计算时，对平滑后的Precision曲线进行均匀采样出11个点（每个点间隔0.1），然后计算这11个点的平均Precision。具体如下：

![img](https://pic3.zhimg.com/v2-a5fe941c6ff5c620914133ccd3a2e1d1_b.jpg)

![img](https://pic4.zhimg.com/v2-82d073094b183851073c012035990f13_b.jpg)

在本例子中， ![AP = (1 × 0.666 + 0.4285 + 0.4285 + 0.4285 + 0 + 0 + 0+ 0 + 0 + 0 )/11=26.84\%](http://www.zhihu.com/equation?tex=AP+%3D+%281+%C3%97+0.666+%2B+0.4285+%2B+0.4285+%2B+0.4285+%2B+0+%2B+0+%2B+0%2B+0+%2B+0+%2B+0+%29%2F11%3D26.84%5C%25) 。这种计算方法也叫插值AP（Interpolated AP）。对于PASCAL VOC有20个类别，那么mAP就是对20个类别的AP进行平均。

### AP (Area under curve AUC，PASCAL VOC2010–2012评测指标)

上述11点插值的办法由于插值点数过少，容易导致结果不准。一个解决办法就是内插所有点。所谓内插所有点，其实就是对上述平滑之后的曲线算曲线下面积。

这样计算之所以会更准确一点，可以这么看！原先11点采样其实算的是曲线下面积的近似，具体近似办法是：取10个宽为0.1，高为Precision的小矩形的面积平均。现在这个则不然，现在这个算了无数个点的面积平均，所以结果要准确一些。示意图如下：

![img](https://pic3.zhimg.com/v2-15b9d199e7bda605052c9f35616a8cc1_b.jpg)

在本例中，

![img](https://pic3.zhimg.com/v2-371a2538c66dde9a6609af9aef50b543_b.jpg)

## COCO mAP

COCO mAP使用101个点的内插mAP（Interpolated AP），此外，COCO还使用了不同IOU阈值，不同尺度下的AP平均来作为评测结果，比如**AP @ [.5 : .95]**对应于IoU的平均AP，从0.5到0.95，步长为0.05。下面是具体评价指标介绍：

![img](https://pic3.zhimg.com/v2-4a7836a4c65b1160626cbab5138a4ac5_b.jpg)

值得注意的是，COCO的AP就是指mAP，没有刻意区分二者。

## ImageNet目标检测评测指标

在ImageNet目标检测数据集里面则采用上面介绍的AUC方法来计算mAP。一般来说，不同的数据集mAP介绍方法会有一些细微差异。

参考资料：

1. [https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173](http://link.zhihu.com/?target=https%3A//medium.com/%40jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)

2. [https://github.com/rafaelpadilla/Object-Detection-Metrics](



作者：我爱馒头

链接：https://zhuanlan.zhihu.com/p/60834912



mAP是目标检测领域的一个常用指标，但却少有教程能真正说清楚，说明白这个东西。自己也是看了无数次，每次看了忘，忘了又去看。这次写下来，希望能记得牢固一些。

一句话概括AP：**AP表示Recall从0~1的平均精度值**。比如说Recall=0.1时，Precision=y1；Recall=0.2时，Precision=y2；...... ；Recall=1.0时，Precision=yn。那么可以以Recall为横轴，Precision为纵轴画一条曲线，曲线下的面积就是AP。

一句话概括mAP：**mAP表示多个物体类别上的平均AP**。注意，mAP有两次平均，一次在Recall取不同值时平均，一次在类别上平均。

http://link.zhihu.com/?target=https%3A//github.com/rafaelpadilla/Object-Detection-Metrics)