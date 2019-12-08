## BN层合并

转自:[BN层合并](https://hey-yahei.cn/2018/08/08/MobileNets-SSD/index.html#BN%E5%B1%82%E5%90%88%E5%B9%B6)

对比chuanqi305的 <a href="https://github.com/chuanqi305/MobileNet-SSD/blob/master/template/MobileNetSSD_train_template.prototxt" target="_blank" rel="noopener">train模型</a> 和 <a href="https://github.com/chuanqi305/MobileNet-SSD/blob/master/template/MobileNetSSD_deploy_template.prototxt" target="_blank" rel="noopener">deploy模型</a> 还能发现一件有趣的事情 ——  **deploy模型中的BN层和scale层都不见啦!!** BN层是这样随随便便就能丢弃的么？没道理啊!

几经辗转，查阅资料之后发现，原来BN层是可以合并进前一层的卷积层或全连接层的，而且这还有利于减少预测用时。参考《<a href="http://machinethink.net/blog/object-detection-with-yolo/#converting-to-metal" target="_blank" rel="noopener">Real-time object detection with YOLO - Converting to Metal</a>》

合并的原理：卷积层、全连接层和BN层都是纯粹的线性转换。

数学推导也很简单：假设图片为 $x$ ，卷积层权重为 $w$ 。那么对于卷积运算有，$$ conv[j] = x[i]w[0] + x[i+1]w[1] + x[i+2]w[2] + … + x[i+k]w[k] + b $$BN层运算为，$$ bn[j] = \frac{\gamma (conv[j] - mean)}{\sqrt{variance}} + \beta = \frac{\gamma \cdot conv[j]}{\sqrt{variance}} - \frac{\gamma \cdot mean}{\sqrt{variance}} + \beta $$代入$conv[j]$变为，$$ bn[j] = x[i] \frac{\gamma \cdot w[0]}{\sqrt{variance}} + x[i+1] \frac{\gamma \cdot w[1]}{\sqrt{variance}} + … + x[i+k] \frac{\gamma \cdot w[k]}{\sqrt{variance}} + \frac{\gamma \cdot b}{\sqrt{variance}} - \frac{\gamma \cdot mean}{\sqrt{variance}} + \beta $$两式对比可以得到，$$ w_{new} = \frac{\gamma \cdot w}{\sqrt{variance}} $$$$ b_{new} = \beta + \frac{\gamma \cdot b}{\sqrt{variance}} - \frac{\gamma \cdot mean}{\sqrt{variance}} = \beta + \frac{\gamma (b-mean)}{\sqrt{variance}} $$注意，其中 $\gamma$、$mean$、$variance$、$\beta$ 都是训练出来的量，在预测阶段相当于一个常量。

原文摘录如下：

[converting-to-metal](http://machinethink.net/blog/object-detection-with-yolo/#converting-to-metal)

However, there was a small wrinkle… YOLO uses a regularization technique called *batch normalization* after its convolutional layers.

The idea behind “batch norm” is that neural network layers work best when the data is clean. Ideally, the input to a layer has an average value of 0 and not too much variance. This should sound familiar to anyone who’s done any machine learning because we often use a technique called “feature scaling” or “whitening” on our input data to achieve this.

Batch normalization does a similar kind of feature scaling for the data in between layers. This technique really helps neural networks perform better because it stops the data from deteriorating as it flows through the network.

To give you some idea of the effect of batch norm, here is a histogram of the output of the first convolution layer without and with batch normalization:

![Histogram of first layer output with and without batch norm](https://machinethink.net/images/yolo/BatchNorm@2x.png)

Batch normalization is important when training a deep network, but it turns out we can get rid of it at inference time. Which is a good thing because not having to do the batch norm calculations will make our app faster. And in any case, Metal does not have an `MPSCNNBatchNormalization` layer.

Batch normalization usually happens after the convolutional layer but *before* the activation function gets applied (a so-called “leaky” ReLU in the case of YOLO). Since both convolution and batch norm perform a linear transformation of the data, we can combine the batch normalization layer’s parameters with the weights for the convolution. This is called “folding” the batch norm layer into the convolution layer.

Long story short, with a bit of math we can get rid of the batch normalization layers but it does mean we have to change the weights of the preceding convolution layer.

A quick recap of what a convolution layer calculates: if `x` is the pixels in the input image and `w` is the weights for the layer, then the convolution basically computes the following for each output pixel:

```nohighlight
out[j] = x[i]*w[0] + x[i+1]*w[1] + x[i+2]*w[2] + ... + x[i+k]*w[k] + b
```

This is a dot product of the input pixels with the weights of the convolution kernel, plus a bias value `b`.

And here’s the calculation performed by the batch normalization to the output of that convolution:

```nohighlight
        gamma * (out[j] - mean)
bn[j] = ---------------------- + beta
            sqrt(variance)
```

It subtracts the mean from the output pixel, divides by the variance, multiplies by a scaling factor gamma, and adds the offset beta. These four parameters — `mean`, `variance`, `gamma`, and `beta` — are what the batch normalization layer learns as the network is trained.

To get rid of the batch normalization, we can shuffle these two equations around a bit to compute new weights and bias terms for the convolution layer:

```nohighlight
           gamma * w
w_new = --------------
        sqrt(variance)

        gamma*(b - mean)
b_new = ---------------- + beta
         sqrt(variance)
```

Performing a convolution with these new weights and bias terms on input `x` will give the same result as the original convolution plus batch normalization.

Now we can remove this batch normalization layer and just use the convolutional layer, but with these adjusted weights and bias terms `w_new` and `b_new`. We repeat this procedure for all the convolutional layers in the network.

**Note:** The convolution layers in YOLO don’t actually use bias, so `b` is zero in the above equation. But note that after folding the batch norm parameters, the convolution layers *do* get a bias term.

Once we’ve folded all the batch norm layers into their preceding convolution layers, we can convert the weights to Metal. This is a simple matter of transposing the arrays (Keras stores them in a different order than Metal) and writing them out to binary files of 32-bit floating point numbers.

If you’re curious, check out the conversion script [yolo2metal.py](https://github.com/hollance/Forge/blob/master/Examples/YOLO/yolo2metal.py) for more details. To test that the folding works the script creates a new model without batch norm but with the adjusted weights, and compares it to the predictions of the original model.