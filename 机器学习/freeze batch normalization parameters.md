**背景**：基于`PyTorch`的模型，想固定主分支参数，只训练子分支，结果发现在不同`epoch`相同的测试数据经过主分支输出的结果不同。

**原因**：未固定主分支`BN`层中的`running_mean`和`running_var`。

**解决方法**：将需要固定的`BN`层状态设置为`eval`。

问题示例：

**环境**：`torch`：1.7.0

```python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def print_parameter_grad_info(net):
    print('-------parameters requires grad info--------')
    for name, p in net.named_parameters():
        print(f'{name}:\t{p.requires_grad}')

def print_net_state_dict(net):
    for key, v in net.state_dict().items():
        print(f'{key}')

if __name__ == "__main__":
    net = Net()

    print_parameter_grad_info(net)
    net.requires_grad_(False)
    print_parameter_grad_info(net)

    torch.random.manual_seed(5)
    test_data = torch.rand(1, 1, 32, 32)
    train_data = torch.rand(5, 1, 32, 32)

    # print(test_data)
    # print(train_data[0, ...])
    for epoch in range(2):
        # training phase, 假设每个epoch只迭代一次
        net.train()
        pre = net(train_data)
        # 计算损失和参数更新等
        # ....

        # test phase
        net.eval()
        x = net(test_data)
        print(f'epoch:{epoch}', x)
```

运行结果：

```
-------parameters requires grad info--------
conv1.weight:   True
conv1.bias:     True
bn1.weight:     True
bn1.bias:       True
conv2.weight:   True
conv2.bias:     True
bn2.weight:     True
bn2.bias:       True
fc1.weight:     True
fc1.bias:       True
fc2.weight:     True
fc2.bias:       True
fc3.weight:     True
fc3.bias:       True
-------parameters requires grad info--------
conv1.weight:   False
conv1.bias:     False
bn1.weight:     False
bn1.bias:       False
conv2.weight:   False
conv2.bias:     False
bn2.weight:     False
bn2.bias:       False
fc1.weight:     False
fc1.bias:       False
fc2.weight:     False
fc2.bias:       False
fc3.weight:     False
fc3.bias:       False
epoch:0 tensor([[-0.0755,  0.1138,  0.0966,  0.0564, -0.0224]])
epoch:1 tensor([[-0.0763,  0.1113,  0.0970,  0.0574, -0.0235]])
```

可以看到：

`net.requires_grad_(False)`已经将网络中的各参数设置成了不需要梯度更新的状态，但是同样的测试数据`test_data`在不同`epoch`中前向之后出现了不同的结果。

调用`print_net_state_dict`可以看到`BN`层中的参数`running_mean`和`running_var`并没在可优化参数`net.parameters`中

```
bn1.weight
bn1.bias
bn1.running_mean
bn1.running_var
bn1.num_batches_tracked
```

但在`training pahse`的前向过程中，这两个参数被更新了。导致整个网络在`freeze`的情况下，同样的测试数据出现了不同的结果

> Also by default, during training this layer keeps running estimates of its computed mean and variance, which are then used for normalization during evaluation. The running estimates are kept with a default`momentum`of 0.1. [source](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batchnorm#torch.nn.BatchNorm2d)

因此在training phase时对BN层显式设置`eval`状态：

```python
if __name__ == "__main__":
    net = Net()
    net.requires_grad_(False)

    torch.random.manual_seed(5)
    test_data = torch.rand(1, 1, 32, 32)
    train_data = torch.rand(5, 1, 32, 32)

    # print(test_data)
    # print(train_data[0, ...])
    for epoch in range(2):
        # training phase, 假设每个epoch只迭代一次
        net.train()
        net.bn1.eval()
        net.bn2.eval()
        pre = net(train_data)
        # 计算损失和参数更新等
        # ....

        # test phase
        net.eval()
        x = net(test_data)
        print(f'epoch:{epoch}', x)

```

可以看到结果正常了：

```
epoch:0 tensor([[ 0.0944, -0.0372,  0.0059, -0.0625, -0.0048]])
epoch:1 tensor([[ 0.0944, -0.0372,  0.0059, -0.0625, -0.0048]])
```

参考：

[Cannot freeze batch normalization parameters](https://discuss.pytorch.org/t/cannot-freeze-batch-normalization-parameters/38696)

[BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batchnorm#torch.nn.BatchNorm2d)
