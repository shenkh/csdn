# 决策树ID3

$$
P\left(X=x_{i}\right)=p_{i}, \quad i=1,2, \cdots, n
$$

$$
H(X)=-\sum_{i=1}^{n} p_{i} \log p_{i}
$$

$$
0 \leqslant H(p) \leqslant \log n
$$

当随机变量只取两个值时，
$$
P(X=1)=p, \quad P(X=0)=1-p, \quad 0 \leqslant p \leqslant 1
$$

熵为

$$
H(p)=-p \log _{2} p-(1-p) \log _{2}(1-p)
$$


$$
H(Y | X)=\sum_{i=1}^{n} p_{i} H\left(Y | X=x_{i}\right)
$$

其中
$$
p_{i}=P\left(X=x_{i}\right), \quad i=1,2, \cdots, n
$$

信息增益（互信息）

$$
g(D, A)=H(D)-H(D | A)
$$

数据集$D$的经验熵$H(D)$:
$$
H(D)=-\sum_{k=1}^{K} \frac{\left|C_{k}\right|}{|D|} \log _{2} \frac{\left|C_{k}\right|}{|D|}
$$

特征$A$对数据集$D$的经验条件熵$H(D|A)$:

$$
H(D | A)=\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} H\left(D_{i}\right)=-\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{D| } \sum_{k=1}^{K} \frac{\left|D_{i k}\right|}{\left|D_{i}\right|} \log _{2} \frac{\left|D_{i k}\right|}{\left|D_{i}\right|}
$$

计算信息增益
$$
g(D, A)=H(D)-H(D | A)
$$

![决策树（decision tree）(三)——连续值处理](https://img-blog.csdn.net/20180228140218532)

图片来自[决策树（decision tree）(三)——连续值处理](https://blog.csdn.net/u012328159/article/details/79396893)

对连续值采用二分法计算信息增益。

```python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np


class DecisionTree:

    def __init__(self, features, labels):
        '''
        features : dataframe
        labels： series
        '''
        self.features = features
        self.labels = labels

    def __info_entropy(self, labels):
        # labels : Series类型
        # https://stackoverflow.com/questions/36004976/count-frequency-of-values-in-pandas-dataframe-column
        total = labels.count()  # 标签有效总数
        counts = labels.value_counts().to_dict()  # {标签名：数量}
        entropy = 0
        for _, value in counts.items():
            p = value / total
            entropy += -p * np.log2(p)
        return entropy

    def __dataset_info(self, labels):
        return self.__info_entropy(labels)

    def get_info_entropy(self, feature, labels):
        feature_values = feature.unique()
        feature_info = 0
        for value in feature_values:
            feature_i_labels = labels[feature == value]
            info = self.__info_entropy(feature_i_labels)
            feature_info += feature_i_labels.size / feature.size * info

        dataset_info = self.__dataset_info(labels)  # G(D)

        return dataset_info-feature_info, feature.name

    def get_continous_var_entropy(self, feature, labels):
        df_feature = pd.concat([feature, labels], axis=1)
        df_feature = df_feature.sort_values(feature.name)
        feature = df_feature[feature.name]
        labels = df_feature[labels.name]

        median = feature.to_numpy(dtype=np.float32)
        median = (median[0:-1] + median[1:]) / 2

        info = self.__dataset_info(labels)
        res = 0
        split_val = median[0]
        for boundary in median:
            label_left = labels[feature < boundary]
            label_right = labels[feature > boundary]
            cur_info_left = self.__info_entropy(label_left)
            cur_info_right = self.__info_entropy(label_right)
            cur_info = label_left.size / feature.size * cur_info_left + \
                label_right.size / feature.size * cur_info_right
            # print(info - cur_info)
            if info - cur_info > res:
                res = info - cur_info
                split_val = boundary

        return res, feature.name, split_val

    def feature_choose(self, discrete_features_name, continue_features_name):
        entropy = [self.get_info_entropy(
            self.features[name], self.labels) for name in discrete_features_name]
        entropy2 = [self.get_continous_var_entropy(
            self.features[name], self.labels) for name in continue_features_name]
        entropy += entropy2
        return max(entropy)


def build_tree(features, labels):

    if labels.unique().size == 1:
        print(labels[0:1])
        return

    decision_tree = DecisionTree(features, labels)

    # info = decision_tree.feature_choose(
    #     ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'], ['密度', '含糖率'])

    header = features.columns.to_list()
    info = decision_tree.feature_choose(header[0:-2], header[-2:])

    print(info)
    decision_tree.features[labels.name] = labels
    for name, group in decision_tree.features.groupby(info[1] if len(info) == 2 else
                                                      decision_tree.features[info[1]] < info[2]):
        print(name)
        if info[1] in ['密度', '含糖率']:
            group = group
        else:
            group = group.drop(info[1], axis=1)
        labels = group[labels.name]
        group = group.drop(labels.name, axis=1)
        build_tree(group, labels)


if __name__ == "__main__":

    df = pd.read_csv('datasets.txt', sep=',', encoding='gb2312')
    build_tree(df[df.columns[1:9]], df[df.columns[9]])

    decision_tree = DecisionTree(df[df.columns[1:9]], df[df.columns[9]])
    info = decision_tree.get_continous_var_entropy(df['密度'], df['好瓜'])
    print(info)

```