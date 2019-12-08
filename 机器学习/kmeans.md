# kmeans实现

kmeans是经典的无监督聚类方法。步骤可以分为以下几步：

- 确定聚类的数目`k`

- 随机初始化`k`个聚类中心
- 根据准则（一般计算欧式距离）将数据分配到对应的聚类中心
- 更新每个类别的聚类中心（均值）
- 重复步骤2-步骤4若干次



```python
# -*- coding:utf-8 -*-
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt


class KMeans():

    def __init__(self, data, k, max_iter=5):
        self._data = data
        self._k = k
        self._max_iter = max_iter
        self._example_num = data.shape[0]
        self._centroids = None
        self._cluster_data_indices = None

    def _random_init_centroid(self):
        random_centroid_indices = random.sample(
            range(0, self._example_num), self._k)
        centroids = self._data[random_centroid_indices]
        self._centroids = centroids

    @staticmethod
    def get_closest_centroid(data, centroids):
        '''data: 1*C, centroid: k*C'''
        distance = np.sum(np.power((data - centroids), 2), 1)
        indices = np.argmin(distance)
        return indices

    def _assign_data_closest_centroid(self):
        ddict = defaultdict(list)
        for row in range(0, self._example_num):
            cloest_centroid = self.get_closest_centroid(
                self._data[row, :], self._centroids)
            ddict[cloest_centroid].append(row)
        self._cluster_data_indices = ddict

    def _update_centroids(self):
        for i in range(0, self._k):
            data_indices = self._cluster_data_indices[i]
            self._centroids[i] = np.mean(self._data[data_indices, :], 0)

    def _plot_clusters(self):
        plt.figure()
        for i in range(0, self._k):
            data_indices = self._cluster_data_indices[i]
            plt.scatter(self._data[data_indices, 0], self._data[data_indices, 1], s=10)
        plt.savefig('kmeans_{}.png'.format(self._index))

    def fit(self):
        self._random_init_centroid()
        for i in range(self._max_iter):
            self._index = i
            self._assign_data_closest_centroid()
            self._update_centroids()
            self._plot_clusters()


if __name__ == "__main__":
    arra = np.random.uniform(1, 6, (200, 2))
    arrb = np.random.uniform(4, 10, (200, 2))
    arrc = np.random.uniform(9, 15, (200, 2))
    arr = np.concatenate((arra, arrb, arrc), axis=0)
    kmeans = KMeans(arr, 3)
    kmeans.fit()
    print('done')

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190518144214688.gif)

参考：[K-Means Algorithm from Scratch](https://dfrieds.com/machine-learning/k-means-from-scratch-python)

