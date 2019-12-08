# Dijkstra

`Dijkstra`是求解单源最短路径的典型方法，指定源点`s`，求它到其余各个结点的最短路。思路是将所有结点分为两部分，一部分是已经确定最短路径的结点集`P`，剩余的结点集保存在`Q`中。初始时`P`中只有一个结点`s`，`s`距离`Q`中各个结点的距离`dis`初始化为`INT_MAX`。每次在`Q`中找到距离源点`s`最近的结点`u`，将`u`加入到集合`P`中，这样即确定了`s`到`u`的最短距离`dis[u]`；同时更新由`s`到`Q`中剩余结点的最短距离，即对于`Q`中的特定结点`v`,比较原始的最短的路径值和经过了中间结点`u`的最短路径，取两者中的较小值:`dis[v] = min(dis[v], dis[u]+matrix[u][v])`，其中`matrix`为邻接矩阵。

如果需要记录到达某个结点的最短路径所经过的是哪些结点，那么需要维护`path[i]`这个数组，其中`path[t]`表示由源点`s`到达结点`t`最短路径中，`t`的前驱结点。如`s->...->t0->t`为`s`到`t`的最短路径，那么`path[t]=t0`，再去求解到达`t0`的最短路径，直到找到源点`s`。`Dijkstra`本质是一种贪心算法，无法处理有负权重的问题。

![示意图](https://upload-images.jianshu.io/upload_images/2295192-b54b0c630ff6dada.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/621/format/webp)
注：图片来源于<https://www.jianshu.com/p/92e46d990d17>

```cpp
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<limits>
using namespace std;
#define INF 1000

//输入的path应初始化为source，表示每个节点的前驱节点
vector<int> Dijkstra(vector<vector<int>>& matrix, int source, vector<int>& path)
{
    int n = matrix.size();
    vector<bool> visited(n, false);
    visited[source] = true;                //初始时P集合只有source这个节点
    vector<int> dis(matrix[source].begin(), matrix[source].end());

    //循环n-1次就能将Q遍历空
    for (int time = 1; time < n; ++time) {
        int min = INT_MAX;
        int u = source;
        //寻找Q中最近的结点，保存到u中
        for (int i = 0; i < n; ++i) {
            if (!visited[i] && dis[i] < min) {
                min = dis[i];
                u = i;
            }
        }

        visited[u] = true;                   //将结点u加入到P集合
        for (int v = 0; v < n; ++v) {        //更新u的所有出边
            if (!visited[v] && matrix[u][v] != INT_MAX && dis[u] + matrix[u][v] < dis[v]) {
                dis[v] = dis[u] + matrix[u][v];
                path[v] = u;
            }
        }
    }
    return dis;
}

int main() {
    int num = 6;     //结点数
    vector<vector<int>> matrix(num, vector<int>(num, INT_MAX));
    for (int i = 0; i < num; ++i) {
        matrix[i][i] = 0;
    }

    int s, t, weight;
    while (cin >> s >> t >> weight) {
        matrix[s][t] = weight;
    }

    int source = 0;
    vector<int> path(num, source);
    vector<int> dis = Dijkstra(matrix, source, path);

    //输出到每个结点的最短路线
    for (int i = 0; i < path.size(); ++i) {
        cout << dis[i] << ",";
        int j = i;
        vector<int> shortest_path;
        while (j != source) {
            shortest_path.push_back(j);
            j = path[j];
        }
        shortest_path.push_back(source);
        for (auto iter = shortest_path.rbegin(); iter != shortest_path.rend(); ++iter){
            cout << "->" << *iter;
        }
        cout << endl;
    }
}
```

```
输入：
0 2 10
0 4 30
0 5 100
1 2 5
2 3 50
3 5 10
4 3 20
4 5 60

输出：
0,->0
2147483647,->0->1
10,->0->2
50,->0->4->3
30,->0->4
60,->0->4->3->5
```