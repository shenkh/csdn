# Bellman_Ford

`Dijkstra`求解单源最短路径时，运用的是贪心算法，因此无法处理图中存在负权边的情况，`Bellman_Ford`则可以解决负权边的问题。用`w[i]`表示边`u[i]->v[i]`的权重，`dist[t]`表示由源点到目标结点`t`的最短路径。`Bellman_Ford`的思想是第一次只考虑经过一条边的最短路径，然后第二次考虑经过两条边到达目标结点的最短路径，这样依次增多。如果`dist[v[i]] > dist[u[i]] + w[i]`，说明到达结点`s->v[i]`的最短路径可以通过`s->u[i]->v[i]`进行松弛。对每条边进行松弛（内层比较`edge_num`次），外层共循环`node_num-1`次。

如果外层进行第`node_num`次循环，仍能对某个`dist[t]`进行松弛，说明存在负权环。

![https://blog.csdn.net/anlian523/article/details/80953767](https://img-blog.csdn.net/2018070720090497)
注：图片来源于<https://blog.csdn.net/anlian523/article/details/80953767>

```cpp
#include<iostream>
#include<vector>
#include<algorithm>
#include<string>
#include<limits>
using namespace std;

/***************************************
Bellman_Ford可以处理有负边的情况，但是无法处理负权环的问题。
***************************************/
bool Bellman_Ford(vector<int>& u, vector<int>& v, vector<int>& w, vector<int>& dist,
    vector<int>& path, int node_num, int edge_num, int source) 
{
    //n个结点的单源最短路径最多经过n-1条边
    for (int k = 0; k < node_num - 1; ++k) {
        //对存在的边进行松弛
        for (int i = 0; i < edge_num; ++i) {
            if (dist[u[i]] + w[i] < dist[v[i]]) {
                dist[v[i]] = dist[u[i]] + w[i];
                path[v[i]] = u[i];
            }
        }
    }
    //如果在进行了n-1次松弛之后，仍然存在某个dist[v[i]] > dist[u[i]] + w[i] 的情况，还可以继续成功松弛，那么必然存在回路了
    //因为正常来讲最短路径包含的边最多只会有n - 1条
    for (int i = 0; i < edge_num; ++i) {
        if (dist[u[i]] + w[i] < dist[v[i]])
            return false;
    }
    return true;
}

vector<int> shortest_path(vector<int>& path, int source, int target) {
    cout << source << "->" << target << "最短路线：";
    vector<int> res;
    while (target != source)
    {
        res.push_back(target);
        target = path[target];
    }
    res.push_back(source);
    reverse(res.begin(), res.end());
    for (auto item : res)
        cout << item << ",";
    cout << endl;
    return res;
}

int main() {
    int node_num, edge_num;
    cin >> node_num >> edge_num;
    vector<int> u(edge_num), v(edge_num), w(edge_num);
    int i = 0;
    while (i < edge_num){
        cin >> u[i] >> v[i] >> w[i];
        ++i;
    }

    int source;
    cin >> source;
    vector<int> dist(node_num, INT_MAX / 3);
    dist[source] = 0;
    vector<int> path(node_num, source);

    bool flag = Bellman_Ford(u, v, w, dist, path, node_num, edge_num, source);
    if (flag == false) {
        cout << "存在负权环，无法使用Bellman-Fold算法" << endl;
        return -1;
    }

    for (int i = 0; i < node_num; ++i) {
        shortest_path(path, source, i);
        cout << dist[i] << endl;
    }
}
```

```
输入：
5
8
0 1 -1
0 2 4
1 2 3
1 3 2
1 4 2
3 2 5
3 1 1
4 3 -3
0

输出：
0->0最短路线：0,
0
0->1最短路线：0,1,
-1
0->2最短路线：0,1,2,
2
0->3最短路线：0,1,4,3,
-2
0->4最短路线：0,1,4,
1
```

-----
最短路径算法的对比

![最短路径对比](https://images2018.cnblogs.com/blog/886183/201806/886183-20180626153322366-1810228590.jpg)

注：图片来自于<https://www.cnblogs.com/thousfeet/p/9229395.html>

-----
参考：

[看完就懂了！一篇搞定图论最短路径问题](https://www.cnblogs.com/thousfeet/p/9229395.html) **（推荐）**

[Bellman-ford算法详解——负权环分析](https://blog.csdn.net/anlian523/article/details/80953767)
