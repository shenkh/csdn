# Bellman_Ford

`Dijkstra`求解单源最短路径时，运用的是贪心算法，因此无法处理图中存在负权边的情况，`Bellman_Ford`则可以解决负权边的问题。用`w[i]`表示边`u[i]->v[i]`的权重，`dist[t]`表示由源点到目标顶点`t`的最短路径。`Bellman_Ford`的思想是第一次只考虑经过一条边的最短路径，然后第二次考虑经过两条边到达目标顶点的最短路径，这样依次增多。如果`dist[v[i]] > dist[u[i]] + w[i]`，说明到达顶点`s->v[i]`的最短路径可以通过`s->u[i]->v[i]`进行松弛。对每条边进行松弛（内层比较`edge_num`次），外层共循环`node_num-1`次。

如果外层进行第`node_num`次循环，仍能对某个`dist[t]`进行松弛，说明存在负权环。

![https://blog.csdn.net/anlian523/article/details/80953767](https://img-blog.csdn.net/2018070720090497)
注：图片来源于<https://blog.csdn.net/anlian523/article/details/80953767>

```cpp
#include<iostream>
#include<vector>
#include<algorithm>
#include<limits>
using namespace std;

/**************************************************************
Bellman_Ford可以处理有负边的情况，但是无法处理负权环问题。
**************************************************************/
class BellmanFord {
public:
    BellmanFord(int nvert, int nedge) : nvert(nvert), nedge(nedge), dist(nvert, INT_MAX) {}
public:
    int nvert, nedge;  //顶点数，边数
    vector<int> dist; //最短路径
private:
    vector<int> u, v, w;
    vector<int> path;
    int src;
public:
    //顶点的序号从0开始
    void setEdge(int ui, int vi, int wi) {
        u.push_back(ui);
        v.push_back(vi);
        w.push_back(wi);
    }

    //如果返回false表示有负权环，bellman_ford无法处理
    bool bellman_ford(int src) {
        path.resize(nvert, src);
        this->src = src;
        dist[src] = 0;
        //n个顶点的单源最短路径最多经过n-1条边
        for (int k = 0; k < nvert - 1; ++k) {
            //对存在的边进行松弛
            for (int i = 0; i < nedge; ++i) {
                if (dist[u[i]] == INT_MAX)
                    continue;
                if (dist[u[i]] + w[i] < dist[v[i]]) {
                    dist[v[i]] = dist[u[i]] + w[i];
                    path[v[i]] = u[i];
                }
            }
        }
        //如果在进行了n-1次松弛之后，仍然存在某个dist[v[i]] > dist[u[i]] + w[i] 的情况，
        //还可以继续成功松弛，那必然存在回路。因为正常来讲最短路径包含的边最多只会有n - 1条。
        for (int i = 0; i < nedge; ++i) {
            if (dist[u[i]] + w[i] < dist[v[i]]) {
                return false;
            }
        }
        return true;
    }

    vector<int> show_shortest_path(int dst) {
        if (dst >= nvert) {
            cout << "dst参数范围有误" << endl;
            return vector<int>{};
        }
        cout << src << "->" << dst << "最短路线：";
        vector<int> res;
        while (dst != src){
            res.push_back(dst);
            dst = path[dst];
        }
        res.push_back(src);
        reverse(res.begin(), res.end());
        for (auto item : res)
            cout << item << ",";
        cout << endl;
        return res;
    }
};

int main() {
    int n, edge;
    cin >> n >> edge;

    BellmanFord bellmanford(n, edge);
    while (edge--)
    {
        int s, e, w;
        cin >> s >> e >> w;
        bellmanford.setEdge(s, e, w);
    }

    if (bellmanford.bellman_ford(0)) {
        for (auto item : bellmanford.dist) {
            cout << item << endl;
        }
        bellmanford.show_shortest_path(3);
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

输出：
0
-1
2
-2
1
0->3最短路线：0,1,4,3,
```

-----
最短路径算法的对比

![最短路径对比](https://images2018.cnblogs.com/blog/886183/201806/886183-20180626153322366-1810228590.jpg)

注：图片来自于<https://www.cnblogs.com/thousfeet/p/9229395.html>

-----
参考：

[看完就懂了！一篇搞定图论最短路径问题](https://www.cnblogs.com/thousfeet/p/9229395.html) **（推荐）**

[Bellman-ford算法详解——负权环分析](https://blog.csdn.net/anlian523/article/details/80953767)
