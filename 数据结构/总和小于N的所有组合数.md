## 总和小于N的组合数

**[题目描述](https://www.nowcoder.com/practice/bf877f837467488692be703735db84e6?tpId=98&tqId=32831&tPage=1&rp=8&ru=%2Fta%2F2019test&qru=%2Fta%2F2019test%2Fquestion-ranking)**

牛牛准备参加学校组织的春游, 出发前牛牛准备往背包里装入一些零食, 牛牛的背包容量为w。

牛牛家里一共有n袋零食, 第i袋零食体积为v[i]。

牛牛想知道在总体积不超过背包容量的情况下,他一共有多少种零食放法(总体积为0也算一种放法)。

**输入描述:**

```
输入包括两行第一行为两个正整数n和w(1 <= n <= 30, 1 <= w <= 2 * 10^9),表示零食的数量和背包的容量。第二行n个正整数v[i](0 <= v[i] <= 10^9),表示每袋零食的体积。
```

**输出描述:**

```
输出一个正整数, 表示牛牛一共有多少种零食放法。
```

**示例1**

**输入**

```
3 10
1 2 4
```

**输出**

```
8
```

**说明**

三种零食总体积小于10,于是每种零食有放入和不放入两种情况，一共有`2*2*2 = 8`种情况。

-----

```cpp
#include<iostream>
#include<numeric>
#include<vector>
#include<algorithm>
using namespace std;

void helper(vector<int>& nums, int start, int remain, long& res) {
	if (remain < 0)
		return;
	if (remain >= 0)
		++res;
	for (int i = start; i < nums.size(); ++i) {
		helper(nums, i + 1, remain - nums[i], res);
	}
	return;
}

int main() {
	int n;
	long w;
	cin >> n >> w;
	vector<int> nums(n);
	long sum = 0;
	for (int i = 0; i < n; ++i) {
		cin >> nums[i];
		sum += nums[i];
	}

	long res = 0;
	if (sum <= w)
		res = pow(2, n);
	else
		helper(nums, 0, w, res);
	cout << res << endl;
	return 0;
}
```
