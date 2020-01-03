# [剑指offer](https://www.nowcoder.com/ta/coding-interviews)

## 二维数组中的查找

```cpp
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int r = array.size();
        if(r == 0) return false;
        int c = array[0].size();
        int i = 0, j = c - 1;
        while(i < r && j >= 0){
            if(array[i][j] > target)
                --j;
            else if(array[i][j] < target)
                ++i;
            else
                return true;
        }
        return false;
    }
};
```

## 替换空格

```cpp
class Solution {
public:
    void replaceSpace(char *str,int length) {
        if(str == nullptr || length <= 0)
            return;
        int num_blanks = 0;

        int i = 0;
        int len = 0;
        while(str[i] != '\0'){
            ++len;
            if(str[i++] == ' ')
                ++num_blanks;
        }

        int newlen = len + 2*num_blanks;
        if(newlen > length)
            return;
        for(int i = len; i >=0; --i){
            if(str[i] == ' '){
                str[newlen--] = '0';
                str[newlen--] = '2';
                str[newlen--] = '%';
            }
            else
                str[newlen--] = str[i];
        }
        return;
    }
};
```

## 从尾到头打印链表

```cpp
/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> res;
        helper(head, res);
        return res;
    }
private:
    void helper(ListNode* head, vector<int>& res){
        if(head == nullptr)
            return;
        helper(head->next, res);
        res.push_back(head->val);
    }
};
```

## 用两个栈实现队列

以下分析转自[MaXboy](https://www.nowcoder.com/questionTerminal/54275ddae22f475981afa2244dd448c6)

> 1，整体思路是元素先依次进入栈1，再从栈1依次弹出到栈2，然后弹出栈2顶部的元素，整个过程就是一个队列的先进先出；
>
> 2，但是在交换元素的时候需要判断两个栈的元素情况：
>
> **进队列时**，队列中是还还有元素，若有，说明栈2中的元素不为空，此时就先将栈2的元素倒回到栈1 中，保持在“进队列状态”。
>
> **出队列时**，将栈1的元素全部弹到栈2中，保持在“出队列状态”。
>
> 所以要做的判断是，进时，栈2是否为空，不为空，则栈2元素倒回到栈1，出时，将栈1元素全部弹到栈2中，直到栈1为空。
>

```cpp
class Solution
{
public:
    void push(int node) {
        while(!stack2.empty()){
            int num = stack2.top();
            stack2.pop();
            stack1.push(num);
        }
        stack1.push(node);
    }

    int pop() {
        while(!stack1.empty()){
            int num = stack1.top();
            stack1.pop();
            stack2.push(num);
        }
        if(stack2.empty())
            return -1;
        int res = stack2.top();
        stack2.pop();
        return res;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

## 旋转数组的最小值（有重复数值）

```cpp
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int hi = rotateArray.size() - 1;
        if(hi < 0) return 0;
        int lo = 0;
        while(lo < hi){
            int mid = lo + (hi - lo)/2;
            if(rotateArray[mid] > rotateArray[hi])
                lo = mid + 1;
            else if(rotateArray[mid] < rotateArray[hi])
                hi = mid;
            else
                hi = hi - 1;
        }
        return rotateArray[lo];
    }
};
```

## 斐波那契数列

```cpp
class Solution {
public:
    int Fibonacci(int n) {
        if(n <=0) return 0;
        if(n == 1) return 1;
        int first = 0, second = 1;
        int res = 0;
        for(int i = 2; i <= n; ++i){
            res = first + second;
            first = second;
            second = res;
        }
        return res;
    }
};
```

## 跳台阶

```cpp
class Solution {
public:
    int jumpFloor(int number) {
        if(number == 1) return 1;
        if(number == 2) return 2;
        if(number <= 0) return 0;
        int first = 1;
        int second = 2;
        int res = 0;
        for(int i = 3; i <= number; ++i){
            res = first + second;
            first = second;
            second = res;
        }
        return res;
    }
};
```

## 矩形覆盖

同斐波那契数列。

## 1的个数

```cpp
class Solution {
public:
     int  NumberOf1(int n) {
         int cnt = 0;
         while(n){
             n = n & (n-1);
             ++cnt;
         }
         return cnt;
     }
};
```

## 数的整数次方

```cpp
class Solution {
public:
    double Power(double base, int exponent) {
        if(exponent < 0){
            if(base == 0)
                return -1;
            base = 1 / base;
            exponent = -exponent;
        }
        
        double res = 1;
        double cur = base;
        while(exponent != 0){
            if(exponent & 1 == 1)
                res *= cur;
            cur *= cur;
            exponent >>= 1;
        }
        return res;
    }
};
```

## 调整数组顺序使奇数位于偶数前面

```cpp
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        vector<int> even;
        int last = 0;
        for(int i = 0; i < array.size(); ++i){
            if(array[i] % 2 == 1)
                swap(array[i], array[last++]);
            else
                even.push_back(array[i]);
        }
        for(int i = last; i < array.size(); ++i)
            array[i] = even[i-last];
    }
};
```

## 链表中倒数第k个结点

```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        ListNode* slow = pListHead, *fast = pListHead;
        for(int i = 0; i < k; ++i){
            if(fast == nullptr)
                return nullptr;
            fast = fast->next;
        }
        while(fast != nullptr){
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
};
```

## 反转链表

```cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
            val(x), next(NULL) {
    }
};*/
class Solution {
public:
    ListNode* ReverseList(ListNode* pHead) {
        if(pHead == nullptr)
            return nullptr;
        ListNode* pre = nullptr, *cur = pHead;

        while(cur != nullptr){
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }

        return pre;
    }
};
```

## 合并有序链表

```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        ListNode* dummy = new ListNode(0);
        ListNode* head = dummy;
        while(pHead1 != nullptr && pHead2 != nullptr){
            if(pHead1->val < pHead2->val){
                head->next = pHead1;
                pHead1 = pHead1->next;
            }
            else{
                head->next = pHead2;
                pHead2 = pHead2->next;
            }
            head = head->next;
        }
        
        if(pHead1 != nullptr)
            head->next = pHead1;
        if(pHead2 != nullptr)
            head->next = pHead2;
        return dummy->next;
    }
};
```

## 树的子结构

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(pRoot1 == nullptr || pRoot2 == nullptr)
            return false;
        return helper(pRoot1, pRoot2) || HasSubtree(pRoot1->left, pRoot2) ||
            HasSubtree(pRoot1->right, pRoot2);
    }
private:
    bool helper(TreeNode* root1, TreeNode* root2){
        if(root2 == nullptr)
            return true;
        if(root1 == nullptr)
            return false;
        return (root1->val == root2->val) && helper(root1->left, root2->left) && 
            helper(root1->right, root2->right);
    }
};
```

## 二叉树的镜像

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};*/
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if(pRoot == nullptr)
            return;
        swap(pRoot->left, pRoot->right);
        if(pRoot->left != nullptr)
            Mirror(pRoot->left);
        if(pRoot->right != nullptr)
            Mirror(pRoot->right);
    }
};
```

## 顺时针打印矩阵

```cpp
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        const int dc[4] = {1, 0, -1, 0};
        const int dr[4] = {0, 1, 0, -1};
        int rows = matrix.size();
        int cols = rows > 0 ? matrix[0].size() : 0;
        vector<vector<int>> seen(rows, vector<int>(cols, false));
        vector<int> res;
        int di = 0, r = 0, c = 0;
        for(int i = 0; i < rows*cols; ++i){
            res.push_back(matrix[r][c]);
            seen[r][c] = true;
            int rr = r + dr[di];
            int cc = c + dc[di];
            if(0 <= rr && rr < rows && 0 <= cc && cc < cols && !seen[rr][cc]){
                r = rr;
                c = cc;
            }
            else{
                di = (di + 1) % 4;
                r += dr[di];
                c += dc[di];
            }
        }
        return res;
    }
};
```

## 栈的压入、弹出序列

```cpp
// https://www.nowcoder.com/questionTerminal/d77d11405cc7470d82554cb392585106
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        int n = pushV.size();
        if(n == 0)
            return false;
        stack<int> sk;
        int j = 0;
        for(auto num : pushV){
            sk.push(num);
            while(j < popV.size() && sk.top() == popV[j]){
                sk.pop();
                ++j;
            }
        }
        return sk.empty();
    }
};
```


## 从上往下打印二叉树（层序遍历）

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            if(pRoot == nullptr)
                return {};
            queue<TreeNode*> q;
            q.push(pRoot);
            vector<vector<int>> res;
            while(!q.empty()){
                int n = q.size();
                vector<int> vec;
                for(int i = 0; i < n; ++i){
                    pRoot = q.front();
                    q.pop();
                    vec.push_back(pRoot->val);
                    if(pRoot->left != nullptr)
                        q.push(pRoot->left);
                    if(pRoot->right != nullptr)
                        q.push(pRoot->right);
                }
                res.push_back(vec);
            }
            return res;
        }
};
```

## 二叉搜索树的后续遍历

<https://www.nowcoder.com/practice/a861533d45854474ac791d90e447bafd?tpId=13&tqId=11176&tPage=2&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking>

```cpp
class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        if(sequence.size() <= 0)
            return false;
        return VerifySequenceOfBST(sequence, 0, sequence.size()-1);
    }
private:
    bool VerifySequenceOfBST(vector<int>& sequence, int start, int end){
        if(start >= end)
            return true;

        int root = sequence[end];
        int index = end; //注意这个初始化很重要
        for(int i=start; i < end; ++i){
            if(sequence[i] > root){
                index = i;
                break;
            }
        }
        //注意比较的数字不包含sequence[end]
        for(int i=index; i < end; ++i){
            if(sequence[i] < root)
                return false;
        }

        return VerifySequenceOfBST(sequence, start, index-1) &&
             VerifySequenceOfBST(sequence, index, end-1);
    }
};
```

## 二叉树中和为某一值的路径

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        vector<vector<int>> res;
        vector<int> vec;
        helper(root, res, vec, expectNumber);
        return res;
    }
private:
    void helper(TreeNode* root, vector<vector<int>>& res, vector<int>& vec, 
                int remain){
        if(root == nullptr)
            return;
        if(root->left == nullptr && root->right == nullptr && root->val == remain){
            vec.push_back(root->val);
            res.push_back(vec);
            vec.pop_back();
            return;
        }
        vec.push_back(root->val);
        helper(root->left, res, vec, remain - root->val);
        helper(root->right, res, vec, remain - root->val);
        vec.pop_back();
        return;
    }
};
```

## 二叉搜索树与双向链表

<https://www.cnblogs.com/grandyang/p/9615871.html>

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};*/
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(pRootOfTree == nullptr)
            return nullptr;
        TreeNode* pre = nullptr;
        TreeNode* head = nullptr;
        helper(pRootOfTree, pre, head);
        return head;
    }
private:
    void helper(TreeNode* root, TreeNode*& pre, TreeNode*& head){
        if(root == nullptr)
            return;
        helper(root->left, pre, head);
        if(head == nullptr){
            head = root;
            pre = root;
        }
        else{
            pre->right = root;
            root->left = pre;
            pre = root;
        }
        helper(root->right, pre, head);
        return;
    }
};
```



## 字符串的排列

```cpp
class Solution {
public:
    vector<string> Permutation(string str) {
        int n = str.size();
        vector<string> res;
        if(n == 0) return res;
        sort(str.begin(), str.end());
        string st;
        vector<bool> seen(n);
        helper(str, res, st, seen);
        return res;
    }
private:
    void helper(string& s, vector<string>& res, string& st, vector<bool>& seen){
        if(st.size() == s.size()){
            res.push_back(st);
            return;
        }
        
        for(int i = 0; i < s.size();){
            if(seen[i] == true){
                ++i; continue;
            }
            st.push_back(s[i]);
            seen[i] = true;
            helper(s, res, st, seen);
            seen[i] = false;
            st.pop_back();
            while(++i < s.size() && s[i] == s[i-1])
                continue;
        }
        return;
    }
};
```



## 数组中出现次数超过一半的数字

```cpp
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        unordered_map<int, int> m1;
        for(auto item : numbers)
            m1[item]++;
        int len = numbers.size() / 2;
        for(auto iter = m1.begin(); iter != m1.end(); ++iter)
            if(iter->second > len)
                return iter->first;
        return 0;
    }
};
```

> 链接：https://www.nowcoder.com/questionTerminal/e8a1b01a2df14cb2b228b30ee6a92163
>
> 采用阵地攻守的思想：
> 第一个数字作为第一个士兵，守阵地；count = 1；
> 遇到相同元素，count++;
> 遇到不相同元素，即为敌人，同归于尽,count--；当遇到count为0的情况，又以新的i值作为守阵地的士兵，继续下去，到最后还留在阵地上的士兵，有可能是主元素。
> 再加一次循环，记录这个士兵的个数看是否大于数组一般即可。  

```cpp
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int n = numbers.size();
        if(n == 0) return 0;
        int num = numbers[0];
        int cnt = 1;
        for(int i = 1; i < n; ++i){
            if(cnt == 0){
                num = numbers[i];
                cnt = 1;
            }
            else{
                if(numbers[i] == num)
                    ++cnt;
                else
                    --cnt;
            }
        }
        
        cnt = 0;
        for(auto item : numbers){
            if(item == num)
                ++cnt;
        }
        
        return cnt > n/2 ? num : 0;
    }
};
```

## 最小的K个数

```cpp
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        int n = input.size();
        if(k <= 0 || k > n) return {};
        if(k == n) return input;
        int l = 0, r = n-1;
        int index = partition(input, l, r);
        while(index != k){
            if(index > k){
                r = index - 1;
            }
            else{
                l = index + 1;
            }
	    index = partition(input, l, r);
        }
        vector<int> res(input.begin(), input.begin()+k);
        return res;
    }
private:
    int partition(vector<int>& nums, int l, int r){
        int pivot = nums[l];
        while(l < r){
            while(l < r && nums[r] > pivot)
                --r;
            if(l < r) nums[l++] = nums[r];
            while(l < r && nums[l] <= pivot)
                ++l;
            if(l < r) nums[r--] = nums[l];
        }
        nums[l] = pivot;
        return l;
    }
};
```

[LeetCode](https://leetcode.com/problems/k-closest-points-to-origin/discuss/220235/Java-Three-solutions-to-this-classical-K-th-problem.)


## 连续子数组的最大和

```cpp
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        int res = INT_MIN;
        vector<int> dp(array.size());
        dp[0] = array[0];
        for(int i = 1; i < array.size(); ++i){
            dp[i] = array[i] + max(dp[i-1], 0);
            res = max(res, dp[i]);
        }
        return res;
    }
};
```

## 从1到n整数中1出现的次数

```cpp
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n)
    {
        int head = 0, cur = 0, tail = 0;
        int res = 0;
        int i = 1;
        while(n/i){
            cur = (n/i) % 10;
            head = n / (i*10);
            tail = n - (n/i)*i;
            if(cur == 0){
                res += head * i;
            }
            else if(cur == 1){
                res += head * i + tail + 1;
            }
            else{
                res += head * i + i;
            }
            i *= 10;
        }
        return res;
    }
};
```

## 数组排成最小的数

```cpp
class Solution {
public:
    string PrintMinNumber(vector<int> numbers) {
        sort(numbers.begin(), numbers.end(), cmp);
        string res;
        for(auto num : numbers)
            res += to_string(num);
        return res;
    }
private:
    static bool cmp(int a, int b){
        string s1 = to_string(a);
        string s2 = to_string(b);
        return s1 + s2 < s2 + s1;
    }
};
```

## 第n个丑数

```cpp
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if(index < 7) return index;
        vector<int> arr(index);
        int t2 = 0, t3 = 0, t5 = 0;
        int n = 1;
        arr[0] = 1;
        while(n < index){
            arr[n] = min(min(arr[t2]*2, arr[t3]*3), arr[t5]*5);
            //这三个if有可能进入一个或者多个，进入多个是三个队列头最小的数有多个的情况
            if(arr[n] == arr[t2]*2)
                ++t2;
            if(arr[n] == arr[t3]*3)
                ++t3;
            if(arr[n] == arr[t5]*5)
                ++t5;
            ++n;
        }
        return arr[index-1];
    }
};
```

## 第一个只出现一次的字符

```cpp
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        unordered_map<char, int> mp;
        for(auto c : str)
            ++mp[c];
        for(int i = 0; i < str.size(); ++i){
            if(mp[str[i]] == 1)
                return i;
        }
        return -1;
    }
};
```

## 数组中的逆序对

<https://www.nowcoder.com/practice/96bd6684e04a44eb80e6a68efc0ec6c5?tpId=13&tqId=11188&tPage=2&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking>

```cpp
class Solution {
public:
    int InversePairs(vector<int> data) {
        int n = data.size();
        if(n == 0) return 0;
        long long res = helper(data, 0, n-1);
        return res % 1000000007;
    }
private:
    long long helper(vector<int>& nums, int l, int r){
        if(l >= r) return 0;
        int mid = l + (r - l) / 2;
        long long res = helper(nums, l, mid) + helper(nums, mid+1, r);
        vector<int> merged(r-l+1);
        int k = 0, i = l, j = mid + 1;
        while(i <= mid && j <= r){
            if(nums[i] > nums[j]){
                // a[i]>a[j]，从a[i]开始到a[mid]必定都是大于这个a[j]的，因为此时分治的两边已经是各自有序了
                res += mid - i + 1;
                merged[k++] = nums[j++];
            }
            else{
                merged[k++] = nums[i++];
            }
        }
        while(i <= mid){
            merged[k++] = nums[i++];
        }
        
        while(j <= r){
            merged[k++] = nums[j++];
        }
        
        copy(merged.begin(), merged.end(), nums.begin() + l);
        return res;
    }
};
```



```cpp
class Solution {
public:
    int InversePairs(vector<int> data) {
        return InversePairs(data, 0, data.size()-1) % 1000000007;
    }
private:
    long long InversePairs(vector<int>& data, int l, int r){
        if(l >= r) return 0;
        int m = l + (r - l)/2;
        long long res = InversePairs(data, l, m) + InversePairs(data, m+1, r);

        vector<int> merge(r - l + 1);
        int i = l, j = m + 1, k = 0, p = m + 1;

        while(i <= m){
            while(p <= r && data[i] > data[p]){
                ++p;
            }
            res += p - (m+1);

            while(j <= r && data[i] > data[j]){
                merge[k++] = data[j++];
            }

            merge[k++] = data[i++];
        }

        while(j <= r){
            merge[k++] = data[j++];
        }

        copy(merge.begin(), merge.end(), data.begin() + l);
        return res;
    }
};
```

## 两个链表的第一个公共结点

```cpp
/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        int len1 = get_len(pHead1);
        int len2 = get_len(pHead2);
        ListNode* fast=nullptr, *slow=nullptr;
        if(len1 > len2){
            fast = pHead1;
            slow = pHead2;
        }
        else{
            fast = pHead2;
            slow = pHead1;
        }
        
        for(int i = 0; i < abs(len2-len1); ++i){
            fast = fast->next;
        }
        
        while(fast != slow){
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
private:
    int get_len(ListNode* head){
        int res = 0;
        while(head != nullptr){
            ++res;
            head = head->next;
        }
        return res;
    }
};
```

## 二叉树的深度

```cpp
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    int TreeDepth(TreeNode* pRoot)
    {
        if(pRoot == nullptr)
            return 0;
        int left = TreeDepth(pRoot->left);
        int right = TreeDepth(pRoot->right);
        return max(left, right) + 1;
    }
};
```



## 判断是否是平衡二叉树

```cpp
class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        return getDepth(pRoot) != -1;
    }
private:
    int getDepth(TreeNode* root){
        if(root == nullptr)
            return 0;
        int left = getDepth(root->left);
        if(left == -1) return -1;
        int right = getDepth(root->right);
        if(right == -1) return -1;
        if(abs(right - left) > 1)
            return -1;
        else
            return 1 + max(left, right);
    }
};
```

## 数组中只出现一次的数字

<https://www.nowcoder.com/practice/e02fdb54d7524710a7d664d082bb7811?tpId=13&tqId=11193&tPage=2&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking>

```cpp
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        int res = 0;
        for(auto item : data)
            res ^= item;
        if(res == 0) return;

        //从后往前找到xor第一个非0的位
        int index = 0;
        while((res & 1) == 0){
            res = res >> 1;
            ++index;
        }

        //根据第n位是不是1将数据分为两部分
        *num1 = 0, *num2 = 0;
        for(auto item : data){
            if(nPosIsNonZero(item, index))
                *num1 ^= item;
            else
                *num2 ^= item;
        }
    }
private:
    //判断输入的num的第n位是不是二进制1
    bool nPosIsNonZero(int num, int n){
        num = num >> n;
        return (num & 1);
    }
};
```

```cpp
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        int res = 0;
        for(auto num : data){
            res ^= num;
        }
        int i = 1;
        while((res & i) == 0){
            i <<= 1;
        }
        
        *num1 = 0, *num2 = 0;
        for(auto num : data){
            if((num & i) == 0)
                *num1 ^= num;
            else
                *num2 ^= num;
        }
        return;
    }
};
```

## 和为S的连续正数序列

```cpp
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int>> res;
        
        int l = 1, r = 2;
        while(l < r){
            int tmp = (l + r) * (r-l+1) / 2;
            if(tmp < sum)
                ++r;
            else if(tmp > sum)
                ++l;
            else{
                vector<int> vec;
                for(int i = l; i <= r; ++i)
                    vec.push_back(i);
                res.push_back(vec);
                ++l;
            }
        }
        return res;
    }
};
```

## 和为S的两个数字

```cpp
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        int n = array.size();
        int i = 0, j = n-1;
        while(i < j){
            int tmp = array[i] + array[j];
            if(tmp > sum)
                --j;
            else if(tmp < sum)
                ++i;
            else
                return {array[i], array[j]};
        }
        return {};
    }
};
```



## 左旋转字符串

```cpp
class Solution {
public:
    string LeftRotateString(string str, int n) {
        int len = str.size();
        if(len < n)
            return "";
        n = n % len;
        reverse(str.begin(), str.begin() + n);
        reverse(str.begin() + n, str.end());
        reverse(str.begin(), str.end());
        return str;
    }
};
```



## 扑克牌顺子

<https://www.nowcoder.com/questionTerminal/762836f4d43d43ca9deb273b3de8e1f4>

```cpp
class Solution {
public:
    // 满足条件 1. max - min <5 （min ,max 都不记0）；2. 除0外没有重复的数字(牌)；3. 数组长度为5
    bool IsContinuous( vector<int> numbers ) {
        if(numbers.size() < 5) return false;
        vector<int> elements(14);
        int max = -1, min = 14;
        for(auto num : numbers){
            if(num == 0) continue;
            ++elements[num];
            if(elements[num] > 1) return false;
            if(num > max) max = num;
            if(num < min) min = num;
        }

        if(max - min > 4) return false;
        else return true;
    }
};
```

## 孩子们的游戏(圆圈中最后剩下的数)

```cpp
class Solution {
public:
    int LastRemaining_Solution(int n, int m)
    {
        if(m <= 0 || n <= 0)
            return -1;
        int res = 0;  // 只有一个人的时候，留下的编号为0
        for(int i = 2; i <=n; ++i){
            res = (res + m) % i;
        }
        return res;
    }
};

```

## 字符串转为数字

```cpp
class Solution {
public:
    int StrToInt(string str) {
        int n = str.size();
        int i = 0;
        while(i < n && str[i] == ' ')
            ++i;
        int flag = 1;
        if(str[i] == '+' || str[i] == '-')
            flag = (str[i++] == '+') ? 1 : -1;
        int base = 0;
        while(i < n && str[i] >= '0' && str[i] <= '9'){
            if(base > INT_MAX/10 || (base == INT_MAX/10) && (flag == 1) && (str[i]-'0' > INT_MAX%10)
              || (base == INT_MAX/10)&&(flag == -1) &&(str[i]-'0' > abs(INT_MIN %10))){
                return 0;
            }
            base = base * 10 + str[i] - '0';
            ++i;
        }
        return i == n ? base*flag : 0;
    }
};
```

## 构建乘积数组

```cpp
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int n = A.size();
        if(n == 0) return {};
        vector<int> res(n);
        res[0] = 1;
        for(int i = 1; i < n; ++i)
            res[i] = res[i-1]*A[i-1];
        int right = 1;
        for(int i = n-1; i >=0; --i){
            res[i] = res[i]*right;
            right *= A[i];
        }
        return res;
    }
};
```

## 字符流中第一个不重复的字符

```cpp
class Solution
{
public:
  //Insert one char from stringstream
    void Insert(char ch)
    {
        vec.push_back(ch);
        ++mp[ch];
        return;
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        for(auto c : vec)
            if(mp[c] == 1)
                return c;
        return '#';
    }
private:
    vector<char> vec;
    unordered_map<char, int> mp;
};
```

## 链表环的入口

```cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        if(pHead == nullptr || pHead->next == nullptr)
            return nullptr;
        ListNode* slow = pHead;
        ListNode* fast = pHead;

        do{
            if(fast == nullptr || fast->next == nullptr)
                return nullptr;
            slow = slow->next;
            fast = fast->next->next;
        }while(slow != fast);

        slow = pHead;
        while(slow != fast){
            slow = slow->next;
            fast = fast->next;
        }
        return slow;
    }
};
```

```cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        ListNode* fast = pHead, *slow = pHead;
        while(true){
            if(fast == nullptr || fast ->next == nullptr)
                return nullptr;
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow)
                break;
        }
        slow = pHead;
        while(fast != slow){
            fast = fast->next;
            slow = slow->next;
        }
        return slow;
    }
};
```

## 删除链表重复结点

```cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead)
    {
        ListNode* dummy = new ListNode(0);
        ListNode* pre = dummy, *cur = pHead;
        pre->next = cur;
        while(cur != nullptr){
            while(cur->next != nullptr && cur->val == cur->next->val){
                cur = cur->next;
            }
            if(pre->next == cur)
                pre = cur;
            else
                pre->next = cur->next;
            cur = cur->next;
        }
        return dummy->next;
    }
};
```

## 对称的二叉树

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    bool isSymmetrical(TreeNode* pRoot)
    {
        if(pRoot == nullptr)
            return true;
        return helper(pRoot->left, pRoot->right);
    }
private:
    bool helper(TreeNode* left, TreeNode* right){
        if(left == nullptr && right == nullptr)
            return true;
        if(left == nullptr || right == nullptr)
            return false;
        if(left->val != right->val)
            return false;
        return helper(left->left, right->right) && helper(left->right, right->left);
    }

};
```

## 按之字形顺序打印二叉树

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        vector<vector<int>> res;
        if(pRoot == nullptr)
            return res;
        queue<TreeNode*> que1;
        queue<TreeNode*> que2;
        que1.push(pRoot);
        while(que1.empty() == false || que2.empty() == false){
            vector<int> vec;
            if(que1.empty() == false){
                int n = que1.size();
                for(int i = 0; i < n; ++i){
                    TreeNode* root = que1.front();
                    que1.pop();
                    vec.push_back(root->val);
                    if(root->left != nullptr)
                        que2.push(root->left);
                    if(root->right != nullptr)
                        que2.push(root->right);
                }
            }
            else{
                int n = que2.size();
                for(int i = 0; i < n; ++i){
                    TreeNode* root = que2.front();
                    que2.pop();
                    vec.push_back(root->val);
                    if(root->left != nullptr)
                        que1.push(root->left);
                    if(root->right != nullptr)
                        que1.push(root->right);
                }
                reverse(vec.begin(), vec.end());
            }
            res.push_back(vec);
        }
        return res;
    }
};

```

## 正则表达式匹配

<https://www.nowcoder.com/questionTerminal/45327ae22b7b413ea21df13ee7d6429c>

```cpp
class Solution {
public:
    bool match(char* str, char* pattern)
    {
        if(*str == '\0' && *pattern == '\0')
            return true;
        if(*str != '\0' && *pattern == '\0')
            return false;

        if(*(pattern + 1) != '*'){
            if(*str == *pattern || (*pattern == '.' && *str != '\0'))
                return match(str+1, pattern+1);
            else
                return false;
        }
        else{
            if(*str == *pattern || (*pattern == '.' && *str != '\0'))
                return match(str+1, pattern) || match(str, pattern+2);
            else
                return match(str, pattern+2);
        }
    }
};
```

## 二叉搜索树的第k个结点

中序遍历。

```cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        stack<TreeNode*> sk;
        while(!sk.empty() || pRoot != nullptr){
            if(pRoot != nullptr){
                sk.push(pRoot);
                pRoot = pRoot->left;
            }
            else{
                pRoot = sk.top();
                sk.pop();
                if(--k == 0){
                    return pRoot;
                }
                pRoot = pRoot->right;
            }
        }
        return nullptr;
    }
};
```

## 数据流中的中位数

```cpp
class Solution {
private:
    priority_queue<int> p;  // 大顶堆
    priority_queue<int, vector<int>, greater<int>> q;  // 小顶堆
public:
    void Insert(int num)
    {
        // 插入的元素应小于大顶堆堆顶值
        if(p.empty() || num < p.top())
            p.push(num);
        else
            q.push(num);
        
        // 如果两个堆中的数量差大于等于2
        if(p.size() == q.size() + 2){
            q.push(p.top());
            p.pop();
        }
        // 保证P中元素的数量大于等于q中
        if(p.size() + 1 == q.size()){
            p.push(q.top());
            q.pop();
        }
    }

    double GetMedian()
    {
        return p.size()==q.size() ? (p.top()+q.top())/2.0 : p.top();
    }

};
```

## 滑动窗口的最大值

```cpp
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        vector<int> res;
        int n = num.size();
        if(n < size)
            return res;
        deque<int> q;
        for(int i = 0; i < n; ++i){
            while(!q.empty() && (num[q.back()] < num[i])){
                q.pop_back();
            }
            while(!q.empty() && i - q.front() + 1 > size){
                q.pop_front();
            }
            q.push_back(i);
            if(!q.empty() && i >= size - 1){
                res.push_back(num[q.front()]);
            }
        }
        return res;
    }
};
```

## 矩阵中的路径

```cpp
class Solution {
public:
    bool hasPath(char* matrix, int rows, int cols, char* str)
    {
        this->matrix = matrix;
        this->cols = cols;
        this->rows = rows;
        vector<vector<bool>> seen(rows, vector<bool>(cols));
        for(int i = 0; i < rows; ++i){
            for(int j = 0; j < cols; ++j){
                if(helper(str, seen, i, j))
                    return true;
            }
        }
        return false;
    }
private:
    bool helper(char* str, vector<vector<bool>>& seen, int i, int j){
        if(*str == '\0')
            return true;
        if(i >= rows || i < 0 || j >= cols || j < 0 || seen[i][j] || 
           *str != matrix[i*cols+j])
            return false;
        seen[i][j] = true;
        bool res = helper(str+1, seen, i+1, j) || helper(str+1, seen, i, j+1) ||
            helper(str+1, seen, i-1, j) || helper(str+1, seen, i, j-1);
        seen[i][j] = false;
        return res;
    }
    char* matrix;
    int cols;
    int rows;
};
```

## 机器人的运动范围

```cpp
class Solution {
public:
    int movingCount(int threshold, int rows, int cols)
    {
        if(rows <= 0 || cols <= 0 || threshold < 0)
            return 0;
        vector<vector<bool>> seen(rows, vector<bool>(cols, false));
        int res = 0;
        helper(rows, cols, threshold, seen, 0, 0, res);
        return res;
    }
private:
    void helper(int rows, int cols, int k, vector<vector<bool>>& seen,
                int i, int j, int& res){
        if(i < 0 || j < 0 || i >= rows || j >= cols || seen[i][j])
            return;
        int t = 0, x = i, y = j;
        while(x != 0 || y != 0){
            t += x % 10 + y % 10;
            x /= 10;
            y /= 10;
        }
        if(t > k) return;
        seen[i][j] = true;
        res += 1;
        helper(rows, cols, k, seen, i+1, j, res);
        helper(rows, cols, k, seen, i-1, j, res);
        helper(rows, cols, k, seen, i, j+1, res);
        helper(rows, cols, k, seen, i, j-1, res);
        return;
    }
};
```

