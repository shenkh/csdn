
# KMP

## next数组

传统的暴力匹配对于目标串`s`和模式串`p`，如果`s[i] != p[j]`，则回溯`i`并将`j`置为0；而KMP算法则是在失配时不回溯`i`，通过`next[j]`来调整模式串`j`的位置。

`next[i]`表示对于模式串`p[0...i-1]`相同的前缀和后缀的最大长度。注意这里的前缀不包括`p[0...i-1]`，只有`p[0]`，`p[0..1]`...`p[0..i-2]`。。

失配时，模式串向右移动的位数为：失配字符所在位置 - 失配字符对应的`next`值。

如果已知`next[j] = k`，说明`p[0..k-1] == p[j-k...j-1]`，那么对于`next[j]`的求解分为两种情况：

- 如果`p[k] == p[j]`，那么`next[j+1] = next[j] + 1`.
- 如果`p[k] != p[j]`，那么假设`next[p[k]] = t`，则`s[0..t-1] == s[k-t...k-1]`。参照下图，即第一个蓝色区域等于第二个蓝色区域，同时由于`p[0..k-1] == p[j-k...j-1]`，则第一个蓝色区域等于第三个蓝色区域，第二个蓝色区域等于第四个蓝色区域，由此可推出第一个蓝色区域等于第四个蓝色区域，如果`p[j] == p[t]`，则`next[j+1] = next[p[k]] + 1`；依次递归迭代。

![next数组求解](https://img-blog.csdn.net/20150812214857858)

以上部分参见：[从头到尾彻底理解KMP](https://blog.csdn.net/v_july_v/article/details/7041827)

-----

## next数组求解

```cpp
vector<int> calculate_next(string pattern) {
    int n = pattern.size();
    if (n <= 0) return vector<int>{};
    vector<int> next(n);
    next[0] = -1;

    for (int i = 0; i < n-1; ++i) {
        int k = next[i];
        if (k != -1 && pattern[k] == pattern[i])
            next[i+1] = next[i] + 1;
        else {
            while (k != -1 && pattern[k] != pattern[i]) {
                k = next[k];
            }
            next[i+1] = k + 1;
        }
    }
    return next;
}
```

    pattern: AABA
    next: -1,0,1,0

-----

## KMP代码实现(转)

**注意**：以下`lps`数组和上述的`next`数组有一定的区别，`lps[i]`表示子串`pattern[0...i]`（包含了`pattern[i]`）对应的最大的相同的前后缀长度。`next`数组可有`lps`数组整体右移一位，同时`next[0]`置为-1获得。

*以下部分转自：[KMP Algorithm for Pattern Searching](https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/)*

Given a text `txt[0..n-1]` and a pattern `pat[0..m-1]`, write a function `search(char pat[], char txt[])` that prints all occurrences of `pat[]` in `txt[]`. You may assume that `n > m`.

    Input:  txt[] = "THIS IS A TEST TEXT"
            pat[] = "TEST"
    Output: Pattern found at index 10

    Input:  txt[] =  "AABAACAADAABAABA"
            pat[] =  "AABA"
    Output: Pattern found at index 0
            Pattern found at index 9
            Pattern found at index 12

![example](https://www.geeksforgeeks.org/wp-content/uploads/Pattern-Searching-2-1.png)

**Preprocessing Overview:**

- KMP algorithm preprocesses `pat[]` and constructs an auxiliary `lps[]` of size `m` (same as size of pattern) which is used to skip characters while matching.
- name `lps` indicates longest proper prefix which is also suffix.. A proper prefix is prefix with whole string not allowed. For example, prefixes of “ABC” are “”, “A”, “AB” and “ABC”. Proper prefixes are “”, “A” and “AB”. Suffixes of the string are “”, “C”, “BC” and “ABC”.
- We search for lps in sub-patterns. More clearly we focus on sub-strings of patterns that are either prefix and suffix.
For each sub-pattern `pat[0..i]` where `i` = 0 to `m-1`, `lps[i]` stores length of the maximum matching proper prefix which is also a suffix of the sub-pattern `pat[0..i]`.

```cpp
#include <iostream>
#include<bitset>
#include<vector>
#include<string>
using namespace std;

// C++ program for implementation of KMP pattern searching algorithm

void computeLPSArray(string pat, int M, vector<int>& lps);

// Prints occurrences of txt[] in pat[]
int KMPSearch(string pat, string txt)
{
    int M = pat.size();
    int N = txt.size();

    // create lps[] that will hold the longest prefix suffix
    // values for pattern
    vector<int> lps(M);

    // Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps);

    int i = 0; // index for txt[]
    int j = 0; // index for pat[]
    int res = 0;
    while (i < N) {
        if (pat[j] == txt[i]) {
            j++;
            i++;
        }

        if (j == M) {
            printf("Found pattern at index %d ", i - j);
            j = lps[j - 1];
            ++res;
        }

        // mismatch after j matches
        else if (i < N && pat[j] != txt[i]) {
            // Do not match lps[0..lps[j-1]] characters,
            // they will match anyway
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }
    }
    return res;
}

// Fills lps[] for given patttern pat[0..M-1]
void computeLPSArray(string pat, int M, vector<int>& lps)
{
    // length of the previous longest prefix suffix
    int len = 0;

    lps[0] = 0; // lps[0] is always 0

    // the loop calculates lps[i] for i = 1 to M-1
    int i = 1;
    while (i < M) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        }
        else // (pat[i] != pat[len])
        {
            // This is tricky. Consider the example.
            // AAACAAAA and i = 7. The idea is similar
            // to search step.
            if (len != 0) {
                len = lps[len - 1];

                // Also, note that we do not increment
                // i here
            }
            else // if (len == 0)
            {
                lps[i] = 0;
                i++;
            }
        }
    }
}

// Driver program to test above function
int main()
{
    string txt = "AABAACAADAABAABA";
    string pat = "AABA";
    int res = KMPSearch(pat, txt);
    cout << res << endl;
    return 0;
}
```

-----

## kmp代码实现

计算模式串`pattern`在目标串`s`出现的次数和位置。注意这里的`next`数组比`pattern`的长度大一，和上述的略微不同。

```cpp
vector<int> calculate_next(string pattern) {
    int n = pattern.size();
    if (n <= 0) return vector<int>{};
    vector<int> next(n+1);
    next[0] = -1;

    for (int i = 0; i < n; ++i) {
        int k = next[i];
        if (k != -1 && pattern[k] == pattern[i])
            next[i+1] = next[i] + 1;
        else {
            while (k != -1 && pattern[k] != pattern[i]) {
                k = next[k];
            }
            next[i+1] = k + 1;
        }
    }
    return next;
}

vector<int> calculate_lps(string pattern) {
    int n = pattern.size();
    if (n <= 0) return vector<int>{};
    vector<int> lps(n);
    lps[0] = 0;

    for (int i = 1; i < n; ++i) {
        int k = lps[i-1];
        if (pattern[k] == pattern[i])
            lps[i] = lps[i-1] + 1;
        else {
            while (k != 0 && pattern[k] != pattern[i]) {
                k = lps[k-1];
            }
            lps[i] = (k == 0 ? pattern[k] == pattern[i] : k + 1);
        }
    }
    return lps;
}

int kmp(string s, string pattern) {
    int n = s.size();
    int m = pattern.size();
    vector<int> next = calculate_next(pattern);
    int res = 0;
    int i =0, j = 0;
    while (i < n) {
        if (j == -1 || s[i] == pattern[j]) {
            ++i, ++j;
            if (j == m) {
                cout << "find at index " << i - m << endl;
                ++res;
                j = next[j];
            }
        }
        else{
            j = next[j];
        }
    }
    return res;
}

int main()
{
    string txt = "AABAACAADAABAABA";
    string pat = "AABA";
    cout << kmp(txt, pat) << endl;
    return 0;
}
```

    find at index 0
    find at index 9
    find at index 12
    3