
# string与整型互换

```cpp
int aa = 30;
stringstream ss;
ss << aa;
string s1 = ss.str();
cout << s1 << endl; // 30

string s2;
ss >> s2;
cout << s2 << endl; // 30
```

```cpp
string s = "17";

stringstream ss;
ss<<s;

int i;
ss >> i;
cout << i << endl; // 17
```

<http://www.cnblogs.com/nzbbody/p/3504199.html>

-----

```cpp
    int a = 30;
    stringstream ss;
    ss << a;

    string s1;
    ss >> s1;
    cout << s1 << endl;

    string s = ss.str();
    cout << s << endl;

    ss.clear(), ss.str(""); //清空缓冲区
    ss << s1;
    int i;
    ss >> i;
    cout << i << endl;
```