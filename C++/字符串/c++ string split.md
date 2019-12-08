# C++ Split string

```cpp
#include<iostream>
#include<sstream>
#include<vector>
#include<string>
using std::vector;
using std::string;

vector<string> split(const string& s, vector<string>& res, char delim) {
    std::stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) {
        res.push_back(item);
    }
    return res;
}

int main()
{
    string s("this is a test");
    vector<string> res;
    split(s, res, 'i');
    for (auto item : res) {
        std::cout << item << std::endl;
    }
    std::cout << "Hello World!\n";
}
```