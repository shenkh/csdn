
# [How to execute a command and get return code stdout and stderr of command in C++](https://stackoverflow.com/questions/52164723/how-to-execute-a-command-and-get-return-code-stdout-and-stderr-of-command-in-c)

**目的**：调用外部exe时，捕获其返回值及屏幕输出。

**环境**：Windows 10

```cpp
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <utility>
using namespace std;

pair<string, int> exec(const char* cmd) {
    array<char, 128> buffer;
    string result;
    int return_code = -1;
    auto pclose_wrapper = [&return_code](FILE* cmd) { return_code = _pclose(cmd); };
    { // scope is important, have to make sure the ptr goes out of scope first
        const unique_ptr<FILE, decltype(pclose_wrapper)> pipe(_popen(cmd, "r"), pclose_wrapper);
        if (pipe) {
            while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
                result += buffer.data();
            }
        }
    }
    return make_pair(result, return_code);
}
```

若要捕获stderr，在cmd参数中添加`2>&1`。如：

```cpp
auto info = exec("test.exe 2>&1");
```

捕获到的stderr：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208110943800.png)

# [How do I execute a command and get the output of the command within C++ using POSIX?](https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po)

仅获取exe屏幕输出 C++ 11

```cpp
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}
```

C++ 11以前

```cpp
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>

std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = _popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    }
    catch (...) {
        _pclose(pipe);
        throw;
    }
    _pclose(pipe);
    return result;
}
```
