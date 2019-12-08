
## memcpy

```cpp
void* my_memcpy(void* dst, const void* src, size_t count) {
    if (dst == nullptr || src == nullptr)
        return nullptr;

    char* pdst = static_cast<char*>(dst);
    const char* psrc = static_cast<const char*>(src);

    // 地址如果存在重叠，从后往前复制
    if (psrc < pdst && psrc + count > pdst) {
        for (int i = count - 1; i >= 0; --i)
            * (pdst + i) = *(psrc + i);
    }
    // 否则从前往后进行内存复制
    else {
        for (int i = 0; i < count; ++i)
            * pdst++ = *psrc++;
    }
    return dst;
}

int main() {
    const char* src = "this";
    char dst[6];
    my_memcpy(dst, src, 5);
    for (auto i : dst)
        cout << i << endl;
    return 0;
}
```

-----

## strcpy

```cpp
//https://blog.csdn.net/okawari_richi/article/details/57411796
char* strcpy_(char* dst, const char* src) {
    if (dst == nullptr || src == nullptr)
        return nullptr;
    char* res = dst;
    while ((*dst++ = *src++) != '\0')
        continue;
    return res;
}

char* strcpy_s_(char* dst, size_t dst_size, const char* src) {
    if (dst == nullptr || src == nullptr)
        return nullptr;

    char* res = dst;
    int i = 0;
    while ( i < dst_size && (*dst++ = *src++) != '\0') {
        ++i;
    }
    if(i == dst_size) res[dst_size - 1] = '\0';
    return res;
}
```