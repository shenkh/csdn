# [How to convert std::string to LPCSTR?](https://stackoverflow.com/questions/1200188/how-to-convert-stdstring-to-lpcstr)

`LPCSTR` 、`LPSTR`、 `LPWSTR` and `LPCWSTR`为指向不同类型字符串的指针（VC下）。


>
> Call `c_str()` to get a `const char *` (`LPCSTR`) from a `std::string`.
>
> It's all in the name:
>
> `LPSTR` - (long) pointer to string - `char *`
> `LPCSTR` - (long) pointer to constant string - `const char *`
> `LPWSTR` - (long) pointer to Unicode (wide) string - `wchar_t *`
> `LPCWSTR` - (long) pointer to constant Unicode (wide) string - `const wchar_t *`
> `LPTSTR` - (long) pointer to TCHAR (Unicode if UNICODE is defined, ANSI if not) string - `TCHAR *`
> `LPCTSTR` - (long) pointer to constant TCHAR string - `const TCHAR *`
>
> You can ignore the L (long) part of the names -- it's a holdover from 16-bit Windows.



```cpp
std::string s = SOME_STRING;
// get temporary LPSTR (not really safe)
LPSTR pst = &s[0];
// get temporary LPCSTR (pretty safe)
LPCSTR pcstr = s.c_str();
// convert to std::wstring
std::wstring ws; 
ws.assign( s.begin(), s.end() );
// get temporary LPWSTR (not really safe)
LPWSTR pwst = &ws[0];
// get temporary LPCWSTR (pretty safe)
LPCWSTR pcwstr = ws.c_str();
```

因为LPWSTR是指针，所以作为函数返回值时不能像上述代码简单使用，应注意在堆上进行创建。

```cpp
LPWSTR ConvertToLPWSTR( const std::string& s )
{
    LPWSTR ws = new wchar_t[s.size()+1]; // +1 for zero at the end
    copy( s.begin(), s.end(), ws );
    ws[s.size()] = 0; // zero at the end
    return ws;
}

void f()
{
    std::string s = SOME_STRING;
    LPWSTR ws = ConvertToLPWSTR( s );

    // some actions

    delete[] ws; // caller responsible for deletion
}
```
