## string to char*

<https://stackoverflow.com/a/7352131/9288778>

```cpp
std::string str = "string";
const char *cstr = str.c_str();
```

Note that it returns a `const char *`; you aren't allowed to change the C-style string returned by `c_str()`. If you want to process it you'll have to copy it first:

```cpp
std::string str = "string";
char *cstr = new char[str.length() + 1];
strcpy(cstr, str.c_str());
// do stuff
delete [] cstr;
```

Or in modern C++:

```cpp
std::vector<char> cstr(str.c_str(), str.c_str() + str.size() + 1);
// use &chars[0] as a char*
```

-----

<https://stackoverflow.com/a/42308974/9288778>

```cpp
std::string str = "string";
char* chr = const_cast<char*>(str.c_str())
```

## char* to string

<https://stackoverflow.com/questions/8438686/convert-char-to-string-c>

    std::string str(buffer, buffer + length);

Or, if the string already exists:

    str.assign(buffer, buffer + length);

Edit: I'm still not completely sure I understand the question. But if it's something like what JoshG is suggesting, that you want up to length characters, or until a null terminator, whichever comes first, then you can use this:

    std::string str(buffer, std::find(buffer, buffer + length, '\0'));