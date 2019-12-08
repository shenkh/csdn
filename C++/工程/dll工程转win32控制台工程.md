1.将配置类型改为【应用程序】。

![在这里插入图片描述](../../g3doc/dll2exe1.JPG)

2.将【窗口】改为【控制台】

![在这里插入图片描述](../../g3doc/dll2exe2.JPG)

3.修改预处理器定义

![在这里插入图片描述](../../g3doc/dll2exe3.JPG)

改为第二步对应的_CONSOLE

![在这里插入图片描述](../../g3doc/dll2exe4.JPG)

4.删除对应的接口DLL函数，添加main函数。

![在这里插入图片描述](../../g3doc/dll2exe5.JPG)

5.可移除dllmain.cpp

```cpp
// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "stdafx.h"

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

```
