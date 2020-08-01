**背景**：在vscode中使用remote-ssh插件时，需要在远程服务器安装vscode-server。在无法正常访问网络时，可按以下步骤进行离线安装。

<https://stackoverflow.com/questions/56671520/how-can-i-install-vscode-server-in-linux-offline>

1. First get commit id
2. Download vscode server from url: `https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable`
3. Upload the `vscode-server-linux-x64.tar.gz` to server
4. Unzip the downloaded `vscode-server-linux-x64.tar.gz` to `~/.vscode-server/bin/${commit_id}` without vscode-server-linux-x64 dir
5. Create `0` file under `~/.vscode-server/bin/${commit_id}`


其中`commit_id`与使用的vscode版本对应，可通过查看[tags](https://github.com/microsoft/vscode/tags)进行确定。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200801182113315.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dybGxlcnk=,size_16,color_FFFFFF,t_70)
如vscode-stable-1.47.3对应的`commit_id`为`91899dcef7b8110878ea59626991a18c8a6a1b3e`


```sh
commit_id=f06011ac164ae4dc8e753a3fe7f9549844d15e35

# Download url is: https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable
curl -sSL "https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable" -o vscode-server-linux-x64.tar.gz

mkdir -p ~/.vscode-server/bin/${commit_id}
# assume that you upload vscode-server-linux-x64.tar.gz to /tmp dir
tar zxvf /tmp/vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/${commit_id} --strip 1
touch ~/.vscode-server/bin/${commit_id}/0
```
