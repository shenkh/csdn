# [Transferring a repository](https://help.github.com/en/articles/transferring-a-repositor)

将GitHub仓库转移至某个新账号下。

1. On GitHub, navigate to the main page of the repository.

2. Under your repository name, click  **Settings**.

   ![Repository settings button](https://help.github.com/assets/images/help/repository/repo-actions-settings.png)

   

3. Under "Danger Zone", click **Transfer**.

   ![Transfer button](https://help.github.com/assets/images/help/repository/repo-transfer.png)

   

4. Read the information about transferring a repository, then type the name of the user or organization you'd like to transfer ownership of the repository to.

   ![Information about repository transfer and field to type the new owner's username](https://help.github.com/assets/images/help/repository/transfer-repo-new-owner-name.png)

   

5. Read the warnings about potential loss of features depending on the new owner's subscription.

   ![Warnings about transferring a repository to a person using a free product](https://help.github.com/assets/images/help/repository/repo-transfer-free-plan-warnings.png)

   

6. Type the name of the repository you'd like to transfer, then click **I understand, transfer this repository**.

   ![Transfer button](https://help.github.com/assets/images/help/repository/repo-transfer-complete.png)



更新本地仓库对应的远程链接：

All links to the previous repository location are automatically redirected to the new location. When you use git clone, git fetch, or git push on a transferred repository, these commands will redirect to the new repository location or URL. However, to avoid confusion, we strongly recommend updating any existing local clones to point to the new repository URL. You can do this by using git remote on the command line:

```
$ git remote set-url origin new_url
```

注意：如果当前用户有两个账号，需先登录要转移的目标账户，然后打开对应的邮件链接，同意接收。

[Github responds with “Repository cant be finished”](https://stackoverflow.com/questions/50263663/github-responds-with-repository-cant-be-finished)

>  Seems like for me, the problem was that I needed to log into the receiving account before accepting from the email.