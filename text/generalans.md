# 分支与合并
分支就是创建当前版本分出的一个独立仓库副本，这个副本不参加主版本迭代，有自己的迭代系统  
合并就是把这个副本和主版本的仓库对比，把副本有的主版本没有的加入主版本  
注意这个操作不会删除分支

# 说明命令
cd 改变路径  
mkdir 创建新路径  
ls 列出该路径下的文件  
git init 初始化为git仓库  
git add <filename> 把xx加入仓库  
git commit -m "这是一些注释" 提交更改  
git push origin master(或其他分支) 把这个仓库推送到远程的服务器的仓库上  
git checkout "xx" 移动到xx分支下     
git branch "xx" 创建xx分支  
git merge "xx" 在除了xx分支时可以使用，把xx合并到该分支下  
git branch -d "xx" 删除xx分支  

# 选择MIT协议的原因
1.最简单、最宽松，对所有权的争议小。  
2.是绝大多数项目的选择。

# 错误信息查证
