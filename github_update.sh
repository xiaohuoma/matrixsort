# 配置用户身份（如果尚未配置）
if [ -z "$(git config --global user.email)" ] || [ -z "$(git config --global user.name)" ]; then
    git config --global user.email "git@github.com"
    git config --global user.name "Elegancejiang"
fi
# 查看哪些文件被修改、新增或删除
git status
# 添加所有文件到暂存区
git add .
# 描述更改内容
git commit -m "25.04.05 Uncoarsening IMB | timer | reset_conn_DS_warp error"
# 移除旧的远程仓库
# git remote remove origin
# git remote add origin git@github.com:Elegancejiang/HunyuanGraph_SC25_0.0.0.git
# 上传
git push origin main

# 下载
# git pull origin main