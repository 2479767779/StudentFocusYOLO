# GitHub推送指南

由于网络或认证问题，这里提供几种方法将项目推送到GitHub。

## 方法1: 手动推送 (推荐)

### 步骤1: 在GitHub上创建仓库
1. 访问 https://github.com/new
2. 填写仓库信息:
   - **Repository name**: `StudentFocusYOLO`
   - **Description**: 基于YOLO的课堂学生专注度实时监控系统
   - **选择**: 公开仓库 (Public)
   - **不要勾选**: "Initialize this repository with a README"
3. 点击 "Create repository"

### 步骤2: 获取仓库URL
创建成功后，你会看到类似这样的页面:
```
Quick setup — if you’ve done this kind of thing before
HTTPS — https://github.com/你的用户名/StudentFocusYOLO.git
```

### 步骤3: 在命令行中执行
```bash
# 进入项目目录
cd StudentFocusYOLO

# 添加远程仓库 (替换下面的URL为你的仓库URL)
git remote add origin https://github.com/你的用户名/StudentFocusYOLO.git

# 推送代码
git push -u origin master
```

## 方法2: 使用GitHub Desktop

1. 下载并安装 GitHub Desktop: https://desktop.github.com/
2. 打开 GitHub Desktop
3. 添加本地仓库: File -> Add local repository -> 选择 StudentFocusYOLO 文件夹
4. 创建仓库: Repository -> Create repository on GitHub
5. 推送: Branch -> Push

## 方法3: 使用SSH密钥 (更安全)

### 生成SSH密钥
```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
# 一路回车即可
```

### 添加SSH密钥到GitHub
1. 复制公钥内容:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
2. 访问 https://github.com/settings/keys
3. 点击 "New SSH key"
4. 粘贴公钥内容

### 使用SSH推送
```bash
cd StudentFocusYOLO
git remote add origin git@github.com:你的用户名/StudentFocusYOLO.git
git push -u origin master
```

## 方法4: 使用Python脚本

如果上面的方法都不行，可以运行提供的Python脚本:

```bash
cd StudentFocusYOLO
python setup_github.py
```

按照提示输入GitHub Personal Access Token即可。

## 常见问题

### 问题1: Authentication failed
**解决方案**: 
- 使用Personal Access Token代替密码
- 访问 https://github.com/settings/tokens 生成Token
- 在密码输入时粘贴Token

### 问题2: Permission denied
**解决方案**:
- 检查仓库是否是你的 (或者你有写权限)
- 确保仓库是Public而不是Private

### 问题3: 推送失败
**解决方案**:
```bash
# 强制推送
git push -f origin master

# 或者先拉取
git pull origin master --allow-unrelated-histories
git push origin master
```

## 验证推送成功

访问你的仓库URL: `https://github.com/你的用户名/StudentFocusYOLO`

应该能看到所有项目文件，包括:
- README.md
- main.py
- src/ 目录
- app/ 目录
- 等等...

## 后续步骤

推送成功后，你可以:

1. **添加徽章**: 在README中添加构建状态徽章
2. **启用GitHub Pages**: 用于展示文档
3. **创建Release**: 发布稳定版本
4. **设置CI/CD**: 自动测试和部署
5. **添加License**: MIT License已包含

---

**注意**: 如果遇到任何问题，请检查网络连接和GitHub账户权限。</parameter>
</parameter>
</write_to_file>