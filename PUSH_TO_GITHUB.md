# 🚀 推送到GitHub指南

## 📋 前置要求

1. **GitHub账户**: 如果没有，请注册 https://github.com/join
2. **Git已安装**: 检查 `git --version`
3. **网络连接**: 确保可以访问github.com

## 🎯 快速开始 (3步)

### 第1步: 在GitHub上创建仓库

**方法A: 网页创建 (推荐)**
1. 访问 https://github.com/new
2. 填写:
   - Repository name: `StudentFocusYOLO`
   - Description: `基于YOLO的课堂学生专注度实时监控系统`
   - 选择: **Public**
   - **不要勾选** "Initialize with README"
3. 点击 "Create repository"

**方法B: 使用GitHub CLI**
```bash
# 安装GitHub CLI (如果未安装)
# https://cli.github.com/

# 登录
gh auth login

# 创建仓库
gh repo create StudentFocusYOLO --public --description "基于YOLO的课堂学生专注度实时监控系统" --push
```

### 第2步: 配置Git并推送

在项目目录中执行:

```bash
# 进入项目
cd StudentFocusYOLO

# 配置用户信息 (如果未配置)
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"

# 添加远程仓库 (替换下面的URL)
git remote add origin https://github.com/你的用户名/StudentFocusYOLO.git

# 推送代码
git push -u origin master
```

### 第3步: 验证

访问: `https://github.com/你的用户名/StudentFocusYOLO`

应该能看到所有项目文件！

## 🔑 认证问题解决

### 如果要求输入密码

**解决方案**: 使用Personal Access Token

1. 生成Token:
   - 访问 https://github.com/settings/tokens
   - 点击 "Generate new token (classic)"
   - 勾选 `repo` 权限
   - 点击 "Generate token"
   - **复制Token (只显示一次)**

2. 在Git推送时:
   - 用户名: 你的GitHub用户名
   - 密码: **粘贴Token**

### 使用SSH (更安全)

```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
# 一路回车

# 复制公钥
cat ~/.ssh/id_ed25519.pub

# 添加到GitHub
# 访问 https://github.com/settings/keys
# 点击 "New SSH key"
# 粘贴公钥

# 使用SSH推送
git remote add origin git@github.com:你的用户名/StudentFocusYOLO.git
git push -u origin master
```

## 🛠️ 使用Python脚本

如果命令行操作困难，使用提供的Python脚本:

```bash
cd StudentFocusYOLO
python setup_github.py
```

按照提示输入GitHub Personal Access Token。

## 📦 项目结构确认

推送前确保项目结构完整:

```
StudentFocusYOLO/
├── README.md              # 项目说明
├── QUICKSTART.md          # 快速开始
├── USAGE_EXAMPLES.md      # 使用示例
├── GITHUB_PUSH_GUIDE.md   # 推送指南
├── main.py                # 主程序
├── requirements.txt       # 依赖包
├── configs/               # 配置文件
├── src/                   # 核心源码
├── app/                   # Web界面
├── scripts/               # 工具脚本
├── tests/                 # 测试代码
└── .gitignore            # Git忽略文件
```

## 🔍 故障排除

### 错误: "Repository not found"
- 检查仓库URL是否正确
- 确保仓库已创建

### 错误: "Authentication failed"
- 使用Personal Access Token代替密码
- 检查Token权限

### 错误: "Permission denied"
- 确保你有仓库的写权限
- 检查仓库是否是你的

### 错误: "Connection timed out"
- 检查网络连接
- 尝试使用SSH

### 推送成功但看不到文件
- 检查分支: `git branch` (应该是 master)
- 强制推送: `git push -f origin master`

## 🎉 成功后的配置

### 1. 添加徽章到README
```markdown
![Build Status](https://github.com/你的用户名/StudentFocusYOLO/workflows/CI/badge.svg)
```

### 2. 启用GitHub Pages (用于文档)
- Settings -> Pages -> Source: master branch
- 访问: `https://你的用户名.github.io/StudentFocusYOLO/`

### 3. 创建Release
- Tags: `v1.0.0`
- Title: `Initial Release`
- Description: `完整的学生专注度监控系统`

### 4. 添加License
项目已包含 MIT License，GitHub会自动识别

## 📞 获取帮助

如果仍有问题:
1. 查看GitHub帮助: https://help.github.com
2. 检查网络代理设置
3. 尝试在不同网络环境下操作

---

**提示**: 推送成功后，记得更新README中的仓库链接！</parameter>
</parameter>
</write_to_file>