# 快速开始指南

## 🎯 5分钟快速启动

### 步骤1: 环境准备

```bash
# 创建虚拟环境
conda create -n focus_monitor python=3.9
conda activate focus_monitor

# 安装依赖
pip install ultralytics opencv-python mediapipe streamlit fastapi
```

### 步骤2: 下载模型

```bash
# 运行下载脚本
python scripts/download_models.py

# 或手动下载
# 访问 https://github.com/ultralytics/assets/releases
# 下载 yolov8n.pt 到 data/models/
```

### 步骤3: 启动系统

```bash
# 方式1: 命令行模式
python main.py --source 0 --display

# 方式2: Web界面
streamlit run app/dashboard.py

# 方式3: API服务
uvicorn app.api:app --reload
```

## 📱 界面说明

### 命令行界面
```
=== 课堂专注度监控系统已启动 ===
按 'q' 退出
按 's' 保存当前统计
按 'r' 重置统计
```

### Web界面功能
- **实时监控**: 摄像头画面 + 专注度叠加
- **数据分析**: 趋势图表 + 统计报表
- **导出数据**: CSV格式的历史数据

## 🎬 演示示例

### 示例1: 实时监控演示

```bash
# 启动实时监控
python main.py --source 0 --display --save-results

# 屏幕将显示:
# - 左上角: 课堂标题和统计
# - 中间: 摄像头画面，带人脸框和专注度标签
# - 右侧: 学生列表和专注度分数
# - 底部: 专注度柱状图
```

### 示例2: 视频文件分析

```bash
# 准备测试视频
mkdir -p test_videos
# 将课堂视频放入 test_videos/

# 分析视频
python scripts/evaluate.py --video test_videos/classroom.mp4 --output results/analysis.json

# 查看结果
cat results/analysis.json
```

### 示例3: Web界面演示

```bash
# 启动Web界面
streamlit run app/dashboard.py

# 浏览器将打开 http://localhost:8501
# 包含三个标签页:
# 1. 实时监控 - 摄像头实时分析
# 2. 视频分析 - 上传视频文件分析
# 3. 数据分析 - 历史数据可视化
```

## 🔧 配置示例

### 摄像头配置

```python
# 在 main.py 中修改
video_processor = VideoProcessor(
    source=0,              # 摄像头索引
    target_fps=30,         # 目标帧率
    resolution=(1280, 720) # 分辨率
)
```

### 模型配置

```yaml
# configs/model_config.yaml
yolo:
  model_type: "yolov8n"    # 可选: yolov8n, yolov8s, yolov8m
  img_size: 640
  conf_threshold: 0.25
  device: "cuda"          # cuda 或 cpu
```

### 专注度评分配置

```yaml
# configs/focus_scoring.yaml
focus_scoring:
  weights:
    posture: 0.30      # 姿态权重
    gaze: 0.30         # 视线权重
    expression: 0.20   # 表情权重
    temporal: 0.20     # 时序权重
  
  thresholds:
    excellent: 85      # 优秀
    good: 70           # 良好
    average: 50        # 一般
    poor: 30           # 较差
```

## 🎓 教学场景应用

### 场景1: 课堂教学监控

```python
# 教师监控脚本
from src.detection.face_detector import FaceDetector
from src.analysis.focus_analyzer import FocusAnalyzer
import cv2

def classroom_monitor():
    # 初始化
    detector = FaceDetector()
    analyzer = FocusAnalyzer()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测和分析
        faces = detector.detect(frame)
        
        for face in faces:
            # 分析专注度
            # 显示结果
            pass
        
        # 显示监控界面
        cv2.imshow("Classroom Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classroom_monitor()
```

### 场景2: 在线学习评估

```python
# 分析录制的在线课程
def analyze_online_learning(video_path):
    # 1. 分析视频
    # 2. 生成专注度报告
    # 3. 识别学习难点
    # 4. 提供改进建议
    pass
```

## 📊 结果解读

### 专注度分数含义

| 分数范围 | 等级 | 含义 | 建议 |
|---------|------|------|------|
| 85-100 | 优秀 | 高度专注 | 保持现状 |
| 70-84 | 良好 | 偶尔分心 | 适当提醒 |
| 50-69 | 一般 | 注意力不集中 | 加强互动 |
| 30-49 | 较差 | 频繁分心 | 需要干预 |
| 0-29 | 严重 | 严重分心 | 立即关注 |

### 行为类型说明

- **FOCUSED**: 专注听讲，视线向前
- **SLIGHTLY_DISTRACTED**: 轻微分心，偶尔看别处
- **DISTRACTED**: 明显分心，视线偏离
- **SLEEPING**: 打瞌睡，头部低垂
- **PHONE_USAGE**: 使用手机，低头动作
- **YAWNING**: 打哈欠，疲劳表现
- **TALKING**: 与他人交谈

## 🚀 进阶功能

### 1. 多摄像头支持

```bash
# 同时监控多个摄像头
python main.py --source 0 --display &
python main.py --source 1 --display &
```

### 2. 批量视频处理

```bash
# 处理目录下所有视频
for video in *.mp4; do
    python scripts/evaluate.py --video "$video" --output "results/${video%.mp4}.json"
done
```

### 3. API集成

```python
import requests

# 发送图片进行分析
with open("classroom.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/image",
        files={"file": f}
    )

# 获取分析结果
results = response.json()
print(f"检测到 {len(results)} 个学生")
```

## ⚡ 性能优化

### GPU加速
```bash
# 使用GPU (需要CUDA)
python main.py --device cuda --model yolov8n
```

### CPU优化
```bash
# 使用轻量模型
python main.py --device cpu --model yolov8n --img-size 416

# 降低处理频率
python main.py --skip-frames 2
```

## 🎯 常见问题

**Q: 摄像头打不开?**
```bash
# 检查摄像头索引
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# 尝试其他索引
python main.py --source 1
```

**Q: 模型下载失败?**
```bash
# 手动下载
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt
mv yolov8n.pt data/models/
```

**Q: 运行太慢?**
```bash
# 降低分辨率
python main.py --img-size 416

# 使用CPU模式
python main.py --device cpu --model yolov8n
```

## 📞 获取帮助

```bash
# 查看帮助信息
python main.py --help

# 查看版本
python -c "import ultralytics; print(ultralytics.__version__)"

# 检查依赖
pip list | grep -E "ultralytics|opencv|mediapipe"
```

---

现在你已经掌握了基本使用方法！开始你的课堂专注度监控之旅吧！🚀