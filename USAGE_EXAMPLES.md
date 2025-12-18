# 使用示例和教程

## 🚀 快速入门示例

### 示例1: 基础实时监控

```bash
# 启动摄像头监控
python main.py --source 0 --display

# 使用特定摄像头
python main.py --source 1 --display

# 使用视频文件
python main.py --source "classroom_video.mp4" --display --save-results
```

### 示例2: 命令行参数详解

```bash
python main.py \
  --source 0 \              # 视频源: 0=摄像头, 或文件路径
  --model yolov8n \         # 模型类型: yolov8n, yolov8s, yolov8m
  --device cuda \           # 计算设备: cuda, cpu
  --img-size 640 \          # 推理图像尺寸
  --conf-threshold 0.25 \   # 置信度阈值
  --iou-threshold 0.45 \    # IoU阈值
  --display \               # 显示实时画面
  --save-results \          # 保存结果
  --output-dir results \    # 输出目录
  --skip-frames 0 \         # 跳帧处理
  --max-students 10         # 最大检测人数
```

### 示例3: Python API调用

```python
from src.detection.face_detector import FaceDetector
from src.detection.pose_estimator import PoseEstimator
from src.analysis.focus_analyzer import FocusAnalyzer
import cv2

# 初始化组件
face_detector = FaceDetector(model_type="yolov8n", device="cpu")
pose_estimator = PoseEstimator()
focus_analyzer = FocusAnalyzer()

# 读取图像
frame = cv2.imread("classroom.jpg")

# 检测人脸
faces = face_detector.detect(frame)

# 分析每个学生
for face in faces:
    landmarks = pose_estimator.extract_landmarks(frame, face.bbox)
    if landmarks:
        head_pose = pose_estimator.estimate_head_pose(frame, landmarks)
        gaze = pose_estimator.estimate_gaze_direction(landmarks)
        
        focus_score = focus_analyzer.analyze(
            "student_1", head_pose, gaze, landmarks
        )
        
        print(f"专注度: {focus_score.total:.1f}")
```

## 🎯 高级使用场景

### 场景1: 批量视频分析

```bash
# 分析多个视频文件
for video in videos/*.mp4; do
    python scripts/evaluate.py \
        --video "$video" \
        --output "results/$(basename "$video" .mp4)_analysis.json"
done
```

### 场景2: Web界面监控

```bash
# 启动Streamlit界面
streamlit run app/dashboard.py --server.port 8501

# 启动FastAPI服务
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### 场景3: 实时API调用

```python
import requests
import json

# 分析图片
with open("classroom.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/analyze/image", files=files)

results = response.json()
print(json.dumps(results, indent=2))
```

## 📊 数据分析示例

### 专注度趋势分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取历史数据
df = pd.read_csv("results/focus_data.csv")

# 绘制趋势图
plt.figure(figsize=(12, 6))
for student in df['student_id'].unique():
    student_data = df[df['student_id'] == student]
    plt.plot(student_data['timestamp'], student_data['total'], label=student)

plt.xlabel('时间')
plt.ylabel('专注度分数')
plt.title('学生专注度趋势')
plt.legend()
plt.show()
```

### 行为统计分析

```python
from collections import Counter

# 统计行为频率
behavior_counts = Counter(df_behavior['behavior'])

# 可视化
plt.pie(behavior_counts.values(), labels=behavior_counts.keys(), autopct='%1.1f%%')
plt.title('行为分布')
plt.show()
```

## 🔧 性能优化配置

### GPU加速配置

```bash
# 使用GPU并启用半精度
python main.py --device cuda --model yolov8n --img-size 640

# 批量处理优化
python main.py --source 0 --batch-size 4 --skip-frames 2
```

### CPU优化配置

```bash
# 使用轻量模型
python main.py --device cpu --model yolov8n --img-size 416

# 降低帧率处理
python main.py --source 0 --skip-frames 3
```

## 🎓 教学场景应用

### 场景1: 课堂实时监控

```python
# 教师端监控脚本
import cv2
from datetime import datetime

class ClassroomMonitor:
    def __init__(self):
        self.setup_system()
    
    def setup_system(self):
        # 初始化检测器
        pass
    
    def monitor_session(self, duration_minutes=45):
        # 持续监控并记录
        pass
    
    def generate_report(self):
        # 生成课堂报告
        pass
```

### 场景2: 在线教育评估

```python
# 远程学习专注度分析
def analyze_online_session(video_path):
    # 分析录制的在线课程
    # 识别专注度变化
    # 生成学习效果报告
    pass
```

## 📈 故障排除指南

### 问题1: 模型加载失败

```bash
# 解决方案: 手动下载模型
python scripts/download_models.py

# 或手动下载
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt -O data/models/yolov8n.pt
```

### 问题2: 摄像头权限问题

```python
# 检查摄像头
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())  # 应该输出 True
cap.release()
```

### 问题3: 内存不足

```bash
# 使用更小的模型
python main.py --model yolov8n --img-size 320

# 降低处理频率
python main.py --skip-frames 5
```

## 🔍 调试技巧

### 启用详细日志

```python
from src.utils.logger import Logger

logger = Logger(log_level="DEBUG", enable_console=True)
logger.info("系统启动")
logger.debug("检测到人脸", extra={"count": len(faces)})
```

### 性能监控

```python
from src.utils.logger import PerformanceLogger

perf_logger = PerformanceLogger()
perf_logger.start_timer("detection")
# 执行检测
duration = perf_logger.end_timer("detection")
print(f"检测耗时: {duration:.3f}s")
```

## 📦 部署示例

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py", "--source", "0", "--display"]
```

### 服务器部署

```bash
# 后台运行
nohup python main.py --source 0 --display --save-results &

# 使用screen
screen -S focus_monitor
python main.py --source 0 --display
# 按 Ctrl+A, 然后按 D 分离会话
```

## 🎯 最佳实践

1. **模型选择**: 根据硬件选择合适的模型大小
2. **分辨率**: 平衡质量和性能
3. **跳帧**: 高分辨率下适当跳帧
4. **批量处理**: 视频文件建议分段处理
5. **数据备份**: 定期保存分析结果

---

更多示例和教程请参考项目文档或联系开发团队。