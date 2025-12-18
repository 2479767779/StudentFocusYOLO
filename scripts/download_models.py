"""
模型下载脚本
下载YOLO和MediaPipe所需模型
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile

def download_yolo_models():
    """下载YOLO模型"""
    print("正在下载YOLO模型...")
    
    # YOLOv8模型URL (使用Ultralytics官方模型)
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8m.pt"
    }
    
    # 创建模型目录
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    for model_name, url in models.items():
        model_path = model_dir / model_name
        
        # 检查是否已存在
        if model_path.exists():
            print(f"✓ {model_name} 已存在")
            downloaded.append(model_name)
            continue
        
        try:
            print(f"下载 {model_name}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"✓ {model_name} 下载完成")
            downloaded.append(model_name)
        except Exception as e:
            print(f"✗ 下载 {model_name} 失败: {e}")
    
    return downloaded

def download_mediapipe_models():
    """MediaPipe模型 (自动下载，无需手动处理)"""
    print("\nMediaPipe模型将在首次使用时自动下载")
    print("包括:")
    print("- Face Mesh 模型")
    print("- Face Detection 模型")
    print("- Iris 模型")

def create_sample_config():
    """创建示例配置文件"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # 创建本地配置模板
    local_config = config_dir / "local.yaml"
    if not local_config.exists():
        content = """# 本地配置文件
# 复制此文件为 local.yaml 并根据需要修改

camera:
  index: 0
  resolution: [1280, 720]
  fps: 30
  brightness: 1.0
  contrast: 1.1

model:
  device: cuda  # cuda 或 cpu
  yolo_model: yolov8n
  conf_threshold: 0.25

focus_monitoring:
  max_students: 10
  skip_frames: 0
  alert_threshold: 50

output:
  save_video: true
  save_logs: true
  output_dir: results
"""
        with open(local_config, 'w') as f:
            f.write(content)
        print(f"✓ 创建示例配置: {local_config}")

def main():
    """主函数"""
    print("=== 学生专注度监控系统 - 模型下载工具 ===\n")
    
    # 下载YOLO模型
    yolo_models = download_yolo_models()
    
    # MediaPipe信息
    download_mediapipe_models()
    
    # 创建示例配置
    create_sample_config()
    
    print(f"\n=== 下载完成 ===")
    print(f"已下载YOLO模型: {len(yolo_models)}个")
    print(f"模型位置: data/models/")
    
    if len(yolo_models) > 0:
        print("\n可以开始使用系统了!")
        print("运行: python main.py --source 0 --display")
    else:
        print("\n警告: 没有成功下载模型")
        print("可以手动下载模型或检查网络连接")

if __name__ == "__main__":
    main()