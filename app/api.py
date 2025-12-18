"""
FastAPI接口
提供RESTful API服务
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import io
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.face_detector import FaceDetector
from src.detection.pose_estimator import PoseEstimator
from src.analysis.focus_analyzer import FocusAnalyzer
from src.analysis.behavior_classifier import BehaviorClassifier
from src.utils.visualization import Visualizer

app = FastAPI(title="课堂专注度监控API", version="1.0.0")

# 全局组件
components = {
    "face_detector": None,
    "pose_estimator": None,
    "focus_analyzer": None,
    "behavior_classifier": None,
    "visualizer": None
}

def initialize_components():
    """初始化组件"""
    if components["face_detector"] is None:
        components["face_detector"] = FaceDetector(model_type="yolov8n", device="cpu")
        components["pose_estimator"] = PoseEstimator()
        components["focus_analyzer"] = FocusAnalyzer()
        components["behavior_classifier"] = BehaviorClassifier()
        components["visualizer"] = Visualizer()

class AnalysisResult(BaseModel):
    """分析结果模型"""
    timestamp: str
    student_id: str
    focus_score: float
    components: dict
    behaviors: List[str]

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    initialize_components()

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "课堂专注度监控API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze/image": "分析单张图片",
            "/analyze/video": "分析视频文件",
            "/stats": "获取统计信息",
            "/health": "健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze/image", response_model=List[AnalysisResult])
async def analyze_image(file: UploadFile = File(...)):
    """
    分析单张图片
    
    Args:
        file: 上传的图片文件
        
    Returns:
        分析结果列表
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "必须上传图片文件")
    
    try:
        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(400, "无法解码图片")
        
        # 分析
        results = await analyze_frame(frame)
        
        return results
        
    except Exception as e:
        raise HTTPException(500, f"分析失败: {str(e)}")

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """
    分析视频文件
    
    Args:
        file: 上传的视频文件
        
    Returns:
        分析结果
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "必须上传视频文件")
    
    try:
        # 保存临时文件
        temp_path = Path("temp_video.mp4")
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # 分析视频
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            raise HTTPException(400, "无法打开视频")
        
        all_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔10帧分析一次
            if frame_count % 10 == 0:
                frame_results = await analyze_frame(frame)
                all_results.extend(frame_results)
            
            frame_count += 1
            
            # 限制分析帧数
            if frame_count >= 300:
                break
        
        cap.release()
        temp_path.unlink(missing_ok=True)
        
        # 汇总结果
        summary = await summarize_results(all_results)
        
        return summary
        
    except Exception as e:
        raise HTTPException(500, f"视频分析失败: {str(e)}")

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    # 这里可以返回历史统计信息
    return {
        "message": "统计功能需要结合数据库使用",
        "suggestion": "使用 /analyze/image 或 /analyze/video 获取实时分析结果"
    }

async def analyze_frame(frame: np.ndarray) -> List[AnalysisResult]:
    """分析单帧"""
    # 人脸检测
    faces = components["face_detector"].detect(frame)
    
    results = []
    
    for i, face in enumerate(faces[:5]):  # 最多5人
        student_id = f"Student_{i+1}"
        
        # 提取关键点
        landmarks = components["pose_estimator"].extract_landmarks(frame, face.bbox)
        
        if landmarks:
            # 估计姿态和视线
            head_pose = components["pose_estimator"].estimate_head_pose(frame, landmarks)
            gaze = components["pose_estimator"].estimate_gaze_direction(landmarks)
            
            # 分析专注度
            focus_score = components["focus_analyzer"].analyze(
                student_id, head_pose, gaze, landmarks, time.time()
            )
            
            # 分类行为
            behaviors = components["behavior_classifier"].classify(
                head_pose, gaze, landmarks, time.time(), student_id
            )
            
            result = AnalysisResult(
                timestamp=datetime.now().isoformat(),
                student_id=student_id,
                focus_score=round(focus_score.total, 2),
                components={
                    "posture": round(focus_score.posture, 2),
                    "gaze": round(focus_score.gaze, 2),
                    "expression": round(focus_score.expression, 2),
                    "temporal": round(focus_score.temporal, 2)
                },
                behaviors=[b.behavior_type.value for b in behaviors]
            )
            
            results.append(result)
    
    return results

async def summarize_results(results: List[AnalysisResult]) -> dict:
    """汇总分析结果"""
    if not results:
        return {"message": "未检测到人脸"}
    
    # 计算统计
    focus_scores = [r.focus_score for r in results]
    avg_focus = sum(focus_scores) / len(focus_scores)
    
    # 行为统计
    all_behaviors = []
    for r in results:
        all_behaviors.extend(r.behaviors)
    
    from collections import Counter
    behavior_counts = Counter(all_behaviors)
    
    return {
        "summary": {
            "total_frames_analyzed": len(results),
            "average_focus_score": round(avg_focus, 2),
            "focus_distribution": {
                "excellent": sum(1 for s in focus_scores if s >= 85),
                "good": sum(1 for s in focus_scores if 70 <= s < 85),
                "average": sum(1 for s in focus_scores if 50 <= s < 70),
                "poor": sum(1 for s in focus_scores if s < 50)
            },
            "behavior_summary": dict(behavior_counts)
        },
        "detailed_results": results
    }

@app.post("/visualize")
async def visualize_results(file: UploadFile = File(...)):
    """可视化分析结果"""
    try:
        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(400, "无法解码图片")
        
        # 分析
        faces = components["face_detector"].detect(frame)
        
        for i, face in enumerate(faces[:5]):
            landmarks = components["pose_estimator"].extract_landmarks(frame, face.bbox)
            
            if landmarks:
                head_pose = components["pose_estimator"].estimate_head_pose(frame, landmarks)
                gaze = components["pose_estimator"].estimate_gaze_direction(landmarks)
                
                # 可视化
                frame = components["visualizer"].draw_face_with_landmarks(
                    frame, face.bbox, landmarks, head_pose, gaze
                )
        
        # 编码为JPEG
        success, encoded = cv2.imencode('.jpg', frame)
        if not success:
            raise HTTPException(500, "无法编码图像")
        
        return StreamingResponse(
            io.BytesIO(encoded.tobytes()),
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=visualized.jpg"}
        )
        
    except Exception as e:
        raise HTTPException(500, f"可视化失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)