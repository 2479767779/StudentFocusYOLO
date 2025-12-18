"""
人脸检测模块 - 基于YOLOv8
负责实时检测课堂中的人脸，提供高精度的面部位置信息
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import torch
import logging
from dataclasses import dataclass

@dataclass
class FaceDetection:
    """人脸检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]
    face_id: Optional[int] = None

class FaceDetector:
    """
    YOLOv8人脸检测器
    支持实时视频流中的人脸检测和跟踪
    """
    
    def __init__(self, 
                 model_type: str = "yolov8n",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = "cuda",
                 min_face_size: int = 80):
        """
        初始化人脸检测器
        
        Args:
            model_type: YOLO模型类型
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            device: 计算设备
            min_face_size: 最小人脸尺寸
        """
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.min_face_size = min_face_size
        
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            # 使用YOLOv8的人脸检测模型
            # 如果没有专门的人脸模型，可以使用通用目标检测模型
            if self.model_type in ["yolov8n", "yolov8s", "yolov8m"]:
                model_name = f"{self.model_type}-face"  # 假设有专门的人脸版本
                try:
                    self.model = YOLO(model_name)
                except:
                    # 回退到标准模型
                    self.model = YOLO(self.model_type)
                    self.logger.warning(f"未找到{model_name}，使用标准{self.model_type}")
            else:
                self.model = YOLO(self.model_type)
            
            self.logger.info(f"成功加载YOLO模型: {self.model_type}，设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        检测单帧图像中的人脸
        
        Args:
            frame: BGR格式的图像帧
            
        Returns:
            人脸检测结果列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            # 执行推理
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )[0]
            
            detections = []
            
            # 解析检测结果
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # 过滤太小的人脸
                width = x2 - x1
                height = y2 - y1
                if width < self.min_face_size or height < self.min_face_size:
                    continue
                
                # 计算中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                detection = FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    center=(center_x, center_y)
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"人脸检测失败: {e}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """
        批量检测多帧图像
        
        Args:
            frames: 图像帧列表
            
        Returns:
            每帧的检测结果列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            results = self.model(
                frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            batch_detections = []
            for result in results:
                detections = []
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    width = x2 - x1
                    height = y2 - y1
                    if width < self.min_face_size or height < self.min_face_size:
                        continue
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detection = FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        center=(center_x, center_y)
                    )
                    detections.append(detection)
                
                batch_detections.append(detections)
            
            return batch_detections
            
        except Exception as e:
            self.logger.error(f"批量人脸检测失败: {e}")
            return [[] for _ in frames]
    
    def visualize(self, frame: np.ndarray, detections: List[FaceDetection], 
                  show_conf: bool = True) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            frame: 原始图像
            detections: 检测结果
            show_conf: 是否显示置信度
            
        Returns:
            可视化后的图像
        """
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # 绘制边界框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(vis_frame, det.center, 3, (0, 0, 255), -1)
            
            # 绘制置信度
            if show_conf:
                label = f"{det.confidence:.2f}"
                cv2.putText(vis_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_frame
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        if self.model is None:
            return {}
        
        return {
            "model_type": self.model_type,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "min_face_size": self.min_face_size
        }