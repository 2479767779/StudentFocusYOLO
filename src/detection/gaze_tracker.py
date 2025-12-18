"""
视线追踪模块
基于面部关键点估计视线方向
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GazeDirection:
    """视线方向数据类"""
    horizontal: float  # 水平方向 (-1 左, 1 右)
    vertical: float    # 垂直方向 (-1 上, 1 下)
    confidence: float  # 置信度

class GazeTracker:
    """
    视线追踪器
    基于面部关键点估计视线方向
    """
    
    def __init__(self):
        """初始化视线追踪器"""
        # 眼睛关键点索引 (基于MediaPipe Face Mesh)
        self.LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
        self.RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
        self.NOSE_TIP_IDX = 1
        self.LEFT_EYE_CENTER_IDX = 468  # 左眼中心 (Refined landmarks)
        self.RIGHT_EYE_CENTER_IDX = 473  # 右眼中心 (Refined landmarks)
    
    def estimate_gaze(self, landmarks: List[Tuple[float, float]]) -> GazeDirection:
        """
        估计视线方向
        
        Args:
            landmarks: 面部关键点列表
            
        Returns:
            视线方向数据
        """
        if len(landmarks) < 100:
            return GazeDirection(0.0, 0.0, 0.0)
        
        try:
            # 获取眼睛中心点
            left_eye_center = self._get_eye_center(landmarks, self.LEFT_EYE_IDX)
            right_eye_center = self._get_eye_center(landmarks, self.RIGHT_EYE_IDX)
            
            # 获取鼻尖
            nose = landmarks[self.NOSE_TIP_IDX]
            
            # 计算视线向量
            left_vector = self._calculate_gaze_vector(left_eye_center, nose)
            right_vector = self._calculate_gaze_vector(right_eye_center, nose)
            
            # 平均视线向量
            avg_vector = (left_vector + right_vector) / 2
            
            # 归一化
            norm = np.linalg.norm(avg_vector)
            if norm > 0:
                gaze_vector = avg_vector / norm
            else:
                gaze_vector = avg_vector
            
            # 转换为水平和垂直分量
            horizontal = np.clip(gaze_vector[0] * 2, -1, 1)
            vertical = np.clip(-gaze_vector[1] * 2, -1, 1)
            
            # 计算置信度
            confidence = self._calculate_confidence(landmarks, left_eye_center, right_eye_center)
            
            return GazeDirection(horizontal, vertical, confidence)
            
        except Exception as e:
            return GazeDirection(0.0, 0.0, 0.0)
    
    def _get_eye_center(self, landmarks: List[Tuple[float, float]], 
                       eye_indices: List[int]) -> Tuple[float, float]:
        """计算眼睛中心点"""
        points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                points.append(landmarks[idx])
        
        if not points:
            return (0.0, 0.0)
        
        # 使用平均值
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        
        return (center_x, center_y)
    
    def _calculate_gaze_vector(self, eye_center: Tuple[float, float], 
                              nose: Tuple[float, float]) -> np.ndarray:
        """计算视线向量"""
        # 从眼睛指向鼻尖的向量
        vector = np.array([
            nose[0] - eye_center[0],
            nose[1] - eye_center[1]
        ], dtype=np.float64)
        
        return vector
    
    def _calculate_confidence(self, landmarks: List[Tuple[float, float]],
                             left_eye: Tuple[float, float],
                             right_eye: Tuple[float, float]) -> float:
        """计算置信度"""
        # 基于眼睛距离和位置的置信度
        if len(landmarks) < 50:
            return 0.0
        
        # 眼睛间距
        eye_distance = np.sqrt(
            (right_eye[0] - left_eye[0])**2 + 
            (right_eye[1] - left_eye[1])**2
        )
        
        # 简单的置信度计算
        if eye_distance > 20:  # 眼睛距离足够大
            return 0.8
        elif eye_distance > 10:
            return 0.6
        else:
            return 0.3
    
    def visualize_gaze(self, frame: np.ndarray, 
                      landmarks: List[Tuple[float, float]],
                      gaze: GazeDirection,
                      position: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        可视化视线方向
        
        Args:
            frame: 原始图像
            landmarks: 面部关键点
            gaze: 视线方向
            position: 显示位置
            
        Returns:
            可视化图像
        """
        vis_frame = frame.copy()
        
        if len(landmarks) < 100:
            return vis_frame
        
        # 获取眼睛中心
        left_eye = self._get_eye_center(landmarks, self.LEFT_EYE_IDX)
        right_eye = self._get_eye_center(landmarks, self.RIGHT_EYE_IDX)
        
        # 绘制视线方向
        scale = 50  # 视线向量长度
        
        # 左眼视线
        left_end = (
            int(left_eye[0] + gaze.horizontal * scale),
            int(left_eye[1] + gaze.vertical * scale)
        )
        cv2.arrowedLine(vis_frame, 
                       (int(left_eye[0]), int(left_eye[1])), 
                       left_end, 
                       (255, 255, 0), 2)
        
        # 右眼视线
        right_end = (
            int(right_eye[0] + gaze.horizontal * scale),
            int(right_eye[1] + gaze.vertical * scale)
        )
        cv2.arrowedLine(vis_frame, 
                       (int(right_eye[0]), int(right_eye[1])), 
                       right_end, 
                       (255, 255, 0), 2)
        
        # 显示数值
        cv2.putText(vis_frame, f"H: {gaze.horizontal:.2f}", 
                   position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(vis_frame, f"V: {gaze.vertical:.2f}", 
                   (position[0], position[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return vis_frame