"""
姿态估计模块 - 基于MediaPipe
负责估计学生的头部姿态和身体姿态，用于专注度分析
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import mediapipe as mp
from dataclasses import dataclass

@dataclass
class HeadPose:
    """头部姿态数据类"""
    yaw: float   # 偏航角 (左右转动)
    pitch: float # 俯仰角 (上下点头)
    roll: float  # 滚转角 (左右倾斜)
    confidence: float

@dataclass
class GazeDirection:
    """视线方向数据类"""
    horizontal: float  # 水平方向 (-1 左, 1 右)
    vertical: float    # 垂直方向 (-1 上, 1 下)
    confidence: float

@dataclass
class FaceLandmarks:
    """面部关键点数据类"""
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose_tip: Tuple[float, float]
    mouth_left: Tuple[float, float]
    mouth_right: Tuple[float, float]
    left_eyebrow: Tuple[float, float]
    right_eyebrow: Tuple[float, float]

class PoseEstimator:
    """
    MediaPipe姿态估计器
    提供头部姿态、视线方向和面部关键点检测
    """
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        初始化姿态估计器
        
        Args:
            model_complexity: 模型复杂度 (0, 1, 2)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # 初始化MediaPipe人脸网格
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 面部关键点索引
        self.LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
        self.RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
        self.LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW_IDX = [300, 293, 334, 296, 336]
        self.NOSE_TIP_IDX = 1
        self.MOUTH_LEFT_IDX = 61
        self.MOUTH_RIGHT_IDX = 291
        
        self.logger = None
    
    def estimate_head_pose(self, face_image: np.ndarray, 
                          landmarks: List[Tuple[float, float]]) -> HeadPose:
        """
        估计头部姿态
        
        Args:
            face_image: 人脸图像
            landmarks: 面部关键点
            
        Returns:
            头部姿态数据
        """
        if len(landmarks) < 2:
            return HeadPose(0, 0, 0, 0.0)
        
        # 3D模型点 (基于MediaPipe的3D人脸模型参考点)
        # 使用鼻尖、左右眼角、嘴角作为参考点
        model_points = np.array([
            (0.0, 0.0, 0.0),          # 鼻尖
            (0.0, -330.0, -65.0),     # 下巴
            (-225.0, 170.0, -135.0),  # 左眼角
            (225.0, 170.0, -135.0),   # 右眼角
            (-150.0, -150.0, -125.0), # 左嘴角
            (150.0, -150.0, -125.0)   # 右嘴角
        ], dtype=np.float64)
        
        # 2D图像点 (从landmarks中提取)
        image_points = np.array([
            landmarks[self.NOSE_TIP_IDX],                    # 鼻尖
            landmarks[152] if 152 < len(landmarks) else landmarks[0],  # 下巴
            landmarks[self.LEFT_EYE_IDX[0]],                # 左眼角
            landmarks[self.RIGHT_EYE_IDX[0]],               # 右眼角
            landmarks[self.MOUTH_LEFT_IDX],                 # 左嘴角
            landmarks[self.MOUTH_RIGHT_IDX]                 # 右嘴角
        ], dtype=np.float64)
        
        # 相机内参 (假设)
        focal_length = face_image.shape[1]
        center = (face_image.shape[1] / 2, face_image.shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 假设无畸变
        dist_coeffs = np.zeros((4, 1))
        
        try:
            # 求解PnP问题
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # 将旋转向量转换为旋转矩阵
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # 提取欧拉角 (弧度)
                # 注意：这里使用近似计算，实际应用可能需要更精确的方法
                sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
                
                singular = sy < 1e-6
                
                if not singular:
                    x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    y = np.arctan2(-rotation_matrix[2, 0], sy)
                    z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                    y = np.arctan2(-rotation_matrix[2, 0], sy)
                    z = 0
                
                # 转换为度数
                yaw = np.degrees(x)   # 左右转头
                pitch = np.degrees(y) # 上下点头
                roll = np.degrees(z)  # 左右倾斜
                
                # 计算置信度 (基于landmarks的分布)
                confidence = self._calculate_pose_confidence(landmarks)
                
                return HeadPose(yaw, pitch, roll, confidence)
            
        except Exception as e:
            pass
        
        return HeadPose(0, 0, 0, 0.0)
    
    def estimate_gaze_direction(self, landmarks: List[Tuple[float, float]]) -> GazeDirection:
        """
        估计视线方向
        
        Args:
            landmarks: 面部关键点
            
        Returns:
            视线方向数据
        """
        if len(landmarks) < 2:
            return GazeDirection(0, 0, 0.0)
        
        try:
            # 左右眼中心
            left_eye_center = np.mean([landmarks[i] for i in self.LEFT_EYE_IDX if i < len(landmarks)], axis=0)
            right_eye_center = np.mean([landmarks[i] for i in self.RIGHT_EYE_IDX if i < len(landmarks)], axis=0)
            
            # 鼻尖
            nose = landmarks[self.NOSE_TIP_IDX]
            
            # 眼睛向量
            left_eye_vector = left_eye_center - nose
            right_eye_vector = right_eye_center - nose
            
            # 平均视线向量
            gaze_vector = (left_eye_vector + right_eye_vector) / 2
            
            # 归一化
            norm = np.linalg.norm(gaze_vector)
            if norm > 0:
                gaze_vector = gaze_vector / norm
            
            # 水平方向 (-1 左, 1 右)
            horizontal = np.clip(gaze_vector[0] * 2, -1, 1)
            
            # 垂直方向 (-1 上, 1 下) - 注意图像坐标系
            vertical = np.clip(-gaze_vector[1] * 2, -1, 1)
            
            # 置信度
            confidence = 0.7  # 简化计算
            
            return GazeDirection(horizontal, vertical, confidence)
            
        except Exception as e:
            return GazeDirection(0, 0, 0.0)
    
    def extract_landmarks(self, frame: np.ndarray, 
                         face_bbox: Tuple[int, int, int, int]) -> Optional[List[Tuple[float, float]]]:
        """
        从人脸区域提取面部关键点
        
        Args:
            frame: 原始图像
            face_bbox: 人脸边界框 (x1, y1, x2, y2)
            
        Returns:
            面部关键点列表
        """
        x1, y1, x2, y2 = face_bbox
        
        # 裁剪人脸区域
        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return None
        
        # 转换为RGB
        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # 检测人脸网格
        results = self.face_mesh.process(face_region_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # 取第一个检测结果
        face_landmarks = results.multi_face_landmarks[0]
        
        # 转换为相对于原图的坐标
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * (x2 - x1) + x1)
            y = int(landmark.y * (y2 - y1) + y1)
            landmarks.append((x, y))
        
        return landmarks
    
    def get_face_landmarks(self, landmarks: List[Tuple[float, float]]) -> Optional[FaceLandmarks]:
        """
        获取标准化的面部关键点
        
        Args:
            landmarks: 原始关键点
            
        Returns:
            标准化关键点对象
        """
        if len(landmarks) < 100:
            return None
        
        try:
            left_eye = np.mean([landmarks[i] for i in self.LEFT_EYE_IDX if i < len(landmarks)], axis=0)
            right_eye = np.mean([landmarks[i] for i in self.RIGHT_EYE_IDX if i < len(landmarks)], axis=0)
            nose = landmarks[self.NOSE_TIP_IDX]
            mouth_left = landmarks[self.MOUTH_LEFT_IDX]
            mouth_right = landmarks[self.MOUTH_RIGHT_IDX]
            left_eyebrow = np.mean([landmarks[i] for i in self.LEFT_EYEBROW_IDX if i < len(landmarks)], axis=0)
            right_eyebrow = np.mean([landmarks[i] for i in self.RIGHT_EYEBROW_IDX if i < len(landmarks)], axis=0)
            
            return FaceLandmarks(
                left_eye=tuple(left_eye),
                right_eye=tuple(right_eye),
                nose_tip=tuple(nose),
                mouth_left=tuple(mouth_left),
                mouth_right=tuple(mouth_right),
                left_eyebrow=tuple(left_eyebrow),
                right_eyebrow=tuple(right_eyebrow)
            )
        except Exception as e:
            return None
    
    def _calculate_pose_confidence(self, landmarks: List[Tuple[float, float]]) -> float:
        """计算姿态估计的置信度"""
        if len(landmarks) < 50:
            return 0.0
        
        # 基于关键点分布的置信度
        points = np.array(landmarks)
        spread = np.std(points, axis=0).mean()
        
        # 简单的置信度计算
        confidence = min(1.0, spread / 100.0)
        return confidence
    
    def visualize_landmarks(self, frame: np.ndarray, 
                           landmarks: List[Tuple[float, float]]) -> np.ndarray:
        """
        可视化面部关键点
        
        Args:
            frame: 原始图像
            landmarks: 面部关键点
            
        Returns:
            可视化图像
        """
        vis_frame = frame.copy()
        
        if landmarks is None:
            return vis_frame
        
        # 绘制所有关键点
        for x, y in landmarks:
            cv2.circle(vis_frame, (int(x), int(y)), 1, (255, 0, 0), -1)
        
        # 绘制特殊关键点
        special_indices = {
            "左眼": self.LEFT_EYE_IDX,
            "右眼": self.RIGHT_EYE_IDX,
            "鼻尖": [self.NOSE_TIP_IDX],
            "左嘴角": [self.MOUTH_LEFT_IDX],
            "右嘴角": [self.MOUTH_RIGHT_IDX]
        }
        
        colors = {
            "左眼": (0, 255, 0),
            "右眼": (0, 255, 0),
            "鼻尖": (0, 0, 255),
            "左嘴角": (255, 0, 0),
            "右嘴角": (255, 0, 0)
        }
        
        for name, indices in special_indices.items():
            points = []
            for idx in indices:
                if idx < len(landmarks):
                    points.append(landmarks[idx])
            
            if points:
                center = np.mean(points, axis=0)
                cv2.circle(vis_frame, (int(center[0]), int(center[1])), 
                          3, colors[name], -1)
                cv2.putText(vis_frame, name, (int(center[0]), int(center[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)
        
        return vis_frame
    
    def visualize_pose(self, frame: np.ndarray, 
                      head_pose: HeadPose,
                      gaze: GazeDirection,
                      position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        可视化姿态信息
        
        Args:
            frame: 原始图像
            head_pose: 头部姿态
            gaze: 视线方向
            position: 文字显示位置
            
        Returns:
            可视化图像
        """
        vis_frame = frame.copy()
        x, y = position
        
        # 显示头部姿态
        cv2.putText(vis_frame, f"Yaw: {head_pose.yaw:.1f}°", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Pitch: {head_pose.pitch:.1f}°", 
                   (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Roll: {head_pose.roll:.1f}°", 
                   (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示视线方向
        cv2.putText(vis_frame, f"Gaze H: {gaze.horizontal:.2f}", 
                   (x, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_frame, f"Gaze V: {gaze.vertical:.2f}", 
                   (x, y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis_frame