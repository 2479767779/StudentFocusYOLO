"""
人脸检测器测试
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.face_detector import FaceDetector, FaceDetection

class TestFaceDetector:
    """人脸检测器测试类"""
    
    def setup_method(self):
        """测试设置"""
        self.detector = FaceDetector(model_type="yolov8n", device="cpu", conf_threshold=0.25)
    
    def test_initialization(self):
        """测试初始化"""
        assert self.detector is not None
        assert self.detector.model is not None
    
    def test_detect_no_faces(self):
        """测试无脸图像"""
        # 创建纯色图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (100, 100, 100)  # 灰色
        
        detections = self.detector.detect(frame)
        
        assert isinstance(detections, list)
        assert len(detections) == 0
    
    def test_detect_with_faces(self):
        """测试人脸检测 (使用模拟数据)"""
        # 创建一个简单的测试图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 模拟检测结果
        # 注意: 实际测试需要真实的人脸图像
        detections = self.detector.detect(frame)
        
        assert isinstance(detections, list)
        # 可能检测到人脸，也可能没有
    
    def test_visualize(self):
        """测试可视化"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 模拟检测结果
        mock_detection = FaceDetection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            center=(150, 150)
        )
        
        vis_frame = self.detector.visualize(frame, [mock_detection])
        
        assert vis_frame is not None
        assert vis_frame.shape == frame.shape
    
    def test_model_info(self):
        """测试模型信息"""
        info = self.detector.get_model_info()
        
        assert "model_type" in info
        assert "device" in info
        assert info["model_type"] == "yolov8n"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])