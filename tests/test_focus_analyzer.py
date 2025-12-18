"""
专注度分析器测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.focus_analyzer import FocusAnalyzer, FocusScore
from src.detection.pose_estimator import HeadPose, GazeDirection

class TestFocusAnalyzer:
    """专注度分析器测试类"""
    
    def setup_method(self):
        """测试设置"""
        self.analyzer = FocusAnalyzer()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'student_states')
        assert len(self.analyzer.student_states) == 0
    
    def test_analyze_perfect_focus(self):
        """测试完美专注状态"""
        # 完美的头部姿态和视线
        head_pose = HeadPose(yaw=0, pitch=0, roll=0, confidence=1.0)
        gaze = GazeDirection(horizontal=0, vertical=0, confidence=1.0)
        landmarks = [(100, 100)] * 100  # 模拟关键点
        
        result = self.analyzer.analyze("test_student", head_pose, gaze, landmarks)
        
        assert result.total >= 85  # 应该是优秀专注度
        assert result.posture >= 90
        assert result.gaze >= 90
        assert result.expression >= 80
        assert result.temporal >= 80
    
    def test_analyze_distracted(self):
        """测试分心状态"""
        # 分心的头部姿态和视线
        head_pose = HeadPose(yaw=45, pitch=30, roll=20, confidence=0.8)
        gaze = GazeDirection(horizontal=0.8, vertical=0.8, confidence=0.7)
        landmarks = [(100, 100)] * 100
        
        result = self.analyzer.analyze("test_student", head_pose, gaze, landmarks)
        
        assert result.total < 50  # 应该是较差专注度
    
    def test_temporal_score(self):
        """测试时序分数"""
        head_pose = HeadPose(yaw=0, pitch=0, roll=0, confidence=1.0)
        gaze = GazeDirection(horizontal=0, vertical=0, confidence=1.0)
        landmarks = [(100, 100)] * 100
        
        # 第一次分析
        result1 = self.analyzer.analyze("temporal_test", head_pose, gaze, landmarks)
        
        # 第二次分析（应该提高时序分数）
        result2 = self.analyzer.analyze("temporal_test", head_pose, gaze, landmarks)
        
        assert result2.temporal >= result1.temporal
    
    def test_classroom_statistics(self):
        """测试课堂统计"""
        # 模拟多个学生
        head_pose = HeadPose(yaw=0, pitch=0, roll=0, confidence=1.0)
        gaze = GazeDirection(horizontal=0, vertical=0, confidence=1.0)
        landmarks = [(100, 100)] * 100
        
        for i in range(5):
            self.analyzer.analyze(f"student_{i}", head_pose, gaze, landmarks)
        
        stats = self.analyzer.get_classroom_statistics()
        
        assert stats["total_students"] == 5
        assert stats["class_average"] >= 85
        assert stats["excellent_count"] == 5
        assert stats["distracted_count"] == 0
    
    def test_student_report(self):
        """测试学生报告"""
        head_pose = HeadPose(yaw=0, pitch=0, roll=0, confidence=1.0)
        gaze = GazeDirection(horizontal=0, vertical=0, confidence=1.0)
        landmarks = [(100, 100)] * 100
        
        # 多次分析
        for _ in range(3):
            self.analyzer.analyze("report_test", head_pose, gaze, landmarks)
        
        report = self.analyzer.get_student_report("report_test")
        
        assert report is not None
        assert report["student_id"] == "report_test"
        assert "current_focus" in report
        assert "average_focus" in report
        assert "component_scores" in report

if __name__ == "__main__":
    pytest.main([__file__, "-v"])