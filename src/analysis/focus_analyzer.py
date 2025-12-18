"""
专注度分析模块
基于多维度特征计算学生的实时专注度分数
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import yaml
from pathlib import Path

@dataclass
class FocusScore:
    """专注度分数数据类"""
    total: float              # 总分 (0-100)
    posture: float            # 姿态分数
    gaze: float               # 视线分数
    expression: float         # 表情分数
    temporal: float           # 时序分数
    timestamp: float          # 时间戳
    student_id: Optional[str] = None

@dataclass
class StudentFocusState:
    """学生专注状态"""
    student_id: str
    focus_history: deque = field(default_factory=lambda: deque(maxlen=60))  # 60秒历史
    attention_span: float = 0.0  # 持续专注时间(秒)
    last_focus_score: float = 0.0
    distraction_count: int = 0
    yawn_count: int = 0
    phone_usage_detected: int = 0
    
    def add_score(self, score: FocusScore):
        """添加新的专注度分数"""
        self.focus_history.append(score)
        
        # 计算持续专注时间
        if score.total >= 70:  # 达到良好专注度
            if self.last_focus_score >= 70:
                self.attention_span += 1.0  # 假设每秒调用一次
            else:
                self.attention_span = 1.0
        else:
            self.attention_span = 0.0
            
        # 记录分心事件
        if score.total < 50 and self.last_focus_score >= 50:
            self.distraction_count += 1
            
        self.last_focus_score = score.total

class FocusAnalyzer:
    """
    专注度分析器
    综合姿态、视线、表情和时序特征计算专注度
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化专注度分析器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.student_states: Dict[str, StudentFocusState] = {}
        
        # 姿态评分参数
        self.posture_weights = self.config["focus_scoring"]["weights"]["posture"]
        self.gaze_weights = self.config["focus_scoring"]["weights"]["gaze"]
        self.expression_weights = self.config["focus_scoring"]["weights"]["expression"]
        self.temporal_weights = self.config["focus_scoring"]["weights"]["temporal"]
        
        # 姿态阈值
        self.posture_params = self.config["focus_scoring"]["posture_params"]
        self.gaze_params = self.config["focus_scoring"]["gaze_params"]
        self.expression_params = self.config["focus_scoring"]["expression_params"]
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """加载配置文件"""
        if config_path is None:
            # 使用默认配置
            config_path = Path(__file__).parent.parent.parent / "configs" / "focus_scoring.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def analyze(self, 
                student_id: str,
                head_pose,
                gaze_direction,
                landmarks,
                frame_timestamp: Optional[float] = None) -> FocusScore:
        """
        分析单个学生的专注度
        
        Args:
            student_id: 学生ID
            head_pose: 头部姿态
            gaze_direction: 视线方向
            landmarks: 面部关键点
            frame_timestamp: 时间戳
            
        Returns:
            专注度分数
        """
        if frame_timestamp is None:
            frame_timestamp = time.time()
        
        # 计算各维度分数
        posture_score = self._calculate_posture_score(head_pose)
        gaze_score = self._calculate_gaze_score(gaze_direction)
        expression_score = self._calculate_expression_score(landmarks)
        temporal_score = self._calculate_temporal_score(student_id)
        
        # 加权平均计算总分
        total_score = (
            posture_score * self.posture_weights +
            gaze_score * self.gaze_weights +
            expression_score * self.expression_weights +
            temporal_score * self.temporal_weights
        )
        
        # 创建分数对象
        focus_score = FocusScore(
            total=total_score,
            posture=posture_score,
            gaze=gaze_score,
            expression=expression_score,
            temporal=temporal_score,
            timestamp=frame_timestamp,
            student_id=student_id
        )
        
        # 更新学生状态
        if student_id not in self.student_states:
            self.student_states[student_id] = StudentFocusState(student_id)
        self.student_states[student_id].add_score(focus_score)
        
        return focus_score
    
    def _calculate_posture_score(self, head_pose) -> float:
        """计算姿态分数"""
        if head_pose is None:
            return 50.0
        
        yaw = abs(head_pose.yaw)   # 左右偏转
        pitch = abs(head_pose.pitch) # 上下点头
        roll = abs(head_pose.roll)   # 左右倾斜
        
        # 基于配置的评分规则
        yaw_score = self._score_from_ranges(yaw, self.posture_params["max_head_yaw"])
        pitch_score = self._score_from_ranges(pitch, self.posture_params["max_head_pitch"])
        roll_score = self._score_from_ranges(roll, self.posture_params["max_head_roll"])
        
        # 平均分数
        posture_score = (yaw_score + pitch_score + roll_score) / 3
        
        # 考虑置信度
        if hasattr(head_pose, 'confidence'):
            posture_score *= head_pose.confidence
        
        return max(0, min(100, posture_score))
    
    def _calculate_gaze_score(self, gaze_direction) -> float:
        """计算视线分数"""
        if gaze_direction is None:
            return 50.0
        
        h = abs(gaze_direction.horizontal)
        v = abs(gaze_direction.vertical)
        
        # 计算角度
        angle = np.sqrt(h**2 + v**2) * 90  # 转换为近似角度
        
        # 视线评分
        front_angle = self.gaze_params["front_cone_angle"]
        side_angle = self.gaze_params["side_cone_angle"]
        
        if angle <= front_angle:
            gaze_score = 100.0
        elif angle <= side_angle:
            gaze_score = 70.0
        else:
            gaze_score = 30.0
        
        # 考虑置信度
        if hasattr(gaze_direction, 'confidence'):
            gaze_score *= gaze_direction.confidence
        
        return max(0, min(100, gaze_score))
    
    def _calculate_expression_score(self, landmarks) -> float:
        """计算表情分数"""
        if landmarks is None:
            return 50.0
        
        # 眼睛开合度
        eye_score = self._calculate_eye_openness(landmarks)
        
        # 打哈欠检测
        yawn_score = self._detect_yawn(landmarks)
        
        # 眨眼频率 (简化计算)
        blink_score = self._calculate_blink_score(landmarks)
        
        # 综合表情分数
        expression_score = (eye_score * 0.4 + yawn_score * 0.4 + blink_score * 0.2)
        
        return max(0, min(100, expression_score))
    
    def _calculate_eye_openness(self, landmarks) -> float:
        """计算眼睛开合度"""
        try:
            # 左右眼垂直距离
            left_eye_top = landmarks[159] if 159 < len(landmarks) else landmarks[0]
            left_eye_bottom = landmarks[145] if 145 < len(landmarks) else landmarks[0]
            right_eye_top = landmarks[386] if 386 < len(landmarks) else landmarks[0]
            right_eye_bottom = landmarks[374] if 374 < len(landmarks) else landmarks[0]
            
            left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
            right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
            
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            
            # 简单的归一化
            threshold = self.expression_params["eye_aspect_threshold"]
            openness = min(1.0, avg_eye_height / (threshold * 100))
            
            if openness >= threshold:
                return 100.0
            elif openness >= threshold * 0.6:
                return 60.0
            else:
                return 20.0
                
        except Exception:
            return 50.0
    
    def _detect_yawn(self, landmarks) -> float:
        """检测打哈欠"""
        try:
            # 嘴巴垂直距离
            mouth_top = landmarks[13] if 13 < len(landmarks) else landmarks[0]
            mouth_bottom = landmarks[14] if 14 < len(landmarks) else landmarks[0]
            
            mouth_open = abs(mouth_top[1] - mouth_bottom[1])
            
            # 简单的阈值判断
            threshold = self.expression_params["yawn_threshold"] * 100
            
            if mouth_open > threshold * 1.5:
                return 20.0  # 正在打哈欠
            elif mouth_open > threshold:
                return 60.0  # 嘴巴微张
            else:
                return 100.0  # 正常
                
        except Exception:
            return 50.0
    
    def _calculate_blink_score(self, landmarks) -> float:
        """计算眨眼分数 (简化版)"""
        # 这里可以基于历史数据计算眨眼频率
        # 简化：假设正常眨眼频率为15-25次/分钟
        # 由于实时计算复杂，这里返回固定分数
        return 80.0
    
    def _calculate_temporal_score(self, student_id: str) -> float:
        """计算时序分数"""
        if student_id not in self.student_states:
            return 80.0
        
        state = self.student_states[student_id]
        
        # 持续专注时间评分
        attention_span = state.attention_span
        if attention_span >= 60:
            span_score = 100.0
        elif attention_span >= 45:
            span_score = 90.0
        elif attention_span >= 30:
            span_score = 80.0
        elif attention_span >= 15:
            span_score = 60.0
        elif attention_span >= 5:
            span_score = 40.0
        else:
            span_score = 20.0
        
        # 波动性评分
        if len(state.focus_history) > 5:
            scores = [fs.total for fs in state.focus_history]
            variance = np.var(scores)
            
            if variance < 100:
                fluctuation_score = 100.0
            elif variance < 400:
                fluctuation_score = 80.0
            elif variance < 900:
                fluctuation_score = 60.0
            else:
                fluctuation_score = 40.0
        else:
            fluctuation_score = 80.0
        
        # 综合时序分数
        temporal_score = (span_score * 0.6 + fluctuation_score * 0.4)
        return temporal_score
    
    def _score_from_ranges(self, value: float, max_value: float) -> float:
        """根据数值范围计算分数"""
        ratio = value / max_value
        
        if ratio <= 0.33:
            return 100.0
        elif ratio <= 0.66:
            return 70.0
        elif ratio <= 1.0:
            return 40.0
        else:
            return 10.0
    
    def get_student_focus_level(self, student_id: str) -> str:
        """获取学生的专注度等级"""
        if student_id not in self.student_states:
            return "unknown"
        
        state = self.student_states[student_id]
        if len(state.focus_history) == 0:
            return "unknown"
        
        latest_score = state.focus_history[-1].total
        
        levels = self.config["focus_levels"]
        
        for level_name, level_info in levels.items():
            min_val, max_val = level_info["range"]
            if min_val <= latest_score <= max_val:
                return level_name
        
        return "unknown"
    
    def get_classroom_statistics(self) -> Dict:
        """获取课堂整体统计"""
        if not self.student_states:
            return {}
        
        total_students = len(self.student_states)
        avg_focus_scores = []
        excellent_count = 0
        distracted_count = 0
        
        for state in self.student_states.values():
            if len(state.focus_history) > 0:
                latest_score = state.focus_history[-1].total
                avg_focus_scores.append(latest_score)
                
                if latest_score >= 85:
                    excellent_count += 1
                elif latest_score < 50:
                    distracted_count += 1
        
        if avg_focus_scores:
            class_avg = np.mean(avg_focus_scores)
            class_std = np.std(avg_focus_scores)
        else:
            class_avg = 0
            class_std = 0
        
        return {
            "total_students": total_students,
            "class_average": round(class_avg, 2),
            "class_std": round(class_std, 2),
            "excellent_count": excellent_count,
            "distracted_count": distracted_count,
            "attention_rate": round(excellent_count / total_students * 100, 2) if total_students > 0 else 0
        }
    
    def get_student_report(self, student_id: str) -> Optional[Dict]:
        """获取学生的详细报告"""
        if student_id not in self.student_states:
            return None
        
        state = self.student_states[student_id]
        
        if len(state.focus_history) == 0:
            return None
        
        # 计算历史统计
        all_scores = [fs.total for fs in state.focus_history]
        posture_scores = [fs.posture for fs in state.focus_history]
        gaze_scores = [fs.gaze for fs in state.focus_history]
        expression_scores = [fs.expression for fs in state.focus_history]
        temporal_scores = [fs.temporal for fs in state.focus_history]
        
        return {
            "student_id": student_id,
            "current_focus": round(state.last_focus_score, 2),
            "average_focus": round(np.mean(all_scores), 2),
            "attention_span": round(state.attention_span, 1),
            "distraction_events": state.distraction_count,
            "yawn_count": state.yawn_count,
            "focus_level": self.get_student_focus_level(student_id),
            "component_scores": {
                "posture": round(np.mean(posture_scores), 2),
                "gaze": round(np.mean(gaze_scores), 2),
                "expression": round(np.mean(expression_scores), 2),
                "temporal": round(np.mean(temporal_scores), 2)
            }
        }