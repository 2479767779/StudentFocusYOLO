"""
行为分类模块
识别和分类学生的各种行为模式
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class BehaviorType(Enum):
    """行为类型枚举"""
    FOCUSED = "focused"
    SLIGHTLY_DISTRACTED = "slightly_distracted"
    DISTRACTED = "distracted"
    SLEEPING = "sleeping"
    PHONE_USAGE = "phone_usage"
    TALKING = "talking"
    YAWNING = "yawning"

@dataclass
class BehaviorEvent:
    """行为事件数据类"""
    behavior_type: BehaviorType
    start_time: float
    end_time: Optional[float] = None
    confidence: float = 1.0
    duration: float = 0.0

class BehaviorClassifier:
    """
    行为分类器
    基于多维度特征识别学生的具体行为模式
    """
    
    def __init__(self):
        """初始化行为分类器"""
        self.active_behaviors: Dict[str, List[BehaviorEvent]] = {}
        self.behavior_history: Dict[str, List[BehaviorEvent]] = {}
        
        # 行为检测阈值
        self.sleep_threshold = 10.0  # 秒
        self.phone_threshold = 5.0   # 秒
        self.yawn_threshold = 3.0    # 秒
        self.distraction_threshold = 3.0  # 秒
    
    def classify(self, 
                 head_pose,
                 gaze_direction,
                 landmarks,
                 timestamp: float,
                 student_id: str = "default") -> List[BehaviorEvent]:
        """
        分类当前行为
        
        Args:
            head_pose: 头部姿态
            gaze_direction: 视线方向
            landmarks: 面部关键点
            timestamp: 时间戳
            student_id: 学生ID
            
        Returns:
            行为事件列表
        """
        events = []
        
        # 检测睡眠/严重分心
        if self._detect_sleeping(head_pose, landmarks):
            events.append(BehaviorEvent(
                behavior_type=BehaviorType.SLEEPING,
                start_time=timestamp,
                confidence=0.9
            ))
        
        # 检测手机使用
        elif self._detect_phone_usage(head_pose, gaze_direction):
            events.append(BehaviorEvent(
                behavior_type=BehaviorType.PHONE_USAGE,
                start_time=timestamp,
                confidence=0.8
            ))
        
        # 检测打哈欠
        if self._detect_yawning(landmarks):
            events.append(BehaviorEvent(
                behavior_type=BehaviorType.YAWNING,
                start_time=timestamp,
                confidence=0.7
            ))
        
        # 检测说话 (基于嘴巴运动)
        if self._detect_talking(landmarks):
            events.append(BehaviorEvent(
                behavior_type=BehaviorType.TALKING,
                start_time=timestamp,
                confidence=0.6
            ))
        
        # 如果没有检测到特殊行为，判断专注度等级
        if not events:
            focus_level = self._assess_focus_level(head_pose, gaze_direction)
            if focus_level == "focused":
                events.append(BehaviorEvent(
                    behavior_type=BehaviorType.FOCUSED,
                    start_time=timestamp,
                    confidence=0.9
                ))
            elif focus_level == "slightly_distracted":
                events.append(BehaviorEvent(
                    behavior_type=BehaviorType.SLIGHTLY_DISTRACTED,
                    start_time=timestamp,
                    confidence=0.7
                ))
            else:
                events.append(BehaviorEvent(
                    behavior_type=BehaviorType.DISTRACTED,
                    start_time=timestamp,
                    confidence=0.6
                ))
        
        # 更新行为历史
        self._update_behavior_history(student_id, events, timestamp)
        
        return events
    
    def _detect_sleeping(self, head_pose, landmarks) -> bool:
        """检测睡眠状态"""
        if head_pose is None or landmarks is None:
            return False
        
        # 头部严重低垂
        if head_pose.pitch > 30:  # 低头超过30度
            # 眼睛闭合检测
            if self._is_eyes_closed(landmarks):
                return True
        
        return False
    
    def _detect_phone_usage(self, head_pose, gaze_direction) -> bool:
        """检测手机使用"""
        if head_pose is None or gaze_direction is None:
            return False
        
        # 头部向下且视线向下
        if head_pose.pitch > 15 and gaze_direction.vertical < -0.3:
            return True
        
        # 头部偏转 + 视线向下
        if abs(head_pose.yaw) > 20 and gaze_direction.vertical < -0.2:
            return True
        
        return False
    
    def _detect_yawning(self, landmarks) -> bool:
        """检测打哈欠"""
        if landmarks is None:
            return False
        
        try:
            # 嘴巴垂直距离
            mouth_top = landmarks[13] if 13 < len(landmarks) else None
            mouth_bottom = landmarks[14] if 14 < len(landmarks) else None
            
            if mouth_top and mouth_bottom:
                mouth_open = abs(mouth_top[1] - mouth_bottom[1])
                
                # 简单阈值
                if mouth_open > 15:  # 嘴巴张开较大
                    return True
        except Exception:
            pass
        
        return False
    
    def _detect_talking(self, landmarks) -> bool:
        """检测说话"""
        if landmarks is None:
            return False
        
        try:
            # 嘴巴水平距离
            mouth_left = landmarks[61] if 61 < len(landmarks) else None
            mouth_right = landmarks[291] if 291 < len(landmarks) else None
            
            if mouth_left and mouth_right:
                mouth_width = abs(mouth_right[0] - mouth_left[0])
                
                # 说话时嘴巴会开合变化
                # 这里简化判断：嘴巴宽度大于某个阈值
                if mouth_width > 30:
                    return True
        except Exception:
            pass
        
        return False
    
    def _is_eyes_closed(self, landmarks) -> bool:
        """判断眼睛是否闭合"""
        try:
            # 左眼
            left_eye_top = landmarks[159] if 159 < len(landmarks) else None
            left_eye_bottom = landmarks[145] if 145 < len(landmarks) else None
            
            # 右眼
            right_eye_top = landmarks[386] if 386 < len(landmarks) else None
            right_eye_bottom = landmarks[374] if 374 < len(landmarks) else None
            
            if left_eye_top and left_eye_bottom and right_eye_top and right_eye_bottom:
                left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
                right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
                
                avg_eye_height = (left_eye_height + right_eye_height) / 2
                
                # 眼睛高度小于阈值认为闭合
                return avg_eye_height < 2
        except Exception:
            pass
        
        return False
    
    def _assess_focus_level(self, head_pose, gaze_direction) -> str:
        """评估专注度等级"""
        if head_pose is None or gaze_direction is None:
            return "distracted"
        
        # 头部姿态检查
        head_ok = (abs(head_pose.yaw) < 20 and 
                  abs(head_pose.pitch) < 15 and 
                  abs(head_pose.roll) < 15)
        
        # 视线检查
        gaze_ok = (abs(gaze_direction.horizontal) < 0.5 and 
                  abs(gaze_direction.vertical) < 0.5)
        
        if head_ok and gaze_ok:
            return "focused"
        elif head_ok or gaze_ok:
            return "slightly_distracted"
        else:
            return "distracted"
    
    def _update_behavior_history(self, student_id: str, events: List[BehaviorEvent], timestamp: float):
        """更新行为历史记录"""
        if student_id not in self.active_behaviors:
            self.active_behaviors[student_id] = []
            self.behavior_history[student_id] = []
        
        # 结束之前的行为
        for active_event in self.active_behaviors[student_id]:
            if active_event.end_time is None:
                active_event.end_time = timestamp
                active_event.duration = timestamp - active_event.start_time
                self.behavior_history[student_id].append(active_event)
        
        # 开始新的行为
        self.active_behaviors[student_id] = events
    
    def get_current_behaviors(self, student_id: str) -> List[BehaviorEvent]:
        """获取当前活跃的行为"""
        return self.active_behaviors.get(student_id, [])
    
    def get_behavior_summary(self, student_id: str, time_window: float = 300) -> Dict:
        """
        获取行为摘要
        
        Args:
            student_id: 学生ID
            time_window: 时间窗口(秒)
            
        Returns:
            行为摘要
        """
        if student_id not in self.behavior_history:
            return {}
        
        current_time = time.time()
        recent_events = [
            event for event in self.behavior_history[student_id]
            if current_time - event.end_time <= time_window
        ]
        
        if not recent_events:
            return {}
        
        # 统计各类行为时长
        behavior_durations = {}
        for event in recent_events:
            behavior_type = event.behavior_type.value
            if behavior_type not in behavior_durations:
                behavior_durations[behavior_type] = 0
            behavior_durations[behavior_type] += event.duration
        
        # 计算占比
        total_duration = sum(behavior_durations.values())
        behavior_percentages = {
            behavior: (duration / total_duration * 100)
            for behavior, duration in behavior_durations.items()
        }
        
        # 找出主要行为
        main_behavior = max(behavior_durations, key=behavior_durations.get) if behavior_durations else "unknown"
        
        return {
            "total_duration": total_duration,
            "behavior_percentages": behavior_percentages,
            "main_behavior": main_behavior,
            "event_count": len(recent_events)
        }
    
    def detect_anomalies(self, student_id: str) -> List[str]:
        """
        检测异常行为
        
        Args:
            student_id: 学生ID
            
        Returns:
            异常行为列表
        """
        if student_id not in self.behavior_history:
            return []
        
        anomalies = []
        recent_events = self.behavior_history[student_id][-10:]  # 最近10个事件
        
        # 统计异常频率
        sleep_count = sum(1 for e in recent_events if e.behavior_type == BehaviorType.SLEEPING)
        phone_count = sum(1 for e in recent_events if e.behavior_type == BehaviorType.PHONE_USAGE)
        yawn_count = sum(1 for e in recent_events if e.behavior_type == BehaviorType.YAWNING)
        
        if sleep_count >= 2:
            anomalies.append("频繁打瞌睡")
        
        if phone_count >= 2:
            anomalies.append("疑似使用手机")
        
        if yawn_count >= 3:
            anomalies.append("频繁打哈欠，可能疲劳")
        
        return anomalies
    
    def visualize_behaviors(self, frame: np.ndarray, 
                           behaviors: List[BehaviorEvent],
                           position: Tuple[int, int] = (10, 150)) -> np.ndarray:
        """
        可视化行为信息
        
        Args:
            frame: 原始图像
            behaviors: 行为事件列表
            position: 显示位置
            
        Returns:
            可视化图像
        """
        vis_frame = frame.copy()
        x, y = position
        
        # 行为颜色映射
        color_map = {
            BehaviorType.FOCUSED: (0, 255, 0),
            BehaviorType.SLIGHTLY_DISTRACTED: (255, 255, 0),
            BehaviorType.DISTRACTED: (255, 165, 0),
            BehaviorType.SLEEPING: (255, 0, 0),
            BehaviorType.PHONE_USAGE: (138, 43, 226),
            BehaviorType.TALKING: (0, 165, 255),
            BehaviorType.YAWNING: (255, 20, 147)
        }
        
        # 显示行为标签
        for i, behavior in enumerate(behaviors):
            label = behavior.behavior_type.value.upper()
            color = color_map.get(behavior.behavior_type, (255, 255, 255))
            
            cv2.putText(vis_frame, label, 
                       (x, y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame