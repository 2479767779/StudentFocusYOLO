"""
日志模块
提供统一的日志记录功能
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import os

class Logger:
    """
    日志管理器
    支持文件日志、控制台输出和JSON格式
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 enable_file: bool = True,
                 enable_console: bool = True):
        """
        初始化日志器
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
            enable_file: 启用文件日志
            enable_console: 启用控制台日志
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.enable_file = enable_file
        self.enable_console = enable_console
        
        # 日志级别映射
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.log_level = level_map.get(log_level.upper(), logging.INFO)
        
        # 创建日志器
        self.logger = logging.getLogger("StudentFocus")
        self.logger.setLevel(self.log_level)
        
        # 避免重复添加处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 格式化器
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        if self.enable_file:
            log_file = self.log_dir / f"student_focus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
        
        # 控制台处理器
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)
        
        # JSON日志存储
        self.json_logs: List[Dict] = []
        self.session_start = time.time()
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """记录信息日志"""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        """记录警告日志"""
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict] = None):
        """记录错误日志"""
        self._log(logging.ERROR, message, extra)
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """记录调试日志"""
        self._log(logging.DEBUG, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict] = None):
        """记录严重错误日志"""
        self._log(logging.CRITICAL, message, extra)
    
    def _log(self, level: int, message: str, extra: Optional[Dict] = None):
        """内部日志方法"""
        if extra:
            # 将额外信息格式化为JSON字符串
            extra_str = json.dumps(extra, ensure_ascii=False)
            message = f"{message} | Extra: {extra_str}"
        
        self.logger.log(level, message)
        
        # 保存到JSON日志
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": logging.getLevelName(level),
            "message": message,
            "session_time": time.time() - self.session_start
        }
        if extra:
            log_entry["extra"] = extra
        
        self.json_logs.append(log_entry)
    
    def log_focus_score(self, student_id: str, focus_score: float, 
                       components: Optional[Dict] = None):
        """记录专注度分数"""
        extra = {
            "student_id": student_id,
            "focus_score": focus_score,
            "components": components
        }
        self.info(f"专注度分数更新: {focus_score:.1f}", extra)
    
    def log_behavior(self, student_id: str, behavior: str, 
                    confidence: float = 1.0):
        """记录行为事件"""
        extra = {
            "student_id": student_id,
            "behavior": behavior,
            "confidence": confidence
        }
        self.info(f"检测到行为: {behavior}", extra)
    
    def log_session_stats(self, stats: Dict):
        """记录会话统计"""
        extra = {
            "session_duration": stats.get("duration", 0),
            "total_frames": stats.get("frames", 0),
            "avg_fps": stats.get("fps", 0),
            "total_students": stats.get("students", 0)
        }
        self.info("会话统计", extra)
    
    def save_json_logs(self, output_path: Optional[str] = None) -> str:
        """
        保存JSON格式日志
        
        Args:
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        if not output_path:
            output_path = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_logs, f, ensure_ascii=False, indent=2)
        
        self.info(f"JSON日志已保存: {output_path}")
        return str(output_path)
    
    def get_session_summary(self) -> Dict:
        """获取会话摘要"""
        if not self.json_logs:
            return {}
        
        total_logs = len(self.json_logs)
        level_counts = {}
        focus_scores = []
        behaviors = []
        
        for log in self.json_logs:
            # 统计日志级别
            level = log["level"]
            level_counts[level] = level_counts.get(level, 0) + 1
            
            # 提取专注度分数
            if "focus_score" in log.get("extra", {}):
                focus_scores.append(log["extra"]["focus_score"])
            
            # 提取行为
            if "behavior" in log.get("extra", {}):
                behaviors.append(log["extra"]["behavior"])
        
        summary = {
            "total_logs": total_logs,
            "level_distribution": level_counts,
            "session_duration": time.time() - self.session_start,
            "focus_statistics": {
                "count": len(focus_scores),
                "average": sum(focus_scores) / len(focus_scores) if focus_scores else 0,
                "max": max(focus_scores) if focus_scores else 0,
                "min": min(focus_scores) if focus_scores else 0
            } if focus_scores else {},
            "behavior_statistics": {
                "total_events": len(behaviors),
                "unique_behaviors": list(set(behaviors))
            } if behaviors else {}
        }
        
        return summary

class PerformanceLogger:
    """
    性能日志器
    用于记录和分析系统性能
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            return duration
        return 0.0
    
    def record_metric(self, name: str, value: float):
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_stats(self, name: str) -> Optional[Dict]:
        """获取指标统计"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        values = self.metrics[name]
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
        }
    
    def get_all_stats(self) -> Dict:
        """获取所有指标统计"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def reset(self):
        """重置所有指标"""
        self.metrics = {}
        self.start_times = {}