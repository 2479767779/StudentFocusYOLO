"""
主程序入口
实时课堂学生专注度监控系统
"""

import sys
import os
import argparse
import time
import cv2
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.detection.face_detector import FaceDetector
from src.detection.pose_estimator import PoseEstimator
from src.detection.gaze_tracker import GazeTracker
from src.analysis.focus_analyzer import FocusAnalyzer
from src.analysis.behavior_classifier import BehaviorClassifier
from src.utils.video_processor import VideoProcessor
from src.utils.visualization import Visualizer
from src.utils.logger import Logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="课堂学生专注度监控系统")
    
    # 视频源
    parser.add_argument("--source", type=str, default="0", 
                       help="视频源: 0 (摄像头), 或视频文件路径")
    
    # 模型配置
    parser.add_argument("--model", type=str, default="yolov8n",
                       help="YOLO模型类型: yolov8n, yolov8s, yolov8m")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备: cuda, cpu")
    
    # 处理参数
    parser.add_argument("--img-size", type=int, default=640,
                       help="推理图像尺寸")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                       help="置信度阈值")
    parser.add_argument("--iou-threshold", type=float, default=0.45,
                       help="IoU阈值")
    
    # 显示参数
    parser.add_argument("--display", action="store_true",
                       help="显示实时画面")
    parser.add_argument("--save-results", action="store_true",
                       help="保存检测结果")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="输出目录")
    
    # 性能参数
    parser.add_argument("--skip-frames", type=int, default=0,
                       help="跳帧处理，0表示不跳帧")
    parser.add_argument("--max-students", type=int, default=10,
                       help="最大检测学生数")
    
    return parser.parse_args()

class FocusMonitor:
    """专注度监控器"""
    
    def __init__(self, args):
        self.args = args
        
        # 初始化组件
        self.face_detector = FaceDetector(
            model_type=args.model,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            device=args.device
        )
        
        self.pose_estimator = PoseEstimator()
        self.gaze_tracker = GazeTracker()
        self.focus_analyzer = FocusAnalyzer()
        self.behavior_classifier = BehaviorClassifier()
        self.visualizer = Visualizer()
        
        # 视频处理器
        self.video_processor = VideoProcessor(
            source=args.source,
            target_fps=30,
            resolution=(1280, 720)
        )
        
        # 日志器
        self.logger = Logger(log_dir="logs", enable_file=True, enable_console=True)
        
        # 状态管理
        self.frame_count = 0
        self.skip_frames = args.skip_frames
        self.processing_interval = max(1, self.skip_frames + 1)
        
        # 结果保存
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.video_saver = None
        if args.save_results:
            self.video_saver = self.video_processor.save_video(
                str(self.output_dir / "focus_monitoring_output.mp4")
            )
        
        # 统计数据
        self.session_stats = {
            "start_time": time.time(),
            "frames": 0,
            "detections": 0,
            "focus_scores": []
        }
    
    def process_frame(self, frame: np.ndarray, frame_num: int, fps: float) -> np.ndarray:
        """处理单帧图像"""
        self.frame_count += 1
        
        # 跳帧处理
        if self.frame_count % self.processing_interval != 0:
            return frame
        
        # 人脸检测
        faces = self.face_detector.detect(frame)
        
        if not faces:
            # 没有检测到人脸，显示提示
            cv2.putText(frame, "No faces detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # 限制检测人数
        if len(faces) > self.args.max_students:
            faces = faces[:self.args.max_students]
        
        self.session_stats["detections"] += len(faces)
        
        # 处理每个学生
        focus_scores = []
        behaviors = []
        
        for i, face in enumerate(faces):
            student_id = f"Student_{i+1}"
            
            # 提取面部关键点
            landmarks = self.pose_estimator.extract_landmarks(frame, face.bbox)
            
            if landmarks:
                # 估计姿态和视线
                head_pose = self.pose_estimator.estimate_head_pose(frame, landmarks)
                gaze = self.pose_estimator.estimate_gaze_direction(landmarks)
                
                # 分析专注度
                focus_score = self.focus_analyzer.analyze(
                    student_id, head_pose, gaze, landmarks, time.time()
                )
                focus_scores.append(focus_score)
                
                # 分类行为
                behavior_events = self.behavior_classifier.classify(
                    head_pose, gaze, landmarks, time.time(), student_id
                )
                behaviors.extend(behavior_events)
                
                # 记录日志
                self.logger.log_focus_score(student_id, focus_score.total, {
                    "posture": focus_score.posture,
                    "gaze": focus_score.gaze,
                    "expression": focus_score.expression,
                    "temporal": focus_score.temporal
                })
                
                for event in behavior_events:
                    self.logger.log_behavior(student_id, event.behavior_type.value, event.confidence)
                
                # 可视化
                if self.args.display:
                    # 绘制人脸框和关键点
                    frame = self.visualizer.draw_face_with_landmarks(
                        frame, face.bbox, landmarks, head_pose, gaze
                    )
        
        # 获取课堂统计
        classroom_stats = self.focus_analyzer.get_classroom_statistics()
        
        # 显示仪表板
        if self.args.display and focus_scores:
            frame = self.visualizer.draw_focus_dashboard(
                frame, focus_scores, behaviors, classroom_stats
            )
        
        # 显示性能信息
        if self.args.display:
            cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {frame_num}", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (200, 200, 200), 1)
        
        # 保存视频
        if self.video_saver:
            self.video_saver.write(frame)
        
        # 更新统计
        self.session_stats["frames"] += 1
        self.session_stats["focus_scores"].extend([fs.total for fs in focus_scores])
        
        return frame
    
    def run(self):
        """运行监控"""
        self.logger.info("启动课堂专注度监控系统")
        self.logger.info(f"配置: {vars(self.args)}")
        
        # 启动视频处理
        if not self.video_processor.start(self.process_frame):
            self.logger.error("无法启动视频处理器")
            return
        
        print("\n=== 课堂专注度监控系统已启动 ===")
        print("按 'q' 退出")
        print("按 's' 保存当前统计")
        print("按 'r' 重置统计")
        print("=" * 40 + "\n")
        
        try:
            while True:
                # 获取显示帧
                frame = self.video_processor.get_frame(timeout=0.1)
                
                if frame is not None and self.args.display:
                    cv2.imshow("Student Focus Monitor", frame)
                
                # 键盘输入处理
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # 退出
                    break
                elif key == ord('s'):  # 保存统计
                    self.save_statistics()
                elif key == ord('r'):  # 重置统计
                    self.reset_statistics()
                
                # 检查视频源是否结束
                if not self.video_processor.is_running:
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            self.cleanup()
    
    def save_statistics(self):
        """保存统计信息"""
        # 计算会话时长
        duration = time.time() - self.session_stats["start_time"]
        
        # 生成报告
        stats = {
            "duration": duration,
            "frames": self.session_stats["frames"],
            "detections": self.session_stats["detections"],
            "avg_focus": (sum(self.session_stats["focus_scores"]) / 
                         len(self.session_stats["focus_scores"]) 
                         if self.session_stats["focus_scores"] else 0)
        }
        
        # 保存JSON日志
        json_path = self.logger.save_json_logs(
            str(self.output_dir / f"session_{int(time.time())}.json")
        )
        
        # 保存课堂统计
        classroom_stats = self.focus_analyzer.get_classroom_statistics()
        if classroom_stats:
            import json
            stats_path = self.output_dir / f"stats_{int(time.time())}.json"
            with open(stats_path, 'w') as f:
                json.dump({
                    "session_stats": stats,
                    "classroom_stats": classroom_stats
                }, f, indent=2)
            
            print(f"\n统计已保存:")
            print(f"- 日志: {json_path}")
            print(f"- 统计: {stats_path}")
            print(f"- 平均专注度: {stats['avg_focus']:.1f}")
            print(f"- 课堂平均: {classroom_stats.get('class_average', 0):.1f}")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.session_stats = {
            "start_time": time.time(),
            "frames": 0,
            "detections": 0,
            "focus_scores": []
        }
        print("\n统计已重置")
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("正在关闭系统...")
        
        # 停止视频处理
        self.video_processor.stop()
        
        # 保存视频
        if self.video_saver:
            self.video_saver.release()
        
        # 保存最终统计
        self.save_statistics()
        
        # 关闭窗口
        if self.args.display:
            cv2.destroyAllWindows()
        
        # 打印会话摘要
        summary = self.logger.get_session_summary()
        if summary:
            print("\n=== 会话摘要 ===")
            print(f"总日志数: {summary.get('total_logs', 0)}")
            print(f"会话时长: {summary.get('session_duration', 0):.1f}秒")
            
            focus_stats = summary.get('focus_statistics', {})
            if focus_stats:
                print(f"专注度统计: 平均={focus_stats.get('average', 0):.1f}, "
                      f"最高={focus_stats.get('max', 0):.1f}, "
                      f"最低={focus_stats.get('min', 0):.1f}")
            
            behavior_stats = summary.get('behavior_statistics', {})
            if behavior_stats:
                print(f"行为事件: {behavior_stats.get('total_events', 0)}个")
                print(f"行为类型: {behavior_stats.get('unique_behaviors', [])}")
        
        self.logger.info("系统已关闭")
        print("\n系统已安全退出")

def main():
    """主函数"""
    args = parse_args()
    
    # 检查CUDA可用性
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("警告: CUDA不可用，切换到CPU")
                args.device = "cpu"
        except ImportError:
            print("警告: PyTorch未安装，使用CPU")
            args.device = "cpu"
    
    # 创建并运行监控器
    monitor = FocusMonitor(args)
    monitor.run()

if __name__ == "__main__":
    main()