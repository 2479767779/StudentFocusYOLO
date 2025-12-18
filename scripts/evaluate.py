"""
模型评估脚本
用于评估专注度检测的准确性和性能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import time
import json
from typing import List, Dict
import argparse

from src.detection.face_detector import FaceDetector
from src.detection.pose_estimator import PoseEstimator
from src.analysis.focus_analyzer import FocusAnalyzer
from src.analysis.behavior_classifier import BehaviorClassifier

class FocusEvaluator:
    """专注度评估器"""
    
    def __init__(self, test_video: str, ground_truth: str = None):
        """
        初始化评估器
        
        Args:
            test_video: 测试视频路径
            ground_truth: 真实标签文件路径 (可选)
        """
        self.test_video = test_video
        self.ground_truth = ground_truth
        
        # 初始化组件
        self.face_detector = FaceDetector(model_type="yolov8n", device="cpu")
        self.pose_estimator = PoseEstimator()
        self.focus_analyzer = FocusAnalyzer()
        self.behavior_classifier = BehaviorClassifier()
        
        # 评估结果
        self.results = {
            "video_path": test_video,
            "total_frames": 0,
            "processing_time": 0,
            "detection_stats": {},
            "focus_scores": [],
            "performance_metrics": {}
        }
    
    def run_evaluation(self, output_path: str = None) -> Dict:
        """
        运行评估
        
        Args:
            output_path: 结果输出路径
            
        Returns:
            评估结果
        """
        print(f"开始评估视频: {self.test_video}")
        
        # 打开视频
        cap = cv2.VideoCapture(self.test_video)
        if not cap.isOpened():
            print(f"无法打开视频: {self.test_video}")
            return {}
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {width}x{height}, {fps:.1f} FPS, {total_frames} 帧")
        
        self.results["video_info"] = {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames
        }
        
        # 处理每一帧
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 进度显示
            if frame_idx % 30 == 0:
                print(f"处理进度: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
            
            # 检测人脸
            faces = self.face_detector.detect(frame)
            
            # 处理每个检测到的人脸
            for face in faces:
                # 提取关键点
                landmarks = self.pose_estimator.extract_landmarks(frame, face.bbox)
                
                if landmarks:
                    # 估计姿态和视线
                    head_pose = self.pose_estimator.estimate_head_pose(frame, landmarks)
                    gaze = self.pose_estimator.estimate_gaze_direction(landmarks)
                    
                    # 分析专注度
                    focus_score = self.focus_analyzer.analyze(
                        f"test_{frame_idx}", head_pose, gaze, landmarks, time.time()
                    )
                    
                    # 分类行为
                    behaviors = self.behavior_classifier.classify(
                        head_pose, gaze, landmarks, time.time(), f"test_{frame_idx}"
                    )
                    
                    # 记录结果
                    self.results["focus_scores"].append({
                        "frame": frame_idx,
                        "focus_score": focus_score.total,
                        "components": {
                            "posture": focus_score.posture,
                            "gaze": focus_score.gaze,
                            "expression": focus_score.expression,
                            "temporal": focus_score.temporal
                        },
                        "behaviors": [b.behavior_type.value for b in behaviors]
                    })
            
            frame_idx += 1
            
            # 限制处理帧数（可选）
            if frame_idx >= 300:  # 最多处理300帧
                break
        
        # 计算性能指标
        processing_time = time.time() - start_time
        avg_fps = frame_idx / processing_time
        
        self.results["total_frames"] = frame_idx
        self.results["processing_time"] = processing_time
        self.results["performance_metrics"] = {
            "total_time": processing_time,
            "avg_fps": avg_fps,
            "frame_time_ms": (processing_time / frame_idx) * 1000 if frame_idx > 0 else 0
        }
        
        # 统计检测结果
        if self.results["focus_scores"]:
            scores = [r["focus_score"] for r in self.results["focus_scores"]]
            self.results["detection_stats"] = {
                "total_detections": len(scores),
                "avg_focus": np.mean(scores),
                "std_focus": np.std(scores),
                "min_focus": np.min(scores),
                "max_focus": np.max(scores),
                "excellent_count": sum(1 for s in scores if s >= 85),
                "good_count": sum(1 for s in scores if 70 <= s < 85),
                "average_count": sum(1 for s in scores if 50 <= s < 70),
                "poor_count": sum(1 for s in scores if s < 50)
            }
        
        cap.release()
        
        # 保存结果
        if output_path:
            self.save_results(output_path)
        
        return self.results
    
    def save_results(self, output_path: str):
        """保存评估结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存: {output_path}")
    
    def print_summary(self):
        """打印评估摘要"""
        if not self.results["detection_stats"]:
            print("没有检测结果")
            return
        
        stats = self.results["detection_stats"]
        perf = self.results["performance_metrics"]
        
        print("\n" + "="*50)
        print("评估结果摘要")
        print("="*50)
        
        print(f"\n处理性能:")
        print(f"  总帧数: {self.results['total_frames']}")
        print(f"  处理时间: {perf['total_time']:.2f}秒")
        print(f"  平均FPS: {perf['avg_fps']:.2f}")
        print(f"  单帧耗时: {perf['frame_time_ms']:.1f}ms")
        
        print(f"\n专注度统计:")
        print(f"  检测总数: {stats['total_detections']}")
        print(f"  平均分数: {stats['avg_focus']:.2f}")
        print(f"  分数标准差: {stats['std_focus']:.2f}")
        print(f"  最低分数: {stats['min_focus']:.2f}")
        print(f"  最高分数: {stats['max_focus']:.2f}")
        
        print(f"\n专注度分布:")
        print(f"  优秀 (≥85): {stats['excellent_count']} ({stats['excellent_count']/stats['total_detections']*100:.1f}%)")
        print(f"  良好 (70-84): {stats['good_count']} ({stats['good_count']/stats['total_detections']*100:.1f}%)")
        print(f"  一般 (50-69): {stats['average_count']} ({stats['average_count']/stats['total_detections']*100:.1f}%)")
        print(f"  较差 (<50): {stats['poor_count']} ({stats['poor_count']/stats['total_detections']*100:.1f}%)")
        
        print("="*50)

def compare_videos(video_paths: List[str], output_dir: str = "evaluation"):
    """
    比较多个视频的评估结果
    
    Args:
        video_paths: 视频路径列表
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    all_results = {}
    
    for video_path in video_paths:
        if not Path(video_path).exists():
            print(f"视频不存在: {video_path}")
            continue
        
        evaluator = FocusEvaluator(video_path)
        results = evaluator.run_evaluation()
        
        if results:
            video_name = Path(video_path).stem
            all_results[video_name] = results
            
            # 保存单个结果
            output_path = Path(output_dir) / f"{video_name}_evaluation.json"
            evaluator.save_results(str(output_path))
            
            # 打印摘要
            print(f"\n视频: {video_path}")
            evaluator.print_summary()
    
    # 保存比较结果
    if all_results:
        comparison_path = Path(output_dir) / "comparison.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n比较结果已保存: {comparison_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="专注度评估脚本")
    parser.add_argument("--video", type=str, required=True, help="测试视频路径")
    parser.add_argument("--ground-truth", type=str, help="真实标签文件路径")
    parser.add_argument("--output", type=str, default="evaluation/result.json", help="输出路径")
    parser.add_argument("--compare", type=str, nargs="+", help="比较多个视频")
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较模式
        compare_videos(args.compare)
    else:
        # 单个视频评估
        evaluator = FocusEvaluator(args.video, args.ground_truth)
        results = evaluator.run_evaluation(args.output)
        evaluator.print_summary()

if __name__ == "__main__":
    main()