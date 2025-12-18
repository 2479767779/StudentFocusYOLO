"""
可视化模块
提供丰富的可视化功能，包括实时显示、图表生成等
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

class Visualizer:
    """
    可视化工具类
    提供多种可视化功能
    """
    
    def __init__(self, 
                 color_scheme: str = "default",
                 font_scale: float = 0.7):
        """
        初始化可视化器
        
        Args:
            color_scheme: 颜色方案
            font_scale: 字体缩放
        """
        self.color_scheme = color_scheme
        self.font_scale = font_scale
        
        # 颜色定义
        self.colors = {
            "excellent": (0, 255, 0),      # 绿色
            "good": (50, 255, 50),         # 浅绿
            "average": (255, 255, 0),      # 黄色
            "poor": (255, 165, 0),         # 橙色
            "critical": (255, 0, 0),       # 红色
            "text": (255, 255, 255),       # 白色
            "background": (0, 0, 0),       # 黑色
            "border": (100, 100, 100)      # 灰色
        }
    
    def draw_focus_dashboard(self, 
                           frame: np.ndarray,
                           focus_scores: List,
                           behaviors: List,
                           classroom_stats: Dict) -> np.ndarray:
        """
        绘制专注度仪表板
        
        Args:
            frame: 原始帧
            focus_scores: 专注度分数列表
            behaviors: 行为列表
            classroom_stats: 课堂统计
            
        Returns:
            可视化帧
        """
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # 绘制顶部信息栏
        vis_frame = self._draw_top_bar(vis_frame, classroom_stats)
        
        # 绘制学生信息
        y_offset = 60
        for i, score in enumerate(focus_scores[:5]):  # 最多显示5个学生
            if i >= 3:  # 只显示前3个详细信息
                break
            vis_frame = self._draw_student_info(vis_frame, score, behaviors[i] if i < len(behaviors) else None, y_offset)
            y_offset += 80
        
        # 绘制底部图表区域
        vis_frame = self._draw_chart_area(vis_frame, focus_scores, y_offset)
        
        return vis_frame
    
    def _draw_top_bar(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """绘制顶部信息栏"""
        height, width = frame.shape[:2]
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 课堂名称
        cv2.putText(frame, "课堂专注度监控系统", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 统计信息
        if stats:
            text = f"学生数: {stats.get('total_students', 0)} | 平均分: {stats.get('class_average', 0)} | 专注率: {stats.get('attention_rate', 0)}%"
            cv2.putText(frame, text, (width - 450, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def _draw_student_info(self, frame: np.ndarray, 
                          score, behavior, y_offset: int) -> np.ndarray:
        """绘制学生信息"""
        height, width = frame.shape[:2]
        
        # 背景框
        x_start, x_end = 10, width - 10
        y_start, y_end = y_offset, y_offset + 70
        
        # 根据分数确定颜色
        color = self._get_focus_color(score.total)
        
        # 绘制边框
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
        
        # 学生ID和分数
        student_id = score.student_id or "Unknown"
        cv2.putText(frame, f"学生: {student_id}", (x_start + 10, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"专注度: {score.total:.1f}", (x_start + 10, y_start + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 分数组成
        cv2.putText(frame, f"姿态: {score.posture:.0f} 视线: {score.gaze:.0f} 表情: {score.expression:.0f}", 
                   (x_start + 200, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 行为状态
        if behavior:
            behavior_text = f"状态: {behavior.behavior_type.value}"
            cv2.putText(frame, behavior_text, (x_start + 200, y_start + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return frame
    
    def _draw_chart_area(self, frame: np.ndarray, scores: List, y_offset: int) -> np.ndarray:
        """绘制图表区域"""
        height, width = frame.shape[:2]
        
        if len(scores) < 2:
            return frame
        
        # 简单的柱状图 - 专注度分布
        chart_height = height - y_offset - 20
        chart_width = width - 20
        chart_x = 10
        chart_y = y_offset + 10
        
        # 绘制背景
        cv2.rectangle(frame, (chart_x, chart_y), 
                     (chart_x + chart_width, chart_y + chart_height), 
                     (50, 50, 50), -1)
        
        # 绘制柱状图
        bar_width = chart_width // len(scores)
        max_score = 100
        
        for i, score in enumerate(scores):
            bar_height = int((score.total / max_score) * chart_height)
            bar_x = chart_x + i * bar_width + 2
            bar_y = chart_y + chart_height - bar_height
            
            color = self._get_focus_color(score.total)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width - 4, chart_y + chart_height), 
                         color, -1)
            
            # 分数标签
            if bar_height > 20:
                cv2.putText(frame, f"{int(score.total)}", 
                           (bar_x + 2, bar_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 标题
        cv2.putText(frame, "专注度趋势", (chart_x + 5, chart_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _get_focus_color(self, score: float) -> Tuple[int, int, int]:
        """根据分数获取颜色"""
        if score >= 85:
            return self.colors["excellent"]
        elif score >= 70:
            return self.colors["good"]
        elif score >= 50:
            return self.colors["average"]
        elif score >= 30:
            return self.colors["poor"]
        else:
            return self.colors["critical"]
    
    def draw_focus_trend_chart(self, 
                             focus_history: List[Dict],
                             output_path: Optional[str] = None) -> plt.Figure:
        """
        绘制专注度趋势图
        
        Args:
            focus_history: 历史数据
            output_path: 输出路径
            
        Returns:
            matplotlib图形
        """
        if not focus_history:
            return None
        
        # 转换为DataFrame
        df = pd.DataFrame(focus_history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 专注度趋势
        if 'timestamp' in df.columns and 'total' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'], unit='s')
            ax1.plot(df['time'], df['total'], marker='o', linewidth=2, markersize=4)
            ax1.set_ylabel('专注度分数')
            ax1.set_title('专注度趋势')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
        
        # 分数组成
        if all(col in df.columns for col in ['posture', 'gaze', 'expression', 'temporal']):
            components = ['posture', 'gaze', 'expression', 'temporal']
            means = [df[comp].mean() for comp in components]
            
            bars = ax2.bar(components, means, color=['#2ecc71', '#3498db', '#f39c12', '#9b59b6'])
            ax2.set_ylabel('平均分数')
            ax2.set_title('各维度平均分数')
            ax2.set_ylim(0, 100)
            
            # 添加数值标签
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{mean:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            return fig
        
        return None
    
    def create_interactive_dashboard(self, 
                                   focus_data: List[Dict],
                                   behavior_data: List[Dict],
                                   output_path: str):
        """
        创建交互式仪表板 (HTML)
        
        Args:
            focus_data: 专注度数据
            behavior_data: 行为数据
            output_path: 输出HTML路径
        """
        if not focus_data:
            return
        
        # 创建DataFrame
        df_focus = pd.DataFrame(focus_data)
        df_behavior = pd.DataFrame(behavior_data)
        
        # 创建图形
        fig = go.Figure()
        
        # 专注度趋势
        if 'timestamp' in df_focus.columns:
            df_focus['datetime'] = pd.to_datetime(df_focus['timestamp'], unit='s')
            
            for student_id in df_focus['student_id'].unique():
                student_data = df_focus[df_focus['student_id'] == student_id]
                fig.add_trace(go.Scatter(
                    x=student_data['datetime'],
                    y=student_data['total'],
                    mode='lines+markers',
                    name=f'学生 {student_id}',
                    hovertemplate='时间: %{x}<br>专注度: %{y:.1f}<extra></extra>'
                ))
        
        # 布局
        fig.update_layout(
            title='课堂专注度实时监控',
            xaxis_title='时间',
            yaxis_title='专注度分数',
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            template='plotly_white'
        )
        
        # 保存HTML
        fig.write_html(output_path)
    
    def draw_face_with_landmarks(self, 
                                frame: np.ndarray,
                                face_bbox: Tuple[int, int, int, int],
                                landmarks: List[Tuple[float, float]],
                                head_pose=None,
                                gaze=None) -> np.ndarray:
        """
        绘制人脸和关键点
        
        Args:
            frame: 原始帧
            face_bbox: 人脸边界框
            landmarks: 关键点
            head_pose: 头部姿态
            gaze: 视线方向
            
        Returns:
            可视化帧
        """
        vis_frame = frame.copy()
        
        # 绘制人脸框
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制关键点
        if landmarks:
            for x, y in landmarks:
                cv2.circle(vis_frame, (int(x), int(y)), 1, (255, 0, 0), -1)
        
        # 绘制姿态信息
        if head_pose:
            cv2.putText(vis_frame, f"Yaw: {head_pose.yaw:.1f}", 
                       (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis_frame, f"Pitch: {head_pose.pitch:.1f}", 
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制视线
        if gaze:
            # 从人脸中心画视线方向
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            end_x = int(center_x + gaze.horizontal * 50)
            end_y = int(center_y + gaze.vertical * 50)
            cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), 
                           (255, 255, 0), 2)
        
        return vis_frame
    
    def draw_statistics_overlay(self, 
                              frame: np.ndarray,
                              stats: Dict,
                              position: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        绘制统计信息覆盖层
        
        Args:
            frame: 原始帧
            stats: 统计信息
            position: 显示位置
            
        Returns:
            可视化帧
        """
        vis_frame = frame.copy()
        x, y = position
        
        # 半透明背景
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (x-5, y-20), (x+350, y+120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        # 显示统计信息
        cv2.putText(vis_frame, f"学生总数: {stats.get('total_students', 0)}", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(vis_frame, f"平均专注度: {stats.get('class_average', 0):.1f}", 
                   (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(vis_frame, f"专注率: {stats.get('attention_rate', 0)}%", 
                   (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.putText(vis_frame, f"分心人数: {stats.get('distracted_count', 0)}", 
                   (x, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        return vis_frame
    
    def create_summary_report(self, 
                            classroom_stats: Dict,
                            student_reports: List[Dict],
                            output_path: str):
        """
        生成总结报告图片
        
        Args:
            classroom_stats: 课堂统计
            student_reports: 学生报告列表
            output_path: 输出路径
        """
        # 创建大图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('课堂专注度分析报告', fontsize=16, fontweight='bold')
        
        # 1. 课堂整体统计
        ax1 = axes[0, 0]
        if classroom_stats:
            labels = ['专注', '分心']
            values = [classroom_stats.get('excellent_count', 0), 
                     classroom_stats.get('distracted_count', 0)]
            colors = ['#2ecc71', '#e74c3c']
            ax1.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
            ax1.set_title('课堂专注度分布')
        
        # 2. 学生排名
        ax2 = axes[0, 1]
        if student_reports:
            students = [r['student_id'] for r in student_reports[:8]]
            scores = [r['average_focus'] for r in student_reports[:8]]
            colors = ['#2ecc71' if s >= 85 else '#3498db' if s >= 70 else '#f39c12' if s >= 50 else '#e74c3c' for s in scores]
            
            bars = ax2.barh(students, scores, color=colors)
            ax2.set_xlabel('平均专注度')
            ax2.set_title('学生专注度排名')
            ax2.set_xlim(0, 100)
            
            # 添加数值标签
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{score:.1f}', va='center')
        
        # 3. 维度分析
        ax3 = axes[1, 0]
        if student_reports:
            # 取第一个学生的详细维度
            first_report = student_reports[0]
            if 'component_scores' in first_report:
                components = list(first_report['component_scores'].keys())
                values = list(first_report['component_scores'].values())
                colors = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']
                
                bars = ax3.bar(components, values, color=colors)
                ax3.set_ylabel('分数')
                ax3.set_title(f'维度分析 (学生: {first_report["student_id"]})')
                ax3.set_ylim(0, 100)
                
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{val:.1f}', ha='center', va='bottom')
        
        # 4. 专注度趋势
        ax4 = axes[1, 1]
        if student_reports:
            # 模拟时间序列数据
            students = [r['student_id'] for r in student_reports[:5]]
            avg_scores = [r['average_focus'] for r in student_reports[:5]]
            
            ax4.scatter(range(len(students)), avg_scores, s=100, alpha=0.6)
            ax4.plot(range(len(students)), avg_scores, alpha=0.3)
            ax4.set_xticks(range(len(students)))
            ax4.set_xticklabels(students, rotation=45)
            ax4.set_ylabel('专注度')
            ax4.set_title('学生专注度散点图')
            ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()