"""
Streamlit Web界面
提供实时专注度监控和数据分析
"""

import streamlit as st
import sys
from pathlib import Path
import cv2
import numpy as np
import time
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection.face_detector import FaceDetector
from src.detection.pose_estimator import PoseEstimator
from src.analysis.focus_analyzer import FocusAnalyzer
from src.analysis.behavior_classifier import BehaviorClassifier
from src.utils.visualization import Visualizer

# 页面配置
st.set_page_config(
    page_title="课堂专注度监控系统",
    page_icon="🎓",
    layout="wide"
)

# 全局状态管理
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.face_detector = None
    st.session_state.pose_estimator = None
    st.session_state.focus_analyzer = None
    st.session_state.behavior_classifier = None
    st.session_state.visualizer = None
    st.session_state.frame_count = 0
    st.session_state.focus_history = []
    st.session_state.behavior_history = []
    st.session_state.running = False

def initialize_system():
    """初始化系统组件"""
    if not st.session_state.initialized:
        with st.spinner("正在初始化系统..."):
            st.session_state.face_detector = FaceDetector(model_type="yolov8n", device="cpu")
            st.session_state.pose_estimator = PoseEstimator()
            st.session_state.focus_analyzer = FocusAnalyzer()
            st.session_state.behavior_classifier = BehaviorClassifier()
            st.session_state.visualizer = Visualizer()
            st.session_state.initialized = True
        st.success("系统初始化完成！")

def process_frame(frame):
    """处理视频帧"""
    try:
        # 人脸检测
        faces = st.session_state.face_detector.detect(frame)
        
        focus_scores = []
        behaviors = []
        
        for i, face in enumerate(faces[:5]):  # 最多处理5个人
            student_id = f"Student_{i+1}"
            
            # 提取关键点
            landmarks = st.session_state.pose_estimator.extract_landmarks(frame, face.bbox)
            
            if landmarks:
                # 估计姿态和视线
                head_pose = st.session_state.pose_estimator.estimate_head_pose(frame, landmarks)
                gaze = st.session_state.pose_estimator.estimate_gaze_direction(landmarks)
                
                # 分析专注度
                focus_score = st.session_state.focus_analyzer.analyze(
                    student_id, head_pose, gaze, landmarks, time.time()
                )
                focus_scores.append(focus_score)
                
                # 分类行为
                behavior_events = st.session_state.behavior_classifier.classify(
                    head_pose, gaze, landmarks, time.time(), student_id
                )
                behaviors.extend(behavior_events)
                
                # 可视化
                frame = st.session_state.visualizer.draw_face_with_landmarks(
                    frame, face.bbox, landmarks, head_pose, gaze
                )
        
        # 更新历史记录
        if focus_scores:
            st.session_state.frame_count += 1
            for score in focus_scores:
                st.session_state.focus_history.append({
                    "timestamp": datetime.now(),
                    "student_id": score.student_id,
                    "total": score.total,
                    "posture": score.posture,
                    "gaze": score.gaze,
                    "expression": score.expression,
                    "temporal": score.temporal
                })
            
            for behavior in behaviors:
                st.session_state.behavior_history.append({
                    "timestamp": datetime.now(),
                    "student_id": behavior.behavior_type.value,
                    "behavior": behavior.behavior_type.value,
                    "confidence": behavior.confidence
                })
        
        # 绘制仪表板
        if focus_scores:
            classroom_stats = st.session_state.focus_analyzer.get_classroom_statistics()
            frame = st.session_state.visualizer.draw_focus_dashboard(
                frame, focus_scores, behaviors, classroom_stats
            )
        
        return frame, focus_scores, behaviors
        
    except Exception as e:
        st.error(f"处理帧时出错: {e}")
        return frame, [], []

def main():
    """主界面"""
    st.title("🎓 课堂学生专注度监控系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("系统控制")
        
        if st.button("🚀 初始化系统", type="primary"):
            initialize_system()
        
        st.markdown("---")
        
        # 摄像头选择
        camera_index = st.number_input("摄像头索引", min_value=0, max_value=9, value=0)
        
        # 模式选择
        mode = st.radio("运行模式", ["实时监控", "视频文件", "数据分析"])
        
        st.markdown("---")
        
        # 统计信息
        if st.session_state.initialized:
            st.metric("处理帧数", st.session_state.frame_count)
            if st.session_state.focus_history:
                recent_scores = [h["total"] for h in st.session_state.focus_history[-10:]]
                if recent_scores:
                    avg_recent = sum(recent_scores) / len(recent_scores)
                    st.metric("最近平均专注度", f"{avg_recent:.1f}")
        
        st.markdown("---")
        
        # 导出数据
        if st.button("📊 导出数据"):
            if st.session_state.focus_history:
                df = pd.DataFrame(st.session_state.focus_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    "下载专注度数据",
                    csv,
                    "focus_data.csv",
                    "text/csv"
                )
            
            if st.session_state.behavior_history:
                df_behavior = pd.DataFrame(st.session_state.behavior_history)
                csv_behavior = df_behavior.to_csv(index=False)
                st.download_button(
                    "下载行为数据",
                    csv_behavior,
                    "behavior_data.csv",
                    "text/csv"
                )
    
    # 主内容区
    if not st.session_state.initialized:
        st.info("请先点击侧边栏的'初始化系统'按钮来启动系统")
        return
    
    if mode == "实时监控":
        show_realtime_monitor(camera_index)
    elif mode == "视频文件":
        show_video_analysis()
    else:
        show_data_analysis()

def show_realtime_monitor(camera_index):
    """实时监控模式"""
    st.subheader("📹 实时监控")
    
    # 启动/停止按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 开始监控", type="primary") and not st.session_state.running:
            st.session_state.running = True
            st.session_state.cap = cv2.VideoCapture(camera_index)
    
    with col2:
        if st.button("⏹️ 停止监控") and st.session_state.running:
            st.session_state.running = False
            if 'cap' in st.session_state:
                st.session_state.cap.release()
    
    if st.session_state.running:
        # 视频显示区域
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # 实时处理循环
        cap = st.session_state.cap
        
        if not cap.isOpened():
            st.error("无法打开摄像头")
            st.session_state.running = False
            return
        
        # 读取并处理帧
        ret, frame = cap.read()
        if ret:
            # 调整大小
            frame = cv2.resize(frame, (1280, 720))
            
            # 处理帧
            processed_frame, focus_scores, behaviors = process_frame(frame)
            
            # 转换为RGB并显示
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # 显示统计
            with stats_placeholder.container():
                if focus_scores:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("检测人数", len(focus_scores))
                    
                    with col2:
                        avg_focus = sum(s.total for s in focus_scores) / len(focus_scores)
                        st.metric("平均专注度", f"{avg_focus:.1f}")
                    
                    with col3:
                        distracted = sum(1 for s in focus_scores if s.total < 50)
                        st.metric("分心人数", distracted)
                    
                    # 显示详细信息
                    st.markdown("#### 学生详情")
                    for score in focus_scores:
                        st.progress(int(score.total) / 100, text=f"{score.student_id}: {score.total:.1f}")
                
                if behaviors:
                    st.markdown("#### 检测行为")
                    for behavior in behaviors[:5]:
                        st.info(f"{behavior.behavior_type.value} (置信度: {behavior.confidence:.2f})")
        
        # 自动刷新
        time.sleep(0.05)  # 20 FPS
        st.rerun()

def show_video_analysis():
    """视频文件分析模式"""
    st.subheader("📁 视频文件分析")
    
    uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # 保存临时文件
        temp_path = Path("temp_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("开始分析"):
            with st.spinner("正在分析视频..."):
                # 这里可以调用evaluate.py的逻辑
                st.success("视频分析功能需要在命令行模式下运行 evaluate.py")
                st.info(f"视频已保存到: {temp_path}")
                st.markdown("```bash\npython scripts/evaluate.py --video temp_video.mp4\n```")

def show_data_analysis():
    """数据分析模式"""
    st.subheader("📊 数据分析")
    
    if not st.session_state.focus_history:
        st.info("暂无数据，请先运行实时监控或上传视频分析")
        return
    
    # 转换为DataFrame
    df_focus = pd.DataFrame(st.session_state.focus_history)
    df_behavior = pd.DataFrame(st.session_state.behavior_history)
    
    # 专注度趋势图
    st.markdown("#### 专注度趋势")
    if len(df_focus) > 1:
        fig_trend = px.line(
            df_focus, 
            x="timestamp", 
            y="total", 
            color="student_id",
            title="专注度随时间变化"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # 分数组成
    st.markdown("#### 分数组成")
    if len(df_focus) > 0:
        components = df_focus[["posture", "gaze", "expression", "temporal"]].mean()
        fig_components = px.bar(
            x=components.index,
            y=components.values,
            title="各维度平均分数",
            labels={"x": "维度", "y": "分数"}
        )
        st.plotly_chart(fig_components, use_container_width=True)
    
    # 行为分布
    if len(df_behavior) > 0:
        st.markdown("#### 行为分布")
        behavior_counts = df_behavior["behavior"].value_counts()
        fig_behavior = px.pie(
            values=behavior_counts.values,
            names=behavior_counts.index,
            title="行为类型分布"
        )
        st.plotly_chart(fig_behavior, use_container_width=True)
    
    # 统计摘要
    st.markdown("#### 统计摘要")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总记录数", len(df_focus))
    
    with col2:
        avg_focus = df_focus["total"].mean() if len(df_focus) > 0 else 0
        st.metric("平均专注度", f"{avg_focus:.1f}")
    
    with col3:
        if len(df_focus) > 0:
            excellent = (df_focus["total"] >= 85).sum()
            st.metric("优秀专注度", f"{excellent}次")
    
    # 原始数据表格
    with st.expander("查看原始数据"):
        st.dataframe(df_focus, use_container_width=True)

if __name__ == "__main__":
    main()