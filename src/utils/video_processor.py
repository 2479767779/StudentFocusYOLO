"""
视频处理模块
负责视频流的读取、处理和保存
"""

import cv2
import numpy as np
import time
from typing import Optional, Callable, List, Tuple
import threading
from queue import Queue
import logging

class VideoProcessor:
    """
    视频处理器
    支持摄像头、视频文件和实时流处理
    """
    
    def __init__(self, source: str = "0", 
                 target_fps: int = 30,
                 resolution: Tuple[int, int] = (1280, 720),
                 buffer_size: int = 5):
        """
        初始化视频处理器
        
        Args:
            source: 视频源 (0为摄像头，或文件路径)
            target_fps: 目标帧率
            resolution: 目标分辨率 (width, height)
            buffer_size: 帧缓冲区大小
        """
        self.source = source
        self.target_fps = target_fps
        self.resolution = resolution
        self.buffer_size = buffer_size
        
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=buffer_size)
        self.processing_thread = None
        
        self.logger = logging.getLogger(__name__)
        
        # 性能统计
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
    
    def open(self) -> bool:
        """打开视频源"""
        try:
            # 解析源
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                self.logger.error(f"无法打开视频源: {self.source}")
                return False
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            self.logger.info(f"成功打开视频源: {self.source}")
            return True
            
        except Exception as e:
            self.logger.error(f"打开视频源失败: {e}")
            return False
    
    def start(self, frame_callback: Optional[Callable] = None):
        """
        启动视频处理
        
        Args:
            frame_callback: 每帧处理回调函数
        """
        if not self.open():
            return False
        
        self.is_running = True
        self.start_time = time.time()
        
        # 启动采集线程
        self.processing_thread = threading.Thread(
            target=self._capture_loop,
            args=(frame_callback,),
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info("视频处理已启动")
        return True
    
    def _capture_loop(self, frame_callback: Optional[Callable]):
        """采集循环"""
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                self.logger.warning("视频流结束或读取失败")
                break
            
            # 调整帧大小
            if self.resolution:
                frame = cv2.resize(frame, self.resolution)
            
            # 计算FPS
            self.frame_count += 1
            if self.start_time:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
            
            # 执行回调
            if frame_callback:
                try:
                    frame = frame_callback(frame, self.frame_count, self.fps)
                except Exception as e:
                    self.logger.error(f"帧回调执行失败: {e}")
            
            # 放入队列
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            self.frame_queue.put(frame)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        获取最新帧
        
        Args:
            timeout: 超时时间
            
        Returns:
            帧数据
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None
    
    def get_frames(self, count: int = 1) -> List[np.ndarray]:
        """
        获取多帧
        
        Args:
            count: 帧数
            
        Returns:
            帧列表
        """
        frames = []
        for _ in range(count):
            frame = self.get_frame()
            if frame is not None:
                frames.append(frame)
        return frames
    
    def save_video(self, output_path: str, 
                   fourcc: str = 'mp4v',
                   fps: Optional[int] = None) -> 'VideoSaver':
        """
        创建视频保存器
        
        Args:
            output_path: 输出路径
            fourcc: 编码器
            fps: 帧率
            
        Returns:
            VideoSaver对象
        """
        return VideoSaver(output_path, self.resolution, fourcc, fps or self.target_fps)
    
    def stop(self):
        """停止视频处理"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        self.logger.info("视频处理已停止")
    
    def get_stats(self) -> dict:
        """获取处理统计"""
        return {
            "frame_count": self.frame_count,
            "fps": round(self.fps, 2),
            "running_time": time.time() - self.start_time if self.start_time else 0
        }

class VideoSaver:
    """视频保存器"""
    
    def __init__(self, output_path: str, 
                 resolution: Tuple[int, int],
                 fourcc: str = 'mp4v',
                 fps: int = 30):
        self.output_path = output_path
        self.resolution = resolution
        self.fps = fps
        
        # 创建视频写入器
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(output_path, fourcc_code, fps, resolution)
        
        self.frame_count = 0
        self.is_open = True
    
    def write(self, frame: np.ndarray) -> bool:
        """写入帧"""
        if not self.is_open:
            return False
        
        try:
            # 确保帧大小匹配
            if frame.shape[:2] != self.resolution[::-1]:
                frame = cv2.resize(frame, self.resolution)
            
            self.writer.write(frame)
            self.frame_count += 1
            return True
        except Exception as e:
            print(f"写入帧失败: {e}")
            return False
    
    def release(self):
        """释放资源"""
        if self.is_open:
            self.writer.release()
            self.is_open = False
            print(f"视频已保存: {self.output_path}, 帧数: {self.frame_count}")

class CameraManager:
    """摄像头管理器"""
    
    @staticmethod
    def list_cameras(max_check: int = 10) -> List[int]:
        """
        列出可用摄像头
        
        Args:
            max_check: 最大检查索引
            
        Returns:
            可用摄像头索引列表
        """
        available = []
        
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        
        return available
    
    @staticmethod
    def get_camera_info(camera_index: int = 0) -> dict:
        """
        获取摄像头信息
        
        Args:
            camera_index: 摄像头索引
            
        Returns:
            摄像头信息字典
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return {}
        
        info = {
            "index": camera_index,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": cap.get(cv2.CAP_PROP_SATURATION),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE)
        }
        
        cap.release()
        return info
    
    @staticmethod
    def set_camera_properties(camera_index: int = 0, 
                            brightness: Optional[float] = None,
                            contrast: Optional[float] = None,
                            exposure: Optional[float] = None) -> bool:
        """
        设置摄像头属性
        
        Args:
            camera_index: 摄像头索引
            brightness: 亮度
            contrast: 对比度
            exposure: 曝光
            
        Returns:
            是否成功
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return False
        
        success = True
        
        if brightness is not None:
            success &= cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        
        if contrast is not None:
            success &= cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        
        if exposure is not None:
            success &= cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        
        cap.release()
        return success