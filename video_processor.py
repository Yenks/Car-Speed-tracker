import cv2
import numpy as np
from threading import Thread, Lock, Event
from queue import Queue
from typing import Optional, Tuple, List
from loguru import logger
from PyQt5.QtCore import QThread  # Import QThread for sleep functionality

class FrameBuffer:
    """Thread-safe frame buffer for video processing."""
    
    def __init__(self, max_size: int = 120):  # Increased buffer size for smoother playback
        self.buffer = Queue(maxsize=max_size)
        self.lock = Lock()
        self.stop_event = Event()
        self.processing_done = Event()  # New event for synchronization
        
    def clear(self):
        """Clear all frames from buffer."""
        with self.lock:
            while not self.buffer.empty():
                self.buffer.get()
                
    def put(self, frame: np.ndarray) -> bool:
        """Add frame to buffer if not full."""
        try:
            self.buffer.put(frame, block=False)
            return True
        except:
            return False
            
    def get(self) -> Optional[np.ndarray]:
        """Get frame from buffer if available."""
        try:
            return self.buffer.get(block=False)
        except:
            return None
            
    def stop(self):
        """Signal to stop processing."""
        self.stop_event.set()
        
    @property
    def should_stop(self) -> bool:
        return self.stop_event.is_set()


class VideoProcessor:
    """Multi-threaded video processor with frame buffering."""
    
    def __init__(self, frame_skip: int = 1, buffer_size: int = 120):  # Increased default buffer size
        self.frame_skip = frame_skip
        self.frame_buffer = FrameBuffer(max_size=buffer_size)
        self.read_thread: Optional[Thread] = None
        self.process_thread: Optional[Thread] = None
        self.current_frame_no = 0
        self.total_frames = 0
        self.fps = 0
        self.current_frame = None
        self.cap = None
        self.detector = None
        self.current_speed = None
        
    def start_processing(self, cap: cv2.VideoCapture, process_fn):
        """Start video processing threads."""
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set OpenCV buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Minimum buffer size for efficient reading
        
        def read_frames():
            """Thread function to read frames from video."""
            frames_to_skip = 0
            
            while not self.frame_buffer.should_stop:
                if self.frame_buffer.buffer.qsize() >= self.frame_buffer.buffer.maxsize * 0.9:
                    # Buffer almost full, wait a bit
                    QThread.msleep(10)
                    continue
                    
                ret, frame = cap.read()
                if not ret:
                    logger.info("Reached end of video")
                    self.frame_buffer.processing_done.set()
                    break
                    
                self.current_frame_no += 1
                
                # Implement dynamic frame skipping
                if frames_to_skip > 0:
                    frames_to_skip -= 1
                    continue
                    
                # Skip frames if needed
                if self.current_frame_no % self.frame_skip != 0:
                    continue
                    
                if not self.frame_buffer.put(frame):
                    frames_to_skip = self.frame_skip  # Skip next batch if buffer full
                    logger.debug("Frame buffer full, implementing dynamic skip")
                    
        def process_frames():
            """Thread function to process buffered frames."""
            while not (self.frame_buffer.should_stop and self.frame_buffer.processing_done.is_set() and self.frame_buffer.buffer.empty()):
                frame = self.frame_buffer.get()
                if frame is None:
                    QThread.msleep(1)  # Small sleep to prevent busy waiting
                    continue
                    
                try:
                    process_fn(frame)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    
        # Start threads
        self.read_thread = Thread(target=read_frames)
        self.process_thread = Thread(target=process_frames)
        
        self.read_thread.start()
        self.process_thread.start()
        
    def stop_processing(self):
        """Stop video processing threads."""
        self.frame_buffer.stop()
        
        if self.read_thread:
            self.read_thread.join()
            
        if self.process_thread:
            self.process_thread.join()
            
        self.frame_buffer.clear()
        
    @property
    def progress(self) -> float:
        """Get current progress as percentage."""
        return (self.current_frame_no / self.total_frames) * 100 if self.total_frames > 0 else 0
        
    def open_video(self, video_path: str) -> bool:
        """Open a video file for processing."""
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return False
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.current_frame_no = 0
            
            # Initialize detector
            from detector import VehicleSpeedDetector
            self.detector = VehicleSpeedDetector()
            
            # Read first frame to get dimensions
            ret, first_frame = self.cap.read()
            if ret:
                height, width = first_frame.shape[:2]
                self.detector.set_video_info(width, self.fps)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            else:
                logger.error("Could not read first frame")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error opening video: {e}")
            return False
            
    def process_next_frame(self) -> bool:
        """Process the next frame from the video."""
        try:
            if self.cap is None or not self.cap.isOpened():
                return False
                
            ret, frame = self.cap.read()
            if not ret:
                return False
                
            self.current_frame_no += 1
            
            # Process only every nth frame based on frame_skip
            if self.current_frame_no % self.frame_skip == 0:
                # Use a smaller frame size for display if the video is large
                h, w = frame.shape[:2]
                if w > 1280:
                    display_frame = cv2.resize(frame, (1280, int(h * (1280/w))))
                else:
                    display_frame = frame
                
                # Process frame with vehicle detector
                processed_frame, speed = self.detector.process_frame(display_frame)
                self.current_frame = processed_frame
                self.current_speed = speed
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return False
            
    def select_vehicle(self, x: int, y: int):
        """Select a vehicle for tracking at the given coordinates."""
        if self.detector:
            self.detector.select_object(cv2.EVENT_LBUTTONDOWN, x, y, None, None)
            
    def toggle_speed_display(self, show: bool):
        """Toggle speed display on/off."""
        if self.detector:
            self.detector.show_speed = show
            
    def get_current_speed(self) -> Optional[float]:
        """Get the current speed of the tracked vehicle."""
        return self.current_speed