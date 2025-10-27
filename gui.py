from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QStatusBar, QSlider
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from typing import Optional, Tuple
from loguru import logger
from config import Config
import sys

class VideoThread(QThread):
    """Thread for video processing to keep UI responsive."""
    frame_ready = pyqtSignal(np.ndarray)
    progress_update = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_processor):
        super().__init__()
        self.video_processor = video_processor
        self.running = False
        self.base_fps = video_processor.fps if video_processor else 30
        self.fps = self.base_fps  # Current fps will be modified by speed control
        
    def run(self):
        """Main thread loop for video processing."""
        self.running = True
        try:
            while self.running:
                if not self.video_processor.process_next_frame():
                    logger.info("Reached end of video")
                    break
                    
                frame = self.video_processor.current_frame
                if frame is not None:
                    # Convert BGR to RGB for Qt
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_ready.emit(frame_rgb)
                    
                progress = int(self.video_processor.progress)
                self.progress_update.emit(progress)
                
                # Control playback speed based on fps
                delay = int(1000 / self.fps) if self.fps > 0 else 33  # Default to ~30fps if fps is 0
                QThread.msleep(delay)
                
        except Exception as e:
            logger.error(f"Error in video processing thread: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.running = False
            
    def stop(self):
        """Stop video processing."""
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.config = Config.load()
        self.init_ui()
        self.video_thread: Optional[VideoThread] = None
        self.video_processor = None  # Will be initialized when video is loaded
        self.last_click_pos = None
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Vehicle Speed Detector")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add mouse click handling for vehicle selection
        self.mousePressEvent = self.handle_mouse_press
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # File selection
        self.file_btn = QPushButton("Open Video")
        self.file_btn.clicked.connect(self.select_video)
        controls_layout.addWidget(self.file_btn)
        
        # Playback controls
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)
        
        self.restart_btn = QPushButton("Restart")
        self.restart_btn.clicked.connect(self.restart_video)
        controls_layout.addWidget(self.restart_btn)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Playback Speed:")
        speed_layout.addWidget(speed_label)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(25)  # 0.25x speed
        self.speed_slider.setMaximum(400)  # 4x speed
        self.speed_slider.setValue(100)    # 1x speed
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(25)
        self.speed_slider.valueChanged.connect(self.update_playback_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_label)
        
        controls_layout.addLayout(speed_layout)
        
        # Speed unit selection
        self.speed_unit = QComboBox()
        self.speed_unit.addItems(["km/h", "mph"])
        self.speed_unit.setCurrentText(self.config.speed_unit)
        self.speed_unit.currentTextChanged.connect(self.update_speed_unit)
        controls_layout.addWidget(self.speed_unit)
        
        # Distance calibration
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setValue(self.config.known_distance_m)
        self.distance_spin.setSuffix(" m")
        self.distance_spin.valueChanged.connect(self.update_distance)
        controls_layout.addWidget(self.distance_spin)
        
        layout.addLayout(controls_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Settings
        self.create_settings_widget(layout)
        
    def create_settings_widget(self, parent_layout):
        """Create collapsible settings panel."""
        settings_widget = QWidget()
        settings_layout = QHBoxLayout(settings_widget)
        
        # Frame skip
        skip_label = QLabel("Frame Skip:")
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 10)
        self.frame_skip_spin.setValue(self.config.frame_skip)
        self.frame_skip_spin.valueChanged.connect(self.update_frame_skip)
        settings_layout.addWidget(skip_label)
        settings_layout.addWidget(self.frame_skip_spin)
        
        # Buffer size
        buffer_label = QLabel("Buffer Size:")
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(10, 100)
        self.buffer_spin.setValue(self.config.buffer_size)
        self.buffer_spin.valueChanged.connect(self.update_buffer_size)
        settings_layout.addWidget(buffer_label)
        settings_layout.addWidget(self.buffer_spin)
        
        # Show speed toggle
        self.show_speed_check = QCheckBox("Show Speed")
        self.show_speed_check.setChecked(self.config.show_speed)
        self.show_speed_check.stateChanged.connect(self.toggle_speed_display)
        settings_layout.addWidget(self.show_speed_check)
        
        settings_layout.addStretch()
        parent_layout.addWidget(settings_widget)
        
    def select_video(self):
        """Open file dialog to select video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Initialize video processor
                from video_processor import VideoProcessor
                self.video_processor = VideoProcessor(
                    frame_skip=self.config.frame_skip,
                    buffer_size=self.config.buffer_size
                )
                
                # Open the video file
                if self.video_processor.open_video(file_path):
                    self.config.video_path = file_path
                    self.config.save()
                    self.status_bar.showMessage(f"Selected video: {file_path}")
                    logger.info(f"Video loaded successfully: {file_path}")
                else:
                    self.status_bar.showMessage("Failed to open video file")
                    return
            except Exception as e:
                self.status_bar.showMessage(f"Error loading video: {e}")
                logger.error(f"Failed to load video: {e}")
            
    def toggle_playback(self):
        """Toggle video playback."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.play_btn.setText("Play")
        else:
            self.start_video_processing()
            self.play_btn.setText("Pause")
            
    def restart_video(self):
        """Restart video from beginning."""
        if self.video_thread:
            self.video_thread.stop()
            self.start_video_processing()
            
    def update_speed_unit(self, unit):
        """Update speed unit configuration."""
        self.config.speed_unit = unit
        self.config.save()
        
    def update_distance(self, value):
        """Update known distance configuration."""
        self.config.known_distance_m = value
        self.config.save()
        
    def update_frame_skip(self, value):
        """Update frame skip configuration."""
        self.config.frame_skip = value
        self.config.save()
        
    def update_playback_speed(self, value):
        """Update video playback speed."""
        speed = value / 100.0  # Convert percentage to multiplier
        if self.video_thread:
            self.video_thread.fps = self.video_processor.fps * speed
        self.speed_label.setText(f"{speed:.1f}x")
        
    def update_buffer_size(self, value):
        """Update buffer size configuration."""
        self.config.buffer_size = value
        self.config.save()
        
    def toggle_speed_display(self, state):
        """Toggle speed display configuration."""
        self.config.show_speed = bool(state)
        self.config.save()
        
    def update_frame(self, frame):
        """Update video display with new frame."""
        try:
            h, w = frame.shape[:2]
            bytes_per_line = 3 * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            
            # Get label size
            label_size = self.video_label.size()
            
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
            # Update progress bar
            progress = int(self.video_processor.progress)
            self.progress_bar.setValue(progress)
            
            # Update speed display
            self.update_speed_display()
            
        except Exception as e:
            logger.error(f"Error updating frame: {e}")
            self.status_bar.showMessage(f"Error updating frame: {e}")
        
    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        
    def show_error(self, message):
        """Display error in status bar."""
        self.status_bar.showMessage(f"Error: {message}")
        logger.error(message)
        
    def start_video_processing(self):
        """Start video processing in separate thread."""
        if not self.video_thread:
            self.video_thread = VideoThread(self.video_processor)
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.progress_update.connect(self.update_progress)
            self.video_thread.error_occurred.connect(self.show_error)
            
        self.video_thread.start()
        
    def handle_mouse_press(self, event):
        """Handle mouse press events for vehicle selection."""
        if self.video_label.underMouse() and self.video_processor:
            # Convert global coordinates to label coordinates
            pos = self.video_label.mapFrom(self, event.pos())
            
            # Convert label coordinates to video frame coordinates
            label_size = self.video_label.size()
            frame_size = self.video_processor.current_frame.shape if self.video_processor.current_frame is not None else None
            
            if frame_size is not None:
                scale_x = frame_size[1] / label_size.width()
                scale_y = frame_size[0] / label_size.height()
                
                # Calculate click position in video coordinates
                x = int(pos.x() * scale_x)
                y = int(pos.y() * scale_y)
                
                # Select vehicle at click position
                self.video_processor.select_vehicle(x, y)
                self.status_bar.showMessage(f"Selected vehicle at position ({x}, {y})")
                
    def update_speed_display(self):
        """Update the speed display in the status bar."""
        if self.video_processor:
            speed = self.video_processor.get_current_speed()
            if speed is not None:
                self.status_bar.showMessage(f"Current Speed: {speed:.2f} km/h")
                
    def closeEvent(self, event):
        """Handle application closure."""
        if self.video_thread:
            self.video_thread.stop()
        self.config.save()
        super().closeEvent(event)


def start_gui():
    """Start the GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())