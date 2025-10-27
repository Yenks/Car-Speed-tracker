from gui import start_gui
from logger import setup_logging
from loguru import logger

def main():
    """Main entry point of the application."""
    try:
        # Initialize logging
        setup_logging()
        logger.info("Starting Vehicle Speed Detection System")
        
        # Start GUI
        start_gui()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()

from detector import VehicleSpeedDetector
from config import Config
from exceptions import VideoLoadError, ModelLoadError, ConfigurationError, ProcessingError
from logger import setup_logging

# --- File selection dialog ---
def select_video() -> Optional[str]:
    """
    Open a file dialog for video selection.
    
    Returns:
        Optional[str]: Selected video file path or None if cancelled
    """
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        
        if file_path:
            if not os.path.exists(file_path):
                logger.error(f"Selected file does not exist: {file_path}")
                return None
                
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    pass
            except IOError as e:
                logger.error(f"Cannot read selected file: {e}")
                return None
                
            return file_path
        else:
            logger.info("No file selected")
            return None
            
    except Exception as e:
        logger.error(f"Error in file selection dialog: {e}")
        return None

# --- Setup logging ---
setup_logging()
logger.info("Starting Vehicle Speed Detection System")

# --- Load configuration ---
config = Config.load()
if not config.validate():
    raise ConfigurationError("Invalid configuration")

# --- Choose video file ---
video_path = config.video_path
if not os.path.exists(video_path):
    logger.warning(f"Default video not found at {video_path}")
    video_path = select_video()

if not video_path:
    logger.error("No video file selected")
    raise VideoLoadError("No video file selected")

try:
    # --- Load video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoLoadError(f"Cannot open video file: {video_path}")
    
    logger.info(f"Successfully loaded video: {video_path}")
except Exception as e:
    logger.error(f"Failed to load video: {e}")
    raise VideoLoadError(f"Failed to load video: {e}")

def detect_fps(cap: cv2.VideoCapture) -> float:
    """
    Detect video FPS from metadata or by sampling frames.
    
    Args:
        cap: OpenCV VideoCapture object
        
    Returns:
        float: Detected FPS value
        
    Raises:
        ProcessingError: If FPS detection fails
    """
    try:
        # Try to get FPS from metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            logger.info(f"FPS from metadata: {fps:.2f}")
            return fps

        # Auto-detect FPS by sampling frames
        logger.warning("FPS metadata missing - auto-detecting FPS...")
        frame_times = []
        start_time = time.time()
        
        for _ in range(10):
            ret, _ = cap.read()
            if not ret:
                break
            frame_times.append(time.time())
        
        if len(frame_times) > 1:
            frame_intervals = [t2 - t1 for t1, t2 in zip(frame_times[:-1], frame_times[1:])]
            fps = 1 / (sum(frame_intervals) / len(frame_intervals))
            logger.info(f"Auto-detected FPS: {fps:.2f}")
        else:
            fps = 30  # fallback
            logger.warning(f"Using fallback FPS: {fps}")
        
        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return fps
        
    except Exception as e:
        logger.error(f"FPS detection failed: {e}")
        raise ProcessingError(f"FPS detection failed: {e}")

# --- Detect FPS ---
try:
    fps = detect_fps(cap)
except ProcessingError as e:
    logger.error(f"Failed to detect FPS: {e}")
    raise

try:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ProcessingError("Invalid frame count detected")
        
    duration = total_frames / fps
    logger.info(f"Video info - FPS: {fps:.2f}, Duration: {duration:.2f}s, Frames: {total_frames}")

    ret, first_frame = cap.read()
    if not ret:
        raise ProcessingError("Could not read first video frame")

    height, width, _ = first_frame.shape
    logger.info(f"Video dimensions: {width}x{height}")

    # --- Scale dynamically ---
    known_distance_m = config.known_distance_m * (width / 1280)  # adjusts for video width
    logger.info(f"Adjusted known distance: {known_distance_m:.2f}m")
    
    try:
        detector = VehicleSpeedDetector(known_distance_m)
        detector.set_video_info(width, fps)
        logger.info("Vehicle detector initialized successfully")
    except Exception as e:
        raise ModelLoadError(f"Failed to initialize vehicle detector: {e}")
        
except Exception as e:
    logger.error(f"Failed to initialize video processing: {e}")
    raise

cv2.namedWindow("Vehicle Speed Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Speed Detector", 1280, 720)
cv2.setMouseCallback("Vehicle Speed Detector", detector.select_object)

paused = False
fullscreen = False
frame_no = 0
start_playback_time = None

print("""
🎮 CONTROLS:
[SPACE] Pause/Resume
[R] Replay
[←] Backward
[→] Forward
[F] Toggle fullscreen
[S] Toggle speed display
[ESC] Exit
""")

def display_frame(frame):
    cv2.imshow("Vehicle Speed Detector", frame)

def process_and_display_frame(frame: np.ndarray):
    """Process and display a single frame with speed detection."""
    try:
        processed_frame, _ = detector.process_frame(frame)
        
        # Add progress information
        cv2.putText(
            processed_frame,
            f"Frame {video_processor.current_frame_no}/{total_frames} ({video_processor.progress:.1f}%)",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2
        )
        
        # Handle real-time synchronization
        if start_playback_time is not None:
            elapsed = time.time() - start_playback_time
            expected_time = video_processor.current_frame_no / fps
            diff = expected_time - elapsed
            if diff > 0:
                time.sleep(diff)
                
        display_frame(processed_frame)
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")

# Initialize video processor with configuration
video_processor = VideoProcessor(
    frame_skip=config.frame_skip,
    buffer_size=config.buffer_size
)

while True:
    if not paused:
        if start_playback_time is None:
            start_playback_time = time.time()
            video_processor.start_processing(cap, process_and_display_frame)
    else:
        video_processor.stop_processing()

    key = cv2.waitKey(1) & 0xFF

    try:
        if key == 27:  # ESC
            logger.info("Exiting application")
            break
        elif key == 32:  # SPACE
            paused = not paused
            if paused:
                video_processor.stop_processing()
                logger.info("Playback paused - click to select vehicle")
            else:
                start_playback_time = time.time() - (video_processor.current_frame_no / fps)
                video_processor.start_processing(cap, process_and_display_frame)
                logger.info("Playback resumed")
                
        elif key in [ord('f'), ord('F')]:
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                "Vehicle Speed Detector",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            )
            logger.info(f"Fullscreen mode: {'enabled' if fullscreen else 'disabled'}")
            
        elif key in [ord('r'), ord('R')]:
            video_processor.stop_processing()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_processor.current_frame_no = 0
            detector.prev_center = None
            start_playback_time = None
            video_processor.start_processing(cap, process_and_display_frame)
            logger.info("Video restarted")
            
        elif key == 81:  # ←
            video_processor.stop_processing()
            new_pos = max(video_processor.current_frame_no - int(fps), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            video_processor.current_frame_no = new_pos
            video_processor.start_processing(cap, process_and_display_frame)
            logger.info("Skipped backward")
            
        elif key == 83:  # →
            video_processor.stop_processing()
            new_pos = min(video_processor.current_frame_no + int(fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            video_processor.current_frame_no = new_pos
            video_processor.start_processing(cap, process_and_display_frame)
            logger.info("Skipped forward")
            
        elif key in [ord('s'), ord('S')]:
            detector.show_speed = not detector.show_speed
            logger.info(f"Speed display: {'enabled' if detector.show_speed else 'disabled'}")
            
    except Exception as e:
        logger.error(f"Error handling keyboard input: {e}")

# Cleanup
try:
    video_processor.stop_processing()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application closed successfully")
except Exception as e:
    logger.error(f"Error during cleanup: {e}")
