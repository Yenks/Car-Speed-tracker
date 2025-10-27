# utils.py
import cv2

def draw_fps(frame, fps, frame_no, total_frames):
    height, width = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_no}/{total_frames}", (width - 250, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame
