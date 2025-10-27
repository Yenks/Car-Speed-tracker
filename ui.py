import cv2

def display_frame(frame):
    """Display resized frame."""
    screen_res = (1280, 720)
    frame_h, frame_w = frame.shape[:2]
    scale = min(screen_res[0] / frame_w, screen_res[1] / frame_h)
    new_w, new_h = int(frame_w * scale), int(frame_h * scale)
    disp_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow("Vehicle Speed Detector", disp_frame)

def print_controls():
    print("""
🎬 CONTROLS
[SPACE]  Pause/Resume
[R]      Replay video
[←]      Step backward
[→]      Step forward
[F]      Toggle fullscreen
[S]      Toggle speed display
[ESC]    Exit
""")
