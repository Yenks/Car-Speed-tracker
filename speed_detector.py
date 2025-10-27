# import cv2
# import numpy as np
# import os

# # --- CONFIGURATION ---
# VIDEO_PATH = "input.mp4"  # Default video
# KNOWN_DISTANCE_M = 10     # Real-world width covered by frame (meters)

# # --- Optional Upload Prompt ---
# if not os.path.exists(VIDEO_PATH):
#     VIDEO_PATH = input("Enter the path of your video file: ").strip()

# # --- Load video ---
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print("Error: Cannot open video file.")
#     exit()

# fps = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# ret, first_frame = cap.read()
# if not ret:
#     print("Error: Could not read video frame.")
#     exit()

# height, width, _ = first_frame.shape
# scale_m_per_px = KNOWN_DISTANCE_M / width

# # --- Background Subtractor ---
# fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)

# # --- Variables ---
# prev_center = None
# selected_box = None
# object_selected = False
# paused = False
# fullscreen = False
# show_speed = True
# frame_no = 0

# # --- Mouse selection callback ---
# def select_object(event, x, y, flags, param):
#     global selected_box, object_selected
#     if event == cv2.EVENT_LBUTTONDOWN:
#         selected_box = (x, y)
#         object_selected = True
#         print(f"✅ Vehicle selected at: {selected_box}")

# # --- Window Setup ---
# cv2.namedWindow("Vehicle Speed Detector", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Vehicle Speed Detector", 1280, 720)
# cv2.setMouseCallback("Vehicle Speed Detector", select_object)

# print("""
# 🎬 Controls:
# [SPACE]  Pause/Resume
# [R]      Replay video
# [←]      Step backward
# [→]      Step forward
# [F]      Toggle fullscreen
# [S]      Toggle speed display
# [ESC]    Exit
# """)

# def display_frame(frame):
#     """Display high-quality resized frame."""
#     screen_res = (1280, 720)
#     frame_h, frame_w = frame.shape[:2]
#     scale = min(screen_res[0] / frame_w, screen_res[1] / frame_h)
#     new_w, new_h = int(frame_w * scale), int(frame_h * scale)
#     disp_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     cv2.imshow("Vehicle Speed Detector", disp_frame)

# while True:
#     if not paused:
#         ret, frame = cap.read()
#         if not ret:
#             print("Video ended.")
#             break
#         frame_no += 1

#         fgmask = fgbg.apply(frame)
#         fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
#         _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         closest_contour = None
#         min_dist = float("inf")

#         for contour in contours:
#             if cv2.contourArea(contour) > 1000:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 center = (x + w // 2, y + h // 2)

#                 if object_selected:
#                     sel_x, sel_y = selected_box
#                     dist = np.linalg.norm(np.array(center) - np.array((sel_x, sel_y)))
#                     if dist < min_dist:
#                         min_dist = dist
#                         closest_contour = (x, y, w, h, center)
#                 else:
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         if object_selected and closest_contour:
#             x, y, w, h, center = closest_contour
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
#             cv2.putText(frame, "Tracking Vehicle", (20, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3, cv2.LINE_AA)

#             if prev_center is not None:
#                 pixel_distance = np.linalg.norm(np.array(center) - np.array(prev_center))
#                 distance_m = pixel_distance * scale_m_per_px
#                 time_s = 1 / fps
#                 speed_mps = distance_m / time_s
#                 speed_kmh = speed_mps * 3.6

#                 if show_speed:
#                     cv2.putText(frame, f"{speed_kmh:.2f} km/h", (x, y - 15),
#                                 cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
#             prev_center = center

#         cv2.putText(frame, f"Frame {frame_no}/{total_frames}", (20, height - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

#         display_frame(frame)

#     # --- Key Controls ---
#     key = cv2.waitKey(20) & 0xFF

#     if key == 27:  # ESC
#         break
#     elif key == 32:  # SPACE = pause/resume
#         paused = not paused
#         print("⏸️ Paused — click to select vehicle." if paused else "▶️ Resumed playback.")
#     elif key in [ord('f'), ord('F')]:
#         fullscreen = not fullscreen
#         if fullscreen:
#             cv2.setWindowProperty("Vehicle Speed Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#         else:
#             cv2.setWindowProperty("Vehicle Speed Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
#     elif key in [ord('r'), ord('R')]:  # Replay
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         frame_no = 0
#         prev_center = None
#         print("🔁 Replaying video...")
#     elif key == 81:  # ← (Left arrow)
#         new_frame = max(frame_no - int(fps), 0)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
#         frame_no = new_frame
#         print("⏪ Rewound a bit.")
#     elif key == 83:  # → (Right arrow)
#         new_frame = min(frame_no + int(fps), total_frames - 1)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
#         frame_no = new_frame
#         print("⏩ Skipped forward.")
#     elif key in [ord('s'), ord('S')]:  # Toggle speed text
#         show_speed = not show_speed
#         print(f"🔢 Speed display: {'ON' if show_speed else 'OFF'}")

# cap.release()
# cv2.destroyAllWindows()
