import cv2
import numpy as np

class VehicleSpeedDetector:
    def __init__(self, known_distance_m=10):
        self.known_distance_m = known_distance_m
        self.prev_centers = []  # Store multiple previous centers for smoothing
        self.prev_speeds = []   # Store previous speeds for averaging
        self.scale_m_per_px = None
        self.fps = None
        self.show_speed = True
        self.object_selected = False
        self.selected_box = None
        self.speed_window = 5   # Number of frames to average speed over
        self.max_speed_change = 10.0  # Maximum allowed speed change between frames (km/h)
        # Optimized background subtractor parameters
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=100,  # Reduced history for faster processing
            varThreshold=40,  # Adjusted for better detection
            detectShadows=False  # Disable shadow detection for performance
        )
        # Cache for frame processing
        self.frame_size = None
        self.gaussian_kernel = (5, 5)
        self.min_contour_area = 500  # Reduced minimum area for better tracking

    def set_video_info(self, frame_width, fps):
        # Calculate scale with perspective consideration
        # Assume standard 60-degree horizontal field of view
        fov_horizontal = 60  # degrees
        viewing_distance = frame_width / (2 * np.tan(np.radians(fov_horizontal/2)))
        self.scale_m_per_px = self.known_distance_m / (frame_width * np.cos(np.radians(30)))  # Assume 30-degree camera tilt
        self.fps = fps

    def select_object(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_box = (x, y)
            self.object_selected = True
            print(f"✅ Vehicle selected at: {self.selected_box}")

    def process_frame(self, frame):
        # Resize frame for faster processing if it's large
        if self.frame_size is None:
            h, w = frame.shape[:2]
            if w > 1280:  # If width is larger than 1280px
                self.frame_size = (1280, int(h * (1280/w)))
                
        if self.frame_size:
            process_frame = cv2.resize(frame, self.frame_size)
        else:
            process_frame = frame
            
        # Apply background subtraction and noise reduction
        fgmask = self.fgbg.apply(process_frame)
        
        # Use threshold instead of Gaussian blur for faster processing
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours with optimized parameters
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        closest_contour = None
        min_dist = float("inf")
        
        # Process only the largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                
                if self.object_selected:
                    sel_x, sel_y = self.selected_box
                    # Use Manhattan distance for faster calculation
                    dist = abs(center[0] - sel_x) + abs(center[1] - sel_y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_contour = (x, y, w, h, center)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        speed_kmh = None
        if self.object_selected and closest_contour:
            x, y, w, h, center = closest_contour
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, "Tracking Vehicle", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3, cv2.LINE_AA)

            # Store current center
            self.prev_centers.append(center)
            if len(self.prev_centers) > self.speed_window:
                self.prev_centers.pop(0)

            if len(self.prev_centers) > 1:
                # Calculate speed using multiple previous positions
                speeds = []
                for i in range(1, len(self.prev_centers)):
                    prev = np.array(self.prev_centers[i-1])
                    curr = np.array(self.prev_centers[i])
                    
                    # Apply perspective correction (objects further up in frame appear to move slower)
                    perspective_factor = 1.0 + (curr[1] / frame.shape[0]) * 0.5
                    
                    pixel_distance = np.linalg.norm(curr - prev)
                    distance_m = pixel_distance * self.scale_m_per_px * perspective_factor
                    time_s = 1 / self.fps
                    speed_mps = distance_m / time_s
                    speeds.append(speed_mps * 3.6)  # Convert to km/h
                
                # Calculate weighted average of speeds (recent speeds count more)
                weights = np.linspace(0.5, 1.0, len(speeds))
                speed_kmh = np.average(speeds, weights=weights)
                
                # Apply Kalman-like filtering
                if self.prev_speeds:
                    prev_speed = self.prev_speeds[-1]
                    max_change = self.max_speed_change
                    if abs(speed_kmh - prev_speed) > max_change:
                        speed_kmh = prev_speed + max_change * np.sign(speed_kmh - prev_speed)
                
                self.prev_speeds.append(speed_kmh)
                if len(self.prev_speeds) > self.speed_window:
                    self.prev_speeds.pop(0)
                
                # Use median for display to remove outliers
                display_speed = np.median(self.prev_speeds)

                if self.show_speed:
                    # Add solid background rectangle for readability
                    label = f"{display_speed:.1f} km/h"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
                    cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                                  (x + text_size[0] + 10, y), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x + 5, y - 5),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

            self.prev_center = center

        return frame, speed_kmh
