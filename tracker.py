import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
from loguru import logger

@dataclass
class VehicleTrack:
    """Represents a tracked vehicle and its speed measurements."""
    id: int
    positions: List[Tuple[float, float]]  # List of (x, y) positions
    timestamps: List[float]  # List of timestamps for each position
    speeds: List[float]  # List of calculated speeds
    last_update: float  # Last update timestamp
    color: Tuple[int, int, int]  # BGR color for visualization
    
    @property
    def current_speed(self) -> Optional[float]:
        """Get the current speed of the vehicle."""
        return self.speeds[-1] if self.speeds else None
        
    @property
    def average_speed(self) -> Optional[float]:
        """Get the average speed of the vehicle."""
        return np.mean(self.speeds) if self.speeds else None
        
    @property
    def max_speed(self) -> Optional[float]:
        """Get the maximum speed of the vehicle."""
        return max(self.speeds) if self.speeds else None
        
    def is_active(self, timeout: float = 1.0) -> bool:
        """Check if the track is still active based on last update time."""
        return (time.time() - self.last_update) < timeout


class MultiVehicleTracker:
    """Tracks multiple vehicles and their speeds."""
    
    def __init__(self, max_disappeared: int = 30):
        self.next_vehicle_id = 0
        self.vehicles: Dict[int, VehicleTrack] = {}
        self.max_disappeared = max_disappeared
        self.speed_unit = "km/h"
        
    def update(self, detections: List[Tuple[int, int, int, int]], frame: np.ndarray) -> np.ndarray:
        """Update vehicle tracks with new detections."""
        current_time = time.time()
        
        # Create new tracks or update existing ones
        matched_tracks = set()
        for bbox in detections:
            x, y, w, h = bbox
            center = (x + w/2, y + h/2)
            
            # Find closest existing track
            closest_id = None
            min_dist = float('inf')
            for vid, track in self.vehicles.items():
                if not track.is_active():
                    continue
                if track.positions:
                    dist = np.linalg.norm(np.array(center) - np.array(track.positions[-1]))
                    if dist < min_dist and dist < 50:  # threshold for matching
                        min_dist = dist
                        closest_id = vid
                        
            if closest_id is not None:
                # Update existing track
                track = self.vehicles[closest_id]
                track.positions.append(center)
                track.timestamps.append(current_time)
                track.last_update = current_time
                
                # Calculate speed
                if len(track.positions) >= 2:
                    dt = track.timestamps[-1] - track.timestamps[-2]
                    if dt > 0:
                        dx = track.positions[-1][0] - track.positions[-2][0]
                        dy = track.positions[-1][1] - track.positions[-2][1]
                        speed = np.sqrt(dx*dx + dy*dy) / dt
                        track.speeds.append(speed)
                
                matched_tracks.add(closest_id)
            else:
                # Create new track
                color = tuple(np.random.randint(0, 255, 3).tolist())
                self.vehicles[self.next_vehicle_id] = VehicleTrack(
                    id=self.next_vehicle_id,
                    positions=[center],
                    timestamps=[current_time],
                    speeds=[],
                    last_update=current_time,
                    color=color
                )
                matched_tracks.add(self.next_vehicle_id)
                self.next_vehicle_id += 1
                
        # Remove inactive tracks
        self.vehicles = {
            vid: track for vid, track in self.vehicles.items()
            if track.is_active() or vid in matched_tracks
        }
        
        # Draw tracks and speeds on frame
        return self.draw_tracks(frame)
        
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw vehicle tracks and speed information on the frame."""
        for track in self.vehicles.values():
            if not track.is_active():
                continue
                
            # Draw track line
            if len(track.positions) > 1:
                points = np.array(track.positions[-20:], dtype=np.int32)
                cv2.polylines(frame, [points], False, track.color, 2)
                
            # Draw current position and speed
            if track.positions:
                x, y = map(int, track.positions[-1])
                if track.current_speed is not None:
                    speed_text = f"{track.current_speed:.1f} {self.speed_unit}"
                    cv2.putText(frame, speed_text, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, track.color, 2)
                              
            # Draw ID and average speed
            if track.average_speed is not None:
                avg_text = f"ID: {track.id} Avg: {track.average_speed:.1f} {self.speed_unit}"
                cv2.putText(frame, avg_text, (x, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, track.color, 2)
                           
        return frame
        
    def get_speed_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get speed statistics for all tracked vehicles."""
        stats = {}
        for vid, track in self.vehicles.items():
            if track.speeds:
                stats[f"Vehicle {vid}"] = {
                    "current": track.current_speed,
                    "average": track.average_speed,
                    "max": track.max_speed
                }
        return stats
        
    def set_speed_unit(self, unit: str):
        """Set the speed unit (km/h or mph)."""
        self.speed_unit = unit
        
    def export_data(self) -> Dict:
        """Export tracking data for all vehicles."""
        return {
            vid: {
                "positions": track.positions,
                "timestamps": track.timestamps,
                "speeds": track.speeds,
                "average_speed": track.average_speed,
                "max_speed": track.max_speed
            }
            for vid, track in self.vehicles.items()
        }