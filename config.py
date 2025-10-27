from dataclasses import dataclass
from typing import Optional
import json
import os
from loguru import logger

@dataclass
class Config:
    video_path: str = "input.mp4"
    window_width: int = 1280
    window_height: int = 720
    known_distance_m: float = 10.0
    show_speed: bool = True
    speed_unit: str = "km/h"  # or "mph"
    frame_skip: int = 1
    enable_threading: bool = True
    buffer_size: int = 30
    export_path: str = "exports"
    model_path: str = "models/yolov8n.pt"

    def save(self, path: str = "config.json") -> None:
        """Save configuration to JSON file."""
        try:
            with open(path, "w") as f:
                json.dump(self.__dict__, f, indent=4)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    @classmethod
    def load(cls, path: str = "config.json") -> 'Config':
        """Load configuration from JSON file."""
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                    return cls(**data)
            logger.warning(f"No configuration file found at {path}, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return cls()

    def validate(self) -> bool:
        """Validate configuration values."""
        try:
            assert self.window_width > 0, "Window width must be positive"
            assert self.window_height > 0, "Window height must be positive"
            assert self.known_distance_m > 0, "Known distance must be positive"
            assert self.frame_skip > 0, "Frame skip must be positive"
            assert self.buffer_size > 0, "Buffer size must be positive"
            assert self.speed_unit in ["km/h", "mph"], "Invalid speed unit"
            assert os.path.exists(self.model_path), "Model file not found"
            
            if not os.path.exists(self.export_path):
                os.makedirs(self.export_path)
                
            return True
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False