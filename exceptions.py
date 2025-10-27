class VideoLoadError(Exception):
    """Raised when video file cannot be loaded."""
    pass

class ModelLoadError(Exception):
    """Raised when YOLO model cannot be loaded."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass

class ProcessingError(Exception):
    """Raised when frame processing fails."""
    pass

class ExportError(Exception):
    """Raised when exporting results fails."""
    pass