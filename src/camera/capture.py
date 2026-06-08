import cv2
import logging
import os
import threading
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Unified video capture for USB cameras and Raspberry Pi Camera Module.

    When use_threading=True a dedicated daemon thread continuously grabs
    frames so the main loop always gets the most recent one — this avoids
    the built-in OpenCV buffer stacking up stale frames under load.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        use_picamera: bool = False,
        use_threading: bool = True,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.use_picamera = use_picamera
        self.use_threading = use_threading

        self._cap: Optional[cv2.VideoCapture] = None
        self._picam = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraCapture":
        if self.use_picamera:
            self._start_picamera()
        else:
            self._start_opencv()

        if self.use_threading:
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

        logger.info(
            "Camera started — source=%s  %dx%d @ %d fps  threading=%s",
            self.source, self.width, self.height, self.fps, self.use_threading,
        )
        return self

    def read(self) -> Tuple[bool, Optional[object]]:
        """Return (success, frame). Frame is a BGR numpy array or None."""
        if self.use_threading:
            with self._lock:
                if self._latest_frame is None:
                    return False, None
                return True, self._latest_frame.copy()
        frame = self._read_raw()
        return (frame is not None), frame

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.stop()
        logger.info("Camera stopped")

    @property
    def actual_fps(self) -> float:
        if self._cap is not None:
            return float(self._cap.get(cv2.CAP_PROP_FPS))
        return float(self.fps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_opencv(self) -> None:
        # For file paths, give an actionable error before even calling OpenCV
        if isinstance(self.source, str) and not os.path.isfile(self.source):
            raise RuntimeError(
                f"Video file not found: {self.source!r}\n"
                f"  Tip: use the full path, e.g.  --source ~/Downloads/video.mp4"
            )
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source!r}")

        # Camera-only properties — do NOT set these on video files.
        # CAP_PROP_BUFFERSIZE=1 in particular disrupts codecs that use a
        # B-frame reorder buffer (e.g. H.264 with B-frames), causing the
        # decoder to return EOF after the first read on some video files.
        if not isinstance(self.source, str):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _start_picamera(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam = Picamera2()
            cfg = self._picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self._picam.configure(cfg)
            self._picam.start()
        except ImportError:
            logger.warning("picamera2 not available — falling back to OpenCV")
            self.use_picamera = False
            self._start_opencv()

    def _read_raw(self):
        if self.use_picamera and self._picam is not None:
            try:
                import cv2
                frame = self._picam.capture_array()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                logger.error("PiCamera read error: %s", exc)
                return None
        if self._cap is not None:
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    def _capture_loop(self) -> None:
        while self._running:
            frame = self._read_raw()
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()
