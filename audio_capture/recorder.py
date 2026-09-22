import platform
import queue
import threading
from typing import Callable, Optional

from . import config
from .frame import AudioFrame
from .mic_source import MicSource


class DualChannelRecorder:
    def __init__(
        self,
        on_frame: Optional[Callable[[AudioFrame], None]] = None,
        mic_device: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        on_frame : callable, optional
            If given, called for every frame from a dedicated dispatch
            thread as soon as it's available (push model -- lowest
            latency, recommended for the live VAD/STT hot path).
        mic_device : int, optional
            sounddevice device index. None = system default input.
        """
        self._queue: "queue.Queue[AudioFrame]" = queue.Queue(
            maxsize=config.MAX_QUEUE_CHUNKS
        )
        self._mic = MicSource(self._queue, device=mic_device)
        self._system = self._build_system_source()

        self._on_frame = on_frame
        self._dispatch_thread: Optional[threading.Thread] = None
        self._dispatch_stop = threading.Event()

    def _build_system_source(self):
        system_name = platform.system()
        if system_name == "Windows":
            from .system_source_windows import WindowsSystemSource

            return WindowsSystemSource(self._queue)
        elif system_name == "Darwin":
            from .system_source_macos import MacSystemSource

            return MacSystemSource(self._queue)
        else:
            raise NotImplementedError(
                f"System audio capture is not implemented for platform "
                f"'{system_name}'. Only Windows and macOS are supported."
            )

    def start(self) -> None:
        self._mic.start()
        self._system.start()

        if self._on_frame is not None:
            self._dispatch_stop.clear()
            self._dispatch_thread = threading.Thread(
                target=self._dispatch_loop, daemon=True
            )
            self._dispatch_thread.start()

    def _dispatch_loop(self) -> None:
        while not self._dispatch_stop.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._on_frame(frame)
            except Exception:
                # A bug in the caller's handler must never kill audio
                # capture -- log and keep going in production.
                pass

    def get(self, timeout: Optional[float] = None) -> Optional[AudioFrame]:
        """Pull model: fetch the next frame directly (only useful if
        you did NOT pass on_frame to __init__)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._mic.stop()
        self._system.stop()
        self._dispatch_stop.set()
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=1.0)
