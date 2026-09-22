import queue
import threading
import time

import numpy as np

from . import config
from .frame import AudioFrame


class PermissionError_(RuntimeError):
    """Raised when macOS Screen & System Audio Recording permission is missing."""


class MacSystemSource:
    def __init__(self, out_queue: "queue.Queue[AudioFrame]"):
        self.out_queue = out_queue
        self._recorder = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        from pysysaudio import SystemAudioRecorder  # deferred import, macOS-only wheel

        if not SystemAudioRecorder.check_permission():
            raise PermissionError_(
                "macOS Screen & System Audio Recording permission not granted. "
                "Grant it via System Settings -> Privacy & Security -> "
                "Screen & System Audio Recording, for your terminal or "
                "Python interpreter, then restart the app."
            )

        # Request our target format directly -- pysysaudio resamples and
        # remixes channels internally, so frames arrive already in the
        # shape/rate the rest of the pipeline expects.
        self._recorder = SystemAudioRecorder(
            sample_rate=config.TARGET_SAMPLE_RATE,
            channels=config.TARGET_CHANNELS,
            format="numpy",
            dtype="int16",
            buffer_size=100,
        )
        self._recorder.start_recording()  # no output_path -> streaming-only, no disk write

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # stream() is a blocking generator with an internal timeout so
        # the loop can check _stop_event periodically instead of hanging.
        for chunk in self._recorder.stream(timeout=0.1):
            if self._stop_event.is_set():
                break
            if chunk is None or len(chunk) == 0:
                continue

            # numpy format yields shape (frames, channels); we requested
            # channels=1 so this is already mono, just needs flattening.
            pcm = np.asarray(chunk).reshape(-1).astype(np.int16)

            frame = AudioFrame(
                source="system",
                timestamp=time.perf_counter(),
                pcm=pcm,
                sample_rate=config.TARGET_SAMPLE_RATE,
            )
            self._enqueue(frame)

    def _enqueue(self, frame: AudioFrame) -> None:
        try:
            self.out_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._recorder is not None:
            self._recorder.stop_recording()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
