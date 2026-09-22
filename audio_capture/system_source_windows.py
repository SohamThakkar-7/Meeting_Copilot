import queue
import threading
import time

import numpy as np

from . import config
from .frame import AudioFrame
from .resampler import StreamResampler


class WindowsSystemSource:
    def __init__(self, out_queue: "queue.Queue[AudioFrame]"):
        self.out_queue = out_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pa = None
        self._stream = None
        self._resampler: StreamResampler | None = None
        self._frames_per_buffer = 0
        self._channels = 1
        self._read_errors = 0
        self._last_error_log = float("-inf")

    def start(self) -> None:
        import pyaudiowpatch as pyaudio  # deferred import, Windows-only wheel

        self._pa = pyaudio.PyAudio()
        loopback = self._pa.get_default_wasapi_loopback()

        native_sr = int(loopback["defaultSampleRate"])
        self._channels = loopback["maxInputChannels"]
        self._resampler = StreamResampler(native_sr, config.TARGET_SAMPLE_RATE)
        self._frames_per_buffer = int(native_sr * config.CHUNK_MS / 1000)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=native_sr,
            input=True,
            input_device_index=loopback["index"],
            frames_per_buffer=self._frames_per_buffer,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._stream.read(
                    self._frames_per_buffer, exception_on_overflow=False
                )
            except Exception as exc:
                # Device hiccup (e.g. output device switched mid-call).
                # Swallowing this silently means a dead loopback stream is
                # indistinguishable from a quiet one -- say it once, then
                # keep the rate down so one bad device can't flood the log.
                self._read_errors += 1
                now = time.perf_counter()
                if now - self._last_error_log > 5.0:
                    self._last_error_log = now
                    print(
                        f"[system] loopback read failed "
                        f"({self._read_errors} so far): {exc}"
                    )
                continue

            pcm = np.frombuffer(raw, dtype=np.int16)
            if self._channels > 1:
                pcm = pcm.reshape(-1, self._channels).mean(axis=1).astype(np.int16)

            pcm = self._resampler.process(pcm)
            if pcm.size == 0:
                continue

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

        # Order matters. The reader thread sits blocked inside a native
        # stream.read(), and closing the stream under it is a use-after-free in
        # C -- it segfaults the process rather than raising. stop_stream()
        # first unblocks the read so the thread can see the stop event.
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                # Still inside the native read. Leaking the stream costs
                # nothing on the way out; closing it now would crash us.
                print("[system] loopback reader did not stop; leaving stream open")
                return

        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
