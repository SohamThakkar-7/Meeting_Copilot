import queue
import threading
import time
import numpy as np
import sounddevice as sd
from . import config
from .frame import AudioFrame
from .resampler import StreamResampler


class MicSource:
    def __init__(self, out_queue: "queue.Queue[AudioFrame]", device: int | None = None):
        self.out_queue = out_queue
        self.device = device
        self._stream: sd.InputStream | None = None
        self._resampler: StreamResampler | None = None
        self._native_sr = config.TARGET_SAMPLE_RATE
        self._lock = threading.Lock()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        pcm = indata[:, 0] if indata.ndim > 1 else indata

        if pcm.dtype != np.int16:
            pcm = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)

        if self._resampler is not None:
            pcm = self._resampler.process(pcm)

        if pcm.size == 0:
            return

        frame = AudioFrame(
            source="mic",
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

    def start(self) -> None:
        chunk_frames = int(config.TARGET_SAMPLE_RATE * config.CHUNK_MS / 1000)
        try:
            self._stream = sd.InputStream(
                samplerate=config.TARGET_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=chunk_frames,
                latency="low",
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
            self._native_sr = config.TARGET_SAMPLE_RATE
            return
        except Exception:
            pass  # fall through to native-rate path below

        device_info = sd.query_devices(self.device, "input")
        self._native_sr = int(device_info["default_samplerate"])
        self._resampler = StreamResampler(self._native_sr, config.TARGET_SAMPLE_RATE)
        native_chunk_frames = int(self._native_sr * config.CHUNK_MS / 1000)

        self._stream = sd.InputStream(
            samplerate=self._native_sr,
            channels=1,
            dtype="int16",
            blocksize=native_chunk_frames,
            latency="low",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
