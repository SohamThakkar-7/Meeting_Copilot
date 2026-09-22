import numpy as np

WINDOW_SIZE_SAMPLES = 512

class FrameAccumulator:
    def __init__(self, window_size: int = WINDOW_SIZE_SAMPLES):
        self.window_size = window_size
        self._buffer = np.zeros(0, dtype=np.int16)

    def push(self, pcm: np.ndarray) -> list[np.ndarray]:
        """
        Add new samples and return zero or more full-size windows ready
        for VAD. Leftover samples that don't fill a complete window are
        kept internally and prepended to the next push() call.
        """
        self._buffer = np.concatenate([self._buffer, pcm])

        windows = []
        while len(self._buffer) >= self.window_size:
            windows.append(self._buffer[: self.window_size])
            self._buffer = self._buffer[self.window_size :]

        return windows
