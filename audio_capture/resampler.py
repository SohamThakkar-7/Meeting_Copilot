import numpy as np

try:
    import soxr

    _HAS_SOXR = True
except ImportError:
    _HAS_SOXR = False

class StreamResampler:
    def __init__(self, in_rate: int, out_rate: int, channels: int = 1):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._passthrough = in_rate == out_rate

        if self._passthrough:
            return

        if _HAS_SOXR:
            self._rs = soxr.ResampleStream(
                in_rate, out_rate, channels, dtype="int16", quality="QQ"
            )
        else:
            self._rs = None

    def process(self, pcm: np.ndarray) -> np.ndarray:
        if self._passthrough:
            return pcm

        if _HAS_SOXR:
            return self._rs.resample_chunk(pcm)

        ratio = self.out_rate / self.in_rate
        n_out = max(1, int(round(len(pcm) * ratio)))
        if len(pcm) < 2:
            return np.zeros(0, dtype=np.int16)
        x_old = np.linspace(0, len(pcm) - 1, num=len(pcm))
        x_new = np.linspace(0, len(pcm) - 1, num=n_out)
        return np.interp(x_new, x_old, pcm).astype(np.int16)
