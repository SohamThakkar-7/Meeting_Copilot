from dataclasses import dataclass
import numpy as np

@dataclass
class AudioFrame:
    source: str
    timestamp: float
    pcm: np.ndarray
    sample_rate: int

    def duration_ms(self) -> float:
        return (len(self.pcm) / self.sample_rate) * 1000.0
