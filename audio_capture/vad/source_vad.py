import numpy as np

from .accumulator import FrameAccumulator
from .speech_gate import SpeechGate


class SourceVAD:
    def __init__(self, sampling_rate: int = 16000, **speech_gate_kwargs):
        self._accumulator = FrameAccumulator()
        self._gate = SpeechGate(sampling_rate=sampling_rate, **speech_gate_kwargs)

    @property
    def is_speaking(self) -> bool:
        return self._gate.is_speaking

    def process(self, pcm: np.ndarray) -> list[dict]:
        """
        pcm: any-length int16 array (e.g. straight from AudioFrame.pcm).
        Returns a list of events (usually empty, occasionally containing
        one or more {'start': t} / {'end': t} dicts) produced by running
        VAD over however many full 512-sample windows this pcm completed.
        """
        events = []
        for window in self._accumulator.push(pcm):
            event = self._gate.process(window)
            if event is not None:
                events.append(event)
        return events
