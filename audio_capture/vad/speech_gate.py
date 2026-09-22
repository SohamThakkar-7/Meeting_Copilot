import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

from .accumulator import WINDOW_SIZE_SAMPLES


class SpeechGate:
    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
    ):
        self.sampling_rate = sampling_rate
        self._model = load_silero_vad(onnx=True)
        self._iterator = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.is_speaking = False

    def process(self, window: np.ndarray) -> dict | None:
        """
        window: exactly WINDOW_SIZE_SAMPLES (512) int16 samples.
        Returns {'start': <seconds>} or {'end': <seconds>} when a
        speech boundary is crossed, or None most of the time (nothing
        new -- still mid-speech or still mid-silence).
        """
        assert len(window) == WINDOW_SIZE_SAMPLES, (
            f"SpeechGate requires exactly {WINDOW_SIZE_SAMPLES} samples, "
            f"got {len(window)}. Pass windows through FrameAccumulator first."
        )

        audio_float = window.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float)

        event = self._iterator(tensor, return_seconds=True)

        if event is not None:
            if "start" in event:
                self.is_speaking = True
            elif "end" in event:
                self.is_speaking = False

        return event

    def reset(self) -> None:
        self._iterator.reset_states()
        self.is_speaking = False
