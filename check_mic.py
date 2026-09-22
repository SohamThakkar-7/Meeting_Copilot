
import argparse
import queue
import sys
import time

import numpy as np

from audio_capture.frame import AudioFrame
from audio_capture.mic_source import MicSource
from audio_capture.vad.source_vad import SourceVAD

BAR_WIDTH = 40
# int16 full scale. Normal speech sits somewhere around 1000-8000 RMS.
FULL_SCALE = 32768.0


def list_devices() -> int:
    import sounddevice as sd

    print("Input devices:")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            default = " (default)" if index == sd.default.device[0] else ""
            print(
                f"  [{index:2d}] {device['name']}  "
                f"{device['max_input_channels']}ch "
                f"{int(device['default_samplerate'])}Hz{default}"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Silero speech threshold (default 0.5); lower it if the meter "
        "moves but SPEECH never lights",
    )
    args = parser.parse_args()

    if args.list:
        return list_devices()

    frames: "queue.Queue[AudioFrame]" = queue.Queue(maxsize=500)
    mic = MicSource(frames, device=args.device)
    vad = SourceVAD(threshold=args.threshold)

    try:
        mic.start()
    except Exception as exc:
        print(f"Could not open the microphone: {exc}", file=sys.stderr)
        return 1

    print(f"Capturing at {mic._native_sr}Hz for {args.seconds:.0f}s.")
    print("Talk normally. Ctrl+C to stop early.\n")

    peak_seen = 0
    speech_frames = 0
    total_frames = 0
    started = time.perf_counter()

    try:
        while time.perf_counter() - started < args.seconds:
            try:
                frame = frames.get(timeout=0.5)
            except queue.Empty:
                print("\rno frames arriving -- is the device live?", end="", flush=True)
                continue

            total_frames += 1
            vad.process(frame.pcm)
            if vad.is_speaking:
                speech_frames += 1

            samples = frame.pcm.astype(np.float32)
            rms = float(np.sqrt((samples**2).mean())) if samples.size else 0.0
            peak = int(np.abs(samples).max()) if samples.size else 0
            peak_seen = max(peak_seen, peak)

            # Log scale: linear makes speech look like nothing next to silence.
            filled = 0
            if rms > 1:
                filled = min(BAR_WIDTH, int(BAR_WIDTH * np.log10(rms) / np.log10(FULL_SCALE)))
            bar = "#" * filled + "-" * (BAR_WIDTH - filled)
            flag = "SPEECH" if vad.is_speaking else "      "
            print(f"\r[{bar}] rms {rms:7.1f}  peak {peak:6d}  {flag}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()

    print("\n")
    if not total_frames:
        print("No audio frames arrived at all. The device never delivered data.")
        return 1

    pct = speech_frames / total_frames * 100
    print(f"frames         : {total_frames}")
    print(f"speech-gated   : {speech_frames} ({pct:.0f}%)")
    print(f"loudest sample : {peak_seen} of {int(FULL_SCALE)}")
    print()

    if peak_seen < 100:
        print("VERDICT: the mic is delivering near-silence. Check that it isn't")
        print("muted, that Windows privacy settings allow microphone access, and")
        print("try --list to pick a different device.")
        return 1
    if pct < 5:
        print("VERDICT: audio is arriving but VAD almost never opened. If you were")
        print("speaking, the threshold is too high for this mic -- try")
        print("--threshold 0.3.")
        return 1
    print("VERDICT: the microphone path is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
