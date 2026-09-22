"""
The full live pipeline: audio capture -> VAD gating -> Deepgram STT -> session
context -> LLM suggestions -> floating overlay.

Usage:
    python main.py                              # mock LLM: no key, no cost
    python main.py --provider groq              # live suggestions (fastest)
    python main.py --provider gemini
    python main.py --provider gemini --trigger-on system
    python main.py --seconds 120

Ctrl+Alt+J anywhere forces a suggestion (skips debounce and the rate floor).
Right-click the overlay to quit.

Requires DEEPGRAM_API_KEY, plus GROQ_API_KEY or GEMINI_API_KEY for that provider.
"""

import argparse
import sys
import threading

from dotenv import load_dotenv

# Must run before anything reads os.environ. A real environment variable
# still wins over the file, so `$env:X = ...` overrides .env for one run.
load_dotenv()

# Models emit typographic characters -- curly quotes, non-breaking hyphens --
# that the Windows console's default cp1252 codec cannot encode, and printing
# one raises UnicodeEncodeError from inside a callback. Never let a stray
# character be the thing that kills a suggestion.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_capture import AudioFrame, DualChannelRecorder
from audio_capture.vad.source_vad import SourceVAD
from llm import NOTHING_TO_SAY, Suggestion
from llm.providers import ProviderError
from overlay.hotkey import GlobalHotkey
from overlay.window import OverlayWindow
from service.client import BrainUnavailable, RemoteBrain
from service.local import TRIGGER_SOURCES, LocalBrain
from stt.deepgram_client import DeepgramSTTClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--provider", default="mock", choices=["mock", "gemini", "groq"])
    parser.add_argument("--model", default=None, help="provider model id")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument(
        "--trigger-on",
        default="both",
        choices=sorted(TRIGGER_SOURCES),
        help="whose completed turns arm the debounce timer",
    )
    parser.add_argument("--debounce-ms", type=float, default=500.0)
    parser.add_argument(
        "--min-interval-ms",
        type=float,
        default=None,
        help="floor between requests; defaults to the provider's own rate limit",
    )
    parser.add_argument("--max-context-tokens", type=int, default=1500)
    parser.add_argument(
        "--brain",
        default=None,
        metavar="URL",
        help="run context+LLM in a remote service (e.g. http://localhost:8000) "
        "instead of in this process; capture and overlay stay local",
    )
    args = parser.parse_args()

    config = {
        "provider": args.provider,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "debounce_ms": args.debounce_ms,
        "min_interval_ms": args.min_interval_ms,
        "max_context_tokens": args.max_context_tokens,
        "trigger_on": args.trigger_on,
    }

    overlay = OverlayWindow()

    # All three callbacks for one request run on the same llm-request thread,
    # but two requests can overlap while a superseded one drains. Thread-local
    # storage keeps each request's generation id to itself.
    request_state = threading.local()

    def on_suggestion_start(reason: str) -> None:
        request_state.generation = overlay.begin()

    def on_suggestion_token(text: str) -> None:
        overlay.append(request_state.generation, text)

    def on_suggestion_done(suggestion: Suggestion) -> None:
        generation = getattr(request_state, "generation", None)
        if generation is None:
            return
        if suggestion.error:
            print(f"\n[suggestion] failed: {suggestion.error}")
            overlay.finish(generation, f"[{suggestion.error}]")
        elif suggestion.cancelled:
            # A newer suggestion already owns the screen. The overlay's
            # generation filter would drop this anyway; saying so explicitly
            # keeps the intent readable.
            pass
        else:
            ttft = f"{suggestion.ttft_ms:.0f}ms" if suggestion.ttft_ms else "n/a"
            text = suggestion.text.strip()
            if not text or text == NOTHING_TO_SAY:
                # The model judged there was nothing useful to add. Get off
                # the screen rather than showing the sentinel.
                print(f"\n    [nothing to add, {ttft}]")
                overlay.dismiss(generation)
                return
            # Echo to the console as well as the overlay. Overlay-only output
            # means any overlay problem looks identical to a model problem.
            print(f"\n>>> {text}")
            print(f"    [ttft {ttft}, total {suggestion.total_ms:.0f}ms]")
            overlay.finish(generation)

    def on_turn_complete(source: str, text: str) -> None:
        print(f"\n[{source}] FINAL: {text}")

    # The only difference `--brain` makes: where turn assembly, scheduling and
    # the provider call happen. Capture, VAD, STT and the overlay are local
    # either way, because each of them needs this machine's hardware.
    brain_kwargs = dict(
        on_turn=on_turn_complete,
        on_start=on_suggestion_start,
        on_token=on_suggestion_token,
        on_done=on_suggestion_done,
    )
    try:
        if args.brain:
            brain = RemoteBrain(args.brain, config, **brain_kwargs)
        else:
            brain = LocalBrain(config, **brain_kwargs)
        info = brain.start()
    except (ProviderError, ValueError, BrainUnavailable) as exc:
        print(f"Could not start the LLM backend: {exc}", file=sys.stderr)
        return 1

    def on_turn_event(source_label: str, event: dict) -> None:
        transcript = event.get("transcript")
        if transcript and event.get("event") in ("Update", "EagerEndOfTurn"):
            print(f"\r[{source_label}] ...{transcript}", end="", flush=True)
        brain.handle_turn_event(source_label, event)

    def on_connection_change(source_label: str, connected: bool) -> None:
        if not connected:
            brain.flush_pending(source_label)

    vads = {"mic": SourceVAD(), "system": SourceVAD()}
    stt_clients = {
        source: DeepgramSTTClient(
            source, on_turn_event, on_connection_change=on_connection_change
        )
        for source in ("mic", "system")
    }

    frame_counts = {"mic": 0, "system": 0}
    sent_counts = {"mic": 0, "system": 0}

    def on_frame(frame: AudioFrame) -> None:
        frame_counts[frame.source] += 1
        vad = vads[frame.source]
        vad.process(frame.pcm)

        if vad.is_speaking:
            stt_clients[frame.source].send_audio(frame.pcm.tobytes())
            sent_counts[frame.source] += 1

    recorder = DualChannelRecorder(on_frame=on_frame)
    hotkey = GlobalHotkey(brain.trigger_now)

    # --- start everything on background threads, then hand the main
    # --- thread to tkinter. mainloop() is what keeps the process alive.

    where = f"remote brain at {args.brain}" if args.brain else "in-process"
    print(f"LLM provider: {info['provider']} ({where})")
    print("Connecting to Deepgram...")
    for client in stt_clients.values():
        client.start()
    hotkey.start()

    try:
        recorder.start()
    except Exception as exc:
        print(f"\nFailed to start capture: {exc}", file=sys.stderr)
        brain.stop()
        hotkey.stop()
        return 1

    if args.seconds is not None:
        threading.Timer(args.seconds, overlay.close).start()

    print("Capturing. Ctrl+Alt+J for a suggestion. Right-click the overlay to quit.")
    try:
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        hotkey.stop()
        recorder.stop()
        for client in stt_clients.values():
            client.stop()
        brain.stop()

    print("\nStopped.")
    for source in ("mic", "system"):
        total = frame_counts[source]
        sent = sent_counts[source]
        pct = (sent / total * 100) if total else 0
        print(f"  {source}: {total} frames captured, {sent} sent to Deepgram ({pct:.0f}%)")

    print("\nFinal context window:\n")
    print(brain.get_context_window())
    return 0


if __name__ == "__main__":
    sys.exit(main())
