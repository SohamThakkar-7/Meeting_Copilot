"""
Exercise the LLM layer without touching audio.

Testing it through the front door would need a real meeting on demand every
time. This feeds a scripted conversation through the same
SessionContextManager and LLMOrchestrator the live app uses, so you can verify
the provider connects, watch tokens stream, and get a real time-to-first-token
number in a couple of seconds.

Usage:
    python check_llm.py                      # mock: no key, no network
    python check_llm.py --provider gemini    # live call, needs GEMINI_API_KEY
    python check_llm.py --provider gemini --list-models
    python check_llm.py --provider gemini --model gemini-2.0-flash

Get a free key at https://aistudio.google.com/apikey, then:
    $env:GEMINI_API_KEY = "..."      # PowerShell
"""

import argparse
import sys
import threading
import time

from dotenv import load_dotenv

load_dotenv()

from context import SessionContextManager
from llm import LLMOrchestrator, build_provider
from llm.providers import ProviderError

# A short scripted exchange, in the shape Deepgram Flux would deliver it.
SCRIPT = [
    ("system", "Hey, thanks for making the time. So walk me through what you've built so far."),
    ("mic", "Sure. It's a desktop copilot that listens to a call and suggests things in real time."),
    ("system", "Interesting. What's your latency budget on that, end to end?"),
]


def feed(session, source, text):
    session.handle_turn_event(source, {"event": "StartOfTurn", "transcript": ""})
    session.handle_turn_event(source, {"event": "Update", "transcript": text})
    session.handle_turn_event(source, {"event": "EndOfTurn", "transcript": text})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="mock", choices=["mock", "gemini", "groq"])
    parser.add_argument("--model", default=None, help="provider model id")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    kwargs = {}
    if args.model and args.provider == "gemini":
        kwargs["model"] = args.model

    try:
        provider = build_provider(args.provider, **kwargs)
    except ProviderError as exc:
        print(f"Could not build provider: {exc}", file=sys.stderr)
        return 1

    if args.list_models:
        if not hasattr(provider, "list_models"):
            print(f"{provider.name} has no model listing.")
            return 0
        try:
            for name in provider.list_models():
                print(name)
        except ProviderError as exc:
            print(f"Listing failed: {exc}", file=sys.stderr)
            return 1
        return 0

    print(f"provider: {provider.name}\n")

    # --- 1. live request against the chosen provider ----------------------
    session = SessionContextManager(track_active_window=False)
    for source, text in SCRIPT:
        feed(session, source, text)

    print("--- context window sent to the model ---")
    print(session.get_context_window())
    print("---\n")

    done = threading.Event()
    result = {}

    def on_start(reason):
        print(f"[{reason}] ", end="", flush=True)

    def on_token(text):
        print(text, end="", flush=True)

    def on_done(suggestion):
        result["suggestion"] = suggestion
        done.set()

    orchestrator = LLMOrchestrator(
        provider,
        session,
        debounce_ms=0,
        min_interval_ms=0,
        max_tokens=args.max_tokens,
        on_start=on_start,
        on_token=on_token,
        on_done=on_done,
    )
    orchestrator.start()
    orchestrator.trigger_now("hotkey")

    if not done.wait(timeout=60):
        print("\nTimed out waiting for a response.", file=sys.stderr)
        orchestrator.stop()
        return 1
    orchestrator.stop()

    suggestion = result["suggestion"]
    print("\n")
    if suggestion.error:
        print(f"FAILED: {suggestion.error}", file=sys.stderr)
        return 1

    ttft = suggestion.ttft_ms
    print(f"time to first token : {ttft:.0f} ms" if ttft else "no tokens received")
    print(f"total               : {suggestion.total_ms:.0f} ms")
    if ttft:
        budget = "within" if ttft <= 1500 else "OVER"
        print(f"                      ({budget} the ~1-2s end-to-end budget, "
              f"before STT and render)")

    # --- 2. orchestrator guards, always on the mock -----------------------
    # These are about scheduling, not about the model, so they run offline and
    # deterministically no matter which provider was selected above.
    print("\n--- orchestrator guards (mock) ---")
    fired = []
    guard_session = SessionContextManager(track_active_window=False)
    guard = LLMOrchestrator(
        build_provider("mock", ttft_ms=10, token_ms=1),
        guard_session,
        debounce_ms=300,
        min_interval_ms=0,
        on_done=lambda s: fired.append(s),
    )
    guard.start()

    # Four turns in rapid succession should collapse into one request.
    for index in range(4):
        feed(guard_session, "system", f"rapid turn {index}")
        guard.notify_turn(guard_session.last_turn)
        time.sleep(0.05)
    time.sleep(1.0)
    guard.stop()

    ok = len(fired) == 1
    print(f"{'ok  ' if ok else 'FAIL'} 4 rapid turns -> {len(fired)} request(s) "
          f"(expected 1: debounce collapsed the burst)")

    if hasattr(provider, "close"):
        provider.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
