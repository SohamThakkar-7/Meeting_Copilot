

import threading
import time

import pytest

from llm.orchestrator import TRIGGER_HOTKEY, TRIGGER_TURN, LLMOrchestrator


class FakeContext:
    """Stands in for SessionContextManager."""

    def __init__(self, text="Them: hello"):
        self.text = text
        self.reads = 0

    def get_context_window(self):
        self.reads += 1
        return self.text


class FakeProvider:
    """Streams a fixed reply, slowly enough to be cancelled mid-flight."""

    name = "fake"
    suggested_min_interval_ms = 0.0

    def __init__(self, words=("one", "two", "three"), delay=0.05, error=None):
        self.words = words
        self.delay = delay
        self.error = error
        self.calls = 0
        self.prompts = []

    def stream(self, system, user, max_tokens, cancel):
        self.calls += 1
        self.prompts.append((system, user))
        if self.error:
            raise self.error
        for word in self.words:
            if cancel.is_set():
                return
            yield word + " "
            if cancel.wait(self.delay):
                return


class Harness:
    """Collects on_start/on_token/on_done so assertions read as behaviour."""

    def __init__(self):
        self.started = []
        self.tokens = []
        self.done = []
        self._event = threading.Event()

    def on_start(self, reason):
        self.started.append(reason)

    def on_token(self, text):
        self.tokens.append(text)

    def on_done(self, suggestion):
        self.done.append(suggestion)
        self._event.set()

    def wait_for_done(self, count=1, timeout=5.0):
        deadline = time.monotonic() + timeout
        while len(self.done) < count and time.monotonic() < deadline:
            self._event.wait(0.05)
            self._event.clear()
        return len(self.done) >= count


def build(provider, harness, **kwargs):
    kwargs.setdefault("debounce_ms", 100.0)
    kwargs.setdefault("min_interval_ms", 0.0)
    return LLMOrchestrator(
        provider,
        FakeContext(),
        on_start=harness.on_start,
        on_token=harness.on_token,
        on_done=harness.on_done,
        **kwargs,
    )


class FakeTurn:
    def __init__(self, source="mic"):
        self.source = source
        self.text = "hello"


@pytest.fixture
def harness():
    return Harness()


# --- debounce -------------------------------------------------------------


def test_burst_of_turns_collapses_into_one_request(harness):
    provider = FakeProvider()
    orchestrator = build(provider, harness, debounce_ms=200.0)
    orchestrator.start()
    try:
        for _ in range(5):
            orchestrator.notify_turn(FakeTurn())
            time.sleep(0.03)
        assert harness.wait_for_done(1)
        time.sleep(0.4)                      # give a second request time to appear
        assert provider.calls == 1
    finally:
        orchestrator.stop()


def test_turns_spaced_beyond_the_debounce_fire_separately(harness):
    provider = FakeProvider(words=("x",), delay=0.0)
    orchestrator = build(provider, harness, debounce_ms=50.0)
    orchestrator.start()
    try:
        orchestrator.notify_turn(FakeTurn())
        assert harness.wait_for_done(1)
        orchestrator.notify_turn(FakeTurn())
        assert harness.wait_for_done(2)
        assert provider.calls == 2
    finally:
        orchestrator.stop()


# --- rate floor -----------------------------------------------------------


def test_rate_floor_delays_rather_than_drops(harness):
    """A throttled trigger must still produce an answer, just later -- the
    conversation still deserves a suggestion."""
    provider = FakeProvider(words=("x",), delay=0.0)
    orchestrator = build(provider, harness, debounce_ms=10.0, min_interval_ms=600.0)
    orchestrator.start()
    try:
        orchestrator.notify_turn(FakeTurn())
        assert harness.wait_for_done(1)

        started = time.monotonic()
        orchestrator.notify_turn(FakeTurn())
        assert harness.wait_for_done(2, timeout=5.0)
        elapsed = time.monotonic() - started

        assert provider.calls == 2, "throttled trigger was dropped, not delayed"
        assert elapsed >= 0.4, f"rate floor not enforced (fired after {elapsed:.2f}s)"
    finally:
        orchestrator.stop()


def test_hotkey_bypasses_debounce_and_rate_floor(harness):
    provider = FakeProvider(words=("x",), delay=0.0)
    orchestrator = build(
        provider, harness, debounce_ms=5000.0, min_interval_ms=5000.0
    )
    orchestrator.start()
    try:
        started = time.monotonic()
        orchestrator.trigger_now()
        assert harness.wait_for_done(1, timeout=2.0)
        assert time.monotonic() - started < 1.0
        assert harness.started == [TRIGGER_HOTKEY]
    finally:
        orchestrator.stop()


# --- source filtering -----------------------------------------------------


def test_trigger_sources_filters_out_other_sources(harness):
    provider = FakeProvider(words=("x",), delay=0.0)
    orchestrator = build(
        provider, harness, debounce_ms=50.0, trigger_sources=("system",)
    )
    orchestrator.start()
    try:
        orchestrator.notify_turn(FakeTurn(source="mic"))
        time.sleep(0.3)
        assert provider.calls == 0

        orchestrator.notify_turn(FakeTurn(source="system"))
        assert harness.wait_for_done(1)
        assert provider.calls == 1
    finally:
        orchestrator.stop()


# --- supersede ------------------------------------------------------------


def test_new_trigger_cancels_the_in_flight_request(harness):
    provider = FakeProvider(words=[f"w{i}" for i in range(20)], delay=0.08)
    orchestrator = build(provider, harness, debounce_ms=10.0)
    orchestrator.start()
    try:
        orchestrator.trigger_now()
        time.sleep(0.25)                     # let it stream a little
        orchestrator.trigger_now()

        assert harness.wait_for_done(2, timeout=6.0)
        assert harness.done[0].cancelled is True
        assert provider.calls == 2
    finally:
        orchestrator.stop()


# --- reporting ------------------------------------------------------------


def test_suggestion_reports_text_and_ttft(harness):
    provider = FakeProvider(words=("hello", "world"), delay=0.01)
    orchestrator = build(provider, harness, debounce_ms=10.0)
    orchestrator.start()
    try:
        orchestrator.trigger_now()
        assert harness.wait_for_done(1)
        suggestion = harness.done[0]

        assert suggestion.text == "hello world"
        assert suggestion.ttft_ms is not None and suggestion.ttft_ms >= 0
        assert suggestion.total_ms >= suggestion.ttft_ms
        assert suggestion.error is None
        assert suggestion.provider == "fake"
    finally:
        orchestrator.stop()


def test_provider_failure_becomes_an_error_not_a_crash(harness):
    provider = FakeProvider(error=RuntimeError("backend exploded"))
    orchestrator = build(provider, harness, debounce_ms=10.0)
    orchestrator.start()
    try:
        orchestrator.trigger_now()
        assert harness.wait_for_done(1)
        assert "backend exploded" in harness.done[0].error
        assert harness.done[0].ttft_ms is None
    finally:
        orchestrator.stop()


def test_context_is_read_at_fire_time_not_trigger_time(harness):
    """The debounce delay should buy a fuller transcript, which only works
    if the context is snapshotted when the request fires."""
    provider = FakeProvider(words=("x",), delay=0.0)
    context = FakeContext("Them: early")
    orchestrator = LLMOrchestrator(
        provider,
        context,
        debounce_ms=200.0,
        min_interval_ms=0.0,
        on_start=harness.on_start,
        on_token=harness.on_token,
        on_done=harness.on_done,
    )
    orchestrator.start()
    try:
        orchestrator.notify_turn(FakeTurn())
        context.text = "Them: early and then some more"   # arrives during debounce
        assert harness.wait_for_done(1)
        assert "and then some more" in provider.prompts[0][1]
    finally:
        orchestrator.stop()


def test_stop_is_idempotent_and_does_not_hang(harness):
    orchestrator = build(FakeProvider(), harness)
    orchestrator.start()
    orchestrator.stop()
    orchestrator.stop()
