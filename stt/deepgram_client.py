"""
Streaming STT via Deepgram's Flux model (/v2/listen endpoint).

Flux is Deepgram's conversational-audio-native model: unlike a plain
transcription model, it understands turn-taking directly and emits explicit
StartOfTurn / Update / EagerEndOfTurn / EndOfTurn / TurnResumed events. This
is a genuinely useful second opinion alongside our own Silero VAD layer --
VAD still earns its keep as a COST gate (deciding whether audio is worth
sending to Deepgram at all), but "has the person finished their turn" is now
something Flux tells us directly, tuned specifically for this exact purpose.

One DeepgramSTTClient instance per audio source (mic, system) -- same "never
share state across streams" rule as the VAD layer, since each is an
independent conversational stream with its own turn state.

Reconnection
------------
Flux's v2 websocket has no working keepalive (deepgram-python-sdk #649). We
only send audio while VAD says someone is speaking, so a long silence looks
like a dead client and Deepgram force-closes the socket with a 1011 keepalive
timeout. Preventing that isn't currently possible from our side, so instead a
supervisor thread owns the connection lifecycle and simply reconnects: connect,
listen until the socket closes, back off, connect again. A connection that
survived a while resets the backoff, so a routine idle-timeout reconnects
almost immediately while a genuinely broken endpoint backs off instead of
hammering.

The cost is real and unavoidable today: a few hundred ms to a couple of
seconds of audio is missed at each reconnect point. on_connection_change lets
downstream layers know, so a half-captured turn can be flushed rather than
left dangling.

Logging is rate-limited per message kind -- fail loud once, not forever.
"""

import threading
import time
from typing import Callable, Optional

from deepgram import DeepgramClient
from deepgram.core.events import EventType

# A connection that stayed up at least this long counts as healthy; the next
# drop is treated as a routine idle-timeout, not as a failing endpoint.
_HEALTHY_UPTIME_SECONDS = 10.0


class DeepgramSTTClient:
    def __init__(
        self,
        source_label: str,
        on_turn_event: Callable[[str, dict], None],
        model: str = "flux-general-en",
        sample_rate: int = 16000,
        eot_threshold: float = 0.7,
        on_connection_change: Optional[Callable[[str, bool], None]] = None,
        reconnect_cooldown: float = 0.5,
        max_reconnect_delay: float = 15.0,
        log_interval: float = 5.0,
    ):
        """
        Parameters
        ----------
        on_connection_change : callable, optional
            Called as (source_label, connected: bool) whenever the socket
            comes up or goes down. Wire the "down" edge to
            SessionContextManager.flush_pending so an in-flight turn isn't
            stranded by a reconnect.
        reconnect_cooldown : float
            Base delay before reconnecting; doubles up to max_reconnect_delay
            while connections keep failing fast.
        log_interval : float
            Minimum seconds between repeats of the same kind of log line.
        """
        self.source_label = source_label
        self._on_turn_event = on_turn_event
        self._on_connection_change = on_connection_change
        self._model = model
        self._sample_rate = sample_rate
        self._eot_threshold = eot_threshold

        self._cooldown = reconnect_cooldown
        self._max_delay = max_reconnect_delay
        self._log_interval = log_interval

        self._client = DeepgramClient()  # reads DEEPGRAM_API_KEY from env
        self._connect_cm = None
        self._connection = None

        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._supervisor: Optional[threading.Thread] = None

        self._log_lock = threading.Lock()
        self._last_log: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}

    def start(self) -> None:
        if self._supervisor is not None and self._supervisor.is_alive():
            return
        self._stopping.clear()
        self._supervisor = threading.Thread(
            target=self._supervise, daemon=True, name=f"deepgram-{self.source_label}"
        )
        self._supervisor.start()

    def stop(self) -> None:
        self._stopping.set()

        with self._lock:
            connection = self._connection
        if connection is not None:
            try:
                # Closes the stream cleanly, which also unblocks the
                # supervisor's start_listening() call.
                connection.send_close_stream()
            except Exception:
                pass

        self._teardown()
        if self._supervisor is not None:
            self._supervisor.join(timeout=3.0)
            self._supervisor = None

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connection is not None

    def _supervise(self) -> None:
        delay = 0.0
        while not self._stopping.is_set():
            if delay and self._stopping.wait(delay):
                break

            try:
                connection = self._open()
            except Exception as exc:
                self._log("connect", f"connect failed: {exc}")
                delay = min(max(delay * 2, self._cooldown), self._max_delay)
                continue

            self._notify_connection(True)
            opened_at = time.perf_counter()
            try:
                # Blocks until the socket closes -- cleanly on stop(), or by
                # Deepgram's keepalive timeout during a long silence.
                connection.start_listening()
            except Exception as exc:
                self._log("listen", f"stream ended: {exc}")
            finally:
                self._teardown()
                self._notify_connection(False)

            if self._stopping.is_set():
                break

            uptime = time.perf_counter() - opened_at
            if uptime >= _HEALTHY_UPTIME_SECONDS:
                # Routine idle-timeout after a healthy run: come straight back.
                delay = self._cooldown
            else:
                delay = min(max(delay * 2, self._cooldown), self._max_delay)
            self._log(
                "reconnect",
                f"connection closed after {uptime:.0f}s; "
                f"reconnecting in {delay:.1f}s (transcription gap expected)",
            )

    def _open(self):
        connect_cm = self._client.listen.v2.connect(
            model=self._model,
            encoding="linear16",
            sample_rate=self._sample_rate,
            eot_threshold=self._eot_threshold,
        )
        connection = connect_cm.__enter__()
        connection.on(EventType.MESSAGE, self._handle_message)
        connection.on(EventType.ERROR, self._handle_error)

        with self._lock:
            self._connect_cm = connect_cm
            self._connection = connection
        return connection

    def _teardown(self) -> None:
        with self._lock:
            connect_cm = self._connect_cm
            self._connect_cm = None
            self._connection = None
        if connect_cm is not None:
            try:
                connect_cm.__exit__(None, None, None)
            except Exception:
                pass

    def _notify_connection(self, connected: bool) -> None:
        if self._on_connection_change is None:
            return
        try:
            self._on_connection_change(self.source_label, connected)
        except Exception as exc:
            self._log("notify", f"connection callback raised: {exc}")

    def send_audio(self, pcm_int16_bytes: bytes) -> None:
        with self._lock:
            connection = self._connection
        if connection is None:
            # Mid-reconnect. Dropping is the only option -- there's nothing to
            # send to, and buffering would only replay stale audio into a
            # fresh turn.
            self._log("offline", "dropping audio: not connected")
            return
        try:
            connection.send_media(pcm_int16_bytes)
        except Exception as exc:
            self._log("send", f"failed to send audio: {exc}")

    def _handle_message(self, message) -> None:
        if getattr(message, "type", None) != "TurnInfo":
            return

        event = {
            "event": getattr(message, "event", None),
            "transcript": getattr(message, "transcript", "") or "",
            "end_of_turn_confidence": getattr(message, "end_of_turn_confidence", None),
            "turn_index": getattr(message, "turn_index", None),
        }
        try:
            self._on_turn_event(self.source_label, event)
        except Exception as exc:
            # A bug downstream must never take down the STT stream.
            self._log("handler", f"turn handler raised: {exc}")

    def _handle_error(self, error) -> None:
        self._log("error", f"Deepgram error: {error}")

    def _log(self, kind: str, message: str) -> None:
        """Print at most one line per `kind` per `log_interval`, noting how
        many were suppressed. One failure should not become a thousand log
        lines."""
        now = time.perf_counter()
        with self._log_lock:
            if now - self._last_log.get(kind, float("-inf")) < self._log_interval:
                self._suppressed[kind] = self._suppressed.get(kind, 0) + 1
                return
            suppressed = self._suppressed.get(kind, 0)
            self._last_log[kind] = now
            self._suppressed[kind] = 0

        extra = f" ({suppressed} similar suppressed)" if suppressed else ""
        print(f"[{self.source_label}] {message}{extra}")
