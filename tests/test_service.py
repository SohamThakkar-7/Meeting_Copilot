
import queue

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from service.app import _Broadcaster, app, brain  # noqa: E402

MOCK_SESSION = {
    "provider": "mock",
    "debounce_ms": 50.0,
    "min_interval_ms": 50.0,
    "max_context_tokens": 500,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    brain.shutdown()


def _drain(q, timeout=5.0):
    """Collect published events until the first 'done'."""
    collected = []
    while True:
        try:
            payload = q.get(timeout=timeout)
        except queue.Empty:
            return collected
        collected.append(payload)
        if payload["type"] == "done":
            return collected


def say(client, source, text):
    for event in (
        {"event": "StartOfTurn"},
        {"event": "Update", "transcript": text},
        {"event": "EndOfTurn", "transcript": text},
    ):
        response = client.post("/events", json={"source": source, "event": event})
        assert response.status_code == 200


# --- lifecycle ------------------------------------------------------------


def test_health_before_any_session(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["session"] is False


def test_events_without_a_session_are_rejected(client):
    response = client.post(
        "/events", json={"source": "mic", "event": {"event": "StartOfTurn"}}
    )
    assert response.status_code == 409


def test_trigger_without_a_session_is_rejected(client):
    assert client.post("/trigger").status_code == 409


def test_unknown_provider_is_a_client_error(client):
    response = client.post("/session", json={"provider": "nope"})
    assert response.status_code == 400


def test_unknown_trigger_on_is_a_client_error(client):
    response = client.post("/session", json={"provider": "mock", "trigger_on": "nobody"})
    assert response.status_code == 400


def test_session_reports_the_provider(client):
    body = client.post("/session", json=MOCK_SESSION).json()
    assert body["provider"] == "mock"
    assert client.get("/health").json()["session"] is True


def test_closing_a_session_clears_it(client):
    client.post("/session", json=MOCK_SESSION)
    assert client.delete("/session").status_code == 200
    assert client.get("/health").json()["session"] is False
    # And the orchestrator is gone with it, so events are refused again.
    assert client.post("/trigger").status_code == 409


# --- the actual pipeline ---------------------------------------------------


def test_a_completed_turn_produces_a_streamed_suggestion(client):
    client.post("/session", json=MOCK_SESSION)
    q = brain.events.subscribe()

    say(client, "system", "What do you know about consensus protocols?")
    events = _drain(q)

    kinds = [e["type"] for e in events]
    assert kinds[0] == "turn"
    assert "start" in kinds and "token" in kinds and kinds[-1] == "done"

    turn = events[0]
    assert turn["source"] == "system"
    assert "consensus" in turn["text"]

    done = events[-1]
    assert done["provider"] == "mock"
    assert done["error"] is None
    # Every field Suggestion needs must survive the trip, or the host cannot
    # rebuild the dataclass its overlay callback expects.
    for field in ("text", "reason", "provider", "ttft_ms", "total_ms", "cancelled"):
        assert field in done

    # Tokens belong to the generation that started, so the overlay's filter
    # can drop a superseded request's tail.
    generations = {e["generation"] for e in events if e["type"] != "turn"}
    assert len(generations) == 1


def test_speaker_attribution_survives_the_round_trip(client):
    client.post("/session", json=MOCK_SESSION)
    say(client, "system", "Tell me about your last project.")
    say(client, "mic", "It was a streaming pipeline.")

    context = client.get("/context").json()["context"]
    assert "Them: Tell me about your last project." in context
    assert "You: It was a streaming pipeline." in context


def test_hotkey_trigger_fires_without_a_new_turn(client):
    client.post("/session", json=MOCK_SESSION)
    say(client, "mic", "Something to talk about.")
    _drain(brain.events.subscribe())

    q = brain.events.subscribe()
    assert client.post("/trigger").status_code == 200
    events = _drain(q)
    starts = [e for e in events if e["type"] == "start"]
    assert starts and starts[0]["reason"] == "hotkey"


def test_flush_finalizes_a_half_spoken_turn(client):
    client.post("/session", json=MOCK_SESSION)
    q = brain.events.subscribe()

    # StartOfTurn and an Update, but the connection drops before EndOfTurn.
    client.post("/events", json={"source": "mic", "event": {"event": "StartOfTurn"}})
    client.post(
        "/events",
        json={"source": "mic", "event": {"event": "Update", "transcript": "half a th"}},
    )
    assert client.post("/flush", json={"source": "mic"}).status_code == 200

    events = _drain(q)
    assert events[0]["type"] == "turn"
    assert events[0]["text"] == "half a th"


# --- fan-out ---------------------------------------------------------------


def test_broadcaster_reaches_every_subscriber():
    b = _Broadcaster()
    first, second = b.subscribe(), b.subscribe()
    b.publish({"type": "ping"})
    assert first.get_nowait() == {"type": "ping"}
    assert second.get_nowait() == {"type": "ping"}


def test_broadcaster_drops_rather_than_blocking_the_request_thread():
    b = _Broadcaster(maxsize=2)
    q = b.subscribe()
    for i in range(10):
        b.publish({"type": "token", "i": i})  # must not block
    assert q.qsize() == 2


def test_unsubscribed_queues_stop_receiving():
    b = _Broadcaster()
    q = b.subscribe()
    b.unsubscribe(q)
    b.publish({"type": "ping"})
    assert q.empty()
