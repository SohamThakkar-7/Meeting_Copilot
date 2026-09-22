

import threading
import time

import pytest

from context.session_manager import SessionContextManager, Turn


def feed(session, source, text, events=("StartOfTurn", "Update", "EndOfTurn")):
    """Push one complete turn through in the shape Deepgram Flux delivers it."""
    for kind in events:
        session.handle_turn_event(
            source, {"event": kind, "transcript": "" if kind == "StartOfTurn" else text}
        )


@pytest.fixture
def session():
    # Window tracking off: it shells out to Win32 and isn't what we're testing.
    return SessionContextManager(track_active_window=False)


# --- basic assembly -------------------------------------------------------


def test_completed_turn_is_recorded_with_speaker_label(session):
    feed(session, "mic", "hello there")
    turns = session.turns
    assert len(turns) == 1
    assert turns[0].text == "hello there"
    assert turns[0].speaker == "You"


def test_system_source_is_labelled_them(session):
    feed(session, "system", "hello back")
    assert session.turns[0].speaker == "Them"


def test_context_window_renders_both_sides(session):
    feed(session, "system", "what have you built")
    feed(session, "mic", "a copilot")
    assert session.get_context_window(include_window=False) == (
        "Them: what have you built\nYou: a copilot"
    )


def test_turn_is_not_recorded_until_end_of_turn(session):
    feed(session, "mic", "half a sentence", events=("StartOfTurn", "Update"))
    assert session.turns == []


def test_empty_transcript_creates_no_turn(session):
    feed(session, "mic", "")
    assert session.turns == []


def test_update_replaces_rather_than_appends(session):
    # Flux resends the whole turn transcript each time; appending would
    # produce "hellohello therehello there now".
    session.handle_turn_event("mic", {"event": "StartOfTurn", "transcript": ""})
    session.handle_turn_event("mic", {"event": "Update", "transcript": "hello"})
    session.handle_turn_event("mic", {"event": "Update", "transcript": "hello there"})
    session.handle_turn_event("mic", {"event": "EndOfTurn", "transcript": "hello there"})
    assert session.turns[0].text == "hello there"


# --- the ordering guarantee ----------------------------------------------


def test_overlapping_turns_order_by_start_not_arrival(session):
    """The reason turns are inserted with bisect rather than appended.

    Two sockets on two threads: system starts speaking first but finishes
    second. Appending on arrival would read backwards.
    """
    session.handle_turn_event("system", {"event": "StartOfTurn", "transcript": ""})
    time.sleep(0.01)
    session.handle_turn_event("mic", {"event": "StartOfTurn", "transcript": ""})

    # mic finishes first...
    session.handle_turn_event("mic", {"event": "EndOfTurn", "transcript": "second"})
    # ...but system started earlier, so it must come first in the transcript.
    session.handle_turn_event("system", {"event": "EndOfTurn", "transcript": "first"})

    assert [turn.text for turn in session.turns] == ["first", "second"]


# --- reconnect recovery ---------------------------------------------------


def test_update_without_start_of_turn_opens_a_turn_lazily(session):
    """After a reconnect we join mid-turn and never see StartOfTurn. The
    words should still land rather than being dropped."""
    session.handle_turn_event("mic", {"event": "Update", "transcript": "mid flight"})
    session.handle_turn_event("mic", {"event": "EndOfTurn", "transcript": "mid flight"})
    assert session.turns[0].text == "mid flight"


def test_flush_pending_finalizes_a_dangling_turn(session):
    feed(session, "mic", "cut off", events=("StartOfTurn", "Update"))
    assert session.turns == []

    session.flush_pending("mic")
    assert [turn.text for turn in session.turns] == ["cut off"]


def test_flush_pending_is_safe_when_nothing_is_pending(session):
    session.flush_pending("mic")
    assert session.turns == []


def test_new_start_of_turn_discards_a_stale_pending_turn(session):
    feed(session, "mic", "lost to a reconnect", events=("StartOfTurn", "Update"))
    session.handle_turn_event("mic", {"event": "StartOfTurn", "transcript": ""})
    session.handle_turn_event("mic", {"event": "EndOfTurn", "transcript": "fresh"})
    assert [turn.text for turn in session.turns] == ["fresh"]


def test_turn_resumed_keeps_the_turn_open(session):
    session.handle_turn_event("mic", {"event": "StartOfTurn", "transcript": ""})
    session.handle_turn_event("mic", {"event": "EagerEndOfTurn", "transcript": "wait"})
    session.handle_turn_event("mic", {"event": "TurnResumed", "transcript": ""})
    assert session.turns == []          # not finalized by the eager guess

    session.handle_turn_event(
        "mic", {"event": "EndOfTurn", "transcript": "wait, there's more"}
    )
    assert session.turns[0].text == "wait, there's more"


def test_unknown_event_types_are_ignored(session):
    session.handle_turn_event("mic", {"event": "SomethingNew", "transcript": "x"})
    assert session.turns == []


# --- token budget ---------------------------------------------------------


def test_oldest_turns_are_evicted_when_over_budget():
    session = SessionContextManager(max_tokens=20, track_active_window=False)
    for index in range(10):
        feed(session, "mic", f"turn number {index} with some padding text")

    assert session.approx_tokens() <= 20
    assert session.dropped_turns > 0
    # What survives is the most recent, not the first.
    assert "9" in session.turns[-1].text


def test_context_window_admits_that_it_trimmed():
    session = SessionContextManager(max_tokens=20, track_active_window=False)
    for index in range(10):
        feed(session, "mic", f"turn number {index} with some padding text")

    assert "trimmed" in session.get_context_window(include_window=False)


def test_reset_clears_everything(session):
    feed(session, "mic", "hello")
    session.reset()
    assert session.turns == []
    assert session.dropped_turns == 0


# --- partials -------------------------------------------------------------


def test_in_progress_turn_is_marked_still_speaking(session):
    feed(session, "mic", "done")
    session.handle_turn_event("system", {"event": "StartOfTurn", "transcript": ""})
    session.handle_turn_event("system", {"event": "Update", "transcript": "mid"})

    window = session.get_context_window(include_window=False)
    assert "You: done" in window
    assert "Them (still speaking): mid" in window


def test_partials_can_be_excluded(session):
    session.handle_turn_event("mic", {"event": "StartOfTurn", "transcript": ""})
    session.handle_turn_event("mic", {"event": "Update", "transcript": "mid"})
    assert session.get_context_window(include_partial=False, include_window=False) == ""


# --- callback contract ----------------------------------------------------


def test_on_turn_complete_fires_once_per_finalized_turn():
    seen = []
    session = SessionContextManager(
        track_active_window=False, on_turn_complete=seen.append
    )
    feed(session, "mic", "one")
    feed(session, "system", "two")

    assert [turn.text for turn in seen] == ["one", "two"]
    assert all(isinstance(turn, Turn) for turn in seen)


def test_on_turn_complete_fires_outside_the_lock():
    """The callback triggers an LLM call. If it ran under the lock, a slow
    model would block every incoming Deepgram event."""
    reentered = []

    def callback(turn):
        # Would deadlock if the manager's lock were still held.
        reentered.append(session.get_context_window(include_window=False))

    session = SessionContextManager(
        track_active_window=False, on_turn_complete=callback
    )

    done = threading.Event()
    threading.Thread(
        target=lambda: (feed(session, "mic", "hello"), done.set()), daemon=True
    ).start()

    assert done.wait(timeout=2.0), "callback deadlocked against the manager lock"
    assert reentered == ["You: hello"]
