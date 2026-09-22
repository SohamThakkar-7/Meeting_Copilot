

import threading
import time

import pytest

tkinter = pytest.importorskip("tkinter")

from overlay.window import OverlayWindow  # noqa: E402


@pytest.fixture(scope="module")
def overlay():
    try:
        window = OverlayWindow()
    except tkinter.TclError as exc:
        pytest.skip(f"no usable Tk display: {exc}")
    yield window


@pytest.fixture(autouse=True)
def clean_slate(overlay):
    """Each test starts from an empty queue and a blank label."""
    while not overlay._queue.empty():
        overlay._queue.get_nowait()
    overlay._label.config(text="")
    yield


def drain(overlay):
    """Apply whatever is queued, on this thread, right now."""
    overlay._drain()
    return overlay._label.cget("text")


# --- rendering ------------------------------------------------------------


def test_tokens_render_in_order(overlay):
    generation = overlay.begin()
    for word in ("alpha ", "beta ", "gamma"):
        overlay.append(generation, word)
    assert drain(overlay) == "alpha beta gamma"


def test_begin_clears_the_previous_suggestion(overlay):
    first = overlay.begin()
    overlay.append(first, "old text")
    drain(overlay)

    overlay.begin()
    assert drain(overlay) == "..."          # placeholder, previous text gone


def test_finish_note_replaces_the_body(overlay):
    generation = overlay.begin()
    overlay.append(generation, "partial")
    drain(overlay)

    overlay.finish(generation, "[rate limited]")
    assert drain(overlay) == "[rate limited]"


def test_finish_without_a_note_leaves_the_text(overlay):
    generation = overlay.begin()
    overlay.append(generation, "keep me")
    drain(overlay)

    overlay.finish(generation)
    assert drain(overlay) == "keep me"


# --- the generation filter ------------------------------------------------


def test_stale_generation_tokens_are_dropped(overlay):
    """The whole reason generations exist: cancellation is cooperative, so a
    superseded request keeps emitting for a chunk or two."""
    first = overlay.begin()
    overlay.append(first, "OLD ")
    drain(overlay)

    second = overlay.begin()
    overlay.append(first, "STALE ")         # still draining from the dead one
    overlay.append(second, "NEW")

    text = drain(overlay)
    assert text == "NEW"
    assert "STALE" not in text and "OLD" not in text


def test_stale_finish_does_not_hide_a_live_suggestion(overlay):
    first = overlay.begin()
    second = overlay.begin()
    overlay.append(second, "live")
    overlay.finish(first, "[superseded]")   # must not overwrite the live one
    assert drain(overlay) == "live"


def test_begin_returns_strictly_increasing_generations(overlay):
    generations = [overlay.begin() for _ in range(5)]
    assert generations == sorted(generations)
    assert len(set(generations)) == 5


def test_begin_is_safe_from_many_threads(overlay):
    """begin() increments under a lock; without it two threads can collide
    on the same generation and neither's tokens get filtered."""
    seen = []
    lock = threading.Lock()

    def worker():
        generation = overlay.begin()
        with lock:
            seen.append(generation)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert len(seen) == 20
    assert len(set(seen)) == 20, "duplicate generation handed out"


# --- dismiss -------------------------------------------------------------


def test_dismiss_clears_and_hides(overlay):
    generation = overlay.begin()
    overlay.append(generation, "nothing useful")
    drain(overlay)

    overlay.dismiss(generation)
    assert drain(overlay) == ""


def test_stale_dismiss_does_not_hide_a_live_suggestion(overlay):
    first = overlay.begin()
    second = overlay.begin()
    overlay.append(second, "live")
    overlay.dismiss(first)              # superseded request bowing out
    assert drain(overlay) == "live"


# --- shutdown (consumes the root, so it runs last) ------------------------


def test_zz_close_from_another_thread_ends_run(overlay):
    def worker():
        time.sleep(0.15)
        overlay.close()

    threading.Thread(target=worker, daemon=True).start()

    started = time.perf_counter()
    overlay.run()                            # must return, not hang
    assert time.perf_counter() - started < 5.0
