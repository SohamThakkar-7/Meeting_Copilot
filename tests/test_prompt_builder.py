"""prompt_builder: the static/volatile split that prompt caching depends on.

Caching is a prefix match, so the system half must be byte-identical on every
request forever. That's an invariant a test can actually hold you to -- the
failure mode otherwise is silent: no error, just a cache that never hits and
a bill that quietly doubles.
"""

from llm.prompt_builder import SYSTEM_PROMPT, build_prompt


def test_system_half_is_identical_across_calls():
    first, _ = build_prompt("Them: hello")
    second, _ = build_prompt("Them: something else entirely")
    assert first == second == SYSTEM_PROMPT


def test_system_half_carries_nothing_per_request():
    """Anything variable above the cache breakpoint kills the prefix. The
    cheap proxy: the transcript must not appear in the system half."""
    marker = "UNIQUE-TRANSCRIPT-MARKER"
    system, user = build_prompt(f"Them: {marker}")
    assert marker not in system
    assert marker in user


def test_context_window_is_passed_through_verbatim():
    context = "[Active window: Meet]\nThem: hi\nYou (still speaking): hel"
    _, user = build_prompt(context)
    assert context in user


def test_user_half_is_more_than_the_bare_transcript():
    # The transcript needs framing or the model has no instruction to act on.
    _, user = build_prompt("Them: hi")
    assert len(user) > len("Them: hi")


def test_empty_context_still_produces_a_usable_prompt():
    system, user = build_prompt("")
    assert system == SYSTEM_PROMPT
    assert user.strip()
