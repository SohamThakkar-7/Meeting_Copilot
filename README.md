# Meeting_Copilot

A real-time meeting copilot. It listens to both sides of a conversation — your
microphone and the system audio coming out of your speakers — transcribes them
separately, and streams short suggestions into a floating always-on-top overlay
while you talk.

The interesting problem here isn't calling a model. It's deciding *when* to call
one: turns end constantly, and a naive "one end-of-turn, one request" loop
produces a flood of answers that are stale before they finish rendering.

## Pipeline

```
mic ────────┐                  ┌── Deepgram Flux ──┐
            ├─ VAD gating ─────┤                   ├─ session context ─ LLM ─ overlay
system ─────┘                  └── Deepgram Flux ──┘
```

Each source gets its own VAD and its own STT connection, so the transcript
stays speaker-attributed (`You` vs `Them`) without diarization.

| Package | Responsibility |
| --- | --- |
| `audio_capture/` | Dual-channel capture (mic + loopback), resampling, Silero VAD gating |
| `stt/` | Deepgram Flux streaming clients, one per source |
| `context/` | Turn assembly, ordering, token-budget trimming, reconnect recovery |
| `llm/` | Scheduling (debounce / rate floor / supersede), prompt building, providers |
| `overlay/` | Tkinter always-on-top overlay and the global hotkey |

### When it asks the model

Three guards, applied in order, in `llm/orchestrator.py`:

- **Debounce** — a completed turn arms a timer rather than firing. A burst of
  short turns ("yeah" / "right" / "mhm") collapses into one request against the
  fuller transcript.
- **Minimum interval** — a floor between requests regardless of debounce. This
  is also what keeps a free-tier key inside its rate limit.
- **Supersede** — a new request cancels any still-streaming previous one, whose
  answer is about a conversation that has already moved on.

The hotkey path bypasses the first two: the user asked, explicitly, now.

## Requirements

- Python 3.10+
- Windows (loopback capture via `PyAudioWPatch`); macOS capture path exists via
  `pysysaudio` but the overlay hotkey is Windows-only
- A Deepgram API key, plus a Groq or Gemini key if you want live suggestions

## Setup

```bash
git clone https://github.com/SohamThakkar-7/Meeting_Copilot.git
cd Meeting_Copilot

python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

copy .env.example .env         # macOS/Linux: cp .env.example .env
```

Then fill in your keys in `.env`. It is gitignored — don't commit it.

## Running

```bash
python main.py                          # mock LLM: no key, no cost
python main.py --provider groq          # live suggestions (fastest)
python main.py --provider gemini
python main.py --provider gemini --trigger-on system
python main.py --seconds 120            # auto-stop after 2 minutes
```

`Ctrl+Alt+J` anywhere forces a suggestion. Right-click the overlay to quit.

Useful flags: `--debounce-ms`, `--min-interval-ms`, `--max-context-tokens`,
`--max-tokens`, `--model`.

### Diagnostics

```bash
python check_mic.py    # verify capture devices without the rest of the stack
python check_llm.py    # feed a scripted conversation through the LLM layer
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover context assembly, orchestrator scheduling, prompt building and
the overlay's generation filter. None of them need audio hardware or a network.
