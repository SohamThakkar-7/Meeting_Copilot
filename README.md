# Meeting_Copilot

A real-time meeting copilot. It listens to both sides of a conversation — your
microphone and the system audio from your speakers — transcribes them
separately, and streams short suggestions into a floating always-on-top overlay
while you talk.

The interesting problem isn't calling a model. It's deciding *when* to call one:
turns end constantly, and a naive "one turn, one request" loop produces answers
that are stale before they finish rendering. Three guards in
`llm/orchestrator.py` handle it — a **debounce** so a burst of "yeah" / "right"
collapses into one request, a **minimum interval** that keeps a free-tier key
inside its rate limit, and **supersede**, which cancels a still-streaming answer
about a conversation that has already moved on.

```
mic ────────┐                  ┌── Deepgram Flux ──┐
            ├─ VAD gating ─────┤                   ├─ context ─ LLM ─ overlay
system ─────┘                  └── Deepgram Flux ──┘
```

Each source gets its own VAD and STT connection, so the transcript stays
speaker-attributed (`You` vs `Them`) without diarization.

| Package | Responsibility |
| --- | --- |
| `audio_capture/` | Dual-channel capture (mic + loopback), resampling, Silero VAD |
| `stt/` | Deepgram Flux streaming clients, one per source |
| `context/` | Turn assembly, ordering, token-budget trimming, reconnect recovery |
| `llm/` | Scheduling, prompt building, providers |
| `overlay/` | Tkinter overlay and the global hotkey |
| `service/` | The same context + LLM stack behind HTTP, for `--brain` |

## Prerequisites

- **Windows.** Loopback capture uses `PyAudioWPatch`; macOS works via
  `pysysaudio` but has no global hotkey, and Linux is unsupported.
- **Python 3.10+** from python.org — the Microsoft Store build ships without
  `tkinter` and the overlay won't start.
- A **Deepgram** API key, plus a **Groq** or **Gemini** key for live suggestions.

## Setup

```bash
git clone https://github.com/SohamThakkar-7/Meeting_Copilot.git
cd Meeting_Copilot

python -m venv venv
venv\Scripts\activate          # macOS: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root with your keys:

```ini
DEEPGRAM_API_KEY=
GROQ_API_KEY=
GEMINI_API_KEY=
# Optional: override the measured default (gemini-3.5-flash-lite).
# GEMINI_MODEL=
```

`.env` is gitignored — don't commit it. The install pulls PyTorch via
`silero-vad`, so expect a large download.

Set your Windows input and output devices **before** launching; both are bound
once at startup.

## Running

```bash
python main.py                          # mock LLM: no key, no cost
python main.py --provider groq          # live suggestions (fastest)
python main.py --provider groq --trigger-on system
python main.py --seconds 120            # auto-stop after 2 minutes
```

`Ctrl+Alt+J` anywhere forces a suggestion. Right-click the overlay to quit.

`--trigger-on system` means only the other person's turns trigger suggestions —
usually what you want in a meeting, and it halves your request rate. Other
flags: `--debounce-ms`, `--min-interval-ms`, `--max-context-tokens`,
`--max-tokens`, `--model`.

The overlay stays hidden when the model has nothing useful to add, so silence
is a valid response, not a bug.

### Diagnostics

```bash
python check_mic.py --list     # what inputs exist
python check_mic.py            # live level next to the VAD's verdict
python check_llm.py            # a scripted conversation through the LLM layer
```

## Running the brain separately

Everything downstream of the transcript — turn assembly, scheduling, the
provider call — can run as its own HTTP service, while capture and the overlay
stay local (they need this machine's audio devices and display).

```bash
pip install -r requirements-service.txt
uvicorn service.app:app --host 0.0.0.0 --port 8000

python main.py --brain http://localhost:8000 --provider groq
```

The service needs `GROQ_API_KEY` / `GEMINI_API_KEY` but **not**
`DEEPGRAM_API_KEY` — STT stays on the host, next to the microphone. Without
`--brain`, the pipeline runs in one process exactly as before.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers context assembly, orchestrator scheduling, prompt building, the overlay's
generation filter and the brain service. None of it needs audio hardware or a
network.
