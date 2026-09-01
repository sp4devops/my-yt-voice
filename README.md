# my-yt-voice

A focused prototype for generating a calm bilingual **English-through-Tamil sleep-learning podcast** with IMS-Toucan.

## Characters

- **Female Tamil learner** — soft, pleasant, curious, eager to learn.
- **Male English professor** — calm, patient, clear, reassuring.

## Current target

Generate one natural ~4 minute WAV. No video, music, subtitles, or visual work yet.

## How it works

Each dialogue turn is tagged as `tam` or `eng`. IMS-Toucan switches language per turn and uses two fixed artificial speaker embeddings, so the same Tamil female and English male remain consistent across the lesson.

The pacing is intentionally slower with short pauses between teaching beats so it remains comfortable for sleep listening.

## Run

The easiest route is **GitHub Actions**:

1. Open **Actions**.
2. Run **Generate Toucan podcast demo**.
3. Download the `english-through-tamil-demo` artifact.

Local Linux execution is also supported:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 espeak-ng libasound-dev libportaudio2 libsqlite3-dev

git clone --depth 1 --branch MassiveScaleToucan https://github.com/DigitalPhonetics/IMS-Toucan.git vendor/IMS-Toucan
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r vendor/IMS-Toucan/requirements.txt
python generate_podcast.py
```

Output:

`output/english-through-tamil-demo.wav`

## Next tuning step

Listen specifically for:
- whether the Tamil speaker reads Tamil naturally,
- whether the two synthetic speaker seeds are convincingly female/male,
- whether English pronunciation is clean,
- whether the pace is relaxing rather than dull,
- whether pauses feel conversational.

The voice seeds and prosody controls are centralized in `VOICE_CONFIG` inside `generate_podcast.py`.
