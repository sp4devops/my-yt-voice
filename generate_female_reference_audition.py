import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
TOUCAN_DIR = ROOT / "vendor" / "IMS-Toucan"
sys.path.insert(0, str(TOUCAN_DIR))

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

SR = 24000
REFERENCE = ROOT / "reference" / "female-reference.wav"
RAW_DIR = ROOT / "output" / "female-reference-audition" / "raw"
CLEAN_DIR = ROOT / "output" / "female-reference-audition" / "clean"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

TEXT = (
    "வணக்கம். இன்று நாம் மெதுவாகவும் தெளிவாகவும் ஆங்கிலம் கற்றுக்கொள்ளலாம். "
    "தவறு செய்தாலும் பரவாயில்லை. ஒவ்வொரு வாக்கியமாக நிதானமாக பயிற்சி செய்வோம்."
)

# Conservative settings: low creativity, controlled pitch/energy, slightly slower
# delivery. The goal is stability and intelligibility, not dramatic expression.
VARIANTS = [
    {"name": "A-steady",  "duration": 1.00, "pitch": 0.72, "energy": 0.78, "prosody": 0.00},
    {"name": "B-soft",    "duration": 1.04, "pitch": 0.78, "energy": 0.76, "prosody": 0.00},
    {"name": "C-natural", "duration": 1.02, "pitch": 0.84, "energy": 0.82, "prosody": 0.01},
    {"name": "D-slower",  "duration": 1.08, "pitch": 0.76, "energy": 0.78, "prosody": 0.00},
    {"name": "E-warm",    "duration": 1.05, "pitch": 0.82, "energy": 0.80, "prosody": 0.02},
]


def peak_normalize(wav: np.ndarray, peak: float = 0.86) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav if m == 0 else (wav * (peak / m)).astype(np.float32)


def clean_audio(src: Path, dst: Path):
    # Gentle cleanup only: remove sub-bass/ultrasonic-ish content and normalize
    # loudness. Avoid aggressive denoise that can create metallic artifacts.
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af",
        "highpass=f=65,lowpass=f=10500,"
        "acompressor=threshold=-24dB:ratio=1.6:attack=20:release=180,"
        "loudnorm=I=-20:LRA=7:TP=-2",
        "-ar", str(SR),
        "-ac", "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    if not REFERENCE.exists():
        raise FileNotFoundError(
            f"Missing {REFERENCE}. Add a clean 15-30 second, single-speaker, "
            "rights-cleared female Tamil reference WAV before running this workflow."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tts = ToucanTTSInterface(device=device, language="tam")
    # Toucan averages embeddings when given multiple files; for this audition we
    # start with one clean reference and isolate synthesis settings.
    tts.set_utterance_embedding(path_to_reference_audio=str(REFERENCE))

    manifest = {
        "reference": str(REFERENCE.relative_to(ROOT)),
        "text": TEXT,
        "variants": [],
        "notes": [
            "Reference-audio speaker embedding; no synthetic GAN seed is used.",
            "All audition text is pure Tamil to avoid code-switch phonemizer errors.",
            "Prosody creativity is near zero to reduce shakiness and unstable delivery.",
            "Final output receives only gentle filtering/compression/loudness normalization.",
        ],
    }

    for i, cfg in enumerate(VARIANTS, start=1):
        wav, sr = tts(
            TEXT,
            duration_scaling_factor=cfg["duration"],
            pitch_variance_scale=cfg["pitch"],
            energy_variance_scale=cfg["energy"],
            pause_duration_scaling_factor=1.08,
            prosody_creativity=cfg["prosody"],
            loudness_in_db=-28.0,
        )
        if sr != SR:
            raise RuntimeError(f"Unexpected sample rate {sr}")

        wav = peak_normalize(wav)
        raw = RAW_DIR / f"{i:02d}-{cfg['name']}.wav"
        clean = CLEAN_DIR / f"{i:02d}-{cfg['name']}.wav"
        sf.write(raw, wav, SR)
        clean_audio(raw, clean)

        manifest["variants"].append({
            **cfg,
            "raw": str(raw.relative_to(ROOT)),
            "clean": str(clean.relative_to(ROOT)),
            "duration_s": round(len(wav) / SR, 2),
        })
        print(f"Generated {clean.name}")

    with open(CLEAN_DIR.parent / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
