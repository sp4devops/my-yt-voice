import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
TOUCAN_DIR = ROOT / "vendor" / "IMS-Toucan"
sys.path.insert(0, str(TOUCAN_DIR))

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Modules.ControllabilityGAN.GAN import GanWrapper
from huggingface_hub import hf_hub_download
from Utility.storage_config import MODEL_DIR

SR = 24000
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

VOICE_CONFIG = {
    "female": {
        "seed": 14,
        "sliders": [0.25, -0.15, 0.10, -0.10, 0.05, 0.10],
        "duration": 1.08,
        "pitch": 0.92,
        "energy": 0.90,
        "prosody": 0.08,
    },
    "male": {
        "seed": 41,
        "sliders": [-0.20, 0.10, -0.05, 0.10, -0.10, -0.05],
        "duration": 1.10,
        "pitch": 0.82,
        "energy": 0.88,
        "prosody": 0.05,
    },
}

def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)

def make_voice_embedding(wgan: GanWrapper, seed: int, sliders):
    wgan.set_latent(seed)
    vec = torch.tensor(sliders, dtype=torch.float32)
    return wgan.modify_embed(vec)

def normalize_peak(wav: np.ndarray, peak: float = 0.92) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    if m > 0:
        wav = wav * (peak / m)
    return wav.astype(np.float32)

def main():
    with open(ROOT / "script.json", "r", encoding="utf-8") as f:
        turns = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tts = ToucanTTSInterface(device=device)
    gan_path = hf_hub_download(
        cache_dir=MODEL_DIR,
        repo_id="Flux9665/ToucanTTS",
        filename="embedding_gan.pt",
    )
    wgan = GanWrapper(gan_path, num_cached_voices=60, device=device)

    embeddings = {
        name: make_voice_embedding(wgan, cfg["seed"], cfg["sliders"])
        for name, cfg in VOICE_CONFIG.items()
    }

    chunks = [silence(1.5)]
    for idx, turn in enumerate(turns, start=1):
        cfg = VOICE_CONFIG[turn["speaker"]]
        tts.set_language(turn["lang"])
        tts.set_utterance_embedding(embedding=embeddings[turn["speaker"]])

        wav, sr = tts(
            turn["text"],
            duration_scaling_factor=cfg["duration"],
            pitch_variance_scale=cfg["pitch"],
            energy_variance_scale=cfg["energy"],
            pause_duration_scaling_factor=1.12,
            prosody_creativity=cfg["prosody"],
            loudness_in_db=-31.0,
        )
        if sr != SR:
            raise RuntimeError(f"Unexpected sample rate {sr}")

        wav = normalize_peak(np.asarray(wav, dtype=np.float32), 0.88)
        chunks.append(wav)
        chunks.append(silence(float(turn.get("pause_after", 1.0))))
        print(f"[{idx}/{len(turns)}] {turn['speaker']} {turn['lang']}")

    final = np.concatenate(chunks)
    sf.write(OUT_DIR / "english-through-tamil-demo.wav", final, SR)
    duration = len(final) / SR
    print(f"Generated {duration:.1f}s -> {OUT_DIR / 'english-through-tamil-demo.wav'}")

if __name__ == "__main__":
    main()
