import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
TOUCAN_DIR = ROOT / "vendor" / "IMS-Toucan"
sys.path.insert(0, str(TOUCAN_DIR))

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
from Modules.ControllabilityGAN.wgan.resnet_init import init_resnet
from Utility.storage_config import MODEL_DIR

SR = 24000
OUT_DIR = ROOT / "output" / "voice-auditions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEMALE_TEXT = "வணக்கம். இன்று நாம மெதுவாகவும் நிதானமாகவும் ஆங்கிலம் கற்றுக்கொள்ளலாம். தவறு செய்தாலும் பரவாயில்லை."
MALE_TEXT = "Good evening. We will learn English slowly, clearly, and one sentence at a time. There is no need to rush."

# We intentionally sample a spread of deterministic synthetic identities.
SEEDS = [3, 7, 11, 14, 18, 22, 27, 31, 36, 41, 47, 53]

ROLE_CONFIG = {
    "female": {
        "lang": "tam",
        "text": FEMALE_TEXT,
        "duration": 1.08,
        "pitch": 0.96,
        "energy": 0.88,
        "prosody": 0.06,
    },
    "male": {
        "lang": "eng",
        "text": MALE_TEXT,
        "duration": 1.12,
        "pitch": 0.80,
        "energy": 0.84,
        "prosody": 0.04,
    },
}


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def normalize_peak(wav: np.ndarray, peak: float = 0.88) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    if m > 0:
        wav = wav * (peak / m)
    return wav.astype(np.float32)


def load_voice_generator(gan_path: str, device: str):
    checkpoint = torch.load(gan_path, map_location="cpu")
    generator, _ = init_resnet(checkpoint["model_parameters"])
    state = {
        key.replace("module.", ""): value
        for key, value in checkpoint["generator_state_dict"].items()
    }
    generator.load_state_dict(state)
    generator = generator.to(device).eval()
    return generator, checkpoint["dataset_mean"], checkpoint["dataset_std"]


def make_voice_embedding(generator, mean, std, seed: int, device: str):
    torch.manual_seed(seed)
    z = torch.randn((1, generator.z_dim), dtype=torch.float32) * 0.4
    with torch.inference_mode():
        embedding = generator(z.to(device)).cpu()
    return embedding * std.cpu().unsqueeze(0) + mean.cpu().unsqueeze(0)


def median_f0(wav: np.ndarray) -> float:
    # Crude but useful first-pass ranking for synthetic candidates.
    f0 = librosa.yin(
        wav.astype(np.float64),
        fmin=60,
        fmax=350,
        sr=SR,
        frame_length=2048,
        hop_length=256,
    )
    voiced = f0[np.isfinite(f0)]
    if voiced.size == 0:
        return 0.0
    return float(np.median(voiced))


def synthesize(tts, embedding, cfg):
    tts.set_language(cfg["lang"])
    tts.set_utterance_embedding(embedding=embedding)
    wav, sr = tts(
        cfg["text"],
        duration_scaling_factor=cfg["duration"],
        pitch_variance_scale=cfg["pitch"],
        energy_variance_scale=cfg["energy"],
        pause_duration_scaling_factor=1.10,
        prosody_creativity=cfg["prosody"],
        loudness_in_db=-31.0,
    )
    if sr != SR:
        raise RuntimeError(f"Unexpected sample rate {sr}")
    return normalize_peak(np.asarray(wav, dtype=np.float32))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tts = ToucanTTSInterface(device=device)
    gan_path = hf_hub_download(
        cache_dir=MODEL_DIR,
        repo_id="Flux9665/ToucanTTS",
        filename="embedding_gan.pt",
    )
    generator, mean, std = load_voice_generator(gan_path, device)

    generated = {"female": [], "male": []}

    for role, cfg in ROLE_CONFIG.items():
        role_dir = OUT_DIR / role
        role_dir.mkdir(exist_ok=True)

        for seed in SEEDS:
            embedding = make_voice_embedding(generator, mean, std, seed, device)
            wav = synthesize(tts, embedding, cfg)
            f0 = median_f0(wav)

            path = role_dir / f"{role}-seed-{seed:02d}.wav"
            sf.write(path, wav, SR)
            generated[role].append(
                {
                    "seed": seed,
                    "median_f0_hz": round(f0, 1),
                    "path": str(path.relative_to(OUT_DIR)),
                    "duration_s": round(len(wav) / SR, 2),
                }
            )
            print(f"{role} seed={seed:02d} f0={f0:.1f}Hz")

    # Pitch is not a gender detector, but it is a useful way to narrow synthetic
    # auditions. Keep the six higher-pitched Tamil candidates and six lower-pitched
    # English candidates, then let a human ear make the final choice.
    selected = {
        "female": sorted(
            generated["female"], key=lambda x: x["median_f0_hz"], reverse=True
        )[:6],
        "male": sorted(
            generated["male"], key=lambda x: x["median_f0_hz"]
        )[:6],
    }

    manifest = {
        "notes": [
            "These are synthetic Toucan speaker identities, not clones of Deepika Arun or Oli Redman.",
            "Female target: soft, warm, clear Tamil audiobook-style delivery.",
            "Male target: calm, mature, slightly deep British-professor-style delivery.",
            "Median F0 is only a first-pass ranking signal. Choose final voices by listening.",
        ],
        "selected": selected,
        "all_candidates": generated,
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Build two quick comparison reels in ranked order.
    for role in ("female", "male"):
        chunks = [silence(1.0)]
        for item in selected[role]:
            wav, _ = sf.read(OUT_DIR / item["path"], dtype="float32")
            chunks.extend([wav, silence(2.0)])
        reel = np.concatenate(chunks)
        sf.write(OUT_DIR / f"{role}-top6-comparison.wav", reel, SR)
        print(f"Wrote {role}-top6-comparison.wav")

    print("Selected female seeds:", [x["seed"] for x in selected["female"]])
    print("Selected male seeds:", [x["seed"] for x in selected["male"]])


if __name__ == "__main__":
    main()
