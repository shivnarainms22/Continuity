"""Generate the demo narration with Gemini TTS and mux it onto the recorded video.

Committed for the same reason `record-demo.mjs` is: the video artefact itself is
gitignored, so without this the audio track could not be rebuilt.

    uv run python docs/submission/add-narration.py <video.webm> <outdir>

Why the awkward parts are the way they are:

* Timings in `narration.json` were measured off the recording (caption-band change
  detection), not copied from `record-demo.mjs`. The investigation runs live, so the
  intended waits and the real shot lengths differ by seconds. Re-record and they must be
  re-measured.
* TTS length varies run to run for identical input, so each line is generated several
  times and the longest take that still fits its slot wins. Anything still over is nudged
  with `atempo`, capped at 1.12 because past that it is audible.
* Requests are serial with backoff on the request itself. The per-minute quota is
  `global_generate_content_requests_per_minute_per_project_per_base_model`; retrying an
  enclosing loop would re-spend against the very limit it is waiting on.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
import wave

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

SR = 24000
MAX_TEMPO = 1.12
TAKES = 3
STYLE = (
    "Read this as the narrator of a polished technical product demo. "
    "Calm, confident, measured, warm but not salesy. Even pace, clear diction, "
    "no rising sing-song. Plain statement of fact.\n\n"
)
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not on PATH. Install it (winget install Gyan.FFmpeg).")
    return exe


def read_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(path: pathlib.Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())


def trim_silence(a: np.ndarray, thresh_db: float = -42.0, pad_ms: int = 60) -> np.ndarray:
    """Strip the lead-in and tail silence the model pads around each utterance.

    Left in place it inflates a short line by seconds and blows its slot for no speech.
    """
    win = int(SR * 0.01)
    n = len(a) // win
    if n == 0:
        return a
    frames = a[: n * win].astype(np.float32).reshape(n, win)
    db = 20 * np.log10((np.sqrt((frames**2).mean(axis=1)) + 1e-9) / 32768.0)
    loud = np.where(db > thresh_db)[0]
    if not len(loud):
        return a
    pad = pad_ms // 10
    return a[max(0, loud[0] - pad) * win : min(n, loud[-1] + 1 + pad) * win]


def synth(client: genai.Client, model: str, voice: str, text: str) -> np.ndarray:
    for attempt in range(8):
        try:
            response = client.models.generate_content(
                model=model,
                contents=STYLE + text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                ),
            )
            raw = response.candidates[0].content.parts[0].inline_data.data
            return np.frombuffer(raw, dtype=np.int16)
        except Exception as exc:  # noqa: BLE001 - only 429 is retryable, the rest re-raise
            if "RESOURCE_EXHAUSTED" not in str(exc) and "429" not in str(exc):
                raise
            time.sleep(min(60, 6 * (attempt + 1)) + random.uniform(0, 3))
    raise RuntimeError("TTS still rate-limited after 8 attempts")


def retime(exe: str, a: np.ndarray, factor: float, work: pathlib.Path) -> np.ndarray:
    src, dst = work / "_in.wav", work / "_out.wav"
    write_wav(src, a)
    subprocess.run(
        [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-filter:a",
            f"atempo={factor:.4f}",
            str(dst),
            "-y",
        ],
        check=True,
    )
    return read_wav(dst)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    video = pathlib.Path(sys.argv[1]).resolve()
    outdir = pathlib.Path(sys.argv[2]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "narration-parts"
    work.mkdir(exist_ok=True)

    here = pathlib.Path(__file__).parent
    spec = json.loads((here / "narration.json").read_text(encoding="utf-8"))
    load_dotenv(here.parent.parent / ".env")

    exe = ffmpeg()
    client = genai.Client(
        vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global"
    )

    print(f"{'id':>4} {'start':>7} {'budget':>7} {'final':>7} {'slack':>7} {'tempo':>6}")
    placed: list[tuple[float, np.ndarray]] = []
    over = []
    for line in spec["lines"]:
        best: tuple[tuple[int, float], np.ndarray] | None = None
        for _ in range(TAKES):
            take = trim_silence(synth(client, spec["model"], spec["voice"], line["text"]))
            seconds = len(take) / SR
            # Prefer the longest take that still fits; a slot filled edge to edge sounds
            # deliberate, a rushed one does not.
            score = (0, -seconds) if seconds <= line["budget"] else (1, seconds)
            if best is None or score < best[0]:
                best = (score, take)
        assert best is not None
        audio = best[1]
        factor = 1.0
        if len(audio) / SR > line["budget"]:
            factor = min(MAX_TEMPO, (len(audio) / SR) / line["budget"])
            audio = retime(exe, audio, factor, work)
        seconds = len(audio) / SR
        slack = line["budget"] - seconds
        if slack < -0.05:
            over.append(line["id"])
        print(
            f"{line['id']:>4} {line['start']:7.1f} {line['budget']:7.1f} "
            f"{seconds:7.2f} {slack:7.2f} {factor:6.3f}" + ("  OVER" if slack < -0.05 else "")
        )
        placed.append((line["start"], audio))

    if over:
        print(f"\nlines that do not fit even at {MAX_TEMPO}x: {over}")
        print("Shorten their text in narration.json rather than raising the cap.")
        return 1

    total = spec["trim_to_seconds"]
    master = np.zeros(int(total * SR), dtype=np.float32)
    for start, audio in placed:
        i = int(start * SR)
        j = min(len(master), i + len(audio))
        master[i:j] += audio[: j - i].astype(np.float32)
    master *= 0.89 * 32767 / np.abs(master).max()
    track = outdir / "narration.wav"
    write_wav(track, master.astype(np.int16))

    webm = outdir / "continuity-demo-narrated.webm"
    mp4 = outdir / "continuity-demo-narrated.mp4"
    common = [
        "-i",
        str(video),
        "-i",
        str(track),
        "-t",
        str(total),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-af",
        LOUDNORM,
    ]
    # WebM keeps the original VP8 stream untouched; only the container gains an audio track.
    subprocess.run(
        [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            *common,
            "-c:v",
            "copy",
            "-c:a",
            "libopus",
            "-b:a",
            "96k",
            str(webm),
            "-y",
        ],
        check=True,
    )
    # MP4/H.264 for YouTube, which is the actual submission path.
    subprocess.run(
        [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            *common,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(mp4),
            "-y",
        ],
        check=True,
    )
    print(f"\nwrote {webm}\nwrote {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
