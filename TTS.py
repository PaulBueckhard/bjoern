from __future__ import annotations
import os
import subprocess
import shutil
import tempfile
from typing import Optional
from functools import lru_cache

VOICE_MAP = {
    "en": os.path.abspath("tts_models/piper-model-english.onnx"),
    "de": os.path.abspath("tts_models/piper-model-german.onnx"),
}

PIPER_BIN = os.environ.get("PIPER_BIN", "/usr/bin/piper")
APLAY_BIN = os.environ.get("APLAY_BIN", "aplay") 

def _have(cmd: str) -> bool:
    """Return True if 'cmd' exists on PATH."""
    return bool(shutil.which(cmd))

def _voice_for(language: str) -> Optional[str]:
    """Return model path for a language code like 'en' or 'de'."""
    lang = (language or "de").split("-")[0].lower()
    return VOICE_MAP.get(lang)

@lru_cache(maxsize=1)
def _piper_flag_style() -> str:
    """
    Detect which CLI flags Piper supports by inspecting `piper --help`.
    Returns one of: 'short', 'long', or 'unknown'.
    """
    try:
        proc = subprocess.run(
            [PIPER_BIN, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        if "-m" in text and "-f" in text:
            return "short"
        if "--model" in text and "--output_file" in text:
            return "long"
        return "unknown"
    except Exception:
        return "unknown"

def speak(text: str, language: str = "de") -> bool:
    """
    Synthesize `text` in the given language and play it via ALSA.
    Returns True on success, False otherwise.
    """
    text = (text or "").strip()
    if not text:
        return True  # nothing to say, not an error

    if not _have(PIPER_BIN):
        print(f"[TTS] Piper not found at '{PIPER_BIN}'. Set PIPER_BIN or install Piper TTS.")
        return False
    if not _have(APLAY_BIN):
        print("[TTS] 'aplay' not found. Install alsa-utils.")
        return False

    voice = _voice_for(language)
    if not voice or not os.path.exists(voice):
        print(f"[TTS] Voice model for '{language}' not found: {voice}")
        return False

    cfg = voice + ".json"
    use_cfg = os.path.exists(cfg)

    try:
        with tempfile.NamedTemporaryFile(prefix="tts_", suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        style = _piper_flag_style()
        tried_cmds = []

        def run_piper(cmd):
            tried_cmds.append(" ".join(cmd))
            return subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

        if style == "short":
            cmd = [PIPER_BIN, "-m", voice, "-f", wav_path]
            if use_cfg:
                cmd += ["-c", cfg]
            proc = run_piper(cmd)
        elif style == "long":
            cmd = [PIPER_BIN, "--model", voice, "--output_file", wav_path]
            if use_cfg:
                cmd += ["--config", cfg]
            proc = run_piper(cmd)
        else:
            # Unknown: try short first, then long
            cmd_short = [PIPER_BIN, "-m", voice, "-f", wav_path]
            if use_cfg:
                cmd_short += ["-c", cfg]
            try:
                proc = run_piper(cmd_short)
                proc.stdin.write(text.encode("utf-8"))
                proc.stdin.close()
                proc.wait(timeout=30)
                if proc.returncode != 0:
                    raise RuntimeError("short flags failed")
            except Exception:
                cmd_long = [PIPER_BIN, "--model", voice, "--output_file", wav_path]
                if use_cfg:
                    cmd_long += ["--config", cfg]
                proc = run_piper(cmd_long)

        if proc.stdin and not proc.stdin.closed:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()

        proc.wait(timeout=40)

        if proc.returncode != 0:
            err = proc.stderr.read().decode("utf-8", errors="ignore")
            print("[TTS] Piper failed:", err.strip() or "(no error text)")
            print("[TTS] Tried:", " | ".join(tried_cmds))
            return False

        ap = subprocess.run([APLAY_BIN, "-q", wav_path], capture_output=True)
        if ap.returncode != 0:
            print("[TTS] aplay failed:", ap.stderr.decode("utf-8", errors="ignore").strip())
            return False

        return True

    except Exception as e:
        print("[TTS] Error:", e)
        return False
    finally:
        try:
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Piper TTS wrapper")
    p.add_argument("text", help="Text to speak")
    p.add_argument("--lang", default="de", help="Language code (de/en)")
    args = p.parse_args()
    ok = speak(args.text, args.lang)
    raise SystemExit(0 if ok else 1)
