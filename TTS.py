import os
import json
import socket
import subprocess
import shutil
import tempfile
import re
from functools import lru_cache
from typing import Optional

# -------------------------------------------------------------------
# Voice model mapping (language → model file path)
# -------------------------------------------------------------------
VOICE_MAP = {
    "en": os.path.abspath("tts_models/piper-model-english.onnx"),
    "de": os.path.abspath("tts_models/piper-model-german.onnx"),
}

IS_WINDOWS = (os.name == "nt")

# Default binary/model locations (used when env vars not provided)
_default_piper_win = r"C:\piper\piper.exe"
_default_espeak_win = r"C:\piper\espeak-ng-data"
_default_piper_linux = "/usr/bin/piper"

# -------------------------------------------------------------------
# Resolve environment-based config
# -------------------------------------------------------------------
PIPER_BIN = os.environ.get("PIPER_BIN")
ESPEAK_DATA = os.environ.get("ESPEAK_DATA")

# Auto-detect Piper binary if not set
if not PIPER_BIN:
    if IS_WINDOWS and os.path.exists(_default_piper_win):
        PIPER_BIN = _default_piper_win
        print(f"[TTS] Using default Windows Piper path: {PIPER_BIN}")
    else:
        PIPER_BIN = _default_piper_linux

# Auto-detect espeak-ng-data on Windows
if IS_WINDOWS and not ESPEAK_DATA and os.path.exists(_default_espeak_win):
    ESPEAK_DATA = _default_espeak_win
    print(f"[TTS] Using default Windows espeak-ng-data path: {ESPEAK_DATA}")

# Optional playback utility (Linux only)
APLAY_BIN = os.environ.get("APLAY_BIN", "aplay")

# Daemon integration config
DAEMON_HOST = os.environ.get("TTS_DAEMON_HOST", "127.0.0.1")
DAEMON_PORT = int(os.environ.get("TTS_DAEMON_PORT", "50051"))
TTS_FORCE_DAEMON = os.environ.get("TTS_FORCE_DAEMON", "0") == "1"
TTS_DEBUG = os.environ.get("TTS_DEBUG", "0") == "1"

# Thread limit to reduce CPU overuse
OMP_THREADS = os.environ.get("OMP_NUM_THREADS", "2")

# -------------------------------------------------------------------
# Text preprocessing
# -------------------------------------------------------------------
def _sanitize_text(text: str) -> str:
    """
    Normalize text so Piper receives clean, predictable input.
    Removes markup, excessive punctuation, emojis, emoticons,
    and ensures the sentence ends cleanly.
    """
    if not text:
        return ""

    # Normalize line breaks
    text = text.replace("\r", " ").replace("\n", " ")

    # Remove Markdown and inline formatting
    text = re.sub(r"(\*+|_+|`+)", " ", text)

    # Remove emoji shortcodes like :smile:
    text = re.sub(r":[A-Za-z0-9_+-]+:", " ", text)

    # Remove ASCII emoticons
    emoticon_pattern = r"(:\)|:-\)|:\(|:-\(|;\)|;-\)|:D|:-D|:\]|:\[|<3)"
    text = re.sub(emoticon_pattern, " ", text)

    # Collapse repeated punctuation (!!! → !)
    text = re.sub(r"([!?\.])\1+", r"\1", text)

    # Replace unsupported unicode (Piper can choke on some glyphs)
    text = re.sub(
        r"[^\x20-\x7EéèêàáâîïôöùüçÇÉÈÊÀÁÂÎÏÔÖÙÜß]",
        " ",
        text
    )

    # Remove excessive whitespace
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip(" .!,?\u200b")

    # Ensure a natural ending for TTS prosody
    if text and not text.endswith((".", "!", "?")):
        text += "."

    return text.strip()


# -------------------------------------------------------------------
# Daemon Mode (if a TTS daemon is running, use it)
# Protocol: send {"text": "...", "language": "..."}\n and expect {"ok": true}
# -------------------------------------------------------------------
def _daemon_speak(text: str, language: str, timeout: float = 20.0) -> bool:
    """Send a TTS request to the daemon. Returns True if audio was generated."""
    try:
        with socket.create_connection((DAEMON_HOST, DAEMON_PORT), timeout=2.0) as s:
            payload = json.dumps({"text": text, "language": language}).encode() + b"\n"
            s.sendall(payload)

            s.settimeout(timeout)
            data = b""
            while b"\n" not in data:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk

        if not data:
            return False

        resp = json.loads(data.decode().strip() or "{}")
        return bool(resp.get("ok"))
    except Exception:
        return False


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def _have(cmd: str) -> bool:
    """Check whether a given command/binary exists in PATH."""
    return bool(shutil.which(cmd))


def _voice_for(language: str) -> Optional[str]:
    """Resolve the closest matching voice model based on language prefix."""
    lang = (language or "en").split("-")[0].lower()
    return VOICE_MAP.get(lang) or VOICE_MAP.get("en")


@lru_cache(maxsize=1)
def _piper_flag_style() -> str:
    """
    Detect which CLI flag format Piper supports:
    - "short":  -m model  -f output.wav
    - "long":   --model model --output_file output.wav
    """
    try:
        proc = subprocess.run([PIPER_BIN, "--help"], capture_output=True, text=True, timeout=5)
        text = (proc.stdout or "") + (proc.stderr or "")
        if "-m" in text and "-f" in text:
            return "short"
        if "--model" in text and "--output_file" in text:
            return "long"
    except Exception:
        pass

    # Fallback: most distros use the short version
    return "short"


# -------------------------------------------------------------------
# Main public API
# -------------------------------------------------------------------
def speak(text: str, language: str = "en") -> bool:
    """
    Convert text to speech using:
      1. TTS daemon (if available)
      2. Piper binary fallback

    Returns True on success.
    """
    text = (text or "").strip()
    if not text:
        return True

    clean_text = _sanitize_text(text)

    # ------------------------
    # 1. Try TTS daemon first
    # ------------------------
    used_daemon = _daemon_speak(clean_text, language)
    if TTS_DEBUG:
        print(f"[TTS] daemon={used_daemon} host={DAEMON_HOST} port={DAEMON_PORT}")

    if TTS_FORCE_DAEMON and not used_daemon:
        print("[TTS] Daemon required but unavailable.")
        return False

    if used_daemon:
        return True

    # ---------------------------------
    # 2. Local Piper fallback
    # ---------------------------------
    if not _have(PIPER_BIN):
        print(f"[TTS] Piper not found at '{PIPER_BIN}'.")
        return False

    if not IS_WINDOWS and not _have(APLAY_BIN):
        print("[TTS] 'aplay' not found. Install alsa-utils.")
        return False

    # Resolve voice model
    voice = _voice_for(language)
    if not voice or not os.path.exists(voice):
        print(f"[TTS] Voice model not found: {voice}")
        return False

    cfg = voice + ".json"
    use_cfg = os.path.exists(cfg)

    try:
        # Temp WAV output file
        with tempfile.NamedTemporaryFile(prefix="tts_", suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        style = _piper_flag_style()
        tried_cmds = []

        # Ensure predictable thread limits (performance and stability)
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = OMP_THREADS

        def run_piper(cmd):
            tried_cmds.append(" ".join(cmd))
            return subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
            )

        # Build Piper command based on detected style
        if style == "short":
            cmd = [PIPER_BIN, "-m", voice, "-f", wav_path]
            if use_cfg:
                cmd += ["-c", cfg]
        else:
            cmd = [PIPER_BIN, "--model", voice, "--output_file", wav_path]
            if use_cfg:
                cmd += ["--config", cfg]

        # Windows uses JSON stdin and optional espeak data
        if IS_WINDOWS:
            cmd += ["--json-input"]
            if ESPEAK_DATA:
                cmd += ["--espeak_data", ESPEAK_DATA]

        # Run Piper
        proc = run_piper(cmd)
        assert proc.stdin is not None

        # Piper input method differs by OS
        if IS_WINDOWS:
            payload = json.dumps({"text": clean_text}) + "\n"
            proc.stdin.write(payload.encode("utf-8"))
        else:
            proc.stdin.write(clean_text.encode("utf-8"))

        proc.stdin.close()
        proc.wait(timeout=40)

        if proc.returncode != 0:
            err = proc.stderr.read().decode("utf-8", errors="ignore")
            print("[TTS] Piper failed:", err.strip() or "(no error text)")
            print("[TTS] Tried:", " | ".join(tried_cmds))
            return False

        # Playback
        if IS_WINDOWS:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        else:
            ap = subprocess.run([APLAY_BIN, "-q", wav_path], capture_output=True)
            if ap.returncode != 0:
                print("[TTS] aplay failed:", ap.stderr.decode("utf-8", errors="ignore").strip())
                return False

        return True

    except Exception as e:
        print("[TTS] Error:", e)
        return False

    finally:
        # Clean up temporary WAV
        try:
            if "wav_path" in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


# -------------------------------------------------------------------
# CLI Test Mode
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys
    ok = speak(" ".join(_sys.argv[1:]) or "Hello from Piper, running on any system!", "en")
    raise SystemExit(0 if ok else 1)
