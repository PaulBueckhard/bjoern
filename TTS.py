import os
import json
import socket
import subprocess
import shutil
import tempfile
from functools import lru_cache
from typing import Optional

VOICE_MAP = {
    "en": os.path.abspath("tts_models/piper-model-english.onnx"),
    "de": os.path.abspath("tts_models/piper-model-german.onnx"),
}

PIPER_BIN = os.environ.get("PIPER_BIN", "/usr/bin/piper")
APLAY_BIN = os.environ.get("APLAY_BIN", "aplay")

DAEMON_HOST = os.environ.get("TTS_DAEMON_HOST", "127.0.0.1")
DAEMON_PORT = int(os.environ.get("TTS_DAEMON_PORT", "50051"))
TTS_FORCE_DAEMON = os.environ.get("TTS_FORCE_DAEMON", "0") == "1"
TTS_DEBUG = os.environ.get("TTS_DEBUG", "0") == "1"

OMP_THREADS = os.environ.get("OMP_NUM_THREADS", "2")

def _daemon_speak(text: str, language: str, timeout: float = 20.0) -> bool:
    """Send a speak request to the persistent TTS daemon."""
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

def _have(cmd: str) -> bool:
    return bool(shutil.which(cmd))

def _voice_for(language: str) -> Optional[str]:
    lang = (language or "en").split("-")[0].lower()
    return VOICE_MAP.get(lang) or VOICE_MAP.get("en")

@lru_cache(maxsize=1)
def _piper_flag_style() -> str:
    try:
        proc = subprocess.run([PIPER_BIN, "--help"], capture_output=True, text=True, timeout=5)
        text = (proc.stdout or "") + (proc.stderr or "")
        if "-m" in text and "-f" in text:
            return "short"
        if "--model" in text and "--output_file" in text:
            return "long"
    except Exception:
        pass
    return "short"

def speak(text: str, language: str = "en") -> bool:
    text = (text or "").strip()
    if not text:
        return True

    # Try daemon first
    used_daemon = _daemon_speak(text, language)
    if TTS_DEBUG:
        print(f"[TTS] daemon={used_daemon} host={DAEMON_HOST} port={DAEMON_PORT}")
    if TTS_FORCE_DAEMON and not used_daemon:
        print("[TTS] Daemon required but unavailable.")
        return False
    if used_daemon:
        return True

    # Fallback if daemon not available
    if not _have(PIPER_BIN):
        print(f"[TTS] Piper not found at '{PIPER_BIN}'.")
        return False
    if not _have(APLAY_BIN):
        print("[TTS] 'aplay' not found. Install alsa-utils.")
        return False

    voice = _voice_for(language)
    if not voice or not os.path.exists(voice):
        print(f"[TTS] Voice model not found: {voice}")
        return False

    cfg = voice + ".json"
    use_cfg = os.path.exists(cfg)

    try:
        with tempfile.NamedTemporaryFile(prefix="tts_", suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        style = _piper_flag_style()
        tried_cmds = []

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

        if style == "short":
            cmd = [PIPER_BIN, "-m", voice, "-f", wav_path]
            if use_cfg:
                cmd += ["-c", cfg]
        else:
            cmd = [PIPER_BIN, "--model", voice, "--output_file", wav_path]
            if use_cfg:
                cmd += ["--config", cfg]

        proc = run_piper(cmd)
        assert proc.stdin is not None
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
            if "wav_path" in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

# CLI test
if __name__ == "__main__":
    import sys as _sys
    ok = speak(" ".join(_sys.argv[1:]) or "Hello from persistent Piper", "en")
    raise SystemExit(0 if ok else 1)
