import os, json, socket, threading, subprocess, time, sys, signal

HOST = os.environ.get("TTS_DAEMON_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_DAEMON_PORT", "50051"))

PIPER_BIN   = os.environ.get("PIPER_BIN", "/usr/bin/piper")
APLAY_BIN   = os.environ.get("APLAY_BIN", "aplay")
ALSA_DEVICE = os.environ.get("ALSA_DEVICE", "default")
OMP_THREADS = os.environ.get("OMP_NUM_THREADS", "2")

USE_SOX_FADE = os.environ.get("USE_SOX_FADE", "0") == "1"
FADE_MS = float(os.environ.get("FADE_MS", "12")) / 1000.0  # 12 ms default

PRESTART_LANG = os.environ.get("PRESTART_LANG", "")

# Sentence pause and speed tuning
SENTENCE_SILENCE = 0.1
LENGTH_SCALE = 0.98
# Keep defaults
NOISE_SCALE = None   # e.g. 0.667
NOISE_W     = None   # e.g. 0.8


VOICE_MAP = {
    "en": os.path.abspath("tts_models/piper-model-english.onnx"),
    "de": os.path.abspath("tts_models/piper-model-german.onnx"),
}

_env = dict(os.environ)
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    _env[k] = OMP_THREADS

_cur_lang = None
_piper_proc = None
_player_proc = None
_sample_rate = 22050

def _log(*a): print(*a, flush=True)

def _read_sample_rate(model_path: str) -> int:
    cfg = model_path + ".json"
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            j = json.load(f)
        return int(j.get("sample_rate", 22050))
    except Exception as e:
        _log("[DAEMON] Could not read sample_rate from", cfg, "->", e)
        return 22050

def _drain_stderr(tag, proc):
    try:
        for line in iter(proc.stderr.readline, b""):
            if not line: break
            _log(f"[{tag} stderr] {line.decode('utf-8','ignore').rstrip()}")
    except Exception:
        pass

def _player_cmd(sample_rate: int):
    if USE_SOX_FADE:
        return [
            "sox",
            "-t","raw",
            "-r", str(sample_rate),
            "-b","16",
            "-e","signed-integer",
            "-c","1",
            "-",
            "-d",
            "fade","t", f"{FADE_MS:.03f}"
        ]
    else:
        return [APLAY_BIN, "-q", "-D", ALSA_DEVICE, "-f", "S16_LE", "-r", str(sample_rate), "-c", "1"]

def _start_pipeline(lang: str) -> bool:
    global _piper_proc, _player_proc, _cur_lang, _sample_rate
    _stop_pipeline()

    model = VOICE_MAP.get(lang)
    if not model or not os.path.exists(model) or not os.path.exists(model + ".json"):
        _log(f"[DAEMON] Missing model/config for '{lang}': {model}")
        return False

    _sample_rate = _read_sample_rate(model)

    piper_cmd  = [PIPER_BIN, "-m", model, "-c", model + ".json", "--output_raw"]

    if SENTENCE_SILENCE is not None:
        piper_cmd += ["--sentence_silence", str(SENTENCE_SILENCE)]
    if LENGTH_SCALE is not None:
        piper_cmd += ["--length_scale", str(LENGTH_SCALE)]
    if NOISE_SCALE is not None:
        piper_cmd += ["--noise_scale", str(NOISE_SCALE)]
    if NOISE_W is not None:
        piper_cmd += ["--noise_w", str(NOISE_W)]

    player_cmd = _player_cmd(_sample_rate)
    _log("[DAEMON] exec:", " ".join(piper_cmd), "|", " ".join(player_cmd))

    try:
        _piper_proc = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_env,
        )
        _player_proc = subprocess.Popen(
            player_cmd,
            stdin=_piper_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _cur_lang = lang
        _log(f"[DAEMON] Pipeline started for '{lang}' at {_sample_rate} Hz (HOT)")

        threading.Thread(target=_drain_stderr, args=("piper",  _piper_proc), daemon=True).start()
        threading.Thread(target=_drain_stderr, args=("player", _player_proc), daemon=True).start()

        return True
    except Exception as e:
        _log("[DAEMON] Failed to start pipeline:", e)
        _stop_pipeline()
        return False

def _stop_pipeline():
    global _piper_proc, _player_proc, _cur_lang
    for p in (_player_proc, _piper_proc):
        if p:
            try:
                p.terminate(); p.wait(timeout=1)
            except Exception:
                try: p.kill()
                except Exception: pass
    _piper_proc = _player_proc = None
    _cur_lang = None

def _feed_text(line: str) -> bool:
    if not _piper_proc or _piper_proc.poll() is not None:
        return False
    try:
        assert _piper_proc.stdin is not None
        _piper_proc.stdin.write(line.encode("utf-8"))
        _piper_proc.stdin.flush()
        return True
    except Exception as e:
        _log("[DAEMON] Piper stdin error:", e)
        return False

def _speak(text: str, lang: str) -> bool:
    global _cur_lang
    lang = (lang or "en").split("-")[0].lower()
    if lang not in VOICE_MAP:
        lang = "en"
    if _cur_lang != lang:
        if not _start_pipeline(lang):
            return False
    return _feed_text(text.strip() + "\n")

def _handle_conn(conn: socket.socket):
    try:
        data = b""
        while b"\n" not in data:
            chunk = conn.recv(4096)
            if not chunk: break
            data += chunk
        msg = data.decode("utf-8", "ignore").strip()
        if not msg:
            conn.sendall(b'{"ok":false,"error":"empty"}\n'); return
        req = json.loads(msg)
        text = (req.get("text") or "").strip()
        lang = (req.get("language") or "en").strip().lower()
        if not text:
            conn.sendall(b'{"ok":false,"error":"no_text"}\n'); return
        ok = _speak(text, lang)
        conn.sendall(b'{"ok":true}\n' if ok else b'{"ok":false,"error":"speak_failed"}\n')
    except Exception as e:
        try: conn.sendall(b'{"ok":false,"error":"internal"}\n')
        except Exception: pass
        _log("[DAEMON] client error:", e)
    finally:
        try: conn.close()
        except Exception: pass

def _serve():
    _log(f"[DAEMON] Listening on {HOST}:{PORT}")
    if PRESTART_LANG:
        _start_pipeline(PRESTART_LANG)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT)); s.listen(5)
        while True:
            conn, _ = s.accept()
            threading.Thread(target=_handle_conn, args=(conn,), daemon=True).start()

def _shutdown(*_):
    _stop_pipeline(); sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    _serve()
