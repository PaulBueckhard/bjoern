import os, json, socket, threading, subprocess, time, sys, signal, shutil

IS_WINDOWS = (os.name == "nt")

HOST = os.environ.get("TTS_DAEMON_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_DAEMON_PORT", "50051"))

_default_piper_win = r"C:\piper\piper.exe"
_default_espeak_win = r"C:\piper\espeak-ng-data"
_default_piper_linux = "/usr/bin/piper"

PIPER_BIN = os.environ.get("PIPER_BIN") or (_default_piper_win if IS_WINDOWS else _default_piper_linux)
ESPEAK_DATA = os.environ.get("ESPEAK_DATA") if IS_WINDOWS else None
if IS_WINDOWS and not ESPEAK_DATA and os.path.exists(_default_espeak_win):
    ESPEAK_DATA = _default_espeak_win

APLAY_BIN   = os.environ.get("APLAY_BIN", "aplay")
ALSA_DEVICE = os.environ.get("ALSA_DEVICE", "default")
OMP_THREADS = os.environ.get("OMP_NUM_THREADS", "2")

USE_SOX_FADE = (os.environ.get("USE_SOX_FADE", "0") == "1") and not IS_WINDOWS
FADE_MS = float(os.environ.get("FADE_MS", "12")) / 1000.0
PRESTART_LANG = os.environ.get("PRESTART_LANG", "")

VOICE_MAP = {
    "en": os.path.abspath("tts_models/piper-model-english.onnx"),
    "de": os.path.abspath("tts_models/piper-model-german.onnx"),
}

_env = dict(os.environ)
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    _env[k] = OMP_THREADS

_cur_lang = None
_piper_proc = None

# Windows playback
_sd_stream = None
_sd_thread = None
_stop_feeder = threading.Event()
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

def _player_cmd_linux(sample_rate: int):
    if USE_SOX_FADE and shutil.which("sox"):
        return [
            "sox","-t","raw","-r",str(sample_rate),"-b","16","-e","signed-integer","-c","1",
            "-", "-d", "fade","t", f"{FADE_MS:.03f}"
        ]
    else:
        return [APLAY_BIN,"-q","-D",ALSA_DEVICE,"-f","S16_LE","-r",str(sample_rate),"-c","1"]

def _start_pipeline(lang: str) -> bool:
    global _piper_proc, _cur_lang, _sample_rate, _sd_stream, _sd_thread, _stop_feeder

    _stop_pipeline()

    model = VOICE_MAP.get(lang)
    if not model or not os.path.exists(model) or not os.path.exists(model + ".json"):
        _log(f"[DAEMON] Missing model/config for '{lang}': {model}")
        return False

    _sample_rate = _read_sample_rate(model)

    cmd = [PIPER_BIN, "-m", model, "-c", model + ".json", "--output_raw"]
    if IS_WINDOWS:
        cmd += ["--json-input"]
        if ESPEAK_DATA:
            cmd += ["--espeak_data", ESPEAK_DATA]

    _log("[DAEMON] exec:", " ".join(cmd) + (" | (player)" if not IS_WINDOWS else ""))

    try:
        _piper_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_env,
        )
        threading.Thread(target=_drain_stderr, args=("piper", _piper_proc), daemon=True).start()

        if IS_WINDOWS:
            import sounddevice as sd

            _stop_feeder.clear()
            _sd_stream = sd.RawOutputStream(
                samplerate=_sample_rate,
                channels=1,
                dtype="int16",
                blocksize=2048,
            )
            _sd_stream.start()

            def _feeder():
                try:
                    while not _stop_feeder.is_set():
                        chunk = _piper_proc.stdout.read(4096)
                        if not chunk:
                            time.sleep(0.002)
                            continue
                        _sd_stream.write(chunk)
                except Exception as e:
                    _log("[DAEMON] Windows feeder error:", e)

            _sd_thread = threading.Thread(target=_feeder, daemon=True)
            _sd_thread.start()

        else:
            player_cmd = _player_cmd_linux(_sample_rate)
            _player_proc = subprocess.Popen(
                player_cmd,
                stdin=_piper_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            threading.Thread(target=_drain_stderr, args=("player", _player_proc), daemon=True).start()

        _cur_lang = lang
        _log(f"[DAEMON] Pipeline started for '{lang}' at {_sample_rate} Hz (HOT)")
        return True

    except Exception as e:
        _log("[DAEMON] Failed to start pipeline:", e)
        _stop_pipeline()
        return False

def _stop_pipeline():
    global _piper_proc, _cur_lang, _sd_stream, _sd_thread, _stop_feeder

    if IS_WINDOWS:
        try:
            _stop_feeder.set()
            if _sd_thread and _sd_thread.is_alive():
                _sd_thread.join(timeout=0.2)
        except Exception:
            pass
        _sd_thread = None
        try:
            if _sd_stream:
                _sd_stream.stop()
                _sd_stream.close()
        except Exception:
            pass
        _sd_stream = None

    procs = []
    try:
        pass
    except Exception:
        pass

    if _piper_proc:
        try:
            _piper_proc.terminate(); _piper_proc.wait(timeout=1)
        except Exception:
            try: _piper_proc.kill()
            except Exception: pass

    _piper_proc = None
    _cur_lang = None

def _feed_text(line: str) -> bool:
    if not _piper_proc or _piper_proc.poll() is not None:
        return False
    try:
        payload = line.strip() + ("\n" if not line.endswith("\n") else "")
        assert _piper_proc.stdin is not None
        if IS_WINDOWS:
            data = json.dumps({"text": payload.strip()}) + "\n"
            _piper_proc.stdin.write(data.encode("utf-8"))
        else:
            _piper_proc.stdin.write(payload.encode("utf-8"))
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
    return _feed_text(text)

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
    _log(f"[DAEMON] Piper: {PIPER_BIN}")
    if IS_WINDOWS and ESPEAK_DATA:
        _log(f"[DAEMON] eSpeak data: {ESPEAK_DATA}")
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
