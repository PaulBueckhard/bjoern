from flask import Flask, request, jsonify
import requests, os, subprocess, time, json, uuid, threading
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_BIN  = os.getenv("OLLAMA_BIN", "ollama")
MODEL       = os.getenv("OLLAMA_MODEL", "llama3")

MAX_TURNS_PER_SESSION = int(os.getenv("LLM_MAX_TURNS", "10")) 
MEM_DIR = Path(os.getenv("LLM_MEM_DIR", "memory")) 
MEM_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = {}
LOCK = threading.Lock()

def ensure_ollama_running():
    """Check if Ollama is reachable; if not, try to start it."""
    try:
        requests.get("http://localhost:11434/api/tags", timeout=1)
        return True
    except requests.exceptions.RequestException:
        print("[LLM] Ollama not responding. Attempting to start it...")
        try:
            subprocess.Popen(
                [OLLAMA_BIN, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(15):
                try:
                    requests.get("http://localhost:11434/api/tags", timeout=1)
                    print("[LLM] Ollama started successfully.")
                    return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
            print("[LLM] Ollama did not respond after startup attempt.")
            return False
        except Exception as e:
            print(f"[LLM] Could not start Ollama: {e}")
            return False

def _load_session_from_disk(session_id: str):
    path = MEM_DIR / f"session_{session_id}.jsonl"
    if not path.exists():
        return []
    items = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                items.append(json.loads(line))
    except Exception as e:
        print("[LLM] Failed reading memory:", e)
    return items

def _append_to_disk(session_id: str, item: dict):
    path = MEM_DIR / f"session_{session_id}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[LLM] Failed writing memory:", e)

def _get_session(session_id: str):
    with LOCK:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = _load_session_from_disk(session_id)
        return SESSIONS[session_id]

def _reset_session(session_id: str):
    with LOCK:
        SESSIONS[session_id] = []
    path = MEM_DIR / f"session_{session_id}.jsonl"
    if path.exists():
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path.rename(MEM_DIR / f"session_{session_id}_{ts}.jsonl")

def _build_prompt(history, user_text, language: str):
    language = (language or "en").split("-")[0].lower()
    sys_preamble = (
        "System: You are a friendly plush assistant. "
        "Maintain helpful, concise answers. "
        "Use the user's current language. "
        f"Current language: {'German' if language=='de' else 'English'}.\n\n"
        "Conversation:\n"
    )
    recent = history[-MAX_TURNS_PER_SESSION:]
    parts = [sys_preamble]
    for turn in recent:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if not content: continue
        if role == "user":
            parts.append(f"User: {content}\n")
        else:
            parts.append(f"Assistant: {content}\n")
    parts.append(f"User: {user_text.strip()}\nAssistant:")
    return "".join(parts)

@app.route("/talk", methods=["POST"])
def talk():
    body = request.json or {}
    user_text = (body.get("text") or "").strip()
    language  = (body.get("language") or "en").strip()
    session_id = (body.get("session_id") or "default").strip()
    reset = bool(body.get("reset", False))

    if not user_text:
        return jsonify({"error": "Missing text"}), 400

    if reset:
        _reset_session(session_id)

    if not ensure_ollama_running():
        return jsonify({"reply": "Ollama could not be started or reached."}), 503

    history = _get_session(session_id)

    prompt = _build_prompt(history, user_text, language)

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        r.raise_for_status()
        data = r.json()
        reply = (data.get("response") or "").strip()

        user_item = {"role": "user", "content": user_text, "lang": language, "ts": time.time()}
        asst_item = {"role": "assistant", "content": reply, "lang": language, "ts": time.time()}
        with LOCK:
            history.append(user_item)
            history.append(asst_item)
            if len(history) > MAX_TURNS_PER_SESSION * 3:
                del history[: (len(history) - MAX_TURNS_PER_SESSION * 2)]
        _append_to_disk(session_id, user_item)
        _append_to_disk(session_id, asst_item)

        return jsonify({"reply": reply, "session_id": session_id})
    except Exception as e:
        return jsonify({"reply": f"Error contacting Ollama: {e}"}), 500

if __name__ == "__main__":
    ensure_ollama_running()
    app.run(host="0.0.0.0", port=5000)
