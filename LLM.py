from flask import Flask, request, jsonify, Response
import requests, os, subprocess, time, json, threading, re
from pathlib import Path
from datetime import datetime
import random, string

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_BIN = os.getenv("OLLAMA_BIN", "ollama")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")

MAX_TURNS_PER_SESSION = int(os.getenv("LLM_MAX_TURNS", "10"))

MEM_DIR = Path(os.getenv("LLM_MEM_DIR", "memory"))
MEM_DIR.mkdir(parents=True, exist_ok=True)

SESSION_MAP_PATH = MEM_DIR / "session_map.json"
PIN_PATH = MEM_DIR / "pin.json"
SETTINGS_PATH = MEM_DIR / "settings.json"

SESSIONS: dict[str, list[dict]] = {}
LOCK = threading.Lock()

BLOCKLIST = [
    "suicide", "self harm", "kill myself", "sex", "porn", "nsfw", "nude",
    "drugs", "cocaine", "meth", "heroin", "weapon", "gun", "bomb", "murder"
]

REFUSAL_EN = "I can't talk about that. Let's choose a safer topic."
REFUSAL_DE = "Darüber kann ich nicht sprechen. Lass uns ein sicheres Thema wählen."

def ensure_ollama_running() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=1)
        return True
    except Exception:
        try:
            subprocess.Popen(
                [OLLAMA_BIN, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(15):
                try:
                    requests.get("http://localhost:11434/api/tags", timeout=1)
                    return True
                except Exception:
                    time.sleep(1)
        except Exception:
            return False
    return False


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_session_from_disk(session_id: str) -> list[dict]:
    p = MEM_DIR / f"session_{session_id}.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def append_to_disk(session_id: str, item: dict):
    p = MEM_DIR / f"session_{session_id}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_session(session_id: str) -> list[dict]:
    with LOCK:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = load_session_from_disk(session_id)
        return SESSIONS[session_id]


def generate_short_id() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(6))


def blocked(text: str) -> bool:
    text = (text or "").lower()
    return any(w in text for w in BLOCKLIST)


def persona(lang: str, user_name: str) -> str:
    name_hint = f" The child's name is {user_name}." if user_name else ""
    if lang.startswith("de"):
        return (
            "System: Du bist Björn, ein freundlicher Plüschbär-Assistent. "
            "Kurze Sätze, kindgerecht, sicher. Keine Erwachsenenthemen."
            f"{name_hint}\n\nGespräch:\n"
        )
    return (
        "System: You are Björn, a friendly teddy bear assistant. "
        "Short sentences, child-safe, encouraging."
        f"{name_hint}\n\nConversation:\n"
    )


def build_prompt(history: list[dict], user_text: str, lang: str, name: str) -> str:
    base = persona(lang, name)
    out: list[str] = [base]
    for turn in history[-MAX_TURNS_PER_SESSION:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        out.append(f"{role}: {turn.get('content','')}\n")
    out.append(f"User: {user_text}\nAssistant:")
    return "".join(out)


@app.route("/talk", methods=["POST"])
def talk():
    body = request.json or {}
    text = (body.get("text") or "").strip()
    lang = (body.get("language") or "en")
    session_id = body.get("session_id")
    user_name = body.get("user_name", "")

    if not text:
        return jsonify({"error": "missing text"}), 400

    if not ensure_ollama_running():
        return jsonify({"reply": "LLM unavailable"}), 503

    history = get_session(session_id)
    prompt = build_prompt(history, text, lang, user_name)

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        reply_raw = (r.json().get("response") or "").strip()

        if blocked(text):
            reply = REFUSAL_DE if lang.startswith("de") else REFUSAL_EN
        else:
            reply = reply_raw

        user_turn = {
            "role": "user",
            "content": text,
            "lang": lang,
            "ts": time.time(),
        }
        bot_turn = {
            "role": "assistant",
            "content": reply,
            "lang": lang,
            "ts": time.time(),
        }

        with LOCK:
            history.append(user_turn)
            history.append(bot_turn)

        append_to_disk(session_id, user_turn)
        append_to_disk(session_id, bot_turn)

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error: {e}"}), 500


@app.route("/api/create_short_id", methods=["POST"])
def create_short_id():
    body = request.json or {}
    session_id = body.get("session_id")
    parent_pin = body.get("pin")

    if not session_id or not parent_pin:
        return jsonify({"error": "missing session_id or pin"}), 400

    session_map = _load_json(SESSION_MAP_PATH, {})
    pin_data = _load_json(PIN_PATH, {})

    # Note: this design supports only one PIN globally.
    pin_data["pin"] = parent_pin
    _save_json(PIN_PATH, pin_data)

    short_id = generate_short_id()
    while short_id in session_map:
        short_id = generate_short_id()

    session_map[short_id] = session_id
    _save_json(SESSION_MAP_PATH, session_map)

    return jsonify({"short_id": short_id})


@app.route("/api/session/<short_id>", methods=["GET"])
def api_get_session(short_id):
    pin = request.args.get("pin", "")

    session_map = _load_json(SESSION_MAP_PATH, {})
    pin_data = _load_json(PIN_PATH, {})

    correct_pin = pin_data.get("pin")

    # Unknown session
    if short_id not in session_map:
        return jsonify({"ok": False, "error": "invalid_session"}), 404

    # Wrong password
    if not correct_pin or pin != correct_pin:
        return jsonify({"ok": False, "error": "invalid_password"}), 401

    session_id = session_map[short_id]
    history = load_session_from_disk(session_id)

    # Load child's name from settings.json (written by main.py)
    settings = _load_json(SETTINGS_PATH, {})
    child_name = settings.get("user_name", "Your Child")

    safe_msgs = [
        {
            "role": item.get("role"),
            "content": item.get("content"),
            "ts": item.get("ts"),
            "lang": item.get("lang"),
        }
        for item in history
    ]

    return jsonify({
        "ok": True,
        "child_name": child_name,
        "messages": safe_msgs,
    })



@app.route("/api/debug/session_ids")
def list_ids():
    session_map = _load_json(SESSION_MAP_PATH, {})
    return jsonify(session_map)


if __name__ == "__main__":
    ensure_ollama_running()
    app.run(host="0.0.0.0", port=5000)
