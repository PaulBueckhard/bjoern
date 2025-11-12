from flask import Flask, request, jsonify, Response
import requests, os, subprocess, time, json, threading, re
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_BIN = os.getenv("OLLAMA_BIN", "ollama")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")

MAX_TURNS_PER_SESSION = int(os.getenv("LLM_MAX_TURNS", "10"))
MEM_DIR = Path(os.getenv("LLM_MEM_DIR", "memory")); MEM_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = {}
NAMES = {}
NAME_LAST_USE = {}
LOCK = threading.Lock()

REFUSAL_EN = "I can’t talk about that. Let’s choose a safe topic—space, animals, or a riddle?"
REFUSAL_DE = "Darüber kann ich nicht sprechen. Lass uns etwas Sicheres wählen: Weltraum, Tiere oder ein Rätsel?"
BLOCKLIST = ["suicide","self harm","kill myself","sex","porn","nsfw","nude","drugs","cocaine","meth","heroin",
             "weapon","gun","bomb","bleeding","gore","murder","suicide pact","how to make","explosive","pedo",
             "alcohol","strip club","fetish","rape","abuse","violence","steal","shoplift","hack","ddos","virus"]

def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=1); return True
    except requests.exceptions.RequestException:
        try:
            subprocess.Popen([OLLAMA_BIN, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(15):
                try:
                    requests.get("http://localhost:11434/api/tags", timeout=1); return True
                except requests.exceptions.RequestException: time.sleep(1)
            return False
        except Exception: return False

def _load_session_from_disk(session_id):
    p = MEM_DIR / f"session_{session_id}.jsonl"
    if not p.exists(): return []
    items = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line: items.append(json.loads(line))
    except Exception: pass
    return items

def _append_to_disk(session_id, item):
    p = MEM_DIR / f"session_{session_id}.jsonl"
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception: pass

def _get_session(session_id):
    with LOCK:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = _load_session_from_disk(session_id)
        return SESSIONS[session_id]

def _persona(language, user_name):
    name_hint = f" The child's name is {user_name}. Use the name at most once occasionally." if user_name else ""
    if (language or "en").startswith("de"):
        return ("System: Du bist Björn, ein freundlicher Plüschbär-Assistent für Kinder. "
                "Kurze, einfache Sätze (≤30 Wörter). Eine kleine Rückfrage oder ein Tipp. "
                "Kindgerecht und sicher; lehne Gefährliches sanft ab und biete Alternativen. "
                "Beginne nicht jeden Satz mit einer Begrüßung. " 
                f"{name_hint} Antworte ausschließlich auf Deutsch.\n\nGespräch:\n")
    return ("System: You are Björn, a kind teddy-bear helper for kids. "
            "Short, simple sentences (≤30 words). Include a small hint or question. "
            "Be safe; gently refuse unsafe or adult topics and suggest alternatives. "
            "Do not start every sentence with a greeting. "
            f"{name_hint} Reply only in English.\n\nConversation:\n")

def _blocked(t):
    t = (t or "").lower()
    return any(w in t for w in BLOCKLIST)

def _safety_wrap(language, user_text, reply):
    if _blocked(user_text) or _blocked(reply):
        return REFUSAL_DE if (language or "").startswith("de") else REFUSAL_EN
    return reply.strip()

def _build_prompt(history, user_text, language, user_name):
    parts = [_persona(language, user_name)]
    recent = history[-MAX_TURNS_PER_SESSION:]
    for turn in recent:
        role = turn.get("role","user"); content = (turn.get("content","") or "").strip()
        if not content: continue
        parts.append(("User: " if role=="user" else "Assistant: ") + content + "\n")
    parts.append(f"User: {user_text.strip()}\nAssistant:")
    return "".join(parts)

@app.route("/talk", methods=["POST"])
def talk():
    body = request.json or {}
    user_text = (body.get("text") or "").strip()
    language = (body.get("language") or "en").strip()
    session_id = (body.get("session_id") or "default").strip()
    user_name = (body.get("user_name") or "").strip()
    if not user_text: return jsonify({"error":"Missing text"}), 400
    if not ensure_ollama_running(): return jsonify({"reply":"Ollama could not be started or reached."}), 503

    with LOCK:
        if user_name: NAMES[session_id] = user_name
    name_for_session = NAMES.get(session_id, "")

    history = _get_session(session_id)
    prompt = _build_prompt(history, user_text, language, name_for_session)

    try:
        r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=120)
        r.raise_for_status()
        data = r.json()
        raw_reply = (data.get("response") or "").strip()

        with LOCK:
            last_use = NAME_LAST_USE.get(session_id, -999)
            turn_idx = len(history)
            allow_name = (turn_idx - last_use) >= 6
        reply = _safety_wrap(language, user_text, raw_reply)

        if name_for_session and allow_name and re.search(rf"\b{re.escape(name_for_session)}\b", raw_reply, flags=re.I):
            with LOCK: NAME_LAST_USE[session_id] = len(history)

        user_item = {"role":"user","content":user_text,"lang":language,"ts":time.time()}
        asst_item = {"role":"assistant","content":reply,"lang":language,"ts":time.time()}
        with LOCK:
            history.append(user_item); history.append(asst_item)
            if len(history) > MAX_TURNS_PER_SESSION * 3:
                del history[: (len(history) - MAX_TURNS_PER_SESSION * 2)]
        _append_to_disk(session_id, user_item); _append_to_disk(session_id, asst_item)
        return jsonify({"reply": reply, "session_id": session_id})
    except Exception as e:
        return jsonify({"reply": f"Error contacting Ollama: {e}"}), 500

@app.route("/logs", methods=["GET"])
def list_sessions():
    items = [p.stem.replace("session_","") for p in sorted(MEM_DIR.glob("session_*.jsonl"))]
    html = ["<html><body><h2>Björn Sessions</h2><ul>"] + [f'<li><a href="/session/{sid}">{sid}</a></li>' for sid in items] + ["</ul></body></html>"]
    return Response("".join(html), mimetype="text/html")

@app.route("/session/<session_id>", methods=["GET"])
def view_session(session_id):
    path = MEM_DIR / f"session_{session_id}.jsonl"
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try: rows.append(json.loads(line.strip()))
                except Exception: pass
    html = ['<html><body><a href="/logs">← back</a><h2>Session '
            f'{session_id}</h2><div style="font-family:system-ui;max-width:800px">']
    for r in rows:
        role = r.get("role",""); content = (r.get("content","") or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        ts = datetime.utcfromtimestamp(r.get("ts", time.time())).strftime("%Y-%m-%d %H:%M:%S")
        color = "#eef" if role == "user" else "#efe"
        html.append(f'<div style="background:{color};padding:10px;margin:8px 0;border-radius:8px"><div style="opacity:.6">{role} · {ts}</div><div>{content}</div></div>')
    html.append("</div></body></html>")
    return Response("".join(html), mimetype="text/html")

if __name__ == "__main__":
    ensure_ollama_running()
    app.run(host="0.0.0.0", port=5000)
