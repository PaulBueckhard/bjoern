from flask import Flask, request, jsonify
import requests, os, subprocess, time

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_BIN = os.getenv("OLLAMA_BIN", "ollama")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")

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

@app.route("/talk", methods=["POST"])
def talk():
    user_text = (request.json or {}).get("text", "").strip()
    if not user_text:
        return jsonify({"error": "Missing text"}), 400

    if not ensure_ollama_running():
        return jsonify({"reply": "Ollama could not be started or reached."}), 503

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": user_text, "stream": False},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        reply = data.get("response", "").strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error contacting Ollama: {e}"}), 500

if __name__ == "__main__":
    # Auto-start Ollama once when the server boots up
    ensure_ollama_running()
    app.run(host="0.0.0.0", port=5000)
