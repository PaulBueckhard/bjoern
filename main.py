import os, time, json, requests, re, uuid
from pathlib import Path
try:
    import RPi.GPIO as GPIO; ON_PI = True
except Exception:
    GPIO = None; ON_PI = False
from STT import SpeechToText
import TTS

MEM_DIR = Path("memory")
MEM_DIR.mkdir(parents=True, exist_ok=True)

BUTTON_PIN = 17
LLM_SERVER_URL = "http://192.168.2.31:5000/talk"
LOG_PATH = "memory/conversation_log.txt"
SETTINGS_PATH = Path("memory/settings.json")

VOSK_MODEL_EN = "sst_models/vosk-model-english"
VOSK_MODEL_DE = "sst_models/vosk-model-german"
SAMPLERATE = 16000
BLOCKSIZE  = 8000
DEFAULT_STT_DEVICE = None if ON_PI else int(os.getenv("STT_DEVICE", "2"))

def get_session_id():
    p = Path("memory/session_id.txt")
    if p.exists():
        sid = (p.read_text(encoding="utf-8").strip() or "").strip()
        if sid: return sid
    sid = str(uuid.uuid4()); p.write_text(sid, encoding="utf-8"); return sid

def load_settings():
    if SETTINGS_PATH.exists():
        try: return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception: pass
    return {"language":"", "user_name":""}

def save_settings(s):
    SETTINGS_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

class Button:
    def __init__(self, pin):
        self.pin = pin
        if ON_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    def is_pressed(self):
        if ON_PI: return GPIO.input(self.pin) == GPIO.LOW
        import ctypes; return bool(ctypes.windll.user32.GetAsyncKeyState(0x26) & 0x8000)
    def wait_for_press(self):
        if ON_PI:
            while GPIO.input(self.pin) != GPIO.LOW: time.sleep(0.02)
            time.sleep(0.03)
        else:
            print("➡️  Hold ↑ to START…")
            while not self.is_pressed(): time.sleep(0.02)
            print("🎙️  Recording… (release ↑ to stop)")
    def stop_condition(self):
        return GPIO.input(self.pin) == GPIO.HIGH if ON_PI else (not self.is_pressed())
    def cleanup(self):
        if ON_PI: GPIO.cleanup()

def _record_on_next_press(stt, button):
    button.wait_for_press()
    if ON_PI and not button.is_pressed(): return ""
    return stt.transcribe_until(button.stop_condition)

def _detect_language_word(text):
    t = (text or "").strip().lower()
    if any(w in t for w in ["german","deutsch"]): return "de"
    if any(w in t for w in ["english","englisch"]): return "en"
    return ""

def choose_language_via_voice(stt, button):
    TTS.speak("Hello! What language should I use: German or English?", "en")
    while True:
        spoken = _record_on_next_press(stt, button)
        if not spoken:
            TTS.speak("I didn't hear anything. Please say German or English.", "en"); continue
        lang = _detect_language_word(spoken)
        if not lang:
            TTS.speak("Sorry, I didn't understand. Please say German or English.", "en"); continue
        if lang == "de": TTS.speak("Okay, dann spreche ich nun Deutsch.", "de")
        else: TTS.speak("Okay, I will continue to speak English.", "en")
        return lang

def _extract_name(text):
    t = re.sub(r"[^A-Za-zÄÖÜäöüß\-'\s]", " ", (text or "").strip())
    m = re.search(r"(?:ich heiße|mein name ist|i am|i'm|my name is)\s+(.+)$", t, flags=re.I)
    if m: t = m.group(1)
    parts = [p for p in re.split(r"\s+", t) if p]
    if not parts: return ""
    if len(parts) >= 2 and parts[0].lower() in {"i","ich","mein","my"}: parts = parts[1:]
    return parts[0][:32].strip(" -'").title()

def ask_user_name(stt, button, language):
    if language == "de": TTS.speak("Wie heißt du? Halte die Taste und sag deinen Namen.", "de")
    else: TTS.speak("What is your name? Hold the button and say your name.", "en")
    for _ in range(3):
        name = _extract_name(_record_on_next_press(stt, button))
        if name:
            if language == "de": TTS.speak(f"Hallo {name}. Schön, dich kennenzulernen.", "de")
            else: TTS.speak(f"Hi {name}. Nice to meet you.", "en")
            return name
        if language == "de": TTS.speak("Bitte sag nur deinen Vornamen.", "de")
        else: TTS.speak("Please say just your first name.", "en")
    if language == "de": TTS.speak("Ich nenne dich Freund.", "de"); return "Freund"
    TTS.speak("I'll call you Friend.", "en"); return "Friend"

def send_to_llm(text, language, session_id, user_name):
    try:
        r = requests.post(LLM_SERVER_URL, json={
            "text": text, "language": language, "session_id": session_id, "user_name": user_name
        }, timeout=60)
        r.raise_for_status()
        return (r.json().get("reply") or "").strip()
    except Exception as e:
        print("[LLM] Error:", e)
        return "Sorry, I couldn't reach the AI server."

def main():
    button = Button(BUTTON_PIN)
    session_id = get_session_id()
    settings = load_settings()
    stt = SpeechToText(
        model_path_en=VOSK_MODEL_EN, model_path_de=VOSK_MODEL_DE,
        samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
        language=settings.get("language") or "en",
        device=DEFAULT_STT_DEVICE,
    )
    lang = settings.get("language") or ""
    if lang not in ("en","de"):
        print("Starting language setup…")
        lang = choose_language_via_voice(stt, button)
        settings["language"] = lang; save_settings(settings)
    stt.set_language(lang)
    user_name = (settings.get("user_name") or "").strip()
    if not user_name:
        user_name = ask_user_name(stt, button, lang)
        settings["user_name"] = user_name; save_settings(settings)

    if lang == "de": TTS.speak(f"{user_name}, du kannst jetzt sprechen.", "de")
    else: TTS.speak(f"{user_name}, you can speak now.", "en")
    print(f"[Ready] lang={lang} user={user_name} session={session_id}")

    try:
        while True:
            print("Hold ↑ (or button) to talk…")
            text = _record_on_next_press(stt, button)
            if text:
                print(f"You: {text}")
                reply = send_to_llm(text, lang, session_id, user_name)
                print(f"AI:   {reply}")
                if reply.strip(): TTS.speak(reply, language=lang)
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"user": user_name, "lang": lang, "input": text, "reply": reply}) + "\n")
            else:
                print("No speech detected.")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        button.cleanup()

if __name__ == "__main__":
    main()
