import os, time, json, requests, re, uuid
from pathlib import Path
try:
    import RPi.GPIO as GPIO; ON_PI = True
except Exception:
    GPIO = None; ON_PI = False

from STT import SpeechToText
import TTS


BUTTON_PIN = 17
LLM_SERVER_URL = "http://192.168.2.31:5000/talk"
CREATE_SHORT_URL = "http://192.168.2.31:5000/api/create_short_id"

SETTINGS_PATH = Path("memory/settings.json")
LOG_PATH = "memory/conversation_log.txt"

VOSK_MODEL_EN = "sst_models/vosk-model-english"
VOSK_MODEL_DE = "sst_models/vosk-model-german"
SAMPLERATE = 16000
BLOCKSIZE = 8000

DEFAULT_STT_DEVICE = None if ON_PI else int(os.getenv("STT_DEVICE", "2"))


def load_settings():
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "language": "",
        "user_name": "",
        "parent_password": "",
        "session_id": "",
        "short_id": "",
    }


def save_settings(s: dict):
    SETTINGS_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


class Button:
    def __init__(self, pin: int):
        self.pin = pin
        if ON_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def is_pressed(self) -> bool:
        if ON_PI:
            return GPIO.input(self.pin) == GPIO.LOW
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x26) & 0x8000)

    def wait_for_press(self):
        if ON_PI:
            while GPIO.input(self.pin) != GPIO.LOW:
                time.sleep(0.02)
            time.sleep(0.03)
        else:
            print("➡️  Hold ↑ to START…")
            while not self.is_pressed():
                time.sleep(0.02)
            print("🎙️  Recording… (release ↑ to stop)")

    def stop_condition(self) -> bool:
        return GPIO.input(self.pin) == GPIO.HIGH if ON_PI else (not self.is_pressed())

    def cleanup(self):
        if ON_PI:
            GPIO.cleanup()

def _record_on_next_press(stt: SpeechToText, button: Button) -> str:
    button.wait_for_press()
    if ON_PI and not button.is_pressed():
        return ""
    return stt.transcribe_until(button.stop_condition)


def _detect_language_word(text: str) -> str:
    t = (text or "").strip().lower()
    if any(w in t for w in ["german", "deutsch"]):
        return "de"
    if any(w in t for w in ["english", "englisch"]):
        return "en"
    return ""

def speak_spelled(text: str, lang: str):
    spelled = " ".join(list(text.upper()))
    if lang == "de":
        TTS.speak(f"Dein Sitzungs-Code lautet: {spelled}. Ich wiederhole: {spelled}.", "de")
    else:
        TTS.speak(f"Your session code is: {spelled}. I repeat: {spelled}.", "en")

def choose_language_via_voice(stt: SpeechToText, button: Button) -> str:
    TTS.speak("Hello! What language should I use: German or English?", "en")
    while True:
        spoken = _record_on_next_press(stt, button)
        if not spoken:
            TTS.speak("I didn't hear anything. Please say German or English.", "en")
            continue

        lang = _detect_language_word(spoken)
        if not lang:
            TTS.speak("Sorry, I didn't understand. Please say German or English.", "en")
            continue

        if lang == "de":
            TTS.speak("Okay, dann spreche ich nun Deutsch.", "de")
        else:
            TTS.speak("Okay, I will continue to speak English.", "en")

        return lang


def _extract_name(text: str) -> str:
    t = re.sub(r"[^A-Za-zÄÖÜäöüß\-'\s]", " ", text or "")
    m = re.search(r"(?:ich heiße|mein name ist|i am|i'm|my name is)\s+(.+)$", t, flags=re.I)
    if m:
        t = m.group(1)

    parts = [p for p in re.split(r"\s+", t.strip()) if p]
    if not parts:
        return ""

    if parts[0].lower() in {"i", "ich", "mein", "my"} and len(parts) >= 2:
        parts = parts[1:]

    return parts[0][:32].strip(" -'").title()


def ask_user_name(stt: SpeechToText, button: Button, language: str) -> str:
    if language == "de":
        TTS.speak("Wie heißt du? Halte die Taste und sag deinen Namen.", "de")
    else:
        TTS.speak("What is your name? Hold the button and say your name.", "en")

    for _ in range(3):
        name = _extract_name(_record_on_next_press(stt, button))
        if name:
            if language == "de":
                TTS.speak(f"Hallo {name}. Schön, dich kennenzulernen.", "de")
            else:
                TTS.speak(f"Hi {name}. Nice to meet you.", "en")
            return name

        if language == "de":
            TTS.speak("Bitte sag nur deinen Vornamen.", "de")
        else:
            TTS.speak("Please say just your first first name.", "en")

    if language == "de":
        TTS.speak("Ich nenne dich Freund.", "de")
        return "Freund"
    TTS.speak("I'll call you Friend.", "en")
    return "Friend"

_PASSWORD_WORDS = {
    # English
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",

    # German
    "null": "0",
    "eins": "1", "ein": "1",
    "zwei": "2",
    "drei": "3",
    "vier": "4",
    "fuenf": "5", "fünf": "5",
    "sechs": "6",
    "sieben": "7",
    "acht": "8",
    "neun": "9",
}

def _normalize_password_text(text: str) -> str:
    t = text.lower()
    t = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return t

def _parse_password(text: str) -> str:
    """Parse 4 digits from text like 'eins zwei drei vier' or '1 2 3 4'."""
    if not text:
        return ""

    raw = text.strip()

    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 4:
        return digits[:4]

    norm = _normalize_password_text(raw)
    tokens = re.split(r"\s+|[,.;:]+", norm)

    out = []
    for tok in tokens:
        if tok.isdigit():
            out.extend(list(tok))
        elif tok in _PASSWORD_WORDS:
            out.append(_PASSWORD_WORDS[tok])

    if len(out) >= 4:
        return "".join(out[:4])

    return ""


def ask_parent_password(stt: SpeechToText, button: Button, language: str) -> str:
    if language == "de":
        TTS.speak("Bitte lege jetzt ein vierstelliges Eltern-Passwort fest.", "de")
        TTS.speak("Du kannst zum Beispiel eins zwei drei vier sagen.", "de")
    else:
        TTS.speak("Please create a four-digit parent password now.", "en")
        TTS.speak("For example, say: one two three four.", "en")

    while True:
        spoken = _record_on_next_press(stt, button)
        pw = _parse_password(spoken)

        if len(pw) == 4:
            spelled = " ".join(pw)
            if language == "de":
                TTS.speak(f"Dein Eltern-Passwort ist {spelled}.", "de")
            else:
                TTS.speak(f"Your parent password is {spelled}.", "en")
            return pw

        if language == "de":
            TTS.speak("Ich habe das Passwort nicht verstanden. Bitte nochmal genau vier Ziffern.", "de")
        else:
            TTS.speak("I could not understand the password. Please say exactly four digits.", "en")


def send_to_llm(text: str, language: str, session_id: str, user_name: str) -> str:
    try:
        r = requests.post(
            LLM_SERVER_URL,
            json={
                "text": text,
                "language": language,
                "session_id": session_id,
                "user_name": user_name,
            },
            timeout=60,
        )
        r.raise_for_status()
        return (r.json().get("reply") or "").strip()
    except Exception as e:
        print("[LLM] Error:", e)
        return "Sorry, I couldn't reach the AI server."


def register_short_id(session_id: str, parent_password: str) -> str | None:
    try:
        r = requests.post(
            CREATE_SHORT_URL,
            json={"session_id": session_id, "pin": parent_password},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("short_id")
    except Exception as e:
        print("[short_id] Error:", e)
        return None

def main():
    button = Button(BUTTON_PIN)
    settings = load_settings()

    lang = settings.get("language", "")
    user_name = settings.get("user_name", "").strip()
    parent_password = settings.get("parent_password", "").strip()
    session_id = settings.get("session_id", "").strip()
    short_id = settings.get("short_id", "").strip()

    stt = SpeechToText(
        model_path_en=VOSK_MODEL_EN,
        model_path_de=VOSK_MODEL_DE,
        samplerate=SAMPLERATE,
        blocksize=BLOCKSIZE,
        language=lang or "en",
        device=DEFAULT_STT_DEVICE,
    )

    first_setup = False

    if lang not in ("en", "de"):
        first_setup = True
        lang = choose_language_via_voice(stt, button)
        settings["language"] = lang
        save_settings(settings)

    stt.set_language(lang)

    if not user_name:
        first_setup = True
        user_name = ask_user_name(stt, button, lang)
        settings["user_name"] = user_name
        save_settings(settings)

    if not parent_password:
        first_setup = True
        parent_password = ask_parent_password(stt, button, lang)
        settings["parent_password"] = parent_password
        save_settings(settings)

    if not session_id:
        first_setup = True
        session_id = str(uuid.uuid4())
        settings["session_id"] = session_id
        save_settings(settings)

    if not short_id:
        first_setup = True
        short_id = register_short_id(session_id, parent_password) or ""
        settings["short_id"] = short_id
        save_settings(settings)

        if short_id:
            speak_spelled(short_id, lang)
        else:
            if lang == "de":
                TTS.speak("Ich konnte den Sitzungs-Code nicht erstellen.", "de")
            else:
                TTS.speak("I could not create the session code.", "en")

    if first_setup:
        if lang == "de":
            TTS.speak(f"{user_name}, du kannst jetzt sprechen.", "de")
        else:
            TTS.speak(f"{user_name}, you can speak now.", "en")

    print(f"[Ready] lang={lang} user={user_name} session={session_id} short={short_id}")

    ask_code_words_en = ["code", "session", "my code", "session id"]
    ask_code_words_de = ["code", "sitzung", "mein code", "sitzungs id"]

    try:
        while True:
            print("Hold ↑ (or button) to talk…")
            text = _record_on_next_press(stt, button)

            if text:
                lower = text.lower()

                if any(w in lower for w in ask_code_words_en + ask_code_words_de):
                    speak_spelled(short_id, lang)
                    continue

                print(f"You: {text}")
                reply = send_to_llm(text, lang, session_id, user_name)
                print(f"AI:   {reply}")

                if reply.strip():
                    TTS.speak(reply, language=lang)

                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "user": user_name,
                        "lang": lang,
                        "input": text,
                        "reply": reply,
                    }) + "\n")

            else:
                print("No speech detected.")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        button.cleanup()

if __name__ == "__main__":
    main()
