import json, uuid, requests, re
from pathlib import Path
import TTS
from STT import SpeechToText

SETTINGS_PATH = Path("memory/settings.json")
CREATE_SHORT_URL = "http://192.168.2.31:5000/api/create_short_id"


def load_settings():
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except:
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


def _record_on_next_press(stt, button):
    button.wait_for_press()
    return stt.transcribe_until(button.stop_condition)


def _detect_language_word(text: str) -> str:
    t = text.lower().strip()
    if "german" in t or "deutsch" in t:
        return "de"
    if "english" in t or "englisch" in t:
        return "en"
    return ""


def choose_language_via_voice(stt, button):
    TTS.speak("Hello! I am Björn! Your personal plushie assistant! Please tell me what language I should use.", "en")
    while True:
        spoken = _record_on_next_press(stt, button)
        lang = _detect_language_word(spoken)
        if lang == "de":
            TTS.speak("Okay, dann spreche ich nun Deutsch.", "de")
            return "de"
        if lang == "en":
            TTS.speak("Okay, I will continue to speak English.", "en")
            return "en"
        TTS.speak("Sorry, I didn't understand that. Please say if I should speak english or german.", "en")


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


def ask_user_name(stt, button, lang):
    if lang == "de":
        TTS.speak("Könntest du mir deinen Namen verraten?", "de")
    else:
        TTS.speak("May I know what your name is?", "en")

    for _ in range(3):
        name = _extract_name(_record_on_next_press(stt, button))
        if name:
            return name

        if lang == "de":
            TTS.speak("Entschuldige, könntest du das wiederholen? Bitte sag nur deinen Vornamen.", "de")
        else:
            TTS.speak("Sorry, could you repeat that. Please just say your first name.", "en")

    return "mein Freund" if lang == "de" else "my Friend"


def ask_user_name_with_confirmation(stt, button, lang):
    while True:
        name = ask_user_name(stt, button, lang)

        if lang == "de":
            TTS.speak(f"Habe ich das richtig verstanden? Dein Name ist {name}?", "de")
        else:
            TTS.speak(f"Did I understand that correctly? Your name is {name}?", "en")

        answer = _record_on_next_press(stt, button).lower()

        if "ja" in answer or "yes" in answer:
            return name

        if lang == "de":
            TTS.speak("Oh, das tut mir aber leid.", "de")
        else:
            TTS.speak("Oh, I am sorry about that.", "en")


_PASSWORD_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "null": "0", "eins": "1", "ein": "1",
    "zwei": "2", "drei": "3", "vier": "4",
    "fuenf": "5", "fünf": "5", "sechs": "6",
    "sieben": "7", "acht": "8", "neun": "9",
}

def _normalize_password_text(text: str) -> str:
    t = text.lower()
    return (t.replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss"))

def _parse_password(text: str) -> str:
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
            out.extend(tok)
        elif tok in _PASSWORD_WORDS:
            out.append(_PASSWORD_WORDS[tok])

    return "".join(out[:4]) if len(out) >= 4 else ""


def ask_parent_password(stt, button, lang):
    if lang == "de":
        TTS.speak("Der nächste Schritt ist für die Eltern gedacht. Bitte lege ein vier stelliges Elternpasswort fest.", "de")
    else:
        TTS.speak("The next step is for parents only. Please create a four digit parent password.", "en")

    while True:
        spoken = _record_on_next_press(stt, button)
        pw = _parse_password(spoken)

        def spell_digits(pw: str) -> str:
            return " ".join(list(pw))
        
        if len(pw) == 4:
            spoken_pw = spell_digits(pw)

            if lang == "de":
                TTS.speak(f"Dankeschön, dein Elternpasswort lautet {spoken_pw}.", "de")
            else:
                TTS.speak(f"Thank you, your parent password is {spoken_pw}.", "en")
            return pw

        if lang == "de":
            TTS.speak("Bitte wiederhole das Elternpasswort. Nenne nur vier Zahlen.", "de")
        else:
            TTS.speak("Please repeat your parent password. Name four digits only.", "en")


def register_short_id(session_id: str, parent_password: str) -> str | None:
    try:
        r = requests.post(
            CREATE_SHORT_URL,
            json={"session_id": session_id, "pin": parent_password},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("short_id")
    except:
        return None


def run_initial_setup(button):
    settings = load_settings()

    stt = SpeechToText(
        model_path_en="sst_models/vosk-model-english",
        model_path_de="sst_models/vosk-model-german",
        samplerate=16000,
        blocksize=8000,
        language=settings.get("language") or "en",
        device=None,
    )

    lang = settings.get("language", "")
    user_name = settings.get("user_name", "")
    parent_password = settings.get("parent_password", "")
    session_id = settings.get("session_id", "")
    short_id = settings.get("short_id", "")

    first = False

    if lang not in ("en", "de"):
        lang = choose_language_via_voice(stt, button)
        settings["language"] = lang
        save_settings(settings)
        first = True

    stt.set_language(lang)

    if not user_name:
        user_name = ask_user_name_with_confirmation(stt, button, lang)
        settings["user_name"] = user_name
        save_settings(settings)
        first = True

    if not parent_password:
        parent_password = ask_parent_password(stt, button, lang)
        settings["parent_password"] = parent_password
        save_settings(settings)
        first = True

    if not session_id:
        session_id = str(uuid.uuid4())
        settings["session_id"] = session_id
        save_settings(settings)
        first = True

    if not short_id:
        short_id = register_short_id(session_id, parent_password) or ""
        settings["short_id"] = short_id
        save_settings(settings)

        if short_id:
            spelled = " ".join(list(short_id.upper()))
            if lang == "de":
                TTS.speak(f"Dein Sitzungs-Code lautet: {spelled}. Ich wiederhole, dein Sitzungs-Code lautet: {spelled}.", "de")
            else:
                TTS.speak(f"Your session code is: {spelled}. I repeat, your session code is: {spelled}.", "en")

        first = True

    if first:
        if lang == "de":
            TTS.speak(f"Danke {user_name}, du bist nun fertig und kannst mit mir reden.", "de")
        else:
            TTS.speak(f"Thank you {user_name}, you are done and can chat with me now.", "en")

    return settings
