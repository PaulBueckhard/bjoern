import json, uuid, requests, re
from pathlib import Path
import TTS
from STT import SpeechToText

# ------------------------------------------------------------
# Persistent settings & backend service configuration
# ------------------------------------------------------------

SETTINGS_PATH = Path("memory/settings.json")

# Local service generating short session IDs
CREATE_SHORT_URL = "http://192.168.2.31:5000/api/create_short_id"


# ------------------------------------------------------------
# Settings handling
# ------------------------------------------------------------

def load_settings():
    """
    Load user settings from disk.
    Returns default structure if file is missing or unreadable.
    """
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
    """Persist user settings as formatted JSON."""
    SETTINGS_PATH.write_text(
        json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ------------------------------------------------------------
# Helper wrappers around STT + physical button interaction
# ------------------------------------------------------------

def _record_on_next_press(stt, button):
    """
    Wait until the button is pressed, then record speech until the
    button-specific stop_condition returns True.
    """
    button.wait_for_press()
    return stt.transcribe_until(button.stop_condition)


# ------------------------------------------------------------
# Language selection
# ------------------------------------------------------------

def _detect_language_word(text: str) -> str:
    """
    Determine preferred language based on spoken keywords.
    Supports English/German.
    """
    t = text.lower().strip()
    if "german" in t or "deutsch" in t:
        return "de"
    if "english" in t or "englisch" in t:
        return "en"
    return ""


def choose_language_via_voice(stt, button):
    """
    Voice-driven language selection.
    Continues prompting until user clearly says English or German.
    """
    TTS.speak(
        "Hello! I am Björn! Your personal plushie assistant! Please tell me what language I should use.",
        "en",
    )
    while True:
        spoken = _record_on_next_press(stt, button)
        lang = _detect_language_word(spoken)

        if lang == "de":
            TTS.speak("Okay, dann spreche ich nun Deutsch.", "de")
            return "de"

        if lang == "en":
            TTS.speak("Okay, I will continue to speak English.", "en")
            return "en"

        TTS.speak(
            "Sorry, I didn't understand that. Please say if I should speak english or german.",
            "en",
        )


# ------------------------------------------------------------
# Name extraction and confirmation
# ------------------------------------------------------------

def _extract_name(text: str) -> str:
    """
    Attempt to extract a first name from free-form spoken text.
    Handles English/German common intro phrases, strips symbols,
    and sanitizes output.
    """
    # Normalize characters
    t = re.sub(r"[^A-Za-zÄÖÜäöüß\-'\s]", " ", text or "")

    # Extract after common name phrases (English/German)
    m = re.search(
        r"(?:ich heiße|mein name ist|i am|i'm|my name is)\s+(.+)$",
        t,
        flags=re.I,
    )
    if m:
        t = m.group(1)

    parts = [p for p in re.split(r"\s+", t.strip()) if p]
    if not parts:
        return ""

    # Remove leading pronouns if present
    if parts[0].lower() in {"i", "ich", "mein", "my"} and len(parts) >= 2:
        parts = parts[1:]

    # Return first usable token, title-cased, max length 32
    return parts[0][:32].strip(" -'").title()


def ask_user_name(stt, button, lang):
    """Ask user for first name up to 3 attempts, fallback to friendly default."""
    if lang == "de":
        TTS.speak("Könntest du mir deinen Namen verraten?", "de")
    else:
        TTS.speak("May I know what your name is?", "en")

    for _ in range(3):
        name = _extract_name(_record_on_next_press(stt, button))
        if name:
            return name

        if lang == "de":
            TTS.speak(
                "Entschuldige, könntest du das wiederholen? Bitte sag nur deinen Vornamen.",
                "de",
            )
        else:
            TTS.speak(
                "Sorry, could you repeat that. Please just say your first name.",
                "en",
            )

    return "mein Freund" if lang == "de" else "my Friend"


def ask_user_name_with_confirmation(stt, button, lang):
    """
    Ask for name, then confirm using a yes/no question.
    Repeats until confirmation is given.
    """
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


# ------------------------------------------------------------
# Parent password parsing (spoken digits)
# ------------------------------------------------------------

_PASSWORD_WORDS = {
    # English digits
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",

    # German digits
    "null": "0", "eins": "1", "ein": "1",
    "zwei": "2", "drei": "3", "vier": "4",
    "fuenf": "5", "fünf": "5", "sechs": "6",
    "sieben": "7", "acht": "8", "neun": "9",
}


def _normalize_password_text(text: str) -> str:
    """Replace umlauts with ASCII fallback to match _PASSWORD_WORDS keys."""
    t = text.lower()
    return (
        t.replace("ä", "ae")
         .replace("ö", "oe")
         .replace("ü", "ue")
         .replace("ß", "ss")
    )


def _parse_password(text: str) -> str:
    """
    Parse spoken (English/German) 4-digit password from raw STT text.
    Supports:
        "1 2 3 4"
        "eins zwei drei vier"
        "one two three four"
        "password is one five eight nine"
        "4321"
    """
    if not text:
        return ""

    raw = text.strip()

    # First: direct extraction of digits if possible
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 4:
        return digits[:4]

    # Token-based approach
    norm = _normalize_password_text(raw)
    tokens = re.split(r"\s+|[,.;:]+", norm)

    out = []
    for tok in tokens:
        if tok.isdigit():
            out.extend(tok)            # multiple digits inside token
        elif tok in _PASSWORD_WORDS:
            out.append(_PASSWORD_WORDS[tok])

    return "".join(out[:4]) if len(out) >= 4 else ""


def ask_parent_password(stt, button, lang):
    """Voice-guided creation of a 4-digit parent password."""
    if lang == "de":
        TTS.speak(
            "Der nächste Schritt ist für die Eltern gedacht. Bitte lege ein vier stelliges Elternpasswort fest.",
            "de",
        )
    else:
        TTS.speak(
            "The next step is for parents only. Please create a four digit parent password.",
            "en",
        )

    while True:
        spoken = _record_on_next_press(stt, button)
        pw = _parse_password(spoken)

        def spell_digits(pw: str) -> str:
            return " ".join(list(pw))

        if len(pw) == 4:
            spelled = spell_digits(pw)

            if lang == "de":
                TTS.speak(f"Dankeschön, dein Elternpasswort lautet {spelled}.", "de")
            else:
                TTS.speak(f"Thank you, your parent password is {spelled}.", "en")
            return pw

        # Retry explanation
        if lang == "de":
            TTS.speak("Bitte wiederhole das Elternpasswort. Nenne nur vier Zahlen.", "de")
        else:
            TTS.speak("Please repeat your parent password. Name four digits only.", "en")


# ------------------------------------------------------------
# Session + short ID registration
# ------------------------------------------------------------

def register_short_id(session_id: str, parent_password: str) -> str | None:
    """Register a session ID with backend service and receive short code."""
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


# ------------------------------------------------------------
# Main interactive onboarding flow
# ------------------------------------------------------------

def run_initial_setup(button):
    """
    Full multi-step onboarding:
      - Choose language
      - Capture name
      - Capture parent password
      - Generate session ID
      - Register short ID
    Uses saved settings to avoid repeating steps.
    """
    settings = load_settings()

    # STT instance
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

    first = False  # Tracks whether this is the user's first successful setup

    # ------------------
    # Select language
    # ------------------
    if lang not in ("en", "de"):
        lang = choose_language_via_voice(stt, button)
        settings["language"] = lang
        save_settings(settings)
        first = True

    stt.set_language(lang)

    # ------------------
    # User name
    # ------------------
    if not user_name:
        user_name = ask_user_name_with_confirmation(stt, button, lang)
        settings["user_name"] = user_name
        save_settings(settings)
        first = True

    # ------------------
    # Parent password
    # ------------------
    if not parent_password:
        parent_password = ask_parent_password(stt, button, lang)
        settings["parent_password"] = parent_password
        save_settings(settings)
        first = True

    # ------------------
    # Session ID
    # ------------------
    if not session_id:
        session_id = str(uuid.uuid4())
        settings["session_id"] = session_id
        save_settings(settings)
        first = True

    # ------------------
    # Short ID registration
    # ------------------
    if not short_id:
        short_id = register_short_id(session_id, parent_password) or ""
        settings["short_id"] = short_id
        save_settings(settings)

        if short_id:
            spelled = " ".join(list(short_id.upper()))
            if lang == "de":
                TTS.speak(
                    f"Dein Sitzungs-Code lautet: {spelled}. Ich wiederhole, dein Sitzungs-Code lautet: {spelled}.",
                    "de",
                )
            else:
                TTS.speak(
                    f"Your session code is: {spelled}. I repeat, your session code is: {spelled}.",
                    "en",
                )

        first = True

    # ------------------
    # Completion message
    # ------------------
    if first:
        if lang == "de":
            TTS.speak(f"Danke {user_name}, du bist nun fertig und kannst mit mir reden.", "de")
        else:
            TTS.speak(f"Thank you {user_name}, you are done and can chat with me now.", "en")

    return settings
