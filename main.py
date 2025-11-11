import os
import time
import json
import requests
from pathlib import Path
import uuid

try:
    import RPi.GPIO as GPIO
    ON_PI = True
except Exception:
    GPIO = None
    ON_PI = False

from STT import SpeechToText
import TTS

BUTTON_PIN = 17
LLM_SERVER_URL = "http://192.168.2.31:5000/talk"
LOG_PATH = "conversation_log.txt"

VOSK_MODEL_EN = "sst_models/vosk-model-english"
VOSK_MODEL_DE = "sst_models/vosk-model-german"
SAMPLERATE = 16000
BLOCKSIZE  = 8000
DEFAULT_STT_DEVICE = None if ON_PI else int(os.getenv("STT_DEVICE", "2"))


def get_session_id() -> str:
    path = Path("session_id.txt")
    if path.exists():
        sid = (path.read_text(encoding="utf-8").strip() or "").strip()
        if sid:
            return sid
    sid = str(uuid.uuid4())
    path.write_text(sid, encoding="utf-8")
    return sid


class Button:
    def __init__(self, pin: int):
        self.pin = pin
        if ON_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        else:
            pass

    def is_pressed(self) -> bool:
        if ON_PI:
            return GPIO.input(self.pin) == GPIO.LOW
        else:
            import ctypes
            VK_UP = 0x26
            return bool(ctypes.windll.user32.GetAsyncKeyState(VK_UP) & 0x8000)

    def wait_for_press(self):
        if ON_PI:
            while GPIO.input(self.pin) != GPIO.LOW:
                time.sleep(0.02)
            time.sleep(0.03)
        else:
            print("➡️  Hold the ↑ arrow key to START speaking…")
            while not self.is_pressed():
                time.sleep(0.02)
            print("🎙️  Recording... (release ↑ to stop)")

    def stop_condition(self):
        if ON_PI:
            return GPIO.input(self.pin) == GPIO.HIGH
        else:
            return not self.is_pressed()

    def cleanup(self):
        if ON_PI:
            GPIO.cleanup()


def _record_on_next_press(stt: SpeechToText, button: Button) -> str:
    button.wait_for_press()
    if ON_PI and not button.is_pressed():
        return ""
    stop_fn = button.stop_condition
    return stt.transcribe_until(stop_fn)


def _detect_language_word(text: str) -> str:
    t = (text or "").strip().lower()
    if any(w in t for w in ["german", "deutsch"]):
        return "de"
    if any(w in t for w in ["english", "englisch"]):
        return "en"
    return ""


def choose_language_via_voice(stt: SpeechToText, button: Button) -> str:
    TTS.speak("Hello! What language should I use: German or English?", "en")
    while True:
        spoken = _record_on_next_press(stt, button)
        if not spoken:
            TTS.speak("I didn't hear anything. Please try again and say German or English.", "en")
            continue
        lang_choice = _detect_language_word(spoken)
        if not lang_choice:
            TTS.speak("Sorry, I didn't understand. Please say German or English.", "en")
            continue

        if lang_choice == "de":
            TTS.speak("Okay, dann spreche ich nun Deutsch.", "de")
        else:
            TTS.speak("Okay, I will continue to speak English.", "en")
        return lang_choice


def send_to_llm(text: str, language: str, session_id: str) -> str:
    try:
        r = requests.post(
            LLM_SERVER_URL,
            json={"text": text, "language": language, "session_id": session_id},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        return data.get("reply", "").strip()
    except Exception as e:
        print("[LLM] Error contacting server:", e)
        return "Sorry, I couldn't reach the AI server."


def main():
    button = Button(BUTTON_PIN)
    session_id = get_session_id()

    stt = SpeechToText(
        model_path_en=VOSK_MODEL_EN,
        model_path_de=VOSK_MODEL_DE,
        samplerate=SAMPLERATE,
        blocksize=BLOCKSIZE,
        language="en",
        device=DEFAULT_STT_DEVICE,
    )

    print("Starting language setup…")
    language = choose_language_via_voice(stt, button)
    stt.set_language(language)
    print(f"Language set to: {language}. Ready.")

    try:
        while True:
            print("Hold ↑ arrow key (or button) to talk…")
            text = _record_on_next_press(stt, button)
            if text:
                print(f"You said: {text!r}")
                reply = send_to_llm(text, language, session_id)
                print(f"AI replied: {reply!r}")
                if reply.strip():
                    TTS.speak(reply, language=language)
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"lang": language, "input": text, "reply": reply}) + "\n")
            else:
                print("No speech detected.")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
    finally:
        button.cleanup()


if __name__ == "__main__":
    main()
