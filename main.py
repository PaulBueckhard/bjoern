import os, time, json, requests
from pathlib import Path
try:
    import RPi.GPIO as GPIO; ON_PI = True
except:
    GPIO = None; ON_PI = False

from STT import SpeechToText
import TTS
from setup import run_initial_setup


BUTTON_PIN = 17
LLM_SERVER_URL = "http://192.168.2.31:5000/talk"
LOG_PATH = "memory/conversation_log.txt"


class Button:
    def __init__(self, pin):
        self.pin = pin
        if ON_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def is_pressed(self):
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
            print("➡️ Hold ↑ to START…")
            while not self.is_pressed():
                time.sleep(0.02)
            print("🎙️ Recording… (release ↑ to stop)")

    def stop_condition(self):
        return GPIO.input(self.pin) == GPIO.HIGH if ON_PI else (not self.is_pressed())

    def cleanup(self):
        if ON_PI:
            GPIO.cleanup()


def _record_on_next_press(stt, button):
    button.wait_for_press()
    return stt.transcribe_until(button.stop_condition)


def send_to_llm(text, lang, session_id, user_name):
    try:
        r = requests.post(
            LLM_SERVER_URL,
            json={"text": text, "language": lang,
                  "session_id": session_id, "user_name": user_name},
            timeout=60,
        )
        r.raise_for_status()
        return (r.json().get("reply") or "").strip()
    except Exception as e:
        print("[LLM] Error:", e)
        return "Sorry, I couldn't reach the AI server."


def main():
    button = Button(BUTTON_PIN)

    settings = run_initial_setup(button)

    lang         = settings["language"]
    user_name    = settings["user_name"]
    session_id   = settings["session_id"]
    short_id     = settings["short_id"]

    stt = SpeechToText(
        model_path_en="sst_models/vosk-model-english",
        model_path_de="sst_models/vosk-model-german",
        samplerate=16000,
        blocksize=8000,
        language=lang,
        device=None,
    )

    print(f"[Ready] lang={lang} user={user_name} session={session_id} short={short_id}")

    ask_code_phrases = ["code", "session", "my code", "session id",
                        "sitzung", "mein code", "sitzungs id"]

    try:
        while True:
            print("Hold ↑ to talk…")
            text = _record_on_next_press(stt, button)

            if text:
                if any(w in text.lower() for w in ask_code_phrases):
                    spelled = " ".join(list(short_id.upper()))
                    if lang == "de":
                        TTS.speak(f"Dein Sitzungs-Code lautet: {spelled}.", "de")
                    else:
                        TTS.speak(f"Your session code is: {spelled}.", "en")
                    continue

                reply = send_to_llm(text, lang, session_id, user_name)
                TTS.speak(reply, lang)

                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "user": user_name, "lang": lang,
                        "input": text, "reply": reply
                    }) + "\n")

            time.sleep(0.2)

    except KeyboardInterrupt:
        pass
    finally:
        button.cleanup()


if __name__ == "__main__":
    main()
