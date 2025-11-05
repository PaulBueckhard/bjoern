import time
import json
import requests
import RPi.GPIO as GPIO

from STT import SpeechToText
import TTS

BUTTON_PIN = 17
LLM_SERVER_URL = "http://192.168.2.31:5000/talk"
LOG_PATH = "conversation_log.txt"

VOSK_MODEL_EN = "sst_models/vosk-model-english"
VOSK_MODEL_DE = "sst_models/vosk-model-german"
SAMPLERATE = 16000
BLOCKSIZE = 8000


def _record_on_next_press(stt: SpeechToText) -> str:
    while GPIO.input(BUTTON_PIN) != GPIO.LOW:
        time.sleep(0.02)
    time.sleep(0.03)  # debounce
    if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        return ""
    stop_fn = lambda: GPIO.input(BUTTON_PIN) == GPIO.HIGH
    return stt.transcribe_until(stop_fn)


def _detect_language_word(text: str) -> str:
    t = (text or "").strip().lower()
    if any(w in t for w in ["german", "deutsch"]):
        return "de"
    if any(w in t for w in ["english", "englisch"]):
        return "en"
    return ""


def choose_language_via_voice(stt: SpeechToText) -> str:
    TTS.speak("Hello! What language should I use: German or English?", "en")
    while True:
        spoken = _record_on_next_press(stt)
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


def send_to_llm(text: str, language: str) -> str:
    try:
        r = requests.post(
            LLM_SERVER_URL,
            json={"text": text, "language": language},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        return data.get("reply", "").strip()
    except Exception as e:
        print("[LLM] Error contacting server:", e)
        return "Sorry, I couldn't reach the AI server."


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    stt = SpeechToText(
        model_path_en=VOSK_MODEL_EN,
        model_path_de=VOSK_MODEL_DE,
        samplerate=SAMPLERATE,
        blocksize=BLOCKSIZE,
        language="en",
    )

    print("Starting language setup…")
    language = choose_language_via_voice(stt)
    stt.set_language(language)
    print(f"Language set to: {language}. Ready. Hold button to record.")

    try:
        while True:
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.03)
                if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
                    continue

                print("Recording... (release button to stop)")
                stop_fn = lambda: GPIO.input(BUTTON_PIN) == GPIO.HIGH
                text = stt.transcribe_until(stop_fn)

                if text:
                    print(f"You said: {text!r}")
                    reply = send_to_llm(text, language)
                    print(f"AI replied: {reply!r}")

                    if reply.strip():
                        TTS.speak(reply, language=language)

                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"lang": language, "input": text, "reply": reply}) + "\n")

                else:
                    print("No speech detected.")

                time.sleep(0.2)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
