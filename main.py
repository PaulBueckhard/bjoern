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


def send_to_llm(text: str) -> str:
    try:
        r = requests.post(
            LLM_SERVER_URL,
            json={"text": text},
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        return data.get("reply", "").strip()
    except Exception as e:
        print("[LLM] Error contacting server:", e)
        return "Sorry, I couldn’t reach the AI server."


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    stt = SpeechToText(
        model_path=VOSK_MODEL_EN,
        samplerate=SAMPLERATE,
        blocksize=BLOCKSIZE
    )

    print("Ready. Hold button to record.")

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
                    reply = send_to_llm(text)
                    print(f"AI replied: {reply!r}")
                    TTS.speak(reply, language="en")

                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"input": text, "reply": reply}) + "\n")
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
