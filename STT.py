import sounddevice as sd
import vosk
import queue
import sys
import json
import RPi.GPIO as GPIO
import time
from queue import Empty
import requests

import TTS

BUTTON_PIN = 17
LLM_SERVER_URL = "http://192.168.2.31:5000/talk"

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

model_en = vosk.Model("sst_models/vosk-model-english")
model_de = vosk.Model("sst_models/vosk-model-german")
model = model_en  # choose model here

samplerate = 16000

recordings = []

def record_while_button():
    q = queue.Queue()

    def callback(indata, frames, time_, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    rec = vosk.KaldiRecognizer(model, samplerate)

    try:
        with sd.RawInputStream(samplerate=samplerate,
                              blocksize=8000,
                              dtype='int16',
                              channels=1,
                              callback=callback):
            print("Recording... (release button to stop)")
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                try:
                    data = q.get(timeout=0.2)
                except Empty:
                    continue
                rec.AcceptWaveform(data)
            final = json.loads(rec.FinalResult())["text"]
            return final
    except Exception as e:
        print("Audio error:", e, file=sys.stderr)
        return ""

def send_to_llm(text):
    try:
        response = requests.post(
            LLM_SERVER_URL,
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        reply = data.get("reply", "")
        return reply
    except Exception as e:
        print("Error contacting LLM:", e)
        return "Sorry, I couldn’t reach the AI server."

def main():
    print("Ready. Hold button to record.")
    try:
        while True:
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                text = record_while_button()
                if text.strip():
                    print(f"You said: {text!r}")
                    reply = send_to_llm(text)
                    print(f"AI replied: {reply!r}")
                    TTS.speak(reply, language="en")

                    recordings.append({"input": text, "reply": reply})
                    with open("conversation_log.txt", "a", encoding="utf-8") as f:
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
