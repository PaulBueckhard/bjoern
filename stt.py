import sounddevice as sd
import vosk
import queue
import sys
import json
import RPi.GPIO as GPIO
import time

# --- Setup button on GPIO 17 ---
BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # button wired to GND

# --- Audio queue ---
q = queue.Queue()

def callback(indata, frames, time_, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# --- Load Vosk model ---
model_en = vosk.Model("vosk-model-english")
model_de = vosk.Model("vosk-model-german")

model = model_de
samplerate = 16000

# --- Storage for recordings ---
recordings = []  # list of recognized texts

# --- Audio input stream ---
with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    rec = vosk.KaldiRecognizer(model, samplerate)
    print("Ready. Hold button to record.")

    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # button pressed
            print("Recording...")
            rec = vosk.KaldiRecognizer(model, samplerate)  # reset recognizer
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:  # keep recording
                data = q.get()
                rec.AcceptWaveform(data)
            # Button released -> finalize
            result = json.loads(rec.FinalResult())["text"]
            recordings.append(result)
            print("Saved recording:", result)
            # Save to file
            with open("recordings.txt", "a") as f:
                f.write(result + "\n")
            time.sleep(0.2)  # debounce
        else:
            time.sleep(0.05)
