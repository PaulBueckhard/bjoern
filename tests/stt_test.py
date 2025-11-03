import sounddevice as sd
import vosk
import queue
import sys
import json

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

model_en = vosk.Model("sst_models/vosk-model-english")
model_de = vosk.Model("sst_models/vosk-model-german")

model = model_de
samplerate = 16000

with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    rec = vosk.KaldiRecognizer(model, samplerate)
    print("Listening...")
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            print(json.loads(rec.Result())["text"])
        else:
            print(json.loads(rec.PartialResult())["partial"])
