import sounddevice as sd
import vosk
import queue
import sys
import json
from queue import Empty
from typing import Callable


class SpeechToText:
    def __init__(self,
                 model_path: str = "sst_models/vosk-model-english",
                 samplerate: int = 16000,
                 blocksize: int = 8000):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.model = vosk.Model(model_path)

    def transcribe_until(self, stop_fn: Callable[[], bool]) -> str:
        """
        Capture microphone audio and feed to Vosk until stop_fn() returns True.
        Returns final recognized text (lowercase from Vosk) or "".
        """
        q: queue.Queue[bytes] = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        rec = vosk.KaldiRecognizer(self.model, self.samplerate)

        try:
            with sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype='int16',
                channels=1,
                callback=callback
            ):
                while not stop_fn():
                    try:
                        data = q.get(timeout=0.2)
                    except Empty:
                        continue
                    rec.AcceptWaveform(data)

                final = json.loads(rec.FinalResult()).get("text", "")
                return (final or "").strip()
        except Exception as e:
            print("Audio error:", e, file=sys.stderr)
            return ""
