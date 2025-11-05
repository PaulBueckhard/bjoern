import sounddevice as sd
import vosk
import queue
import sys
import json
from queue import Empty
from typing import Callable, Optional


class SpeechToText:
    def __init__(
        self,
        model_path_en: str = "sst_models/vosk-model-english",
        model_path_de: str = "sst_models/vosk-model-german",
        samplerate: int = 16000,
        blocksize: int = 8000,
        language: str = "en",
    ):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._model_paths = {"en": model_path_en, "de": model_path_de}
        self._models: dict[str, vosk.Model] = {}
        self._lang = "en"
        self._model: Optional[vosk.Model] = None
        self.set_language(language)

    def _ensure_loaded(self, lang: str) -> vosk.Model:
        if lang not in self._models:
            path = self._model_paths.get(lang)
            if not path:
                raise ValueError(f"No Vosk model path configured for '{lang}'")
            self._models[lang] = vosk.Model(path)
        return self._models[lang]

    def set_language(self, lang: str):
        lang = (lang or "en").split("-")[0].lower()
        if lang not in ("en", "de"):
            lang = "en"
        self._lang = lang
        self._model = self._ensure_loaded(lang)

    @property
    def language(self) -> str:
        return self._lang

    def transcribe_until(self, stop_fn: Callable[[], bool]) -> str:
        if self._model is None:
            raise RuntimeError("Vosk model not loaded")

        q: queue.Queue[bytes] = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        rec = vosk.KaldiRecognizer(self._model, self.samplerate)

        try:
            with sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype="int16",
                channels=1,
                callback=callback,
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
            print("[STT] Audio error:", e, file=sys.stderr)
            return ""
