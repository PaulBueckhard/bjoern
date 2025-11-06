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
        device: Optional[int | str] = None,
        debug: bool = False,
    ):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.debug = debug

        self._model_paths = {"en": model_path_en, "de": model_path_de}
        self._models: dict[str, vosk.Model] = {}
        self._lang = "en"
        self._model: Optional[vosk.Model] = None
        self.set_language(language)

        if self.debug:
            print("[STT] sounddevice version:", sd.__version__)
            try:
                default_in = sd.default.device
                print("[STT] Default input device:", default_in)
                if self.device is not None:
                    info = sd.query_devices(self.device)
                    print("[STT] Using input device:", info)
            except Exception as e:
                print("[STT] Could not query device info:", e)

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
            if self.debug:
                import numpy as np
                level = 20 * np.log10(np.max(np.abs(np.frombuffer(indata, dtype="int16"))) / 32768 + 1e-9)
                print(f"[STT] level ~ {level:.1f} dBFS")

        rec = vosk.KaldiRecognizer(self._model, self.samplerate)

        try:
            with sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype="int16",
                channels=1,
                device=self.device,
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
