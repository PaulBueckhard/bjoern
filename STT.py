import sounddevice as sd
import vosk
import queue
import sys
import json
from queue import Empty
from typing import Callable, Optional


class SpeechToText:
    """
    Lightweight wrapper around Vosk + sounddevice for streaming speech-to-text.

    Flow:
      - Load English/German Vosk models lazily.
      - Open a RawInputStream and feed PCM blocks into a KaldiRecognizer.
      - Continue until stop_fn() returns True, then return the final utterance.
    """

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
        # Audio capture configuration
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.debug = debug

        # Map language → Vosk model path
        self._model_paths = {"en": model_path_en, "de": model_path_de}

        # Cache for loaded Vosk.Model instances
        self._models: dict[str, vosk.Model] = {}

        # Load initial language/model
        self._lang = "en"
        self._model: Optional[vosk.Model] = None
        self.set_language(language)

        # Optional diagnostics
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

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _ensure_loaded(self, lang: str) -> vosk.Model:
        """Load the Vosk model for a language if not loaded yet."""
        if lang not in self._models:
            path = self._model_paths.get(lang)
            if not path:
                raise ValueError(f"No Vosk model path configured for '{lang}'")
            self._models[lang] = vosk.Model(path)
        return self._models[lang]

    def set_language(self, lang: str):
        """
        Select a language ('en' or 'de').
        Falls back to English if unsupported.
        Loads the corresponding Vosk model.
        """
        lang = (lang or "en").split("-")[0].lower()
        if lang not in ("en", "de"):
            lang = "en"

        self._lang = lang
        self._model = self._ensure_loaded(lang)

    @property
    def language(self) -> str:
        """Return the currently active STT language."""
        return self._lang

    # ------------------------------------------------------------------
    # Transcription loop
    # ------------------------------------------------------------------

    def transcribe_until(self, stop_fn: Callable[[], bool]) -> str:
        """
        Capture microphone audio until stop_fn() becomes True, then return
        Vosk's final recognized text.

        stop_fn:
            A function checked repeatedly; recording stops once it returns True.
        """
        if self._model is None:
            raise RuntimeError("Vosk model not loaded")

        # Thread-safe queue for PCM blocks from the audio callback
        q: queue.Queue[bytes] = queue.Queue()

        def callback(indata, frames, time_info, status):
            """
            Called by sounddevice whenever a new audio block is available.
            Writes PCM bytes into the queue for the recognizer to consume.
            """
            if status:
                print(status, file=sys.stderr)

            q.put(bytes(indata))

            # Optional input-level meter for debugging
            if self.debug:
                import numpy as np
                # Compute approximate amplitude in dBFS
                level = 20 * np.log10(
                    np.max(np.abs(np.frombuffer(indata, dtype="int16"))) / 32768 + 1e-9
                )
                print(f"[STT] level ~ {level:.1f} dBFS")

        # Kaldi recognizer for streaming audio
        rec = vosk.KaldiRecognizer(self._model, self.samplerate)

        try:
            # Create input stream capturing raw PCM16
            with sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype="int16",
                channels=1,
                device=self.device,
                callback=callback,
            ):
                # Consume audio blocks while waiting for stop_fn()
                while not stop_fn():
                    try:
                        data = q.get(timeout=0.2)
                    except Empty:
                        continue

                    # Feed captured audio directly into Vosk
                    rec.AcceptWaveform(data)

                # Return final recognized text
                final = json.loads(rec.FinalResult()).get("text", "")
                return (final or "").strip()

        except Exception as e:
            print("[STT] Audio error:", e, file=sys.stderr)
            return ""


# -------------------- CLI Test --------------------

if __name__ == "__main__":
    import time
    print("Initializing STT...")
    stt = SpeechToText(debug=True)
    print("Speak for 3 seconds...")
    t0 = time.time()
    text = stt.transcribe_until(lambda: time.time() - t0 > 3.0)
    print("Result:", repr(text))
