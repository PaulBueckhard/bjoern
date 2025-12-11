# Björn - The AI-Powered Plushie Assistant

Björn is a **voice-controlled plushie assistant** powered by:

- **Offline Speech-to-Text** (Vosk)  
- **Offline Text-to-Speech** (Piper)  
- **Local LLM Responses** (Ollama)  
- **GPIO button interaction** (Raspberry Pi)  
- **Safety-focused persona shaping**  
- **Parent-protected session logging & short-ID sharing**

Designed for children, Björn uses a warm teddy bear personality, speaks English or German, responds safely, and allows parents to review interactions securely.

---

See [björn-web](https://github.com/PaulBueckhard/bjoern-web) for the project's web logging feature.

---

## Features

### Speech-to-Text (STT)
- Offline transcription using Vosk  
- English + German models  
- Streams microphone data until button release  

### Text-to-Speech (TTS)
- Offline, realistic voice using Piper  
- Supports Windows + Linux  
- Optional low-latency streaming daemon  

### Safe Teddy Bear Persona
- Very short, simple, child-appropriate sentences  
- Warm and playful personality  
- Redirects unsafe questions  
- Provides emotional support  
- Never breaks character  

### Safety System
- Scans both user input and AI responses for unsafe keywords  
- Unsafe output is replaced with a soft refusal message  

### Parent Portal
- Voice-guided setup flow creates:
  - Language preference  
  - Child’s name  
  - Parent PIN  
  - Session ID  
  - Short, easy-to-share session code  
- Parents can view chat logs safely

### Evaluation Framework
- Automatic safety test suite  
- Aggregated passing/failing report  
- Logs per test category  

---
## Architecture
```text
+---------------------+              +-------------------------+
|   Child / User      |              |   Parent (Web UI)       |
|---------------------|              |-------------------------|
| - Press button      |              | - Enters short ID + PIN |
| - Talks to Björn    |              | - Views conversation    |
+----------+----------+              +-----------+-------------+
           |                                     |
           v                                     |
   [GPIO Button / ↑]                             |
           |                                     |
           v                                     |
+----------+----------+                          |
|      main.py        |--------------------------+
|---------------------|              reads logs from
| - Waits for button  |              memory/session_*.jsonl
| - Triggers STT      |
| - Sends text to LLM |
| - Triggers TTS      |
+----+-----------+----+
     |           |
     |           v
     |    +------+-----------------+
     |    |  stt.py (SpeechToText) |
     |    |------------------------|
     |    | - Vosk (EN/DE)         |
     |    | - Microphone stream    |
     |    +------------------------+
     |
     v
+----+-----------------------------+
|   server.py (Flask /talk)        |
|----------------------------------|
| - Builds teddy-bear persona      |
| - Safety check (user + reply)    |
| - Calls Ollama                   |
| - Logs turns to memory/          |
+----+-----------------------------+
     |
     v
+----+-----------------------------+
|     Ollama (LLM backend)         |
|----------------------------------|
| - Model: llama3 (configurable)   |
| - Runs locally                   |
+----+-----------------------------+
     |
     v
+----+-----------------------------+
|        TTS.py / daemon.py        |
|----------------------------------|
| - Piper text-to-speech           |
| - Optional streaming daemon      |
| - Outputs audio to speaker       |
+----------------------------------+

```


---
## Hardware Requirements
You need a:
- Raspberry Pi 4 / 5
- USB microphone
- Speaker or 3.5mm audio output
- GPIO push button (default pin 17)
- (Optional) Plushie shell

Optionally you can test Björn on your Windows Computer, assuming you have a microphone and headphones/speakers.

---


## Installation

Björn runs on:

- Raspberry Pi 4
- Linux x86_64
- Windows (development only)

### Clone the repository:

```bash
git clone https://github.com/PaulBueckhard/bjoern
cd bjoern
```

### Install python dependencies:

```bash
pip install -r requirements.txt
```

### Install Ollama (LLM):

Download from https://ollama.ai

Then pull your model (default: llama3):
```bash
ollama pull llama3

```

### Install Piper (TTS):
#### Linux:
```bash
sudo apt install piper

```

#### Windows:

[Download Piper](https://github.com/rhasspy/piper/releases/tag/2023.11.14-2) binaries and place them here:
```bash
C:\piper\piper.exe
C:\piper\espeak-ng-data

```


---

## Running Björn

Start backend server:

```bash
python server.py
```

Run Piper TTS daemon (optional, but recommended):

```bash
python daemon.py
```

Start the assistant:

```bash
python main.py
```

On first boot, Björn will:
1. Ask which language to use
2. Ask your name
3. Ask for a parent PIN
4. Generate a session ID
5. Register a short code

All via voice.

---

## Interaction Model

Press and hold the button (or ↑ in dev mode) to make Björn listen. 

He will:
1. Record speech until button released
2. Transcribe via Vosk
3. Send text to /talk (server)
4. Speak the reply via Piper

---

## Parent Portal

Every conversation is stored under:

```
memory/session_<uuid>.jsonl
```

Parents get:
- A **short ID** (e.g. A7QK2Z)
- A **PIN** (e.g. 1234)
- And access to their child's conversation through the website (visit [björn-web](https://github.com/PaulBueckhard/bjoern-web))

---

## Safety Evaluation Framework

You can run safety evaluation tests via:

```bash
python tests/test_data/run_thesis_report
```

And view the report summary at:

```bash
memory/eval/thesis_test_report.md
```

These will test Björn's persona stability and ability to block harmful content.


---

## Known Limitations
- Safety system is simple (keyword-based).
- Response times increase on limited hardware (Raspberry Pi is slower than a desktop).
- Ollama response quality depends on model used.
- Vosk STT accuracy varies with microphone quality.
