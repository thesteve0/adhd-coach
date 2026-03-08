# Reachy Mini + Framework Desktop: Architecture Guide

This document covers how to build a locally-running ADHD coach on a Reachy Mini Lite
(USB-connected) using a Framework Desktop with Strix Halo / Ryzen AI 395+.

---

## Hardware Reality Check

### Reachy Mini Lite (USB version — your model)

- **No onboard computer** — the USB connection carries motor/sensor control signals
- **The daemon runs on YOUR desktop**, not on the robot
- Motor control, camera, microphone, and speaker all route through the daemon process
- No IMU (that is wireless-only)

### Framework Desktop (Strix Halo)

- Ryzen AI 395+ with Radeon 890M GPU
- 96 GB unified memory — plenty for 7B–34B parameter models
- ROCm support — can run local LLMs, Whisper, and TTS

**You are correct: there is no meaningful AI compute on the Lite robot itself. All AI runs on your desktop.**

---

## System Architecture

```
+------------------------------------------------------------------+
|  Framework Desktop (Ryzen AI 395+ / ROCm)                        |
|                                                                   |
|  +--------------------+    +----------------------------------+   |
|  | reachy-mini-daemon |    | ADHD Coach Orchestrator (Python) |   |
|  | localhost:8000     |    |                                  |   |
|  |                    |    |  1. Mic audio (from robot)       |   |
|  | REST API:          |◄───|  2. VAD (silero or webrtcvad)    |   |
|  |  /api/state        |    |  3. Whisper STT (ROCm)           |   |
|  |  /api/move         |    |  4. ollama LLM + tool calls      |   |
|  |  /api/volume       |    |  5. Piper/Kokoro TTS             |   |
|  |  /api/motors       |    |  6. Audio → robot speaker        |   |
|  |                    |    |  7. Robot movement (expressions) |   |
|  | WebSocket:         |    |                                  |   |
|  |  /api/state/ws     |    +----------------------------------+   |
|  +--------+-----------+                                          |
|           | USB                                                   |
+-----------+------------------------------------------------------ +
            |
            v
+------------------------------+
|  Reachy Mini Lite            |
|  - Servo motors (head, body, |
|    antennas)                 |
|  - Camera (OpenCV via SDK)   |
|  - Microphone (16kHz, 2ch)   |
|  - Speaker (16kHz)           |
|  - Direction of Arrival      |
+------------------------------+
```

---

## The Daemon: Key Concept

For the **Lite/USB version**, run this once before your app starts:

```bash
reachy-mini-daemon
```

This process:
- Listens on `localhost:8000`
- Exposes a REST + WebSocket API
- Handles all low-level USB motor communication
- Locks the audio card while active (sounddevice backend)

Verify it is running by opening `http://localhost:8000` in a browser — you should see the Reachy Dashboard.

The Python SDK then connects to this daemon transparently:

```python
from reachy_mini import ReachyMini

# Auto-detects Lite mode -> connects to localhost:8000
with ReachyMini() as mini:
    print("Connected!")
```

---

## SDK Capabilities (What You Can Actually Do)

### Movement

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

with ReachyMini() as mini:
    # Head: pitch/roll/yaw + vertical translation
    mini.goto_target(
        head=create_head_pose(z=10, mm=True),   # nod up 10mm
        duration=0.5,
        method="minjerk"    # smooth: linear | minjerk | ease_in_out | cartoon
    )

    # Antennas: radians, range roughly [-1, 1]
    mini.goto_target(antennas=np.deg2rad([45, -45]), duration=0.3)

    # Body rotation
    mini.goto_target(body_yaw=np.deg2rad(30), duration=1.0)

    # All at once
    mini.goto_target(
        head=create_head_pose(z=-5, mm=True),
        antennas=[0.3, 0.3],
        body_yaw=np.deg2rad(-15),
        duration=0.8
    )

    # High-frequency control (bypasses interpolation — for live tracking)
    mini.set_target(head=pose, antennas=[0, 0])
```

### Camera

```python
# Returns numpy array: (height, width, 3) uint8, BGR
frame = mini.media.get_frame()
```

- Uses OpenCV backend for Lite
- ~25 FPS is practical (camera worker polls at 0.04s sleep)
- Use this for posture detection, presence detection

### Audio: Microphone

```python
mini.media.start_recording()

# Returns numpy array: (N_samples, 2) float32, 16kHz stereo
samples = mini.media.get_audio_sample()

# Direction of Arrival detection (built-in!)
# doa: 0=left, pi/2=front/back, pi=right (radians)
# is_speech_detected: bool
doa, is_speech_detected = mini.media.get_DoA()

mini.media.stop_recording()
```

### Audio: Speaker

```python
mini.media.start_playing()

# Push numpy array: (N_samples, 1 or 2) float32, 16kHz
# NON-BLOCKING — returns immediately while audio plays in background
mini.media.push_audio_sample(tts_audio_array)

# To wait for playback to finish:
import time
time.sleep(len(tts_audio_array) / mini.media.get_output_audio_samplerate())

mini.media.stop_playing()
```

### Media Backend Note

For Lite/USB, always use `media_backend="default"` (OpenCV + sounddevice):

```python
with ReachyMini(media_backend="default") as mini:
    ...
```

The daemon locks the audio card while running. If you get audio device errors,
check that nothing else is using the soundcard simultaneously.

### REST API (Direct HTTP, No SDK Needed)

The daemon also exposes a full REST API at `localhost:8000`:

| Endpoint | Method | Description |
|---|---|---|
| `/api/state/full` | GET | Full robot state (head pose, antennas, body yaw) |
| `/api/state/ws/full` | WebSocket | Continuous state stream |
| `/api/move/goto` | POST | Move to target pose |
| `/api/move/set_target` | POST | Instant pose (no interpolation) |
| `/api/volume/speaker` | POST | Set speaker volume |
| `/api/volume/microphone` | POST | Set mic volume |
| `/api/motors` | GET | Motor status |

Browse the full interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## Reference App: What the Official Conversation App Does

The [reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app)
is the canonical reference. It uses **OpenAI Realtime API + fastrtc** for cloud-based
audio streaming. Understanding its architecture helps you replace the cloud parts with
local equivalents.

### Component Map

| Official App Component | Purpose | Your Local Replacement |
|---|---|---|
| `openai_realtime.py` | WebSocket to OpenAI, audio I/O, tool dispatch | `orchestrator.py` with ollama |
| `moves.py` | 100 Hz control loop, pose composition | Keep as-is or adapt |
| `camera_worker.py` | 25 FPS capture, face tracking in thread | Keep as-is |
| `main.py` | Wires all managers together | Your `main.py` |
| OpenAI Realtime API | STT + LLM + TTS in one stream | Whisper + ollama + Piper separately |
| `fastrtc` | Low-latency audio transport | Direct sounddevice via SDK |

### Movement Architecture (from `moves.py`)

The official app uses a sophisticated **100 Hz control loop** in a dedicated thread:

- **Primary moves**: Sequential, mutually exclusive (emotions, dances, breathing, goto poses)
  managed via a queue
- **Secondary moves**: Additive offsets for speech sway and face tracking, blended on top
- **Single write point**: `set_target()` called once per loop with the fused pose
- **Idle behavior**: Breathing animation starts automatically after inactivity threshold
- **Antenna freeze**: During listening, antennas freeze with smooth blend on release

For an ADHD coach you can simplify this considerably — a few named poses
(listening, thinking, affirming, idle) with `goto_target()` is sufficient to start.

### Camera Architecture (from `camera_worker.py`)

A daemon thread captures frames at ~25 FPS using `mini.media.get_frame()`:

```
CameraWorker thread
  loop at ~25 FPS:
    frame = mini.media.get_frame()           # BGR numpy array
    store frame in thread-safe self.latest_frame (Lock)
    if head_tracker enabled:
      eye_center = head_tracker.get_head_position(frame)
      mini.look_at_image(eye_center)         # robot looks at user's face
      update face_tracking_offsets (Lock)

Main thread:
    frame = camera_worker.get_latest_frame()   # safe copy
```

For posture detection you can pass `get_latest_frame()` into a vision model
(SmolVLM2, MediaPipe Pose, or a fine-tuned classifier).

---

## Local AI Stack for ADHD Coach

### STT: faster-whisper (ROCm)

```python
from faster_whisper import WhisperModel

# "base" is fast and accurate enough for short utterances
# device="cuda" works with ROCm via HIP
model = WhisperModel("base", device="cuda", compute_type="float16")

def transcribe(audio_np: np.ndarray) -> str:
    # audio_np: float32, 16kHz, mono
    segments, _ = model.transcribe(audio_np, language="en", vad_filter=True)
    return " ".join(s.text for s in segments).strip()
```

### VAD: Silero VAD

Do not send audio to Whisper on every chunk — use VAD to detect speech boundaries:

```python
import torch

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, read_audio, *_) = utils

def has_speech(audio_chunk: np.ndarray, sample_rate=16000) -> bool:
    tensor = torch.from_numpy(audio_chunk[:, 0])  # mono from stereo
    timestamps = get_speech_timestamps(tensor, vad_model, sampling_rate=sample_rate)
    return len(timestamps) > 0
```

Alternatively, `mini.media.get_DoA()` returns `is_speech_detected` which uses the
robot's built-in beamforming — useful as a first-pass filter before running Silero.

### LLM: ollama (ROCm)

```bash
# Install ollama with ROCm support
# Then pull your model
ollama pull llama3.2
# Or your fine-tuned ADHD coach model:
# ollama create adhd-coach -f Modelfile
```

```python
import ollama

def get_coaching_response(user_text: str, state: dict) -> ollama.ChatResponse:
    return ollama.chat(
        model="adhd-coach",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an ADHD executive function coach on a small desktop robot. "
                    f"Be brief and directive. Never ask open-ended questions. "
                    f"Current user state: {state}"
                )
            },
            {"role": "user", "content": user_text}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "adhd_coach_tool",
                    "description": "Deliver an ADHD task initiation coaching response",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "micro_step": {"type": "string", "description": "The single tiny action to take right now"},
                            "timer_seconds": {"type": "integer", "description": "Optional: seconds for a focus timer"}
                        },
                        "required": ["micro_step"]
                    }
                }
            }
        ]
    )
```

### TTS: Piper (fast, CPU, local)

Piper is the best balance of quality and speed for this use case:

```bash
pip install piper-tts
# Download a voice model:
# https://huggingface.co/rhasspy/piper-voices
```

```python
import subprocess
import numpy as np
import io
import wave

def synthesize_piper(text: str, voice: str = "en_US-lessac-medium") -> np.ndarray:
    """Returns float32 numpy array at 22050 Hz — resample to 16kHz for robot."""
    result = subprocess.run(
        ["piper", "--model", voice, "--output-raw"],
        input=text.encode(),
        capture_output=True
    )
    # result.stdout is raw 16-bit PCM at 22050 Hz
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio
```

For lower latency, use the Python API directly instead of subprocess:

```python
from piper.voice import PiperVoice

voice = PiperVoice.load("en_US-lessac-medium.onnx")

def synthesize(text: str) -> np.ndarray:
    audio_chunks = []
    for audio_bytes in voice.synthesize_stream_raw(text):
        chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_chunks.append(chunk)
    return np.concatenate(audio_chunks)
```

---

## Complete Orchestration Loop

```python
import asyncio
import numpy as np
import time
from scipy.signal import resample
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from faster_whisper import WhisperModel
from piper.voice import PiperVoice
import ollama

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

whisper = WhisperModel("base", device="cuda", compute_type="float16")
piper_voice = PiperVoice.load("en_US-lessac-medium.onnx")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.5                      # how often to read from mic
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
SPEECH_SILENCE_THRESHOLD = 1.5          # seconds of silence to end utterance

# ---------------------------------------------------------------------------
# Helper: resample audio for robot speaker (expects 16kHz)
# ---------------------------------------------------------------------------

def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    n_samples = int(len(audio) * dst_rate / src_rate)
    return resample(audio, n_samples).astype(np.float32)

# ---------------------------------------------------------------------------
# Helper: TTS -> robot speaker
# ---------------------------------------------------------------------------

def speak(mini: ReachyMini, text: str):
    chunks = []
    for audio_bytes in piper_voice.synthesize_stream_raw(text):
        chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        chunks.append(chunk)
    audio = np.concatenate(chunks)

    # Piper outputs 22050 Hz mono; robot speaker expects 16kHz
    audio_16k = resample_audio(audio, 22050, SAMPLE_RATE)
    audio_stereo = np.stack([audio_16k, audio_16k], axis=1)  # (N, 2)

    mini.media.push_audio_sample(audio_stereo)
    time.sleep(len(audio_16k) / SAMPLE_RATE)  # wait for playback

# ---------------------------------------------------------------------------
# Helper: Robot expressions
# ---------------------------------------------------------------------------

def expression_listening(mini: ReachyMini):
    """Antennas up, slight head tilt — attentive."""
    mini.goto_target(antennas=[0.4, 0.4], head=create_head_pose(z=5, mm=True), duration=0.4)

def expression_thinking(mini: ReachyMini):
    """Antennas droop slightly, head tilts."""
    mini.goto_target(antennas=[0.1, -0.1], head=create_head_pose(z=0, mm=True), duration=0.5)

def expression_affirm(mini: ReachyMini):
    """Quick nod + antenna wiggle."""
    mini.goto_target(head=create_head_pose(z=8, mm=True), duration=0.2)
    mini.goto_target(head=create_head_pose(z=0, mm=True), duration=0.2)
    mini.goto_target(antennas=[0.5, 0.5], duration=0.2)
    mini.goto_target(antennas=[0, 0], duration=0.3)

def expression_idle(mini: ReachyMini):
    """Neutral pose."""
    mini.goto_target(antennas=[0, 0], head=create_head_pose(z=0, mm=True), duration=0.8)

# ---------------------------------------------------------------------------
# Helper: ADHD state (plug in your real state tracker here)
# ---------------------------------------------------------------------------

def get_adhd_state() -> dict:
    return {
        "sitting_time_minutes": 45,       # replace with real sensor/timer
        "time_of_day": "afternoon",
        "posture": "slouched",            # replace with camera-based detection
        "exercise_minutes_today": 0,
    }

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_adhd_coach():
    with ReachyMini(media_backend="default") as mini:
        mini.media.start_recording()
        mini.media.start_playing()
        print("ADHD Coach ready. Listening...")

        expression_idle(mini)

        audio_buffer = []
        last_speech_time = time.time()
        in_utterance = False

        try:
            while True:
                # --- Capture audio chunk ---
                chunk = mini.media.get_audio_sample()  # (N, 2) float32 16kHz

                # --- Built-in speech detection (fast, no model needed) ---
                _, is_speech = mini.media.get_DoA()

                if is_speech:
                    if not in_utterance:
                        print("Speech detected, listening...")
                        expression_listening(mini)
                        in_utterance = True
                    audio_buffer.append(chunk)
                    last_speech_time = time.time()

                elif in_utterance:
                    audio_buffer.append(chunk)  # include trailing silence
                    silence_duration = time.time() - last_speech_time

                    if silence_duration >= SPEECH_SILENCE_THRESHOLD:
                        # --- End of utterance: transcribe ---
                        expression_thinking(mini)

                        full_audio = np.concatenate(audio_buffer, axis=0)
                        mono = full_audio[:, 0]  # take left channel

                        segments, _ = whisper.transcribe(
                            mono,
                            language="en",
                            vad_filter=True
                        )
                        user_text = " ".join(s.text for s in segments).strip()
                        audio_buffer = []
                        in_utterance = False

                        if not user_text:
                            expression_idle(mini)
                            continue

                        print(f"User: {user_text}")

                        # --- LLM call ---
                        state = get_adhd_state()
                        response = ollama.chat(
                            model="adhd-coach",  # or "llama3.2" while testing
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an ADHD executive function coach "
                                        "running on a small desktop robot. Be brief "
                                        "and directive. Give one concrete micro-step. "
                                        "Never ask open-ended questions. "
                                        f"User state: {state}"
                                    )
                                },
                                {"role": "user", "content": user_text}
                            ]
                        )

                        reply = response.message.content
                        print(f"Coach: {reply}")

                        # --- Express + speak ---
                        expression_affirm(mini)
                        speak(mini, reply)
                        expression_idle(mini)

                time.sleep(CHUNK_SECONDS)

        finally:
            mini.media.stop_recording()
            mini.media.stop_playing()

if __name__ == "__main__":
    run_adhd_coach()
```

---

## Integration with Your ADHD Env

The orchestration loop above calls `get_adhd_state()` — this is where your existing
OpenEnv environment feeds in. Two options:

### Option A: Direct import (simplest)

```python
# Your ADHD env state tracker runs in-process
from adhd_env.state import StateTracker

tracker = StateTracker()

def get_adhd_state() -> dict:
    return tracker.current_state.dict()
```

### Option B: HTTP call to your deployed env

```python
import requests

def get_adhd_state() -> dict:
    resp = requests.get("http://localhost:8000/state")  # your adhd env endpoint
    return resp.json()
```

Option B keeps the env as a service (matches your existing architecture) and
lets you score robot responses the same way you score text responses in the env.

---

## Adding Camera-Based Posture Detection

The robot's camera gives you a numpy array you can pass directly into a model:

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=0)

def detect_posture(frame_bgr: np.ndarray) -> str:
    import cv2
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(frame_rgb)

    if not results.pose_landmarks:
        return "unknown"

    # Compare shoulder-ear distance to classify slouch
    # (simplified — tune thresholds to your setup)
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
    vertical_diff = abs(left_shoulder.y - left_ear.y)

    return "upright" if vertical_diff > 0.15 else "slouched"

# In your camera thread:
frame = mini.media.get_frame()
posture = detect_posture(frame)
```

---

## Dependency Summary

```toml
# For your Framework Desktop (add to your venv)
[dependencies]
reachy-mini = ">=1.5.0"          # robot SDK
faster-whisper = "*"             # local STT (ROCm via HIP)
piper-tts = "*"                  # local TTS (CPU, fast)
scipy = "*"                      # audio resampling
ollama = "*"                     # local LLM client
mediapipe = "*"                  # optional: posture detection
silero-vad = "*"                 # optional: better VAD than DoA alone
```

Install ollama separately: https://ollama.com/download/linux

---

## Startup Sequence

```bash
# Terminal 1: Start the robot daemon (once, keep running)
reachy-mini-daemon

# Verify: open http://localhost:8000 in browser

# Terminal 2: Start your ADHD coach app
cd /workspaces/adhd-coach
python scripts/reachy_coach.py
```

---

## Known Limitations and Gotchas

### Audio card locking
The daemon plays a sound on startup/exit via sounddevice, which **locks the audio card**.
If your SDK code also opens sounddevice at the same time you'll get a device-busy error.
Start your media session after the daemon has finished its startup sound (~2 seconds).

### `push_audio_sample()` is non-blocking
The robot speaker plays in background. If you `push` a new sample before the previous
one finishes, they overlap. Either wait with `time.sleep(duration)` or track playback state.

### Microphone sample accumulation
`get_audio_sample()` returns whatever has accumulated since the last call. The buffer
size depends on your call frequency. Call it on a tight loop (every 0.1–0.5s) and
concatenate chunks until you detect end of utterance.

### Piper output rate
Piper outputs 22050 Hz; the robot speaker expects 16000 Hz. Always resample with scipy
before pushing. Wrong sample rate = chipmunk voice or slowed-down audio.

### ROCm + faster-whisper
Use `compute_type="float16"` not `"int8"` — ROCm handles fp16 well on Strix Halo
but int8 quantization support varies. If you hit errors, fall back to `device="cpu"`.

### No network needed
With the Lite/USB daemon running on your desktop, everything is localhost.
No WiFi, no network config, no mDNS. This is simpler and more reliable than the
wireless version.

---

## References

### Primary sources (provided)

- **https://huggingface.co/spaces/pollen-robotics/Reachy_Mini**
  Official HuggingFace Space for Reachy Mini — entry point for apps, demos, and the desktop control UI.

- **https://github.com/pollen-robotics/reachy_mini**
  Main SDK repository — source code, examples, and the AGENTS.md guide for AI coding assistants.

- **https://www.jeffgeerling.com/blog/2026/testing-reachy-mini-hugging-face-robot/**
  Independent hands-on review confirming hardware specs, real-world limitations, and that the conversation app offloads AI to the cloud.

- **https://huggingface.co/docs/reachy_mini/SDK/quickstart**
  Official quickstart — covers daemon startup, auto-detection of Lite vs wireless, and the first Python script.

- **https://huggingface.co/docs/reachy_mini/SDK/python-sdk**
  Full Python SDK reference — movement methods, camera/audio APIs, media backend options, and sample formats.

- **https://huggingface.co/docs/reachy_mini/API/rest-api**
  REST + WebSocket API reference for the daemon at `localhost:8000` — all endpoint categories and the Swagger UI location.

- **https://github.com/pollen-robotics/reachy_mini_conversation_app**
  Reference conversation application — production-grade architecture using OpenAI Realtime API, fastrtc, layered motion system, and camera worker; the closest thing to a "how to build a full app" blueprint.

### Additional sources (discovered during research)

- **https://huggingface.co/docs/reachy_mini/SDK/integration**
  AI integration guide — covers LLM tool use patterns, the JavaScript SDK for browser apps, and links to the conversation demo.

- **https://huggingface.co/docs/reachy_mini/SDK/media-architecture**
  Explains how audio/video differs between Lite (OpenCV + sounddevice on the desktop) and wireless (GStreamer + WebRTC on the Pi); explains the audio card locking behavior.

- **https://github.com/pollen-robotics/reachy_mini_conversation_demo**
  Simpler companion demo (vs the full app) — VAD + LLM + TTS loop; referenced in official docs as the canonical "how to wire audio and LLM together" example.
