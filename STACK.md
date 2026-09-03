# Technology Stack of Who-Speak-AI

## 1. System & Language Runtimes

| Technology | Specified Constraint | Resolved / Active Version | Specification Source | Notes |
|------------|----------------------|---------------------------|----------------------|-------|
| **Python** | `>=3.12, <3.13` | `3.12.x` | `app/voice_verification/pyproject.toml` | Target Python runtime required across all backend services, workers, and verification pipelines |
| **Node.js** | `>=20.19.0 \|\| >=22.12.0` | Node.js 20+ / 22+ LTS | `app/web/package-lock.json` (engine constraint) | JavaScript runtime required for Vite dev server and building the React web app |

---

## 2. Voice Agent Runtimes & WebRTC Transport

| Technology / Library | Specified Constraint | Resolved / Pinned Version | Specification Source | Notes |
|----------------------|----------------------|---------------------------|----------------------|-------|
| **Pipecat** (`pipecat-ai[livekit]`) | `==1.8.1` | `1.8.1` | `pyproject.toml [project.optional-dependencies.pipecat]` | Pinned primary conversational agent runtime; manages the full audio pipeline (VAD → Smart Turn → STT → LLM → TTS) |
| **LiveKit Agents** (`livekit-agents[mcp]`) | `>=1.6, <2` | `1.6.x` | `pyproject.toml [project.optional-dependencies.agent]` | Alternate voice agent runtime in `app/assistant/livekit_agent.py` |
| **LiveKit Server** | Latest stable binary | Local daemon binary | `scripts/run_livekit_server_local.sh` | WebRTC SFU server running at `ws://127.0.0.1:7880` for room audio routing |
| **LiveKit Python SDK** (`livekit-api`) | `>=1.0, <2` | `1.0.x` | `pyproject.toml [project.optional-dependencies.gateway]` | Python server SDK for creating room dispatch tokens and signing participant JWTs |
| **LiveKit Transport** | In: 16 kHz, Out: 48 kHz | 20 ms audio chunks | `app/assistant/pipecat_runtime/runtime.py` | Full-duplex WebRTC audio streaming protocol |

---

## 3. Frontend Web Application (`app/web`)

| Technology / Library | Specified Constraint | Resolved / Lockfile Version | Specification Source | Notes |
|----------------------|----------------------|-----------------------------|----------------------|-------|
| **React** (`react`) | `^19.0.0` | `19.2.8` | `app/web/package.json` | Modern React with functional components and hooks (`useLiveKitVoiceSession`, `useAudioLevels`) |
| **React DOM** (`react-dom`) | `^19.0.0` | `19.2.8` | `app/web/package.json` | React virtual DOM renderer for browser interface |
| **Vite** (`vite`) | `^7.0.0` | `7.3.6` | `app/web/package.json` | Next-generation dev server (:5173), HMR, and build bundler; proxies `/api` to `:8020` |
| **Vite React Plugin** (`@vitejs/plugin-react`) | `^5.0.0` | `5.2.0` | `app/web/package.json` | Fast refresh and JSX transformation plugin |
| **TypeScript** (`typescript`) | `^5.7.0` | `5.9.3` | `app/web/package.json` | Statically typed JavaScript with strict typing configuration (`tsconfig.json`) |
| **LiveKit Client SDK** (`livekit-client`) | `^2.22.0` | `2.22.0` | `app/web/package.json` | WebRTC audio publication, subscription, and custom data-channel messaging |
| **Vanilla CSS** | W3C Standard CSS | CSS Custom Properties | `app/web/src/styles.css` | Bespoke styling system: dark mode palette, glassmorphism, responsive grid, and CSS animations (no Tailwind) |
| **Web Audio API** | W3C Browser API | Native browser implementation | `app/web/src/wav-recorder.ts` | Custom mono Float32-to-Int16 PCM WAV encoder for voice enrollment and challenge audio (server canonicalizes to 16 kHz) |
| **Vitest** (`vitest`) | `^3.0.0` | `3.2.7` | `app/web/package.json` | Vite-native unit and component test runner |

---

## 4. Standalone Experiment / Admin UI

| Technology / Library | Specified Constraint | Resolved / Pinned Version | Specification Source | Notes |
|----------------------|----------------------|---------------------------|----------------------|-------|
| **Streamlit** (`streamlit`) | `>=1.52, <2` | `1.52.x` | `pyproject.toml [project.optional-dependencies.ui]` | Interactive web dashboard in `app/voice_verification/streamlit_app.py` for 1:1 and 1:N speaker enrollment/identification testing |

---

## 5. Speech-to-Text (STT / ASR)

| Technology / Library | Specified Constraint | Model / Resolved Version | Specification Source | Notes |
|----------------------|----------------------|--------------------------|----------------------|-------|
| **faster-whisper** | `>=1.1, <2` | `1.1.x` | `pyproject.toml [project.optional-dependencies.providers]` | CTranslate2-accelerated local Whisper inference on CPU or CUDA |
| **Whisper Model** | Configurable (`VOICE_WHISPER_MODEL`) | `base` (CTranslate2) | `app/assistant/config.py` | Local ASR model pre-configured for Vietnamese (`vi`) transcription at 16 kHz mono |

---

## 6. Voice Activity Detection (VAD) & Turn-Taking

| Technology / Library | Specified Constraint | Resolved / Active Version | Specification Source | Notes |
|----------------------|----------------------|---------------------------|----------------------|-------|
| **Silero VAD** | Bundled with Pipecat `1.8.1` | Silero VAD v5 | `app/assistant/pipecat_runtime/runtime.py` | Neural voice activity detector (`SileroVADAnalyzer`) with 0.7 confidence threshold and configurable start/stop intervals |
| **Smart Turn v3** | Local ONNX model | `smart_turn_v3.onnx` | `app/assistant/pipecat_runtime/smart_turn.py` | Local end-of-turn classification model preventing premature assistant cut-offs |
| **ONNX Runtime** (`onnxruntime`) | `>=1.17, <2` | `1.17.x` | `pyproject.toml [project.optional-dependencies.providers]` | High-performance inference engine running Smart Turn and ZeroTTS models |
| **RMS Energy VAD** | Native Python / NumPy | Amplitude threshold (`0.012`) | `app/assistant/audio_turn.py` | In-memory fallback VAD (`LocalSpeechTurnBuffer`) for boundary slicing |

---

## 7. Large Language Model (LLM)

| Technology / Library | Specified Constraint | Active Model / Version | Specification Source | Notes |
|----------------------|----------------------|------------------------|----------------------|-------|
| **OpenAI Python SDK** (`openai`) | `>=1.0, <3` | `1.x` | `pyproject.toml [project.optional-dependencies.providers]` | Async OpenAI API client (`AsyncOpenAI`) using the Responses API |
| **OpenAI Model** | Configurable (`OPENAI_MODEL`) | `gpt-5` (default) | `app/assistant/config.py` | Streams conversational replies from local ASR transcripts; never sees raw audio, biometrics, or encryption keys |

---

## 8. Text-to-Speech (TTS)

| Technology / Library | Specified Constraint | Model / Voice Pinned | Specification Source | Notes |
|----------------------|----------------------|----------------------|----------------------|-------|
| **ZeroTTS** (`zerotts`) | `==0.1.1` | `0.1.1` | `pyproject.toml [project.optional-dependencies.providers]` | Primary local neural streaming TTS (`zeroweight-ai/ZeroTTS`) at 48 kHz mono |
| **ZeroTTS Voice** | `VOICE_ZEROTTS_VOICE` | `maichi` | `app/assistant/config.py` | Preset Vietnamese voice weights (never synthesizes from enrollment audio) |
| **Microsoft Edge-TTS** (`edge-tts`) | `>=7, <8` | `7.x` | `pyproject.toml [project.optional-dependencies.providers]` | Streaming fallback TTS using Azure neural voice `vi-VN-HoaiMyNeural` |
| **SentenceBuffer** | Configurable (default 120 via config, 180 class default) | Custom regex chunking | `app/assistant/streaming.py` | Splits streaming LLM output at sentence boundaries to feed continuous TTS generation |

---

## 9. Speaker Verification & Biometric Embeddings

| Technology / Library | Specified Constraint | Resolved / Pinned Version | Specification Source | Notes |
|----------------------|----------------------|---------------------------|----------------------|-------|
| **RawNet3 Model** | ViMD Fine-Tuned (`best.pt`) | SHA-256 verified checkpoint | `app/voice_verification/src/voiceauth/config.py` | Deep neural network generating 256-D speaker embeddings from raw 16 kHz audio waveforms (`0b06fd3c...`) |
| **PyTorch** (`torch`) | `>=2.0` | Target platform CPU/CUDA | `app/voice_verification/README.md` | Core tensor and deep learning inference framework |
| **asteroid-filterbanks** | `==0.4.0` | `0.4.0` | `pyproject.toml [project.optional-dependencies.model]` | Sinc-convolution filterbanks for time-domain acoustic feature extraction |
| **Hugging Face Hub** (`huggingface-hub`) | `>=1.11, <1.12` | `1.11.x` | `pyproject.toml [project.optional-dependencies.model]` | Downloads and caches model configuration files |
| **NumPy** (`numpy`) | `>=2.0, <2.1` | `2.0.x` | `pyproject.toml [project.dependencies]` | Fundamental numerical arrays, L2-normalization, and tensor manipulation |
| **SciPy** (`scipy`) | `>=1.16, <1.17` | `1.16.x` | `pyproject.toml [project.dependencies]` | Scientific audio signal processing, windowing, and filtering |
| **SoundFile** (`soundfile`) | `>=0.13, <0.14` | `0.13.x` | `pyproject.toml [project.dependencies]` | Reading and writing 16-bit PCM WAV audio files |

---

## 10. Homomorphic Encryption (Privacy-Preserving Biometrics)

| Technology / Library | Specified Constraint | Resolved / Pinned Version | Specification Source | Notes |
|----------------------|----------------------|---------------------------|----------------------|-------|
| **TenSEAL** (`tenseal`) | `==0.3.16` | `0.3.16` | `pyproject.toml [project.optional-dependencies.he]` | Microsoft SEAL wrapper for privacy-preserving computation in Python |
| **CKKS Cryptosystem** | `poly_modulus_degree=8192` | `coeff_mod=[60,40,40,60]` | `app/voice_verification/src/voiceauth/he.py` | Homomorphic scheme enabling encrypted cosine similarity evaluation in ciphertext space |

---

## 11. Secrets & Key Storage

| Technology / Library | Specified Constraint | Resolved / Pinned Version | Specification Source | Notes |
|----------------------|----------------------|---------------------------|----------------------|-------|
| **Python keyring** (`keyring`) | `>=25, <27` | `25.x` | `pyproject.toml [project.optional-dependencies.agent]` | Hardware-backed / OS-level keychain storage (`who-speak.voice-he` service) for CKKS private keys; fails closed if unavailable |

---

## 12. Backend Microservices & Architecture

| Microservice | Port | Framework & Version | ASGI Server | Notes |
|--------------|------|---------------------|-------------|-------|
| **Assistant Gateway** | `:8020` | **FastAPI** `>=0.116, <1` | **Uvicorn** `>=0.35, <1` | User registration, login sessions, voice enrollment, and signed LiveKit tokens |
| **Matcher API** | `:8011` | **FastAPI** `>=0.116, <1` | **Uvicorn** `>=0.35, <1` | Privacy-preserving ciphertext-only speaker matching over encrypted vectors |
| **Pipecat Supervisor** | `:8021` | **FastAPI** `>=0.116, <1` | **Uvicorn** `>=0.35, <1` | Spawns and supervises isolated per-room Pipecat worker processes via signed descriptors |

### Microservice Dependencies
- **Pydantic** (`pydantic`): `>=2.10, <3` for strongly typed data models and settings validation
- **HTTPX** (`httpx`): `>=0.28, <1` for async inter-service communication (`[http2]` extra included only in the `matcher` dependency group)

---

## 13. Database & Persistence

| Component | Technology | Version | Schema / Tables | Notes |
|-----------|------------|---------|-----------------|-------|
| **Gateway Storage** | **SQLite 3** (`sqlite3`) | Built-in Python 3.12 standard library | `users`, `sessions`, `voice_profiles`, `mock_calendar_events` | Local relational database (`who_speak_gateway.db`) |
| **Matcher Storage** | **SQLite 3** (`sqlite3`) | Built-in Python 3.12 standard library | `voice_contexts`, `voice_identities` | Zero-plaintext database storing only base64 ciphertext and public HE contexts (`voice_matcher.db`) |

---

## 14. Security & Cryptography

| Mechanism | Algorithm / Standard | Parameters / Key Size | Implementation Source |
|-----------|----------------------|-----------------------|-----------------------|
| **Password Hashing** | `scrypt` (`hashlib.scrypt`) | $N=16384, r=8, p=1$, 16-byte random salt | `app/assistant_gateway/security.py` |
| **Session Authentication** | Secure HttpOnly Cookie | 32-byte cryptographically secure token, SHA-256 digest in SQLite, 8-hour TTL | `app/assistant_gateway/main.py` |
| **Inter-Service Authorization** | Bearer Tokens & HMAC Shared Secrets | Min 32 characters (`VOICE_MATCHER_TOKEN`, `PIPECAT_SUPERVISOR_SECRET`) | Gateway & Supervisor contracts |

---

## 15. Agent Tools & Protocols

| Technology | Protocol / Provider | Implementation Details | Notes |
|------------|---------------------|------------------------|-------|
| **Model Context Protocol** | MCP (`livekit-agents[mcp]`) | Supported in LiveKit Agent architecture | Tool protocol framework for extensible voice assistant actions |
| **Calendar Tool** | `MockCalendarProvider` | SQLite-backed demo calendar (`app/assistant/tools/calendar.py`) | Deterministic mock tool executed strictly via `PolicyToolExecutor` under authenticated state |

---

## 16. Testing & Quality Assurance

| Tool | Specified Constraint | Resolved / Active Version | Specification Source | Scope |
|------|----------------------|---------------------------|----------------------|-------|
| **pytest** (`pytest`) | `>=8.3, <10` | `8.3.x` | `pyproject.toml [project.optional-dependencies.test]` | Backend unit, integration, and security assertion suites |
| **Vitest** (`vitest`) | `^3.0.0` | `3.2.7` | `app/web/package.json` | Frontend React component and hook test runner |
| **Agent Simulation Suite** | Custom Python Builder | `build_scenarios.py` | `app/assistant/simulations/` | YAML scenario files (`authored.yaml`, `risks.yaml`, `scenarios.yaml`) for deterministic dialogue policy tests |
