# Who-Speak-AI

A privacy-preserving, voice-authenticated virtual assistant. Who-Speak-AI combines fine-tuned speaker verification with homomorphic encryption and real-time conversational voice agents — your biometric data stays private while the assistant stays personal.

---

## What It Does

1. **You register and enroll your voice** — three spoken samples are processed into a 256-dimensional speaker embedding using a fine-tuned RawNet3 model.
2. **Your voiceprint is encrypted** — the embedding is encrypted client-side with CKKS homomorphic encryption (TenSEAL). The server only ever sees ciphertext.
3. **You talk to the assistant** — a real-time voice pipeline connects you through WebRTC (LiveKit), transcribes your speech locally with Whisper, generates replies with GPT-5, and speaks back using ZeroTTS — all in Vietnamese.
4. **Private tools require voice auth** — before the assistant accesses your calendar or personal data, it challenges you to speak and verifies your identity against the encrypted voiceprint. No match, no access.

---

## Architecture Overview

```
Browser (React 19 + LiveKit Client)
    │
    │  WebRTC audio + data channels
    ▼
LiveKit Server (SFU, ws://127.0.0.1:7880)
    │
    ├──► Pipecat Runtime (or LiveKit Agent)
    │       ├── Silero VAD → Smart Turn v3 → Local Whisper STT
    │       ├── OpenAI GPT-5 (streaming LLM)
    │       ├── ZeroTTS / Edge-TTS (streaming TTS)
    │       └── Voice Auth Gate → RawNet3 → HE Matcher
    │
    ├──► Assistant Gateway (:8020, FastAPI)
    │       ├── User accounts & sessions (SQLite)
    │       ├── Voice enrollment
    │       └── LiveKit token issuance
    │
    └──► Matcher API (:8011, FastAPI)
            └── Ciphertext-only speaker matching (SQLite, zero plaintext)
```

---

## Requirements

| Requirement | Version | Why |
|-------------|---------|-----|
| **Python** | 3.12.x (`>=3.12, <3.13`) | Backend services, ML inference, agent workers |
| **Node.js** | 20+ or 22+ LTS (`>=20.19.0 \|\| >=22.12.0`) | Vite dev server and React frontend build |
| **LiveKit Server** | Latest stable binary | WebRTC media routing at `ws://127.0.0.1:7880` |
| **OS Keychain** | macOS Keychain / Linux Secret Service | Stores private CKKS encryption keys securely |

---

## Getting Started

### 1. Clone and set up Python

```bash
git clone <repo-url>
cd Who-Speak-AI

python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU or CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies
pip install -r requirements.txt
```

Or install in editable mode with all extras:

```bash
pip install -e "app/voice_verification[ui,matcher,he,model,agent,gateway,pipecat,providers,test]"
```

### 2. Configure environment

```bash
cp app/voice_verification/.env.example app/voice_verification/.env.local
```

Edit `.env.local` with your secrets:
- `OPENAI_API_KEY` — required for GPT-5 conversation
- `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` — required for WebRTC
- `VOICE_MATCHER_TOKEN` — shared secret for the Matcher API (min 8 characters)
- `PIPECAT_SUPERVISOR_SECRET` — required in Pipecat mode (min 32 characters)

### 3. Start the services

Open five terminals:

```bash
# 1. LiveKit WebRTC Server
app/voice_verification/scripts/run_livekit_server_local.sh

# 2. Matcher API (port 8011)
app/voice_verification/scripts/run_matcher_local.sh

# 3. Assistant Gateway (port 8020)
app/voice_verification/scripts/run_gateway_local.sh

# 4. Pipecat Supervisor (port 8021)
app/voice_verification/scripts/run_pipecat_supervisor_local.sh

# 5. Web frontend (port 5173)
cd app/web && npm install && npm run dev
```

### 4. Use it

Open **http://127.0.0.1:5173**, register an account, enroll your voice (three samples), and start talking to the assistant.

---

## Project Structure

```
Who-Speak-AI/
├── app/
│   ├── web/                        # React 19 + Vite 7 frontend
│   │   ├── src/                    # Components: VoiceStage, ConversationPanel, LiveKitPanel
│   │   ├── package.json            # Frontend dependencies
│   │   └── vite.config.ts          # Dev server config (proxies /api → :8020)
│   │
│   ├── assistant/                  # Real-time voice agent pipeline
│   │   ├── pipecat_runtime/        # Pipecat 1.8.1 pipeline & session supervisor
│   │   ├── providers/              # Whisper, ZeroTTS, Edge-TTS, OpenAI LLM
│   │   ├── tools/                  # PolicyToolExecutor + MockCalendarProvider
│   │   ├── simulations/            # YAML dialogue scenarios for policy tests
│   │   ├── livekit_agent.py        # LiveKit Agents-based runtime (alternate)
│   │   ├── config.py               # All runtime configuration from env vars
│   │   └── streaming.py            # SentenceBuffer for chunked TTS input
│   │
│   ├── assistant_gateway/          # FastAPI gateway (:8020)
│   │   ├── main.py                 # Routes: register, login, enroll, token
│   │   ├── security.py             # scrypt password hashing, session tokens
│   │   └── store.py                # SQLite: users, sessions, voice_profiles
│   │
│   └── voice_verification/         # Speaker verification & encryption
│       ├── apps/matcher_api/       # Matcher service (:8011), ciphertext-only
│       ├── src/voiceauth/          # RawNet3, TenSEAL HE, keychain storage
│       ├── scripts/                # Launch scripts for each service
│       ├── streamlit_app.py        # Standalone testing dashboard
│       └── pyproject.toml          # All Python dependency definitions
│
├── model/                          # RawNet3 training & evaluation pipeline
├── techincal-report/               # Technical report documentation
├── requirements.txt                # Consolidated Python dependencies
├── STACK.md                        # Complete technology stack inventory
└── GUIDE_RAWNET3.md                # RawNet3 model guide
```

---

## Key Technologies

### Voice Pipeline

| Stage | Technology | Details |
|-------|-----------|---------|
| **Voice Activity Detection** | Silero VAD v5 | Neural VAD with 0.7 confidence threshold |
| **Turn Detection** | Smart Turn v3 | Local ONNX model preventing premature cut-offs |
| **Speech-to-Text** | faster-whisper (`base` model) | CTranslate2-accelerated, configured for Vietnamese |
| **Language Model** | OpenAI GPT-5 | Streaming responses via the Responses API |
| **Text-to-Speech** | ZeroTTS (`maichi` voice) | Local 48 kHz neural streaming, Edge-TTS fallback |

### Speaker Verification

| Component | Details |
|-----------|---------|
| **Model** | RawNet3, fine-tuned on ViMD dataset, 256-D embeddings |
| **Encryption** | CKKS homomorphic encryption (TenSEAL), `poly_modulus_degree=8192` |
| **Matching** | Encrypted cosine similarity — server never sees plaintext embeddings |
| **Key Storage** | OS keychain via `keyring` (`who-speak.voice-he` service) |

### Security

| Mechanism | Implementation |
|-----------|---------------|
| **Passwords** | scrypt (N=16384, r=8, p=1) with 16-byte random salt |
| **Sessions** | HttpOnly cookie, 32-byte token, SHA-256 digest stored, 8-hour TTL |
| **Inter-service auth** | Bearer tokens with min 32-character shared secrets |

---

## Testing

```bash
# Backend tests
pytest app/voice_verification/tests/

# Frontend tests
cd app/web && npm test

# Dialogue simulation tests
# Uses YAML scenarios in app/assistant/simulations/
```

---

## Documentation

- [STACK.md](STACK.md) — Full technology stack with exact version constraints
- [GUIDE_RAWNET3.md](GUIDE_RAWNET3.md) — RawNet3 model training and evaluation
- [app/voice_verification/README.md](app/voice_verification/README.md) — Speaker verification, HE details, agent lifecycle
- [app/web/README.md](app/web/README.md) — Web client architecture and setup

---

## Contributors

| Name | Student ID | Email |
|------|-----------|-------|
| Nguyen Manh Cuong | 23127034 | nmcuong23@clc.fitus.edu.vn |
| Nguyen Tran Thien An | 23127315 | nttan23@clc.fitus.edu.vn |
| Nguyen Dong Thanh | 23127538 | ndthanh23@clc.fitus.edu.vn |
