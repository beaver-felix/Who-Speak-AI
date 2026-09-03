# Who Speak AI — Voice Verification MVP

This package implements the speaker-verification portion of Who Speak AI. It
uses the selected RawNet3/ViMD checkpoint to create a local 256-D embedding,
encrypts it with a local CKKS context, and sends only ciphertext to the matcher.

## Setup

Use a dedicated Python 3.12 environment. Install PyTorch for the target CPU or
CUDA platform first, then install the local research package and this app:

```bash
cd model/Thanh2
python -m pip install -e ".[data,rawnet3]"
cd ../../app/voice_verification
python -m pip install -e ".[ui,matcher,he,model,test]"
cp .env.example .env.local
```

Place the verified fine-tuned `best.pt` artifact at the path in `.env.local`.
The app deliberately refuses the base RawNet3 checkpoint or an artifact with a
different SHA-256.

The launch scripts load `.env.local`; do not commit that file. The matcher
refuses to start without `VOICE_MATCHER_TOKEN`, and should remain bound to
`127.0.0.1` for this local MVP.

## Run locally

In one terminal:

```bash
scripts/run_matcher_local.sh
```

In another terminal:

```bash
scripts/run_streamlit.sh
```

Raw audio and plaintext embeddings remain in the Streamlit/model-worker process.
The matcher stores only ciphertext, public HE context, profile metadata, and a
display name.

## Local LiveKit voice Agent

The LiveKit Agent keeps authentication separate from conversation:

```text
room join -> GUEST -> explicit private-mode request -> AUTH_PENDING
          -> 5-second audio challenge -> 1:1 SV -> AUTHENTICATED or GUEST

authenticated/guest microphone audio -> local VAD -> local Whisper ASR
  -> final transcript event -> OpenAI response -> response event
  -> ZeroTTS streaming PCM -> browser playback
```

Install its optional local dependencies:

```bash
python -m pip install -e ".[agent,providers,test]"
```

Set `VOICE_HE_CONTEXT_MODE=keychain` in `.env.local`, enroll the owner again in
Streamlit, then copy the displayed identity UUID into `VOICE_OWNER_ID`. Existing
enrollment made with the default memory-only context cannot be reused;
re-enrollment is intentional. The private HE context is stored in macOS Keychain
only. If Keychain is unavailable, the Agent fails closed and never falls back to
a plaintext file.

Start a local LiveKit server binary in one terminal, then the Agent in another:

```bash
scripts/run_livekit_server_local.sh
scripts/run_livekit_auth_agent.sh
```

The browser sends `request_private_mode` on the `voice-auth` data topic and
receives authentication state on `voice-auth-status`. Conversation events use
the separate `voice-agent-event` topic and contain only a safe `turn_id`,
monotonic event sequence, final ASR transcript, incremental response text,
public processing state, and an optional `Demo Calendar · Mock MCP` label.
The browser never receives a score, embedding, HE context, matcher token, raw
audio, or OpenAI key. Voice-challenge audio is never sent into the conversation
pipeline. The challenge captures exactly five seconds, then waits for an
explicit `resume_conversation` or `continue_as_guest` command; speaking after
the five-second boundary or while the matcher is processing is discarded and
does not become an ASR transcript.

Enable conversation explicitly in `.env.local`:

```env
VOICE_AGENT_CONVERSATION_ENABLED=true
OPENAI_API_KEY=...
OPENAI_MODEL=...
VOICE_WHISPER_MODEL=base
VOICE_WHISPER_DEVICE=cpu
VOICE_DEFAULT_TIMEZONE=Asia/Ho_Chi_Minh
VOICE_TTS_PROVIDER=zerotts
VOICE_TTS_FALLBACK_PROVIDER=edge
VOICE_ZEROTTS_MODEL=zeroweight-ai/ZeroTTS
VOICE_ZEROTTS_VOICE=maichi
VOICE_ZEROTTS_CACHE_DIR=./app/voice_verification/data/model-cache/zerotts
VOICE_ZEROTTS_WARMUP=true
VOICE_ZEROTTS_STARTUP_BUFFER_MS=250
VOICE_ZEROTTS_INTRA_OP_THREADS=4
VOICE_ZEROTTS_CODEC_THREADS=
VOICE_TTS_QUEUE_MAX_CHUNKS=8
VOICE_TTS_MAX_SENTENCE_CHARS=120
VOICE_EDGE_TTS_VOICE=vi-VN-HoaiMyNeural
VOICE_VAD_MIN_SPEECH_SECONDS=0.3
VOICE_VAD_SILENCE_SECONDS=0.45
VOICE_VAD_MAXIMUM_SECONDS=15
MCP_PROVIDER=mock
```

### ZeroTTS startup buffer

`VOICE_ZEROTTS_STARTUP_BUFFER_MS` là thời gian audio đầu tiên được giữ lại
trước khi phát ra LiveKit. Mặc định `250` ms là mức cân bằng giữa độ trễ và
khả năng tránh hụt tiếng.

- Tăng lên `400` hoặc `600` ms nếu tiếng bị ngắt ngay lúc bắt đầu câu hoặc
  giữa các chunk đầu tiên.
- Giảm xuống `100` hoặc `0` ms khi audio đã liên tục nhưng muốn Agent phản hồi
  sớm hơn.
- Không tăng biến này để sửa tình trạng model sinh chậm liên tục; buffer hữu
  hạn chỉ trì hoãn lúc bắt đầu và sẽ cạn nếu tốc độ sinh thấp hơn tốc độ phát.
- Mỗi lần đổi giá trị phải restart Pipecat Agent. Nên đo ít nhất 5–10 lượt và
  chỉ thay đổi một biến tại một thời điểm.

The OpenAI adapter receives only the local final transcript and policy-selected
capabilities. `MockCalendarProvider` is deterministic demo storage: it is not
Google Calendar and all calendar responses must remain labelled as demo data.

## Pipecat runtime (optional local validation)

The original LiveKit Agents runtime remains the default fallback. Pipecat is
selected explicitly and uses a separate gateway-managed supervisor, so the two
runtimes must never be started for the same room:

```bash
python -m pip install -e ".[agent,gateway,pipecat,providers,test]"
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Put the generated value in `PIPECAT_SUPERVISOR_SECRET` in the local
`.env.local` used by the gateway and supervisor. For Pipecat mode, create or
sign in to the account in the React app and enroll there. This is important:
gateway enrollment uses the account-scoped Keychain context, so an identity
enrolled only by the standalone Streamlit flow is not automatically the same
identity used by the unified account flow. Then configure:

```env
VOICE_AGENT_RUNTIME=pipecat
VOICE_AGENT_CONVERSATION_ENABLED=true
PIPECAT_SUPERVISOR_URL=http://127.0.0.1:8021
MCP_PROVIDER=mock
MOCK_CALENDAR_ENABLED=true
OPENAI_API_KEY=...
```

Start five local processes in separate terminals. The matcher must be running
before the Pipecat worker attempts a voice challenge:

```bash
scripts/run_livekit_server_local.sh
scripts/run_pipecat_supervisor_local.sh
scripts/run_matcher_local.sh
scripts/run_gateway_local.sh
cd ../web && npm run dev
```

The gateway starts exactly one Pipecat worker for each room through a signed
descriptor. The descriptor includes only room/account identifiers; private HE
context, RawNet3, raw audio, matcher tokens, and the OpenAI key remain in the
local Python processes. Pipecat mode uses local Silero VAD, the project's
faster-whisper adapter, the existing policy-first supervisor, and a custom
ZeroTTS streaming PCM adapter with a buffered Edge-TTS fallback. ZeroTTS uses
the preset voice configured by `VOICE_ZEROTTS_VOICE`; it does not read voice
enrollment audio. The Pipecat bundled Whisper module is intentionally not
installed because it can initialize MLX/Metal during import on headless CPU
machines. During the Mock MCP phase, calendar routing is deterministic and
passes through `PolicyToolExecutor`; this is deliberate so a prompt cannot
invent a tool or trigger an unvalidated side effect. A future OpenAI tool-call
loop can use the same typed executor without changing the AuthDecision or
provider boundary.

## Validation

```bash
cd ../..
source .venv-voice/bin/activate
PYTHONPATH=. pytest app/voice_verification/tests -q
python .agents/skills/livekit-simulations/scripts/build_scenarios.py assemble \
  --in app/assistant/simulations/authored.yaml \
  --agent-description-file app/assistant/simulations/description.md \
  --risks app/assistant/simulations/risks.yaml --strict \
  --out app/assistant/simulations/scenarios.yaml
```

The generated simulations are a policy regression suite for the conversational
Agent. RawNet3, VAD, ASR, and LiveKit audio transport still require deterministic
local fixture/integration tests; run `lk agent simulate` only after installing
the LiveKit CLI and enabling its beta/cloud simulation access.
