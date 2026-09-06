# Who Speak AI — Local Runbook

This directory contains the current voice-verification and realtime voice-agent
implementation.

There are two local workflows:

1. **Streamlit verification demo** — enroll and verify voices without the
   conversational Agent.
2. **React + LiveKit + Pipecat** — run the account flow and realtime
   `SV → VAD → ASR → LLM → TTS` Agent.

The matcher receives only encrypted voice templates and encrypted queries. Raw
audio, plaintext embeddings, the HE private context, matcher tokens, and model
keys must remain in the trusted Python process.

## Requirements

- Python 3.12
- `uv` or an equivalent Python environment manager
- Node.js 20+ for the React client
- A verified RawNet3 ViMD `best.pt` artifact
- A local LiveKit server binary for the realtime workflow
- Microphone permission for the browser

The RawNet3 artifact, required hashes, and model contract are documented in
the repository root: [GUIDE_RAWNET3.md](../../GUIDE_RAWNET3.md).

## 1. Install

Run these commands from the repository root:

```bash
uv venv --python 3.12 .venv-voice
source .venv-voice/bin/activate

# Install a PyTorch build suitable for the machine.
# For CPU-only local development:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the research model package first.
uv pip install -e "model/Thanh2[data,rawnet3]"

# Install the voice app, matcher, HE, Agent, providers, and tests.
uv pip install -e "app/voice_verification[ui,matcher,he,model,agent,gateway,providers,pipecat,test]"

cp -n app/voice_verification/.env.example app/voice_verification/.env.local
```

Set at least these values in
`app/voice_verification/.env.local`:

```env
VOICE_MATCHER_TOKEN=use-a-private-local-token
VOICE_MODEL_CHECKPOINT=../../rawnet3/runs/vimd/checkpoints/best.pt
VOICE_MODEL_DEVICE=cpu
```

The application verifies the checkpoint SHA-256 at startup and refuses the
wrong artifact. Do not replace the fine-tuned `best.pt` with `last.pt` or
the base pretrained `model.pt`.

## 2. Streamlit verification demo

Use this mode to test RawNet3 enrollment, HE encryption, and 1:N speaker
identification.

Open two terminals and activate the environment in each:

```bash
source .venv-voice/bin/activate
```

Terminal 1 — matcher:

```bash
app/voice_verification/scripts/run_matcher_local.sh
```

Terminal 2 — Streamlit:

```bash
app/voice_verification/scripts/run_streamlit.sh
```

Open <http://127.0.0.1:8501> and:

1. Open **Enroll voice**.
2. Record three separate Vietnamese voice samples, each around 4–8 seconds.
3. Enter a display name and select **Encrypt and enroll**.
4. Open **Verify voice** and record a new sample.
5. Confirm the matched identity or **Not registered** result.

Check the matcher with:

```bash
curl http://127.0.0.1:8011/health
```

Streamlit uses a new in-memory HE context by default. An identity enrolled in
this standalone flow is not automatically available to the account-based React
flow. Use `VOICE_HE_CONTEXT_MODE=keychain` only when intentionally creating a
persistent local owner context.

## 3. Realtime React + LiveKit + Pipecat workflow

This is the current full application flow. Use the Pipecat runtime configured
by:

```env
VOICE_AGENT_RUNTIME=pipecat
VOICE_AGENT_CONVERSATION_ENABLED=true
MCP_PROVIDER=mock
```

Also set `OPENAI_API_KEY`. The local mock MCP provider is recommended for the
first run; it avoids requiring a separate Google Calendar MCP deployment.

### Use the local Google Calendar MCP server

The repository also contains a self-hosted Google Calendar MCP server in
`google-calendar-mcp/`. It is optional. Use it when you want the Agent to read
real calendar events instead of the deterministic mock data.

Set these values in `app/voice_verification/.env.local`:

```env
MCP_PROVIDER=local
GOOGLE_MCP_ENDPOINT=http://127.0.0.1:3000
```

The local MCP server has its own Google OAuth token store. Authenticate the
same Google email that you connect later from the Who Speak web UI. The gateway
OAuth connection and the local MCP OAuth connection are separate by design.

In Google Cloud, enable Google Calendar API and create OAuth 2.0 credentials
for a **Desktop app**. Save the JSON file as
`google-calendar-mcp/gcp-oauth.keys.json`. This file is ignored by Git; never
commit it or put its contents in a frontend `.env` file.

From the repository root, use a separate terminal:

```bash
cd google-calendar-mcp
npm install
npm run build

# Authenticate the Google account. A browser window should open.
GOOGLE_OAUTH_CREDENTIALS="$PWD/gcp-oauth.keys.json" npm run auth

# Keep this process running as the local HTTP MCP server.
GOOGLE_OAUTH_CREDENTIALS="$PWD/gcp-oauth.keys.json" \
  ENABLED_TOOLS=list-events npm run start:http
```

Verify the MCP server before starting the gateway:

```bash
curl http://127.0.0.1:3000/health
curl http://127.0.0.1:3000/api/accounts
```

The account returned by `/api/accounts` must have the same email as the
Google account connected in **Connect Google Calendar**. `ENABLED_TOOLS=list-events`
is intentional: the current application rollout only exposes read-only event
listing. Do not start the local MCP with `0.0.0.0` for this local workflow.

After the MCP server is running, start the LiveKit, matcher, gateway,
Pipecat, and React processes below. In the web UI, sign in, enroll your voice,
select **Connect Google Calendar**, then complete the voice challenge before
asking about personal events.

Generate a supervisor secret and put the same value in `.env.local`:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Start these processes in separate terminals from the repository root:

```bash
# Terminal 1 — LiveKit WebRTC server
app/voice_verification/scripts/run_livekit_server_local.sh

# Terminal 2 — ciphertext-only voice matcher
app/voice_verification/scripts/run_matcher_local.sh

# Terminal 3 — account/session gateway on port 8020
app/voice_verification/scripts/run_gateway_local.sh

# Terminal 4 — gateway-managed Pipecat supervisor on port 8021
app/voice_verification/scripts/run_pipecat_supervisor_local.sh

# Terminal 5 — React client on port 5173
cd app/web
npm install
./scripts/run_web_local.sh
```

Open <http://127.0.0.1:5173>, then:

1. Register or sign in to a local account.
2. Enroll three voice samples.
3. Join the local voice room.
4. Allow browser microphone access.
5. Select **Start voice challenge** and speak for about five seconds.
6. After successful SV, select **Resume conversation**.

The conversation flow is:

```text
LiveKit audio → authentication gate → RawNet3 SV → local VAD
→ local Vietnamese Whisper → OpenAI LLM → TTS → LiveKit playback
```

The voice challenge is isolated from the conversation pipeline. Audio captured
for authentication is not reused as an ASR transcript.

## Alternative: LiveKit Agents runtime

The original Python LiveKit Agents runtime is available as an alternative. In
`.env.local`, set:

```env
VOICE_AGENT_RUNTIME=livekit
VOICE_AGENT_CONVERSATION_ENABLED=true
```

Then replace the Pipecat supervisor command with:

```bash
app/voice_verification/scripts/run_livekit_auth_agent.sh
```

Do not run the Pipecat supervisor and the LiveKit Agent for the same room.

For this standalone Agent flow, set
`VOICE_HE_CONTEXT_MODE=keychain`, enroll the owner in Streamlit again, and set
the resulting identity UUID as `VOICE_OWNER_ID`. Re-enrollment is intentional
because the session-only HE context cannot be reused by the persistent Agent
process.

## Troubleshooting

### `best.pt` was not found or hash validation failed

Acquire the artifact described in [GUIDE_RAWNET3.md](../../GUIDE_RAWNET3.md),
place it at the configured path, and check it with:

```bash
shasum -a 256 rawnet3/runs/vimd/checkpoints/best.pt
```

### Matcher is unavailable

Confirm the matcher is running:

```bash
curl http://127.0.0.1:8011/health
```

Both the Streamlit/Agent process and matcher must use the same
`VOICE_MATCHER_TOKEN`.

### Microphone is unavailable

Allow microphone access to the browser in macOS **System Settings → Privacy &
Security → Microphone**. Close other applications currently using the
microphone and reload the web page.

### Agent does not start

Check the services:

```bash
curl http://127.0.0.1:8020/health
curl http://127.0.0.1:8021/health
```

For Pipecat, confirm `PIPECAT_SUPERVISOR_SECRET` is at least 32 characters
and is the same in the gateway and supervisor environment. For either Agent
runtime, confirm `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and
`LIVEKIT_URL` match the local LiveKit server.

### Local Google Calendar MCP is unavailable

Confirm that the MCP process is running and that the authenticated account is
visible:

```bash
curl http://127.0.0.1:3000/health
curl http://127.0.0.1:3000/api/accounts
```

If `/api/accounts` is empty, run `npm run auth` again with the same
`GOOGLE_OAUTH_CREDENTIALS` path. Confirm the application environment uses
`MCP_PROVIDER=local` and `GOOGLE_MCP_ENDPOINT=http://127.0.0.1:3000`, then
restart the gateway. `MCP_PROVIDER=mock` intentionally does not contact
Google Calendar.

### ZeroTTS starts slowly or audio underruns

The first run downloads and warms the local TTS model. Increase
`VOICE_ZEROTTS_STARTUP_BUFFER_MS` to `400` or `600`, then restart the
Pipecat supervisor. Do not run both Agent runtimes while testing audio output.

## Tests

From the repository root:

```bash
source .venv-voice/bin/activate
PYTHONPATH=. pytest app/voice_verification/tests -q

cd app/web
npm test
npm run typecheck
```

The automated tests do not replace the real microphone, RawNet3 model, LiveKit
transport, or external provider acceptance checks. Run one full manual flow on
the target Mac before presenting the application.
