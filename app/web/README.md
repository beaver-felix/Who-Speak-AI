# Who Speak AI local web client

React + Vite client for the unified local workflow:

```text
local account → enroll voice → local LiveKit room → explicit voice challenge
```

The browser talks only to the FastAPI gateway. It never receives the LiveKit
API secret, OpenAI key, HE private context, matcher token, or RawNet3 embedding.
The calendar integration is deliberately a **Mock MCP provider** in this phase;
it has no Google OAuth connection and every result is marked as demo data.

## Run

Keep these processes running first:

```bash
cd ../voice_verification
scripts/run_livekit_server_local.sh
scripts/run_matcher_local.sh
scripts/run_gateway_local.sh
scripts/run_pipecat_supervisor_local.sh
```

The checked-in local configuration uses `VOICE_AGENT_RUNTIME=pipecat`. Use
`scripts/run_livekit_auth_agent.sh` only when intentionally switching that
setting to `livekit`; do not run both agent runtimes for the same room.

Install and start the web client in a new terminal:

```bash
cd ../web
npm install
npm run dev
```

Open <http://127.0.0.1:5173> or <http://localhost:5173>. The Vite development
server proxies `/api` to the Gateway so the browser keeps its HttpOnly session
cookie on the same origin. Register a local account, record three 4–8 second
samples, then join the local room. Allow microphone access and choose **Start
voice challenge**; speak naturally for around five seconds.

Leave `VITE_GATEWAY_URL` unset during local development. Set it only when the
gateway is hosted separately. Vite environment variables are public: do not put
a secret in them.
