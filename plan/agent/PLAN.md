# Revised plan: Local-first VoiceAgent với OpenAI LLM

## Quyết định chính

Toàn bộ phần nhạy cảm và model voice chạy local:

```text
Browser
   ↓ local WebRTC
LiveKit Server local
   ↓
Python LiveKit Agent local
   ├── RawNet3 local
   ├── HE private context local
   ├── VoiceAuthGate local
   └── SQLite matcher local
```

Chỉ LLM được gọi qua OpenAI API:

```text
Local ASR transcript
        ↓
OpenAI Responses API
        ↓
Local Agent
        ↓
LiveKit audio response
```

Raw audio, RawNet3 embedding, private HE key và encrypted template không được gửi tới OpenAI.

OpenAI SDK tự đọc `OPENAI_API_KEY` từ environment; key chỉ nằm trong Agent/backend process, không nằm trong Next.js browser hoặc LiveKit metadata. [OpenAI API quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request), [Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)

## Phase 0 — Giữ nguyên SV baseline

Không thay đổi flow Streamlit đang chạy:

```text
Enroll 3 samples
   ↓
RawNet3 local
   ↓
HE encryption
   ↓
Local matcher
   ↓
Client-side decision
```

Streamlit tiếp tục dùng cho:

- enrollment;
- manual verification;
- threshold debug;
- model evaluation;
- local administration.

## Phase 1 — Sửa contract SID/SV và AuthDecision

### SID

```python
VoiceAuthGate.identify(audio) -> IdentificationResult
```

SID là 1:N identification.

SID dùng để:

- nhận diện owner;
- personalization;
- hiển thị tên;
- chọn user profile.

SID không cấp private permission.

### SV

```python
VoiceAuthGate.verify(
    audio,
    target_identity_id: UUID,
) -> VerificationResult
```

SV là 1:1 và bắt buộc target identity.

Luồng auth:

```text
ROOM_JOIN
   ↓
GUEST
   ↓ user requests private mode
AUTH_PENDING
   ↓ challenge audio 4–6 giây
1:1 SV
   ↓
AUTHENTICATED hoặc GUEST
```

Không chạy SV tự động khi room join.

Nếu sau này muốn auto-identify owner khi join, đó là flow riêng:

```text
ROOM_JOIN
   ↓ short audio sample
SID 1:N
   ↓
OWNER / UNKNOWN
```

SID không được biến thành SV.

### AuthDecision

```python
AuthDecision(
    state=AuthState.AUTHENTICATED,
    identity_id=owner_id,
    display_name="An",
    expires_at=timestamp,
)
```

Các state:

```text
GUEST
AUTH_PENDING
AUTHENTICATED
SESSION_EXPIRED
```

LLM không được tạo hoặc sửa `AuthDecision`.

## Phase 2 — Persistent local voice session

Current Streamlit tạo HE context theo session, không phù hợp cho Agent restart. Với local single-owner, chuyển sang:

```text
macOS Keychain
└── private HE context

SQLite
├── stable context metadata
├── encrypted voice template
└── owner identity metadata
```

Quy tắc:

- Private HE context không nằm trong SQLite.
- Không lưu plaintext embedding.
- Không lưu raw audio.
- Nếu Keychain không khả dụng, app fail closed.
- Không fallback sang plaintext file.
- Context cũ dạng ephemeral được backup và enroll lại.
- Agent và Streamlit dùng chung `VoiceAuthSessionFactory`.
- Matcher vẫn chỉ nhận public context + ciphertext.

## Phase 3 — LiveKit Auth Gate local-only

Đây là milestone tiếp theo, chưa làm ASR/LLM/TTS.

```text
Local browser
   ↓
Local LiveKit room
   ↓
Python Agent local
   ↓
Audio frame buffer
   ↓
AudioRecording adapter
   ↓
VoiceAuthGate.verify()
   ↓
AuthDecision
```

Behavior:

- Room join bắt đầu ở `GUEST`.
- User yêu cầu private mode.
- Agent chuyển sang `AUTH_PENDING`.
- Agent thu challenge 4–6 giây.
- Audio được xử lý local bởi RawNet3.
- Agent gọi 1:1 SV với owner identity.
- Pass → `AUTHENTICATED`.
- Fail → giữ `GUEST`.
- Session hết hạn → `SESSION_EXPIRED`.
- Reconnect không tự động bypass authentication.

Phase này không có:

- ASR;
- LLM;
- TTS;
- typed tools;
- MCP.

Chỉ dùng một LiveKit client tối giản để test room và authentication. Chưa xây full frontend.

## Phase 4 — Local ASR và OpenAI LLM

Sau khi Auth Gate local ổn định:

```text
LiveKit audio
   ↓
Local VAD / turn detection
   ↓
Local Whisper ASR
   ↓
Transcript
   ↓
OpenAI Responses API
   ↓
Text response
   ↓
Local TTS
   ↓
LiveKit audio
```

Provider mặc định:

- ASR: Whisper chạy local;
- LLM: OpenAI Responses API;
- TTS: Edge-TTS chạy local/network provider.

Environment:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

Quy tắc:

- `OPENAI_API_KEY` chỉ được load trong Python Agent.
- Không đưa key vào frontend.
- Không ghi key vào logs.
- Không gửi raw audio tới OpenAI.
- Chỉ gửi transcript tối thiểu cần thiết.
- Không gửi private HE context, voice embedding hoặc identity template.
- LLM streaming được thêm sau khi non-streaming path ổn định.
- Model name phải cấu hình qua environment để đổi model không sửa code.

Tạo interface:

```python
class LLMProvider:
    async def respond(
        self,
        transcript: str,
        *,
        allowed_tools: set[str],
        auth_context: AuthDecision,
    ) -> str:
        ...
```

OpenAI là provider đầu tiên. Gemini/Groq chỉ thêm sau khi OpenAI path ổn định, không chạy nhiều provider trong phase đầu.

## Phase 5 — Supervisor và typed tools

Auth Gate nằm trước Supervisor:

```text
Audio
  ↓
VoiceAuthGate
  ↓
trusted AuthDecision
  ↓
Supervisor
  ↓
allowed tool set
  ↓
LLM
```

Tool policy:

```text
GUEST:
  weather
  news
  general_qa

AUTHENTICATED:
  weather
  news
  general_qa
  read_diary
  write_diary
  delete_diary
```

LLM chỉ nhìn thấy tools đã được cấp.

Bắt buộc chặn:

- Guest gọi private tool;
- LLM tự thêm private tool;
- transcript tuyên bố “tôi là owner”;
- auth hết hạn nhưng vẫn gọi diary;
- prompt yêu cầu bỏ qua Auth Gate;
- tool failure bị giả thành thành công.

Không thêm LangGraph ở phase này.

## Phase 6 — Unified Next.js frontend

Sau khi local Agent flow ổn định, tạo frontend user-facing:

```text
/
├── Home
├── Enroll voice
├── Verify voice
├── Assistant
└── Settings
```

Next.js dùng chung:

- design system;
- navigation;
- microphone states;
- Authenticated/Guest status;
- error handling;
- loading states.

Streamlit chỉ còn là dashboard/debug nội bộ.

Frontend không được biết:

- OpenAI key;
- LiveKit API secret;
- matcher token;
- private HE key;
- model path;
- SQLite path.

Backend cấp temporary LiveKit token và gọi các trusted voice service. Người dùng cuối chỉ thấy một app thống nhất.

## Phase 7 — MCP cuối cùng

Chỉ thêm MCP sau khi typed tools ổn định.

MCP server không được truy cập:

- raw audio;
- RawNet3;
- HE private context;
- plaintext embedding;
- SQLite trực tiếp.

MCP chỉ nhận request đã qua `ToolPolicy`.

## Testing

### Local unit/integration tests

- SID 1:N.
- SV 1:1 bắt buộc target.
- Room join không chạy SV.
- Private mode mới chạy SV.
- Auth state transitions.
- Auth expiration.
- Keychain unavailable → fail closed.
- Audio frame buffer.
- Audio silence/short/invalid.
- Matcher HTTP 500/timeout.
- OpenAI timeout/rate-limit.
- OpenAI key missing → lỗi cấu hình rõ ràng.
- Guest/private tool policy.
- Reconnect không bypass auth.
- OpenAI payload không chứa raw audio hoặc embedding.

### LiveKit tests

- Agent join room local.
- Room disconnect.
- Challenge bắt đầu đúng thời điểm.
- Challenge đủ 4–6 giây.
- SV success/failure.
- User interrupt trong challenge.
- Session expiry.
- Private tool denial.

### LiveKit simulations

Tạo scenario files sau khi Agent có behavior thực tế:

```text
app/assistant/simulations/
├── description.md
├── risks.yaml
├── authored.yaml
└── scenarios.yaml
```

Risk tối thiểu:

- Guest private request;
- explicit unlock;
- SV success;
- SV failure;
- session expiry;
- prompt extraction;
- sensitive data;
- matcher unavailable;
- OpenAI unavailable;
- interruption during auth.

Dùng `build_scenarios.py assemble --strict` để kiểm tra coverage. Trước khi chạy, xác nhận CLI bằng:

```bash
lk agent simulate --help
```

Simulation kiểm tra conversation/policy behavior, không thay thế audio fixture tests cho RawNet3.

## Thứ tự triển khai ít lỗi nhất

```text
1. Giữ Streamlit SV ổn định
2. Sửa SID/SV/AuthDecision contract
3. Làm persistent local HE context
4. Chạy LiveKit local Auth Gate only
5. Test local auth flow
6. Thêm local Whisper
7. Thêm OpenAI LLM
8. Thêm local TTS
9. Thêm Supervisor + typed tools
10. Tạo Next.js unified frontend
11. Thêm MCP cuối cùng
```

Tiêu chí quan trọng: không debug WebRTC, RawNet3, HE, ASR, OpenAI, TTS và MCP trong cùng một phase.
