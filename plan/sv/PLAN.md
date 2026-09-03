# Plan cập nhật: Streamlit Voice Verification MVP

> Cập nhật implementation: Streamlit vẫn giữ vai trò dashboard/debug. User-facing
> flow hiện dùng React + Vite (`app/web`) và FastAPI gateway
> (`app/assistant_gateway`): account local → enroll voice theo account → LiveKit
> local → voice challenge 1:1 → Agent. Calendar hiện là `MockCalendarProvider`
> local, luôn trả `provider=mock, demo=true`; chưa có Google OAuth hay MCP thật.
> Conversation được bật tường minh bằng `VOICE_AGENT_CONVERSATION_ENABLED=true`:
> local VAD → local Whisper → transcript-only OpenAI → Edge TTS → LiveKit audio.
> `VoiceAuthGate` vẫn độc lập và là nguồn duy nhất cấp private capability.

## 1. Mục tiêu phase đầu

Thay PySide6 desktop bằng Streamlit, nhưng giữ nguyên các nguyên tắc bảo mật và model architecture:

- Enroll voice bằng 3 recording.
- RawNet3 fine-tuned trên ViMD tạo embedding 256 chiều.
- Speaker identification 1:N giống app face.
- HE mã hóa template/query trước khi gửi matcher.
- Client giải mã và quyết định kết quả.
- Chuẩn bị sẵn contract cho verification 1:1.
- Chưa triển khai Agent, ASR, LLM, TTS, MCP, diary.

Streamlit chỉ là UI/orchestrator. Logic speaker, HE, matcher và future Agent không được đặt trực tiếp trong page callback.

Streamlit là client MVP/dashboard, không phải transport realtime cuối cùng. Khi triển khai
VoiceAgent, dùng Python LiveKit Agents trên LiveKit WebRTC; toàn bộ voice core phải được
giữ độc lập để Streamlit và LiveKit dùng chung.

## 2. Cấu trúc project đề xuất

Giữ app face nguyên vẹn làm reference. Tạo voice app riêng:

```text
app/voice_verification/
├── streamlit_app.py
├── pages/
│   ├── home.py
│   ├── enroll_voice.py
│   ├── verify_voice.py
│   └── settings.py
├── src/voiceauth/
│   ├── audio/
│   ├── speaker/
│   ├── he/
│   ├── matching/
│   ├── api/
│   ├── session/
│   ├── errors.py
│   └── config.py
├── apps/matcher_api/
├── scripts/
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

Các feature Agent tương lai tách riêng:

```text
app/assistant/
├── livekit_agent.py
├── agents/
│   └── supervisor.py
├── tasks/
│   └── voice_authentication.py
├── providers/
│   ├── asr.py
│   ├── llm.py
│   └── tts.py
├── tools/
└── mcp/
```

`voiceauth` không import bất kỳ module nào từ `agent`, `asr`, `llm`, `tts` hoặc `mcp`.

Agent sau này chỉ gọi interface ổn định:

```python
VoiceAuthGate.identify(audio) -> IdentificationResult
VoiceAuthGate.verify(audio, target_identity_id) -> VerificationResult
```

Agent không được truy cập trực tiếp RawNet3, TenSEAL, database hoặc Streamlit session.

Agent runtime dùng Python LiveKit Agents. LiveKit chịu trách nhiệm WebRTC room,
realtime media, session, turn detection, interruptions và audio delivery; Agent code
chỉ chứa workflow, tools và quyền truy cập. LiveKit Agents hỗ trợ STT-LLM-TTS,
agent handoff và tool workflow trong cùng một session:

- [LiveKit Agents](https://docs.livekit.io/agents/)
- [LiveKit workflows](https://docs.livekit.io/agents/logic/workflows/)
- [LiveKit MCP tools](https://docs.livekit.io/agents/logic/tools/mcp/)

LiveKit Server được self-host ở local development để không phát sinh platform fee.
LiveKit Cloud chỉ là lựa chọn deployment sau; chi phí model/provider vẫn phải được
quản lý riêng.

## 3. Cách dùng Streamlit

Dùng `st.Page` và `st.navigation` để định nghĩa multipage app vì đây là hướng được Streamlit khuyến nghị cho navigation tùy biến. [Streamlit multipage navigation](https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation)

Audio lấy bằng:

```python
audio_value = st.audio_input(
    "Record your voice",
    sample_rate=16000,
    key="verification_audio",
)
```

`st.audio_input` nhận microphone từ browser và trả về object dạng file-like `UploadedFile`, mặc định phù hợp cho speech ở 16 kHz. [Streamlit st.audio_input](https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input)

Luồng privacy cần ghi rõ:

```text
Browser microphone
        ↓
Trusted Streamlit session
        ↓
RawNet3 embedding local
        ↓
HE encryption
        ↓
Voice matcher API
```

Raw audio chỉ được xử lý trong trusted Streamlit session, không lưu lâu dài và không gửi tới matcher. Nếu Streamlit được deploy cloud, Streamlit app được xem là trusted session service; matcher vẫn là untrusted ciphertext-only service.

Không dùng module global để lưu private key, audio hoặc user state. Mỗi user session dùng `st.session_state`, vì Streamlit session state được tách theo session và có thể dùng xuyên các page. [Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)

Các quy tắc bắt buộc:

- Không lưu `UploadedFile` lâu trong `st.session_state`; đọc bytes ngay rồi xóa reference.
- Không xử lý lại cùng một recording sau mỗi rerun.
- Tạo digest cho mỗi audio input để chống duplicate processing.
- Không đặt HE private context trong `st.cache_resource`.
- Chỉ cache model worker immutable nếu đã kiểm tra thread safety.
- Dùng `st.form`, `st.status` và `st.spinner` để kiểm soát rerun và hiển thị tiến trình.
- Không để Streamlit callback chứa logic model hoặc database.

## 4. Những phần copy từ app face

Copy các pattern đã ổn định:

- Matcher FastAPI routes, schemas, service và repository.
- SQLAlchemy/Alembic migration pattern.
- HE key store và Base64 serialization.
- API client, typed errors và timeout handling.
- Client-side decryption và decision flow.
- Public-context-only matcher boundary.
- Logging an toàn, không log dữ liệu sinh trắc học.
- Reset context và encrypted identities.
- Test structure cho API, HE, integration và security boundary.

Không copy:

- PySide6, Qt WebChannel, camera thread.
- InsightFace, DeepFace, Facenet512.
- Face crop/alignment/quality validation.
- Threshold `23.56`.
- Hard-code embedding dimension `512`.
- Face-specific model names hoặc error messages.

## 5. RawNet3 inference

Tạo `voiceauth.speaker.rawnet3_encoder.RawNet3SpeakerEncoder`.

Bắt buộc sử dụng artifact trong `GUIDE_RAWNET3.md`:

- `best.pt`.
- Không dùng `last.pt`.
- Không dùng pretrained `model.pt` thay cho fine-tuned checkpoint.
- Kiểm tra SHA-256:

  `0b06fd3c4644d6c1cf4e7f9c087cf7fba8be493952589ab4321f8319d4215386`

- Giữ `best.pt.json`, `resolved_config.json` và `final_test.json`.
- Load bằng `weights_only=True`, `strict=True`.
- Kiểm tra checkpoint identity và config hash.
- Model chạy `eval()`, frozen parameters và `torch.inference_mode()`.

Tái sử dụng từ `model/Thanh2`:

- `RawNet3Adapter`.
- `canonicalize_audio`.
- `evenly_spaced_segments`.
- `RAWNET3_EVALUATION_SAMPLES`.

Inference contract:

```text
input: mono float32 waveform
sample rate: 16,000 Hz
resampling: SciPy polyphase
amplitude normalization: none
evaluation crop: 64,240 samples
short audio: repeat waveform
embedding: 256-D
normalization: L2
score: cosine similarity
initial threshold: 0.6565317440180312
accept rule: score >= threshold
```

Model được load một lần trong một spawned worker process riêng. Streamlit rerun không được load lại checkpoint.

## 6. Audio subsystem

MVP dùng microphone của browser thông qua Streamlit, không dùng `sounddevice` trực tiếp trong Python server.

Tạo adapter:

```python
AudioDecoder.decode(uploaded_audio) -> AudioRecording
AudioRecording.waveform
AudioRecording.sample_rate
AudioRecording.channels
AudioRecording.duration_seconds
```

Audio pipeline:

1. Đọc WAV bytes từ `st.audio_input`.
2. Decode bằng `soundfile`.
3. Downmix stereo về mono.
4. Resample về 16 kHz.
5. Không amplitude-normalize.
6. Kiểm tra duration, finite samples, silence và empty input.
7. Gửi waveform vào RawNet3 worker.
8. Xóa bytes và object tạm sau khi embedding hoàn tất.

Enrollment cố định:

- 3 recording riêng.
- Mỗi recording khoảng 4–6 giây.
- Tạo 3 embedding.
- Arithmetic mean.
- L2-normalize template.
- Encrypt template.
- Gửi encrypted template lên matcher.

Verification:

- 1 recording khoảng 4–6 giây.
- Tạo encrypted query.
- Matcher so với tất cả templates.
- Client decrypt squared distances.
- Chuyển thành cosine:

```text
cosine = 1 - squared_euclidean_distance / 2
```

- Chọn score cao nhất.
- Nếu dưới threshold, trả `Unknown`.

## 7. HE matcher và database

Tạo voice matcher service/database riêng, không dùng chung context hoặc identity với face app.

Voice profile:

```text
model_profile: rawnet3-vimd-best-epoch-2
embedding_dim: 256
preprocessing_profile: rawnet3-vimd-16khz-v1
score_metric: cosine
threshold_profile: vimd-validation-security-v1
```

Matcher chỉ nhận:

- Public HE context.
- Encrypted 256-D query/template.
- Identity metadata.
- Optional candidate filter.

Matcher không nhận:

- Raw audio.
- Plaintext embedding.
- Private HE key.
- Threshold hoặc final authentication decision.

MVP dùng SQLite local. Cloud Run/Supabase để phase sau, sau khi local integration hoàn tất.

## 8. Implementation status — local VoiceAgent foundation

Implemented locally:

- SID (`VoiceAuthGate.identify`) and target-required 1:1 SV
  (`VoiceAuthGate.verify`) are separate contracts.
- `AuthSession` has `GUEST`, `AUTH_PENDING`, `AUTHENTICATED`, and
  `SESSION_EXPIRED`; a stale decision cannot authorize private tools.
- A 5-second LiveKit raw-audio challenge starts only after the explicit
  `request_private_mode` data-channel command. Join and reconnect start as
  guest, and failure paths fail closed.
- The persistent HE private context uses macOS Keychain with no file fallback;
  Streamlit and the Agent share `VoiceAuthSessionFactory` when
  `VOICE_HE_CONTEXT_MODE=keychain` is enabled. Re-enroll after changing context.
- OpenAI Responses, local Whisper, and Edge-TTS adapters exist behind separate
  provider interfaces but are not connected to the Auth Gate yet.
- Local tests cover auth state, target-bound SV, audio buffering, Keychain
  failure, policy expiry, OpenAI payload boundaries, and the LiveKit test harness.

Deferred deliberately:

- VAD/ASR → LLM → TTS room pipeline;
- typed diary tools and Supervisor execution;
- Next.js user frontend and temporary-token backend;
- MCP integration;
- `lk agent simulate` execution: scenario assets are generated and strict
  coverage-checked, but the local machine still needs the LiveKit CLI and its
  beta/cloud simulation access.

API thiết kế sẵn:

```python
target_identity_id: UUID | None
```

`None` nghĩa là 1:N. Có giá trị nghĩa là chỉ xác thực identity đó, phục vụ 1:1 về sau.

## 8. Streamlit pages

### Home

- Trạng thái model, matcher và session.
- Số identity đã enroll.
- Nút chuyển tới Enroll hoặc Verify.
- Không có chat hoặc assistant logic.

### Enroll Voice

- Nhập display name.
- Record sample 1, 2, 3.
- Hiển thị duration và audio preview tùy chọn.
- Chạy từng embedding.
- Lưu encrypted template.
- Không lưu raw recording.

### Verify Voice

- Record một query audio.
- Hiển thị trạng thái Processing.
- Trả kết quả:

```text
Verified: Thanh
Not registered
Verification failed
```

- Score chỉ hiển thị ở development/debug mode, không gọi là confidence tuyệt đối.

### Settings

- Matcher URL.
- API token chỉ đọc từ environment.
- Timeout.
- Model profile.
- Threshold profile.
- Reset local session/context.
- Không cho user tùy ý nhập threshold trong MVP.

## 9. Kiến trúc VoiceAgent realtime về sau

Sau khi voice verification ổn định, realtime pipeline dùng LiveKit:

```text
Browser/mobile microphone
          ↓ WebRTC
      LiveKit Room
          ↓
   Python LiveKit Agent
          ↓
      Auth Gate
       ├── RawNet3 SV
       ├── HE matcher
       └── Authenticated / Guest
                ↓
        Supervisor Agent
          ├── VAD / turn detection
          ├── ASR
          ├── LLM
          ├── typed tools
          ├── MCP toolsets
          └── TTS
```

Trạng thái phiên:

```text
GUEST → AUTH_PENDING → AUTHENTICATED → SESSION_EXPIRED
```

Nếu người dùng gọi chức năng riêng tư:

1. Agent chuyển sang `AUTH_PENDING`.
2. Thu authentication sample riêng, không dùng transcript làm bằng chứng speaker.
3. Chạy `VoiceAuthGate` với RawNet3 và HE matcher.
4. Chỉ sau khi pass mới cấp private tools hoặc private agent.
5. Khi session hết hạn, thu voice authentication lại.

MVP Agent chỉ dùng một `Supervisor Agent` và typed tools. Chưa thêm LangChain hoặc
LangGraph vì LiveKit đã cung cấp session, tasks, tools và handoff; thêm framework
orchestration thứ hai ngay từ đầu sẽ tạo hai nguồn quản lý workflow.

Quy tắc phân lớp:

- `Agent`: điều khiển hội thoại dài hạn và chọn tool.
- `Task`: bước ngắn bắt buộc phải hoàn tất, ví dụ voice authentication.
- `Tool`: side effect có schema và quyền rõ ràng.
- `MCPToolset`: kết nối MCP server, không chứa logic UI.
- `Provider adapter`: bọc ASR, LLM và TTS cụ thể để thay provider không sửa Agent.

Agent không được nằm trong:

- Streamlit page.
- RawNet3 worker.
- Matcher API.
- Audio decoder.
- HE client.

ASR là bước riêng sau authentication:

```text
Audio input
  ├── VoiceAuthGate
  └── ASR pipeline
```

Không dùng transcript để thay thế speaker verification. LLM, TTS và MCP chỉ được thêm sau khi auth result đã trở thành một typed domain object.

Streamlit tiếp tục được giữ làm dashboard/evaluation UI cho các trạng thái:
`SV`, `VAD`, `ASR`, `LLM`, `TTS`, latency và trace. Streamlit không được dùng để
giả lập microphone streaming hoặc thay thế LiveKit room ở realtime production.

## 10. Phân kỳ triển khai

### Phase 0 — Runtime và artifact

- Tạo Python 3.12 environment riêng.
- Kiểm tra PyTorch, Streamlit, soundfile, TenSEAL.
- Xác minh checkpoint hash.
- Chạy RawNet3 smoke test.

### Phase 1 — Core voice domain

- Tạo `voiceauth` package.
- Implement audio decoder.
- Implement RawNet3 wrapper.
- Implement spawned model worker.
- Implement embedding, enrollment averaging và cosine scoring.

### Phase 2 — HE matcher

- Copy matcher pattern từ app face.
- Đổi dimension thành 256.
- Tạo voice-specific schema và migration.
- Implement encrypted template/query.
- Implement client-side decision.

### Phase 3 — Streamlit UI

- Tạo multipage navigation.
- Implement `st.session_state`.
- Implement enrollment page.
- Implement verification page.
- Implement settings và reset.
- Kiểm tra rerun không lặp inference.

### Phase 4 — Integration và acceptance

- Same-speaker accept.
- Different-speaker reject.
- Unknown speaker branch.
- 1:N default.
- Optional 1:1 target filter.
- Kiểm tra không có raw audio/plaintext embedding trong matcher payload/database.

### Phase 5 — LiveKit VoiceAgent

- Chạy LiveKit Server local bằng self-hosted development mode.
- Tạo Python `livekit_agent.py` dùng `AgentServer` và `AgentSession`.
- Tạo `VoiceAuthGate` dưới dạng task bắt buộc trước private workflow.
- Kết nối LiveKit audio room với RawNet3 voice core.
- Thêm VAD/turn detection và interruption handling.
- Thêm ASR, LLM và TTS qua provider adapters.
- Dùng một Supervisor Agent với typed tools.
- Thêm MCP bằng explicit `MCPToolset`; không để MCP server truy cập raw audio,
  private key hoặc encrypted identity templates.
- Giữ Streamlit làm dashboard quan sát, không đưa LiveKit session state vào
  `st.session_state`.
- Kiểm tra guest mode, authentication timeout, session expiry và private-tool denial.

## 11. Tests và acceptance criteria

Automated tests:

- Audio WAV decode.
- Mono/stereo conversion.
- Sample-rate conversion.
- Audio ngắn được repeat đúng.
- Silence/empty/non-finite audio bị reject.
- Embedding shape `(256,)`, finite và norm xấp xỉ `1`.
- Sai checkpoint hash bị reject.
- Sai checkpoint identity bị reject.
- Worker load model một lần.
- Worker timeout/crash trả lỗi recoverable.
- Enrollment yêu cầu ba recordings.
- Template averaging đúng.
- HE public context không decrypt được.
- Matcher không nhận private context.
- Encrypted distance-to-cosine mapping đúng.
- Same speaker accept.
- Different speaker reject.
- Unknown không bị ép vào user.
- `target_identity_id` giới hạn đúng candidate.
- Streamlit rerun không xử lý trùng audio.
- Session A không nhìn thấy template/session state của session B.

Manual acceptance:

1. Mở Streamlit trên Mac.
2. Cho phép browser dùng microphone.
3. Enroll ba recordings.
4. Verify lại cùng người.
5. Verify bằng người khác.
6. Verify audio im lặng hoặc nhiễu.
7. Refresh page và kiểm tra session handling.
8. Kiểm tra matcher chỉ thấy ciphertext.
9. Đo CPU latency thực tế.
10. Ghi rõ threshold hiện tại chỉ là ViMD validation baseline.

## Assumptions

- UI chính: Streamlit.
- Runtime: Python 3.12 riêng.
- Audio: microphone browser qua `st.audio_input`.
- Privacy: Streamlit là trusted session, matcher chỉ xử lý ciphertext.
- Enrollment: ba recording riêng, mỗi recording 4–6 giây.
- MVP: 1:N.
- API đã chuẩn bị cho 1:1.
- Matcher voice dùng service/database riêng.
- Chưa triển khai Agent, ASR, LLM, TTS, MCP.
- `PLAN_APP.md` sẽ được viết lại thành voice-verification-first plan, không còn mô tả như đã triển khai toàn bộ virtual assistant.
- Streamlit là MVP/dashboard UI; LiveKit + Python Agents là realtime VoiceAgent runtime.
- LiveKit Server tự host ở local/dev; Cloud deployment và model billing là phase sau.
- Agent bắt đầu bằng Supervisor Agent + typed tools; chỉ thêm handoff/task groups khi workflow thực sự cần.
