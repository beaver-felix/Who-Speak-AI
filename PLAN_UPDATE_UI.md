# Plan chỉnh Voice UI theo PuPuPlatter + speech-to-speech

## Tóm tắt

Giữ nguyên React/Vite, LiveKit, Pipecat, RawNet3, HE, Auth Gate, Whisper, OpenAI và Mock Calendar. Chỉ tái sử dụng có chọn lọc pattern UI/audio từ:

- `Voice-Agent-PuPuPlatter`: component structure, voice status, conversation panel, reconnection.
- `speech-to-speech`: voice orb, audio-reactive visualizer, state presentation và playback lifecycle.

Không thay toàn bộ frontend bằng repo bên ngoài và không chuyển audio sang WebSocket/OpenAI Realtime.

Mục tiêu:

```text
Signed in
→ Join local room
→ Guest
→ Explicit voice challenge
→ Authenticated
→ Listening
→ Transcribing
→ Thinking
→ Speaking
→ Completed
```

Desktop dùng orb làm trung tâm với conversation side panel. Mobile dùng orb trước, conversation mở thành drawer/full-screen.

## Phạm vi tác động

| Thành phần | Quyết định |
|---|---|
| Account/enrollment | Giữ nguyên |
| LiveKit transport | Giữ nguyên |
| Auth Gate | Giữ nguyên, không auto-SV khi join |
| Silero VAD + Smart Turn | Giữ một owner duy nhất trong Pipecat |
| Whisper | Giữ transcript final ở bản đầu |
| OpenAI | Giữ streaming response hiện tại |
| TTS | Giữ Pipecat streaming provider hiện tại |
| Frontend state/audio lifecycle | Refactor chính |
| Security boundary | Không thay đổi |

Delay hiện tại nhiều khả năng đến từ chuỗi:

```text
Smart Turn wait
→ Whisper chạy full segment trên CPU
→ chờ LLM delta/sentence boundary
→ ZeroTTS tạo audio
→ LiveKit publish/playback
```

Bản chỉnh đầu tiên sẽ đo từng stage và cải thiện cảm giác realtime ở UI trước, không đổi model tùy tiện.

## Thay đổi implementation

### 1. Chuẩn hóa session/event state

Mở rộng event contract hiện tại bằng event session-level:

```ts
type SessionStatus =
  | "idle"
  | "connecting"
  | "starting"
  | "ready"
  | "reconnecting"
  | "stopping"
  | "failed";

type VoiceAgentEvent =
  | {
      type: "session_status";
      message_id: string;
      status: SessionStatus;
      message?: string;
    }
  | {
      type: "state";
      message_id: string;
      turn_id?: string;
      state:
        | "listening"
        | "transcribing"
        | "thinking"
        | "speaking"
        | "completed"
        | "interrupted"
        | "error";
      stage?: "auth" | "vad" | "stt" | "llm" | "tool" | "tts" | "transport";
      message?: string;
      sequence?: number;
    }
  | {
      type: "transcript";
      message_id: string;
      turn_id: string;
      text: string;
      is_final: true;
      sequence?: number;
    }
  | {
      type: "assistant_response";
      message_id: string;
      turn_id: string;
      text: string;
      is_final: boolean;
      tool?: ToolBadge | null;
      sequence?: number;
    };
```

Quy tắc:

- `session_status` mô tả trạng thái Agent, không phải trạng thái một câu nói.
- `state` mô tả từng turn.
- Auth challenge không tạo `transcript` hoặc `assistant_response`.
- `stage` chỉ dùng để hiển thị lỗi/latency, không chứa score, embedding hoặc dữ liệu nhạy cảm.
- Giữ `message_id`, `turn_id`, `sequence` để chống duplicate và event cũ.
- Giới hạn kích thước `seenEventIds` để history không tăng vô hạn.

Backend phát thêm:

```text
starting
ready
reconnecting
failed
```

Frontend không hiển thị `Offline` trong thời gian model đang warm-up.

### 2. Tách `LiveKitPanel` thành các trách nhiệm nhỏ

`LiveKitPanel.tsx` hiện đang chứa kết nối LiveKit, auth command, event parsing, audio lifecycle, reducer và UI. Tách thành:

```text
LiveKitPanel
├── useLiveKitVoiceSession
├── useAudioLevels
├── VoiceStage
│   ├── VoiceOrb
│   ├── SessionStatus
│   ├── AuthStatus
│   └── VoiceControls
└── ConversationPanel
    ├── ConversationHeader
    ├── ConversationThread
    ├── TurnRow
    ├── ToolBadge
    └── TurnError
```

`useLiveKitVoiceSession` chịu trách nhiệm:

- tạo và đóng `Room`;
- lấy token từ FastAPI;
- register data/text stream handlers;
- join/leave/reconnect;
- auth commands;
- reset conversation khi reconnect;
- attach/detach audio track;
- expose session view model cho UI.

`conversation.ts` vẫn là reducer thuần và nguồn sự thật cho lịch sử. UI chỉ render state từ reducer, không tự giữ message song song.

### 3. Hiển thị conversation theo turn

Thay vì render flat message một cách độc lập, UI tạo một `TurnRow` theo `turn_id`:

```text
TurnRow
├── User transcript
├── Assistant response
├── Tool badge
├── Speaking/Played/Interrupted
└── Error + action phù hợp
```

Behavior:

- User chỉ hiện transcript final.
- Assistant response cập nhật dần trong cùng một bubble khi backend gửi delta.
- Placeholder `Listening`, `Transcribing`, `Thinking` được thay bằng nội dung thật trong cùng turn.
- Lỗi chỉ hiển thị một lần ở turn bị lỗi, không lặp lại dưới cả user và assistant bubble.
- `calendar.*` luôn hiển thị:
  ```text
  Demo Calendar · Mock MCP
  ```
- TTS lỗi giữ nguyên assistant text và hiển thị:
  ```text
  Audio unavailable
  ```
- Retry chỉ hiện với lỗi LLM/tool có thể retry. Không tự retry vô hạn và không retry cả turn chỉ vì TTS lỗi.

### 4. Audio-reactive orb và visualizer thật

Tham khảo ý tưởng từ `VoiceVisualizer.tsx` và `orb-visualizer.js`, nhưng adapt bằng Web Audio API hiện có, không copy provider context.

Tạo `useAudioLevels`:

- tạo một `AudioContext` với `latencyHint: "interactive"`;
- tạo analyser cho microphone track;
- tạo analyser cho assistant remote audio track;
- chạy một `requestAnimationFrame` loop duy nhất;
- ghi mức âm thanh vào CSS custom properties:
  ```css
  --mic-level
  --assistant-level
  ```
- dùng smoothing attack/release để tránh giật;
- cleanup analyser, media source, animation frame và AudioContext khi leave/disconnect.

Không:

- tạo AudioWorklet mới cho audio send path ở phase này;
- gửi raw PCM riêng ngoài LiveKit;
- tạo nhiều audio element cho cùng một remote track;
- dùng animation timer để giả lập mic activity.

Orb state:

```text
ready/connected  → green
auth_pending     → amber
listening        → cyan
transcribing     → amber
thinking/tool    → amber
speaking         → violet
error            → red
```

Màu bão hòa chỉ xuất hiện trên orb và role label nhỏ. Panel, bubble và button dùng nền trung tính theo design language của `speech-to-speech`.

### 5. Layout UI đích

Desktop:

```text
┌────────────────────────────────────────────────────┐
│ Account · Agent status · Auth status                │
├─────────────────────────┬──────────────────────────┤
│                         │ Conversation             │
│       Voice Orb         │ User transcript           │
│   audio-reactive        │ Agent response            │
│                         │ Tool/error status         │
│ Listening / Speaking    │                          │
│ Mic · Leave room        │                          │
└─────────────────────────┴──────────────────────────┘
```

Mobile:

```text
Account/auth status
Voice orb
Current state
Mic and leave controls
Latest response preview
Open conversation
```

Conversation side panel là nguồn hiển thị chính. Không tạo nhiều bubble nổi trùng lặp làm người dùng khó biết đâu là lịch sử thật.

Copy UI:

- `Connecting to local room`
- `Starting local voice engine`
- `Ready · speak when you are ready`
- `Listening`
- `Transcribing`
- `Thinking`
- `Speaking`
- `Played`
- `Interrupted`
- `Audio unavailable`

Không dùng wording gây hiểu nhầm như `Streaming reply` nếu trạng thái thực tế chỉ là final event. Assistant draft có thể cập nhật theo delta, nhưng footer phải nói rõ trạng thái hiện tại.

### 6. Reconnect và audio unlock

Khi user bấm Join:

```text
button gesture
→ api.livekitToken()
→ room.connect()
→ room.startAudio()
→ enable microphone
→ starting
→ ready/guest
```

Khi reconnect:

```text
RoomEvent.Reconnecting
→ UI hiển thị Reconnecting
→ disable challenge/tool controls

RoomEvent.Reconnected
→ không khôi phục Authenticated từ state cũ
→ reset conversation/auth
→ lấy session/token mới
→ trở về Guest
```

Khi `startAudio()` bị browser chặn:

```text
Audio is blocked
→ hiển thị Enable audio
→ tiếp tục sau một user gesture
```

Khi disconnect:

- detach toàn bộ remote audio;
- xóa audio elements;
- stop analyser loop;
- reset conversation;
- reset auth về Guest;
- giải phóng `AudioContext`;
- không để audio cũ phát tiếp.

### 7. Tái sử dụng code PuPuPlatter an toàn

Có thể adapt pattern từ:

```text
VoiceVisualizer.tsx
VoiceStatus.tsx
ReconnectionStatus.tsx
ConversationPanel.tsx
MessageBubble.tsx
```

Không copy:

- provider contexts;
- OpenAI/Gemini direct browser flow;
- `VITE_*_API_KEY`;
- multi-provider tabs;
- camera/particle features;
- API routes riêng;
- tool executor của repo.

Để giảm dependency và bug:

- không thêm `framer-motion`, Tailwind hoặc toàn bộ shadcn chỉ để lấy UI;
- dùng React hiện tại, system font, CSS transitions và SVG inline;
- giữ `livekit-client` hiện tại;
- chỉ copy thuật toán/behavior cần thiết.

Repo PuPuPlatter dùng MIT License. Nếu copy đáng kể source code, phải:

- giữ notice/license của repo;
- thêm attribution trong file adapt hoặc `THIRD_PARTY_NOTICES.md`;
- không đưa cả repo PuPuPlatter vào bundle/app;
- không sửa repo reference gốc.

### 8. Tối ưu latency theo từng bước

Bước đầu không đổi RawNet3, HE, Whisper model hoặc TTS provider.

Thêm structured timing log theo:

```text
room_id_hash
session_id
turn_id
stage
elapsed_ms
```

Các mốc:

```text
speech_started
speech_stopped
smart_turn_released
stt_started
stt_final
llm_started
llm_first_delta
llm_final
tts_started
tts_first_audio
assistant_started_speaking
assistant_stopped_speaking
turn_completed
```

Không log:

- raw audio;
- full private transcript trong production;
- embedding;
- similarity score;
- HE key;
- matcher token;
- OpenAI key.

Sau khi có baseline, benchmark riêng:

```text
Smart Turn wait: 2.0s vs 1.0s
TTS sentence limit: 120 vs 80 characters
Whisper: base vs tiny
```

Không copy `VOICE_VAD_SILENCE_SECONDS=0.064` từ `speech-to-speech`; giá trị đó phụ thuộc transport/model khác. Giữ VAD hiện tại trong UI pass và chỉ thay sau khi có audio fixture accuracy test.

Mục tiêu bản responsive:

- state `Listening` xuất hiện ngay khi VAD báo speech;
- user transcript xuất hiện ngay sau STT final;
- assistant draft xuất hiện ngay khi có LLM delta đầu tiên;
- audio phát theo chunk đầu tiên, không đợi toàn bộ câu;
- interruption dừng audio trong thời gian hữu hạn;
- không tăng memory qua nhiều turns.

## Testing và acceptance

### Unit

- `session_status` parse/reducer.
- Chuyển đúng `starting → ready`.
- Reconnect reset về Guest.
- `turn_id` ghép đúng user/assistant.
- Assistant delta update đúng bubble.
- Duplicate event không tạo message trùng.
- Event sequence cũ không ghi đè state mới.
- Auth challenge không tạo chat message.
- Error stage hiển thị đúng turn.
- Audio level smoothing và cleanup.
- Một remote track chỉ tạo một audio element/analyser.

### Component

- Orb render đúng với từng session/turn state.
- Mic level và assistant level thay đổi CSS variable.
- Conversation side panel tự scroll khi đang ở cuối.
- Không auto-scroll khi user đọc message cũ.
- Mobile mở/đóng conversation drawer.
- Keyboard focus và `aria-live` đúng.
- `prefers-reduced-motion` tắt animation phụ.
- Text wrap đúng ở zoom 200% và 400%.

### LiveKit integration/E2E

- Join room bắt đầu ở `Guest`.
- Agent warm-up hiển thị `Starting`, không hiển thị Offline giả.
- Start voice challenge chỉ hoạt động sau khi room/Agent ready.
- Challenge audio không đi vào Whisper/LLM.
- SV success chuyển Authenticated.
- SV failure giữ Guest.
- Nói câu hỏi tạo đúng một User bubble.
- Assistant tạo đúng một response bubble.
- Mock Calendar badge xuất hiện đúng khi tool được phép.
- Guest không gọi được calendar.
- Remote audio attach và detach đúng lifecycle.
- `room.startAudio()` chỉ được gọi sau user gesture.
- TTS lỗi vẫn giữ assistant text.
- Interruption không phát chồng audio.
- Reconnect không bypass authentication.
- Agent restart không làm UI hiển thị Authenticated giả.

### Visual/browser

Kiểm tra tối thiểu:

```text
Chrome macOS
Safari macOS
360px mobile width
1440px desktop width
keyboard-only
prefers-reduced-motion
zoom 200%
```

Chạy:

```bash
npm test -- --run
npm run typecheck
npm run build
PYTHONPATH=. .venv-voice/bin/python -m pytest app/voice_verification -q
```

LiveKit local E2E chỉ chạy sau khi bốn process hoạt động:

```text
LiveKit server
matcher
gateway
Pipecat supervisor
```

## Thứ tự triển khai

1. Thêm `session_status` và `stage` vào event contract.
2. Tách LiveKit connection/audio lifecycle khỏi UI render.
3. Giữ reducer làm nguồn sự thật và render theo `turn_id`.
4. Tạo audio analyser thật cho mic/output.
5. Xây orb-first desktop layout và mobile drawer.
6. Adapt conversation/status/reconnect patterns từ PuPuPlatter.
7. Cập nhật copy và accessibility.
8. Thêm structured latency logs.
9. Chạy unit/component/frontend checks.
10. Chạy local LiveKit E2E.
11. Chỉ sau khi có latency baseline mới điều chỉnh Smart Turn, Whisper hoặc sentence buffering.
12. Không thêm MCP thật hoặc provider mới trong đợt refactor UI này.

## Acceptance criteria

Bản chỉnh được xem là đạt khi:

- Orb phản ứng theo audio thật, không còn meter animation giả.
- Conversation side panel hiển thị rõ transcript và response theo từng turn.
- Assistant response có thể hiện text delta sớm nhưng vẫn chốt final.
- Trạng thái `Listening → Transcribing → Thinking → Speaking → Played` dễ hiểu.
- Agent warm-up không bị hiển thị là Offline.
- Không duplicate audio element, agent hoặc response.
- Reconnect luôn quay về Guest.
- Auth Gate vẫn nằm trước Supervisor/ToolPolicy.
- Guest không gọi Mock Calendar.
- Không có API key, HE data, embedding hoặc raw audio trong browser payload.
- Không đổi LiveKit transport chỉ để bắt chước `speech-to-speech`.
- Code adapt từ PuPuPlatter có attribution/license phù hợp.
- Test frontend, Python và local LiveKit E2E đều pass.

## Assumptions

- Runtime mặc định là Pipecat.
- UI mặc định ưu tiên tiếng Việt nhưng có thể giữ key/state tiếng Anh trong code.
- User transcript chỉ hiển thị final ở bản đầu.
- Assistant text vẫn cập nhật theo LLM delta hiện tại.
- Không thêm dependency UI lớn nếu không cần thiết.
- Streamlit tiếp tục là dashboard/debug, không dùng làm realtime frontend.
- `MockCalendarProvider` vẫn là provider duy nhất trong phase này.
- `Voice-Agent-PuPuPlatter` và `speech-to-speech` chỉ là source/reference; không stage hoặc sửa trực tiếp các thư mục đó.
