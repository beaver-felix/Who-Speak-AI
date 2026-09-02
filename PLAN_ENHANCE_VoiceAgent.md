# Plan: Tối ưu Voice Chat realtime trên LiveKit

## Tóm tắt

Giữ LiveKit làm transport chính. Không thay bằng WebSocket/WebRTC trực tiếp của `speech-to-speech`.

Học từ `speech-to-speech` ở ba điểm:

- VAD streaming và kết thúc lượt nói nhanh hơn.
- OpenAI trả response theo streaming.
- Edge-TTS phát theo từng câu thay vì chờ toàn bộ response.

Auth Gate vẫn là lớp độc lập và luôn chạy trước conversation pipeline.

```text
LiveKit audio
→ Auth Router
→ Silero/Smart Turn VAD
→ Local Whisper
→ OpenAI streaming
→ Edge-TTS từng câu
→ LiveKit AudioSource
→ Browser chat + audio
```

Không sửa Streamlit SV trong kế hoạch này.

## 1. Sửa correctness trước latency

Thay flow hiện tại bằng `TurnController` duy nhất cho mỗi participant:

```text
ROOM_JOIN
→ GUEST
→ user requests private mode
→ AUTH_PENDING
→ chỉ VoiceAuthGate nhận audio
→ AUTHENTICATED hoặc GUEST
→ conversation được bật
```

Khi `AUTH_PENDING`:

- Không đưa audio vào VAD/ASR/LLM.
- Xóa/reset conversation buffer trước và sau challenge.
- Không tạo chat turn từ câu nói dùng để xác thực.
- Không gửi audio challenge tới OpenAI hoặc TTS.

Khi reconnect:

- Hủy mọi turn đang chạy.
- Xóa audio playback queue.
- Reset UI về `GUEST`.
- Không khôi phục `AUTHENTICATED` từ frontend state.

Các thành phần liên quan:

- `app/assistant/livekit_agent.py`
- `app/assistant/auth_controller.py`
- `app/assistant/audio_turn.py`

## 2. Đo latency làm baseline

Thêm structured timing nội bộ theo `room_id`, participant hash và `turn_id`:

```text
speech_start
speech_end
vad_final
transcript_start
transcript_final
llm_first_token
llm_first_phrase
response_final
tts_first_audio
browser_playback_start
playback_completed
```

Không log:

- raw audio;
- transcript đầy đủ trong production log;
- embedding;
- HE key/context;
- API key;
- matcher token.

Đo riêng:

```text
speech_end → transcript_final
transcript_final → llm_first_phrase
llm_first_phrase → tts_first_audio
speech_end → browser_playback_start
```

Mục tiêu sau tối ưu:

- `tts_first_audio` bắt đầu trước khi toàn bộ câu trả lời hoàn tất.
- Giảm ít nhất 30% p50 thời gian từ `speech_end` đến `browser_playback_start`.
- Không giảm độ chính xác ASR và không tạo duplicate turn.

## 3. VAD và turn detection

Đợt đầu không thay model ASR. Chỉ thay RMS VAD hiện tại bằng adapter streaming có interface ổn định:

```python
class SpeechTurnDetector(Protocol):
    def append(self, pcm: np.ndarray, sample_rate: int) -> list[TurnEvent]:
        ...
```

Event nội bộ:

```text
speech_started
speech_activity
speech_ended
turn_reopened
turn_final
```

Behavior mặc định:

- Nhận audio frame nhỏ 20–40ms.
- Phát hiện speech nhanh.
- Dùng khoảng silence ngắn hơn 0.8 giây hiện tại.
- Có giới hạn thời gian tối đa cho một turn.
- Không kết thúc ngay nếu người dùng chỉ tạm dừng giữa câu.
- Cho phép reopen turn nếu người dùng nói tiếp trong khoảng grace period.

Ưu tiên dùng Silero VAD; Smart Turn chỉ bật sau khi kiểm tra model cache và memory trên máy Mac.

Nếu Smart Turn gây lỗi hoặc tốn tài nguyên, fallback về VAD streaming cơ bản, không quay lại xử lý RMS toàn đoạn trong cùng turn.

## 4. Local ASR

Giữ `LocalWhisperProvider` và model hiện tại trong đợt đầu để giảm rủi ro.

Thay đổi behavior:

- Load và warm-up Whisper khi Agent khởi động.
- Không tải model ở lượt nói đầu tiên.
- Giữ model singleton trong mỗi Agent process.
- Không chạy nhiều lượt ASR đồng thời trên cùng worker.
- Vẫn chỉ gửi transcript final tới LLM.

UI không hiển thị partial ASR ở phiên bản đầu. Thay vào đó:

```text
Listening...
→ Transcribing...
→ transcript final
```

Sau khi pipeline ổn định mới benchmark riêng:

- Faster Whisper hiện tại;
- Parakeet TDT;
- MLX/Apple Silicon nếu máy hỗ trợ.

Không tự động đổi model trong cùng phase vì có thể làm thay đổi chất lượng tiếng Việt.

## 5. OpenAI streaming

Mở rộng interface LLM:

```python
class LLMProvider(Protocol):
    async def respond_stream(
        self,
        transcript: str,
        *,
        allowed_tools: set[str],
        auth_context: AuthDecision,
    ) -> AsyncIterator[LLMEvent]:
        ...
```

Event:

```text
text_delta
phrase_completed
response_completed
provider_error
```

Behavior:

- Chỉ gửi transcript tối thiểu tới OpenAI.
- Không gửi raw audio, embedding, HE context hoặc AuthDecision nội bộ.
- Giữ `store=False`.
- Giới hạn context/prompt để giảm latency.
- LLM không được quyết định AuthDecision hoặc tự thêm tool.
- Nếu OpenAI lỗi, giữ transcript trong UI và hiển thị retry.

Frontend có thể cập nhật câu trả lời theo `phrase_completed`, không cần hiển thị từng token để tránh chữ nhảy quá nhiều.

## 6. Edge-TTS theo từng câu

Giữ Edge-TTS trong phase đầu, nhưng không gọi một lần cho toàn bộ response.

Flow mới:

```text
OpenAI text stream
→ sentence buffer
→ sentence 1 hoàn chỉnh
→ Edge-TTS sentence 1
→ decode/publish
→ sentence 2 tiếp tục xử lý
```

Tạo adapter có behavior tương tự:

```python
class StreamingTTSProvider(Protocol):
    async def synthesize_sentences(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        ...
```

Yêu cầu:

- Phát câu đầu tiên ngay khi TTS câu đầu hoàn tất.
- Không gom toàn bộ MP3 của cả response.
- Giữ thứ tự câu.
- Một câu TTS lỗi không làm thay đổi AuthDecision.
- Nếu TTS lỗi, text response vẫn hiển thị `Audio unavailable`.

Lưu ý: Edge-TTS là network provider, không phải TTS local hoàn toàn. Raw microphone vẫn không gửi ra ngoài, nhưng text cần đọc sẽ được gửi tới dịch vụ Edge-TTS. Nếu yêu cầu privacy nghiêm ngặt hơn, local streaming TTS sẽ là phase sau.

Các thành phần liên quan:

- `app/assistant/providers/openai_llm.py`
- `app/assistant/providers/edge_tts_local.py`
- `app/assistant/livekit_tts.py`

## 7. LiveKit audio playback

Giữ một assistant audio track duy nhất trong room.

Thêm playback controller:

```text
TTS sentence bytes
→ decode PCM
→ playback queue
→ LiveKit AudioSource
```

Yêu cầu:

- Không tạo track mới cho mỗi câu.
- Queue audio theo `turn_id` và sequence number.
- Browser chỉ attach một audio element cho assistant track.
- Có prebuffer nhỏ trước khi phát.
- Phát `speaking` khi audio frame đầu tiên được capture.
- Phát `completed` sau khi queue và LiveKit playout hoàn tất.
- Disconnect phải xóa queue và detach audio element.

Barge-in chưa bật mặc định trong phase đầu. Khi người dùng nói trong lúc Agent phát:

```text
đánh dấu interrupted
→ không tạo concurrent turn
→ giữ session ổn định
```

Sau khi streaming cơ bản ổn định mới thêm cancellation và dừng audio hiện tại.

## 8. UI/UX conversation

Frontend dùng `TurnCard`, không hiển thị transcript và response như hai message rời không có trạng thái.

Một lượt hiển thị:

```text
You
Listening…
→ Transcribing…
→ Lịch hôm nay của tôi có gì?

Who Speak AI
Thinking…
→ Bạn có một cuộc họp demo lúc 14 giờ.

Speaking
→ Played
```

State UI:

```text
idle
listening
transcribing
thinking
speaking
completed
interrupted
error
```

Behavior:

- Tạo placeholder `Listening…` ngay khi VAD phát hiện speech.
- Thay placeholder bằng transcript final.
- Tạo placeholder `Thinking…` ngay sau transcript.
- Cập nhật assistant text theo phrase, sau đó commit bản final.
- Hiển thị `Speaking`, `Played` hoặc `Audio unavailable`.
- Lỗi nằm trong đúng turn.
- Retry chỉ dùng transcript final tạm thời, không dùng raw audio.
- Không auto-scroll nếu người dùng đang đọc message cũ.
- Hiện nút `New response` nếu có turn mới.
- Voice meter dùng RMS thật hoặc hiển thị trạng thái mic rõ ràng; không dùng animation giả liên tục.

Mở rộng `conversation.ts` để hỗ trợ:

```text
turn_id
sequence
message_id
partial/final
audio_state
error
tool badge
```

Event mới tối thiểu:

```text
turn_started
state
transcript_final
assistant_phrase
assistant_response_final
audio_started
audio_completed
interrupted
error
```

Không hiển thị similarity score, embedding, HE material, matcher token hoặc API key.

## 9. Testing và simulation

### Unit tests

- `AUTH_PENDING` không tạo ASR/LLM turn.
- Room join luôn bắt đầu ở `GUEST`.
- VAD phát hiện speech/silence đúng.
- Turn reopen không tạo duplicate.
- Event cùng `message_id` chỉ xử lý một lần.
- Event cũ hoặc sai sequence bị bỏ qua.
- LLM streaming giữ đúng thứ tự phrase.
- TTS sentence queue giữ đúng thứ tự.
- TTS lỗi vẫn giữ text response.
- OpenAI lỗi tạo retryable error.
- Guest không gọi calendar.
- Auth hết hạn khóa calendar.
- Reconnect reset auth và conversation.

### Integration tests

- Audio frame → VAD → final transcript.
- Transcript → OpenAI stream → assistant phrase.
- Assistant phrase → Edge-TTS → LiveKit audio queue.
- Auth challenge không xuất hiện trong chat.
- Mock Calendar vẫn có `provider=mock`, `demo=true`.
- Browser nhận `audio_started` trước `audio_completed`.
- Không tạo nhiều audio element hoặc assistant track.
- OpenAI payload không chứa raw audio/embedding/HE.

### LiveKit E2E

- Join room.
- Bật microphone.
- Start voice challenge.
- Xác thực thành công/thất bại.
- Hỏi câu general.
- Hỏi calendar khi guest.
- Xác thực rồi hỏi calendar.
- Nhận user transcript.
- Nhận assistant response.
- Nghe được TTS.
- Disconnect/reconnect.
- TTS/OpenAI/Whisper failure.
- Nói trong lúc Agent đang speaking.

### Simulations

Cập nhật:

- `app/assistant/simulations/description.md`
- `app/assistant/simulations/risks.yaml`
- `app/assistant/simulations/authored.yaml`

Thêm coverage cho:

- challenge audio không lọt vào conversation;
- transcript/assistant event ordering;
- OpenAI streaming timeout;
- TTS sentence failure;
- audio unavailable nhưng text còn;
- interruption không tạo duplicate;
- stale event/duplicate event;
- guest calendar denial;
- authenticated mock calendar;
- expiry/reconnect;
- prompt injection;
- sensitive-data disclosure.

Trước khi chạy simulation:

```bash
lk agent simulate --help
```

Sau đó assemble bằng `build_scenarios.py assemble --strict`. Simulation chỉ kiểm tra behavior/policy; latency audio và RawNet3 vẫn cần fixture test riêng.

## Thứ tự triển khai

1. Thêm latency measurement.
2. Tách hoàn toàn Auth Gate khỏi conversation pipeline.
3. Tạo `TurnController` và state transition rõ ràng.
4. Warm-up Whisper.
5. Cải thiện VAD/turn detection.
6. Xây `TurnCard` và placeholder UI.
7. Thêm OpenAI streaming.
8. Thêm Edge-TTS theo từng câu.
9. Thêm LiveKit playback queue.
10. Chạy integration/E2E và simulations.
11. Chỉ sau đó mới làm barge-in, partial ASR và local streaming TTS.

## Acceptance criteria

Flow đạt yêu cầu khi:

```text
Join room
→ Guest
→ Start voice challenge
→ Authenticated hoặc Guest
→ Nói câu hỏi
→ Listening placeholder xuất hiện ngay
→ Transcript final xuất hiện
→ Thinking xuất hiện
→ Assistant phrase đầu tiên xuất hiện
→ Audio bắt đầu trước khi toàn bộ response hoàn tất
→ Played hoặc Audio unavailable
```

Các invariant bắt buộc:

- Auth challenge không bao giờ thành chat message.
- Guest không thể gọi private calendar.
- LLM không thể nâng quyền.
- Reconnect không bypass voice verification.
- TTS lỗi không làm thay đổi auth.
- Không gửi raw audio/embedding/HE key tới OpenAI.
- Không tạo duplicate turn hoặc duplicate audio track.

## Assumptions

- Giữ React + Vite và LiveKit local hiện tại.
- Giữ OpenAI làm LLM provider.
- Giữ Whisper local và Edge-TTS trong phase đầu.
- ASR chỉ hiển thị final transcript; chưa hiển thị partial ASR.
- Assistant text được cập nhật theo phrase hoàn chỉnh, không hiển thị từng token.
- Mock Calendar tiếp tục là provider duy nhất.
- Chưa bật barge-in hoàn chỉnh trong đợt đầu.
- Không sửa Streamlit baseline.
- Mọi API LiveKit cụ thể phải được kiểm tra lại theo phiên bản SDK đang cài trước khi triển khai.
