from __future__ import annotations

import asyncio
import base64
import json
from datetime import timedelta

import numpy as np
import pytest
import httpx
from fastapi.testclient import TestClient

from app.assistant.contracts import AuthDecision, AuthState
from app.assistant.auth_controller import VoiceAuthChallengeController
from app.assistant.auth_session import AuthSession
from app.assistant.events_bridge import SessionEventSink
from app.assistant.pipecat_runtime.auth_router import AuthRouterProcessor
from app.assistant.pipecat_runtime.conversation import PipecatConversationProcessor
from app.assistant.pipecat_runtime.runtime import cancel_if_participant_absent
from app.assistant.pipecat_runtime.session import (
    PipecatSessionDescriptor,
    SessionDescriptorError,
    sign_session_descriptor,
    verify_session_descriptor,
)
from app.assistant.pipecat_runtime.smart_turn import SmartTurnGateProcessor
from app.assistant.pipecat_runtime.stt import LocalWhisperSTTService
from app.assistant_gateway.main import GatewaySettings, create_app
from app.assistant_gateway.store import VoiceProfile
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    TranscriptionFrame,
    UserAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.base_object import BaseObject
from voiceauth.matching import VerificationResult


def descriptor() -> PipecatSessionDescriptor:
    return PipecatSessionDescriptor(
        session_id="session-1",
        room_name="room-1",
        participant_identity="web-1",
        user_id="user-1",
        voice_identity_id="voice-1",
        display_name="An",
        issued_at=1_000,
    )


def test_smart_turn_gate_holds_incomplete_vad_stop_until_next_complete_stop() -> None:
    class Analyzer:
        def __init__(self) -> None:
            self.predictions = [EndOfTurnState.INCOMPLETE, EndOfTurnState.COMPLETE]
            self.audio: list[tuple[bytes, bool]] = []
            self.cleared = 0

        def set_sample_rate(self, sample_rate: int) -> None:
            assert sample_rate == 16_000

        def update_vad_start_secs(self, _value: float) -> None:
            pass

        def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
            self.audio.append((buffer, is_speech))
            return EndOfTurnState.INCOMPLETE

        async def analyze_end_of_turn(self):
            return self.predictions.pop(0), None

        def clear(self) -> None:
            self.cleared += 1

        async def cleanup(self) -> None:
            pass

    async def exercise() -> list[object]:
        analyzer = Analyzer()
        processor = SmartTurnGateProcessor(analyzer=analyzer, max_wait_seconds=10)
        await BaseObject.setup(processor, TaskManager())
        emitted: list[object] = []

        async def capture(frame, _direction=FrameDirection.DOWNSTREAM):
            emitted.append(frame)

        processor.push_frame = capture  # type: ignore[method-assign]
        audio = InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16_000, num_channels=1)
        await processor.process_frame(VADUserStartedSpeakingFrame(0.384), FrameDirection.DOWNSTREAM)
        await processor.process_frame(audio, FrameDirection.DOWNSTREAM)
        await processor.process_frame(VADUserStoppedSpeakingFrame(0.2), FrameDirection.DOWNSTREAM)
        assert not any(isinstance(frame, VADUserStoppedSpeakingFrame) for frame in emitted)

        # Resuming speech cancels the bounded fallback. The next stop is
        # evaluated as one continuous user turn and is released downstream.
        await processor.process_frame(VADUserStartedSpeakingFrame(0.384), FrameDirection.DOWNSTREAM)
        await processor.process_frame(audio, FrameDirection.DOWNSTREAM)
        await processor.process_frame(VADUserStoppedSpeakingFrame(0.2), FrameDirection.DOWNSTREAM)
        await processor.cleanup()
        return emitted

    emitted = asyncio.run(exercise())
    assert sum(isinstance(frame, VADUserStoppedSpeakingFrame) for frame in emitted) == 1
    assert sum(isinstance(frame, InputAudioRawFrame) for frame in emitted) == 2


def test_smart_turn_gate_force_stops_after_bounded_wait() -> None:
    class Analyzer:
        def set_sample_rate(self, _sample_rate: int) -> None:
            pass

        def update_vad_start_secs(self, _value: float) -> None:
            pass

        def append_audio(self, _buffer: bytes, _is_speech: bool) -> EndOfTurnState:
            return EndOfTurnState.INCOMPLETE

        async def analyze_end_of_turn(self):
            return EndOfTurnState.INCOMPLETE, None

        def clear(self) -> None:
            pass

        async def cleanup(self) -> None:
            pass

    async def exercise() -> list[object]:
        processor = SmartTurnGateProcessor(analyzer=Analyzer(), max_wait_seconds=0.001)
        await BaseObject.setup(processor, TaskManager())
        emitted: list[object] = []

        async def capture(frame, _direction=FrameDirection.DOWNSTREAM):
            emitted.append(frame)

        processor.push_frame = capture  # type: ignore[method-assign]
        await processor.process_frame(VADUserStartedSpeakingFrame(0.384), FrameDirection.DOWNSTREAM)
        await processor.process_frame(VADUserStoppedSpeakingFrame(0.2), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.01)
        await processor.cleanup()
        return emitted

    emitted = asyncio.run(exercise())
    assert sum(isinstance(frame, VADUserStoppedSpeakingFrame) for frame in emitted) == 1


def test_pipecat_descriptor_is_signed_and_fresh() -> None:
    secret = "s" * 32
    token = sign_session_descriptor(descriptor(), secret)
    assert verify_session_descriptor(token, secret, now=1_050).room_name == "room-1"

    _, signature = token.split(".")
    tampered = base64.urlsafe_b64encode(b'{"session_id":"forged"}').rstrip(b"=").decode() + "." + signature
    with pytest.raises(SessionDescriptorError, match="signature"):
        verify_session_descriptor(tampered, secret, now=1_050)
    with pytest.raises(SessionDescriptorError, match="expired"):
        verify_session_descriptor(token, secret, now=1_200)


def test_auth_router_consumes_pending_challenge_audio_before_stt() -> None:
    class Controller:
        def __init__(self) -> None:
            self.decision = AuthDecision(AuthState.AUTH_PENDING)
            self.samples = None

        def current(self):
            return self.decision

        def append_pcm(self, samples, *, sample_rate):
            self.samples = (samples, sample_rate)
            self.decision = AuthDecision(AuthState.GUEST)
            return self.decision

        def request_private_mode(self):
            return self.decision

        def cancel(self):
            return AuthDecision(AuthState.GUEST)

    controller = Controller()
    decisions = []
    errors = []

    async def on_decision(value):
        decisions.append(value)

    async def on_error(value):
        errors.append(value)

    router = AuthRouterProcessor(
        controller=controller,
        participant_id="web-1",
        on_decision=on_decision,
        on_error=on_error,
    )
    router.set_conversation_ready()
    frame = UserAudioRawFrame(
        audio=np.ones(320, dtype=np.int16).tobytes(), sample_rate=16_000, num_channels=1, user_id="web-1"
    )
    asyncio.run(router.process_frame(frame, FrameDirection.DOWNSTREAM))

    assert router.challenge_frames_consumed == 1
    assert controller.samples[1] == 16_000
    assert len(controller.samples[0]) == 320
    assert decisions[0].state is AuthState.GUEST
    assert errors == []


def test_auth_router_matcher_failure_returns_to_guest_and_reports_error() -> None:
    class Controller:
        def __init__(self) -> None:
            self.decision = AuthDecision(AuthState.AUTH_PENDING)

        def current(self):
            return self.decision

        def append_pcm(self, _samples, *, sample_rate):
            assert sample_rate == 16_000
            raise RuntimeError("matcher unavailable")

        def request_private_mode(self):
            return self.decision

        def cancel(self):
            self.decision = AuthDecision(AuthState.GUEST)
            return self.decision

    controller = Controller()
    decisions: list[AuthDecision] = []
    errors: list[str] = []

    async def on_decision(value):
        decisions.append(value)

    async def on_error(value):
        errors.append(value)

    router = AuthRouterProcessor(
        controller=controller,
        participant_id="web-1",
        on_decision=on_decision,
        on_error=on_error,
    )
    router.set_conversation_ready()
    frame = UserAudioRawFrame(
        audio=np.ones(320, dtype=np.int16).tobytes(), sample_rate=16_000, num_channels=1, user_id="web-1"
    )
    asyncio.run(router.process_frame(frame, FrameDirection.DOWNSTREAM))

    assert controller.current().state is AuthState.GUEST
    assert [decision.state for decision in decisions] == [AuthState.GUEST]
    assert errors == ["Voice verification could not be completed. Please try again."]


def test_auth_router_drops_conversation_audio_until_engine_is_ready() -> None:
    class Controller:
        def current(self):
            return AuthDecision(AuthState.GUEST)

    async def no_op(_value) -> None:
        return None

    router = AuthRouterProcessor(
        controller=Controller(),
        participant_id="web-1",
        on_decision=no_op,
        on_error=no_op,
    )
    frame = UserAudioRawFrame(
        audio=np.ones(320, dtype=np.int16).tobytes(), sample_rate=16_000, num_channels=1, user_id="web-1"
    )

    asyncio.run(router.process_frame(frame, FrameDirection.DOWNSTREAM))

    assert router.conversation_ready is False
    assert router.startup_frames_dropped == 1


def test_auth_router_drops_audio_after_capture_until_explicit_resume() -> None:
    class Gate:
        def verify(self, _recording, *, target_identity_id):
            return VerificationResult(True, target_identity_id, "An", 0.9, 1, target_identity_id)

    async def exercise() -> tuple[list[object], AuthRouterProcessor, list[str]]:
        controller = VoiceAuthChallengeController(
            gate=Gate(),
            target_identity_id="owner",
            session=AuthSession(ttl=timedelta(minutes=5)),
        )
        decisions: list[AuthDecision] = []
        progress_phases: list[str] = []

        async def on_decision(value):
            decisions.append(value)

        async def on_error(_value):
            pass

        async def on_progress(_decision, status):
            progress_phases.append(status.phase.value)

        router = AuthRouterProcessor(
            controller=controller,
            participant_id="web-1",
            on_decision=on_decision,
            on_error=on_error,
            on_auth_progress=on_progress,
            resume_guard_seconds=0.0,
        )
        router.set_conversation_ready()
        emitted: list[object] = []

        async def capture(frame, _direction=FrameDirection.DOWNSTREAM):
            emitted.append(frame)

        router.push_frame = capture  # type: ignore[method-assign]
        challenge = UserAudioRawFrame(
            audio=np.ones(16_000 * 5, dtype=np.int16).tobytes(),
            sample_rate=16_000,
            num_channels=1,
            user_id="web-1",
        )
        extra_speech = UserAudioRawFrame(
            audio=np.ones(320, dtype=np.int16).tobytes(),
            sample_rate=16_000,
            num_channels=1,
            user_id="web-1",
        )
        controller.request_private_mode()
        await router.process_frame(challenge, FrameDirection.DOWNSTREAM)
        await router.process_frame(extra_speech, FrameDirection.DOWNSTREAM)
        assert controller.phase.value == "waiting_for_resume"
        router.resume_conversation()
        await router.process_frame(extra_speech, FrameDirection.DOWNSTREAM)
        return emitted, router, progress_phases

    emitted, router, progress_phases = asyncio.run(exercise())
    assert len(emitted) == 1
    assert isinstance(emitted[0], UserAudioRawFrame)
    assert router.post_challenge_frames_dropped == 1
    assert "processing" in progress_phases


def test_unjoined_pipecat_session_is_cancelled_after_timeout() -> None:
    class Runner:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        async def cancel(self, reason: str) -> None:
            self.reasons.append(reason)

    runner = Runner()
    was_cancelled = asyncio.run(
        cancel_if_participant_absent(
            participant_joined=asyncio.Event(),
            runner=runner,
            timeout_seconds=0.001,
            session_id="session-test",
        )
    )

    assert was_cancelled is True
    assert runner.reasons == ["expected browser did not join"]


def test_joined_pipecat_session_is_not_cancelled() -> None:
    class Runner:
        async def cancel(self, _reason: str) -> None:
            raise AssertionError("joined session must not be cancelled")

    joined = asyncio.Event()
    joined.set()
    was_cancelled = asyncio.run(
        cancel_if_participant_absent(
            participant_joined=joined,
            runner=Runner(),
            timeout_seconds=0.001,
            session_id="session-test",
        )
    )

    assert was_cancelled is False


def test_new_speech_cancels_a_thinking_turn_before_opening_the_next_turn() -> None:
    class Supervisor:
        pass

    emitted: list[dict] = []

    async def send(message, _participant):
        emitted.append(json.loads(message))

    processor = PipecatConversationProcessor(
        supervisor=Supervisor(),
        auth=lambda: AuthDecision(AuthState.GUEST),
        events=SessionEventSink(send, "web-1"),
        tts_enabled=False,
    )

    async def exercise() -> None:
        cancelled = asyncio.Event()

        async def thinking() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        # A Pipeline normally initializes this; the isolated processor test
        # only needs Pipecat's managed task lifecycle.
        await BaseObject.setup(processor, TaskManager())
        processor._active_turn = "old-turn"
        processor._response_task = processor.create_task(thinking(), name="test-thinking")
        await asyncio.sleep(0)
        await processor.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(cancelled.wait(), timeout=0.2)

    asyncio.run(exercise())

    assert processor._active_turn not in {None, "old-turn"}
    assert any(
        event["type"] == "state" and event["state"] == "listening"
        for event in emitted
    )


def test_conversation_processor_drops_queued_transcription_when_auth_ingress_is_closed() -> None:
    class Supervisor:
        async def respond_stream(self, _transcript, *, auth):
            raise AssertionError("a blocked auth frame must never reach the supervisor")

    emitted: list[object] = []

    async def send(_message, _participant):
        pass

    processor = PipecatConversationProcessor(
        supervisor=Supervisor(),
        auth=lambda: AuthDecision(AuthState.GUEST),
        events=SessionEventSink(send, "web-1"),
        tts_enabled=False,
        conversation_ingress_allowed=lambda: False,
    )

    async def capture(frame, _direction=FrameDirection.DOWNSTREAM):
        emitted.append(frame)

    processor.push_frame = capture  # type: ignore[method-assign]
    frame = TranscriptionFrame(
        text="challenge audio must not become a transcript",
        user_id="web-1",
        timestamp="test",
    )

    asyncio.run(processor.process_frame(frame, FrameDirection.DOWNSTREAM))

    assert emitted == []


def test_local_whisper_adapter_emits_final_transcript_without_mlx_import() -> None:
    class Provider:
        def transcribe(self, recording, *, language):
            assert recording.sample_rate == 16_000
            assert language == "vi"
            return "xin chào"

    service = LocalWhisperSTTService(Provider())

    async def collect():
        return [frame async for frame in service.run_stt(np.zeros(320, dtype=np.int16).tobytes())]

    frames = asyncio.run(collect())
    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "xin chào"
    assert frames[0].finalized is True


def test_browser_event_sink_contains_no_voice_private_material() -> None:
    messages = []

    async def send(message, participant):
        messages.append((message, participant))

    sink = SessionEventSink(send, "web-1")
    asyncio.run(sink.send_event("state", turn_id="turn-1", state="thinking"))
    assert messages[0][1] == "web-1"
    assert "embedding" not in messages[0][0]
    assert "audio" not in messages[0][0]
    assert "HE" not in messages[0][0]


def test_pipecat_gateway_starts_supervisor_without_room_agent_dispatch(tmp_path, monkeypatch) -> None:
    settings = GatewaySettings(
        database_path=tmp_path / "gateway.db",
        livekit_url="ws://127.0.0.1:7880",
        livekit_api_key="devkey",
        livekit_api_secret="a-development-secret-with-at-least-32-bytes",
        mcp_provider="mock",
        mock_calendar_enabled=True,
        agent_runtime="pipecat",
        pipecat_supervisor_url="http://127.0.0.1:8021",
        pipecat_supervisor_secret="supervisor-secret-with-at-least-32-bytes",
    )
    app = create_app(settings)
    requests = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    with TestClient(app) as client:
        user = client.post(
            "/api/auth/register",
            json={
                "email": "pipecat@example.test",
                "password": "correct-horse-battery-staple",
                "display_name": "An",
            },
        ).json()
        app.state.store.save_voice_profile(VoiceProfile(user["id"], "context", "voice-1", "An"))
        response = client.post("/api/livekit/token")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["runtime"] == "pipecat"
    assert len(requests) == 1
    descriptor_token = requests[0][1]["json"]["descriptor"]
    verified = verify_session_descriptor(descriptor_token, settings.pipecat_supervisor_secret)
    assert verified.user_id == user["id"]
    encoded_payload = body["participant_token"].split(".")[1]
    jwt_payload = __import__("json").loads(base64.urlsafe_b64decode(encoded_payload + "=" * (-len(encoded_payload) % 4)))
    assert "roomConfig" not in jwt_payload
    assert settings.livekit_api_secret not in response.text
