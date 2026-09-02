"""One-room Pipecat worker for local LiveKit voice conversations."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import time

from app.assistant.auth_controller import VoiceAuthChallengeController
from app.assistant.auth_session import AuthSession
from app.assistant.agents.conversation import ConversationSupervisor
from app.assistant.config import LocalAgentSettings, OpenAISettings
from app.assistant.contracts import AuthDecision, AuthState
from app.assistant.events import VoiceAgentState
from app.assistant.events_bridge import SessionEventSink
from app.assistant.pipecat_runtime.auth_router import AuthRouterProcessor
from app.assistant.pipecat_runtime.conversation import PipecatConversationProcessor
from app.assistant.pipecat_runtime.edge_tts_service import StreamingTTSPipecatService
from app.assistant.pipecat_runtime.session import PipecatSessionDescriptor
from app.assistant.pipecat_runtime.smart_turn import SmartTurnGateProcessor
from app.assistant.pipecat_runtime.stt import LocalWhisperSTTService
from app.assistant.providers.tts_factory import build_tts_stack
from app.assistant.providers.openai_llm import OpenAIResponsesProvider
from app.assistant.providers.whisper_local import LocalWhisperProvider
from app.assistant.tools.calendar import MockCalendarProvider
from app.assistant.voice_service import build_account_voice_auth_gate
from app.assistant_gateway.main import GatewaySettings
from app.assistant_gateway.store import GatewayStore

logger = logging.getLogger(__name__)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def log_runtime_versions() -> None:
    logger.info(
        "pipecat_runtime_versions python=%s pipecat=%s livekit=%s livekit_agents=%s numpy=%s faster_whisper=%s",
        __import__("platform").python_version(),
        _version("pipecat-ai"),
        _version("livekit"),
        _version("livekit-agents"),
        _version("numpy"),
        _version("faster-whisper"),
    )


async def cancel_if_participant_absent(
    *,
    participant_joined: asyncio.Event,
    runner: object,
    timeout_seconds: float,
    session_id: str,
) -> bool:
    """Stop a dispatched job when its signed browser participant never joins."""

    try:
        await asyncio.wait_for(participant_joined.wait(), timeout=timeout_seconds)
        return False
    except TimeoutError:
        logger.warning(
            "pipecat_session_join_timeout session_id=%s timeout_seconds=%s",
            session_id,
            timeout_seconds,
        )
        # WorkerRunner is deliberately kept as an object here so this helper
        # stays testable without importing Pipecat at module import time.
        await runner.cancel("expected browser did not join")  # type: ignore[attr-defined]
        return True


async def run_pipecat_session(
    descriptor: PipecatSessionDescriptor,
    bot_token: str,
    *,
    settings: LocalAgentSettings | None = None,
) -> None:
    """Connect one local Pipecat pipeline to one signed room descriptor."""

    # Adapted from pipecat/examples/transports/transports-livekit.py and
    # pipecat/examples/getting-started/06a-voice-agent-local.py.
    # Pipecat version: 1.8.1.
    active = settings or LocalAgentSettings.from_environment()
    if not active.conversation_enabled:
        raise RuntimeError(
            "VOICE_AGENT_CONVERSATION_ENABLED=true is required for PIPECAT runtime."
        )
    log_runtime_versions()

    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
        from pipecat.workers.runner import WorkerRunner
    except ImportError as error:
        raise RuntimeError(
            'Install the pinned Pipecat extra with `python -m pip install -e ".[pipecat]".`'
        ) from error

    gateway = GatewaySettings.from_environment()
    if gateway.mcp_provider != "mock" or not gateway.mock_calendar_enabled:
        raise RuntimeError("PIPECAT local mode requires MCP_PROVIDER=mock and MOCK_CALENDAR_ENABLED=true.")
    store = GatewayStore(gateway.database_path)
    store.initialize()
    calendar = MockCalendarProvider(store)
    gate = await asyncio.to_thread(build_account_voice_auth_gate, descriptor.user_id)
    auth_controller = VoiceAuthChallengeController(
        gate=gate,
        target_identity_id=descriptor.voice_identity_id,
        session=AuthSession(ttl=active.auth_ttl),
        capture_seconds=5.0,
    )

    from pipecat.frames.frames import InterruptionFrame

    transport = LiveKitTransport(
        active.livekit_url,
        bot_token,
        descriptor.room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16_000,
            audio_in_channels=1,
            audio_in_stream_on_start=True,
            audio_out_enabled=True,
            audio_out_sample_rate=48_000,
            audio_out_channels=1,
            audio_out_10ms_chunks=2,
        ),
    )
    sink = SessionEventSink(transport.send_message, descriptor.participant_identity)
    conversation = ConversationSupervisor(
        llm=OpenAIResponsesProvider(OpenAISettings.from_environment()),
        calendar=calendar,
        account_id=descriptor.user_id,
    )
    conversation_processor: PipecatConversationProcessor
    auth_expiry_task: asyncio.Task | None = None

    async def on_auth_decision(decision: AuthDecision) -> None:
        nonlocal auth_expiry_task
        if auth_expiry_task is not None and not auth_expiry_task.done():
            auth_expiry_task.cancel()
            await asyncio.gather(auth_expiry_task, return_exceptions=True)
        auth_expiry_task = None
        await sink.send_auth(decision)
        if decision.may_use_private_tools and decision.expires_at is not None:
            expires_at = decision.expires_at

            async def publish_expiry() -> None:
                delay = max(0.0, expires_at.timestamp() - time.time())
                await asyncio.sleep(delay)
                expired = auth_controller.current()
                if expired.state is AuthState.SESSION_EXPIRED:
                    await sink.send_auth(expired)

            auth_expiry_task = asyncio.create_task(
                publish_expiry(), name=f"auth-expiry-{descriptor.session_id}"
            )

    async def on_auth_error(message: str) -> None:
        await sink.send_event("state", state=VoiceAgentState.ERROR, message=message)

    auth_router = AuthRouterProcessor(
        controller=auth_controller,
        participant_id=descriptor.participant_identity,
        on_decision=on_auth_decision,
        on_error=on_auth_error,
    )

    async def on_audio_started(turn_id: str) -> None:
        await conversation_processor.audio_started(turn_id)

    async def on_audio_completed(turn_id: str) -> None:
        await conversation_processor.audio_completed(turn_id)

    async def on_audio_error(turn_id: str, message: str) -> None:
        await conversation_processor.audio_error(turn_id, message)

    tts_provider, tts_fallback = build_tts_stack(
        provider_name=active.tts_provider,
        fallback_name=active.tts_fallback_provider,
        zerotts_model=active.zerotts_model,
        zerotts_voice=active.zerotts_voice,
        zerotts_cache_dir=active.zerotts_cache_dir,
        queue_max_chunks=active.tts_queue_max_chunks,
        edge_voice=active.edge_tts_voice,
    )
    if active.zerotts_warmup or tts_provider.name == "edge":
        try:
            logger.info("pipecat_tts_warmup provider=%s", tts_provider.name)
            await tts_provider.warmup()
        except Exception:
            if tts_fallback is None:
                raise
            logger.warning(
                "pipecat_tts_primary_unavailable primary=%s fallback=%s",
                tts_provider.name,
                tts_fallback.name,
                exc_info=True,
            )
            await tts_fallback.warmup()
            await tts_provider.close()
            tts_provider, tts_fallback = tts_fallback, None
    logger.info("pipecat_tts_ready provider=%s", tts_provider.name)
    tts = StreamingTTSPipecatService(
        tts_provider,
        fallback_provider=tts_fallback,
        timeout_seconds=active.tts_timeout_seconds,
        on_audio_started=on_audio_started,
        on_audio_completed=on_audio_completed,
        on_error=on_audio_error,
    )
    conversation_processor = PipecatConversationProcessor(
        supervisor=conversation,
        auth=auth_controller.current,
        events=sink,
        tts_enabled=True,
        max_sentence_chars=active.tts_max_sentence_chars,
    )
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.7,
                start_secs=active.vad_min_speech_seconds,
                stop_secs=active.vad_silence_seconds,
                min_volume=0.6,
            )
        ),
        audio_idle_timeout=1.0,
    )
    smart_turn = SmartTurnGateProcessor(
        analyzer=LocalSmartTurnAnalyzerV3(
            smart_turn_model_path=active.smart_turn_model_path,
            cpu_count=active.smart_turn_cpu_count,
            params=SmartTurnParams(
                stop_secs=active.vad_silence_seconds,
                pre_speech_ms=active.smart_turn_pre_speech_ms,
                max_duration_secs=min(active.vad_maximum_seconds, 8.0),
            ),
        ),
        max_wait_seconds=active.smart_turn_max_wait_seconds,
    )
    whisper_provider = LocalWhisperProvider(
        model_name=active.whisper_model, device=active.whisper_device
    )
    logger.info(
        "pipecat_local_model_warmup session_id=%s model=%s device=%s",
        descriptor.session_id,
        active.whisper_model,
        active.whisper_device,
    )
    await asyncio.to_thread(whisper_provider.warmup)
    stt = LocalWhisperSTTService(
        whisper_provider,
        sample_rate=16_000,
        language="vi",
    )
    @stt.event_handler("on_error")
    async def on_stt_error(_stt, _error) -> None:
        await conversation_processor.report_stt_error()

    pipeline = Pipeline(
        [
            transport.input(),
            auth_router,
            vad,
            smart_turn,
            stt,
            conversation_processor,
            tts,
            transport.output(),
        ]
    )
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(_observer, latency_seconds: float) -> None:
        logger.info(
            "pipecat_user_to_bot_latency session_id=%s seconds=%.3f",
            descriptor.session_id,
            latency_seconds,
        )

    worker = PipelineWorker(
        pipeline,
        name=f"pipecat-{descriptor.session_id}",
        params=PipelineParams(
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=48_000,
            enable_metrics=True,
            enable_usage_metrics=False,
        ),
        observers=[latency_observer],
        enable_rtvi=False,
        idle_timeout_secs=None,
    )
    participant_joined = asyncio.Event()

    async def send_current_status() -> None:
        await sink.send_auth(auth_controller.current())

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(_transport, participant_id: str) -> None:
        if participant_id == descriptor.participant_identity:
            participant_joined.set()
            await send_current_status()

    @transport.event_handler("on_participant_connected")
    async def on_participant_connected(_transport, participant_id: str) -> None:
        if participant_id == descriptor.participant_identity:
            participant_joined.set()
            await send_current_status()

    @transport.event_handler("on_data_received")
    async def on_data_received(_transport, data: bytes, participant_id: str | None) -> None:
        if participant_id != descriptor.participant_identity:
            return
        try:
            text = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            return
        action = text
        payload = None
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and isinstance(payload.get("action"), str):
                action = payload["action"]
        except json.JSONDecodeError:
            pass
        if action == "request_private_mode":
            await auth_router.queue_frame(InterruptionFrame())
            await sink.send_auth(auth_router.request_private_mode())
        elif action == "cancel_private_mode":
            await auth_router.queue_frame(InterruptionFrame())
            await sink.send_auth(auth_router.cancel_private_mode())
        elif action == "retry" and isinstance(payload, dict) and isinstance(payload.get("turn_id"), str):
            await conversation_processor.retry(payload["turn_id"])

    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(_transport, participant_id: str) -> None:
        if participant_id == descriptor.participant_identity:
            # A reconnect receives a new descriptor and a new AuthSession; this
            # worker never keeps an authenticated state after disconnect.
            await runner.cancel()

    join_watchdog: asyncio.Task | None = None

    try:
        await runner.add_workers(worker)
        join_watchdog = asyncio.create_task(
            cancel_if_participant_absent(
                participant_joined=participant_joined,
                runner=runner,
                timeout_seconds=active.participant_join_timeout_seconds,
                session_id=descriptor.session_id,
            ),
            name=f"join-watchdog-{descriptor.session_id}",
        )
        logger.info(
            "pipecat_session_ready session_id=%s room=%s participant=%s",
            descriptor.session_id,
            descriptor.room_name,
            descriptor.participant_identity,
        )
        await runner.run(auto_end=False)
    finally:
        if join_watchdog is not None and not join_watchdog.done():
            join_watchdog.cancel()
            await asyncio.gather(join_watchdog, return_exceptions=True)
        if auth_expiry_task is not None and not auth_expiry_task.done():
            auth_expiry_task.cancel()
            await asyncio.gather(auth_expiry_task, return_exceptions=True)
        try:
            await tts.cleanup()
        except Exception:
            logger.exception("pipecat_tts_cleanup_failed")
        gate.close()
