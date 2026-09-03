"""Local LiveKit transport with an explicit 1:1 speaker-auth gate.

Authentication remains separate from the conversational pipeline. A browser
must send ``request_private_mode`` over the ``voice-auth`` topic before the
Agent buffers microphone audio for speaker verification.
"""

from __future__ import annotations

import asyncio
import json
import logging
from time import monotonic

import numpy as np

from app.assistant.auth_controller import VoiceAuthChallengeController
from app.assistant.auth_session import AuthSession
from app.assistant.agents.conversation import ConversationSupervisor
from app.assistant.config import LocalAgentSettings
from app.assistant.contracts import AuthChallengePhase, AuthDecision, AuthState
from app.assistant.events import VoiceAgentState, event_payload
from app.assistant.events_bridge import auth_status_payload
from app.assistant.livekit_tts import publish_mp3
from app.assistant.providers.edge_tts_local import EdgeTTSProvider
from app.assistant.providers.openai_llm import OpenAIResponsesProvider
from app.assistant.providers.whisper_local import LocalWhisperProvider
from app.assistant.streaming import SentenceBuffer
from app.assistant.tools.calendar import MockCalendarProvider
from app.assistant.turn_controller import ConversationTurnController
from app.assistant.voice_service import build_account_voice_auth_gate, build_persistent_voice_auth_gate
from app.assistant_gateway.main import GatewaySettings
from app.assistant_gateway.store import GatewayStore

logger = logging.getLogger(__name__)
AUTH_COMMAND_TOPIC = "voice-auth"
AUTH_STATUS_TOPIC = "voice-auth-status"
AGENT_EVENT_TOPIC = "voice-agent-event"
AGENT_COMMAND_TOPIC = "voice-agent-command"
REQUEST_PRIVATE_MODE = "request_private_mode"
CANCEL_PRIVATE_MODE = "cancel_private_mode"
RESUME_CONVERSATION = "resume_conversation"
CONTINUE_AS_GUEST = "continue_as_guest"
RETRY_VOICE = "retry_voice"
AGENT_PARTICIPANT_IDENTITY = "voice-auth-gate"


def require_livekit() -> None:
    try:
        import livekit.agents  # noqa: F401
        import livekit.rtc  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install voice_verification with the [agent] extra to run the realtime agent.") from error


def _voice_target_from_dispatch(metadata: str, *, legacy_owner_id: str | None) -> tuple[str | None, str]:
    """Read a backend-signed room dispatch target, with legacy local fallback.

    Metadata is part of the LiveKit token's signed room configuration. The
    browser receives the token but cannot alter this target.
    """
    try:
        payload = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError as error:
        raise ValueError("The LiveKit room dispatch metadata is invalid.") from error
    account_id = payload.get("user_id")
    identity_id = payload.get("voice_identity_id")
    if isinstance(account_id, str) and account_id and isinstance(identity_id, str) and identity_id:
        return account_id, identity_id
    if legacy_owner_id:
        return None, legacy_owner_id
    raise ValueError("Room has no enrolled voice target. Enroll voice before joining the assistant.")


async def accept_auth_gate_job(request) -> None:
    """Give the browser a stable identity for trusted status messages."""
    await request.accept(name="Voice Auth Gate", identity=AGENT_PARTICIPANT_IDENTITY)


async def auth_gate_entrypoint(ctx) -> None:
    """Run one local voice-auth room session.

    This function must remain module-level: macOS uses multiprocessing spawn,
    which cannot pickle a function nested inside ``build_server``.
    """
    from livekit import agents, rtc

    settings = LocalAgentSettings.from_environment()
    account_id, target_identity_id = _voice_target_from_dispatch(
        ctx.job.metadata,
        legacy_owner_id=str(settings.owner_identity_id) if settings.owner_identity_id else None,
    )
    # Model load, HE private context access, and matcher registration occur
    # only in this local Agent process. They are absent from LiveKit data.
    if account_id:
        gate = await asyncio.to_thread(build_account_voice_auth_gate, account_id)
    else:
        gate = await asyncio.to_thread(build_persistent_voice_auth_gate)
    controllers: dict[str, VoiceAuthChallengeController] = {}
    turn_controllers: dict[str, ConversationTurnController] = {}
    turn_tasks: dict[str, asyncio.Task] = {}
    processing_turns: set[str] = set()
    retryable_turns: dict[tuple[str, str], str] = {}
    event_sequences: dict[tuple[str, str], int] = {}
    auth_sequences: dict[str, int] = {}
    auth_progress_elapsed_ms: dict[str, int] = {}
    resume_guard_until: dict[str, float] = {}
    turn_timings: dict[str, dict[str, float]] = {}
    room = ctx.room
    conversation = None
    transcriber = None
    tts = None
    audio_source = None

    if settings.conversation_enabled:
        # Provider construction is trusted/local. Whisper's model remains lazy
        # and audio is never passed to OpenAI; only the resulting transcript is.
        gateway_settings = GatewaySettings.from_environment()
        if gateway_settings.mcp_provider != "mock" or not gateway_settings.mock_calendar_enabled:
            raise RuntimeError("Only MCP_PROVIDER=mock is supported by the local Agent phase.")
        store = GatewayStore(gateway_settings.database_path)
        store.initialize()
        from app.assistant.config import OpenAISettings

        conversation = ConversationSupervisor(
            llm=OpenAIResponsesProvider(OpenAISettings.from_environment()),
            calendar=MockCalendarProvider(store),
            account_id=account_id,
        )
        transcriber = LocalWhisperProvider(model_name=settings.whisper_model, device=settings.whisper_device)
        tts = EdgeTTSProvider(voice=settings.edge_tts_voice)

    def controller_for(identity: str) -> VoiceAuthChallengeController:
        if identity not in controllers:
            controllers[identity] = VoiceAuthChallengeController(
                gate=gate,
                target_identity_id=target_identity_id,
                session=AuthSession(ttl=settings.auth_ttl),
                capture_seconds=settings.auth_challenge_seconds,
            )
        return controllers[identity]

    def turn_controller_for(identity: str) -> ConversationTurnController:
        if identity not in turn_controllers:
            turn_controllers[identity] = ConversationTurnController(
                min_speech_seconds=settings.vad_min_speech_seconds,
                silence_seconds=settings.vad_silence_seconds,
                maximum_seconds=settings.vad_maximum_seconds,
            )
        return turn_controllers[identity]

    def mark_timing(turn_id: str, stage: str) -> None:
        timings = turn_timings.setdefault(turn_id, {})
        timings.setdefault(stage, monotonic())

    def log_timing(turn_id: str) -> None:
        timings = turn_timings.pop(turn_id, None)
        if not timings:
            return
        started = timings.get("speech_start")
        if started is None:
            return
        safe_stages = {
            name: round(timestamp - started, 3)
            for name, timestamp in timings.items()
            if name != "speech_start"
        }
        logger.info("voice_turn_latency turn_id=%s stages=%s", turn_id, safe_stages)

    async def send_status(identity: str, decision: AuthDecision, *, controller=None) -> None:
        auth_sequences[identity] = auth_sequences.get(identity, 0) + 1
        challenge = controller.challenge_status() if controller is not None else None
        session_id = str(getattr(room, "name", "")) or None
        await room.local_participant.send_text(
            auth_status_payload(
                decision,
                challenge=challenge,
                session_id=session_id,
                sequence=auth_sequences[identity],
            ),
            topic=AUTH_STATUS_TOPIC,
            destination_identities=[identity],
        )

    async def send_event(identity: str, *, event_type: str, turn_id: str | None = None, **values) -> None:
        if turn_id is not None:
            key = (identity, turn_id)
            event_sequences[key] = event_sequences.get(key, 0) + 1
            values["sequence"] = event_sequences[key]
        await room.local_participant.send_text(
            event_payload(event_type=event_type, turn_id=turn_id, **values),
            topic=AGENT_EVENT_TOPIC,
            destination_identities=[identity],
        )

    def tool_view(tool_result: dict | None) -> dict | None:
        if tool_result is None:
            return None
        return {
            "name": "calendar.create_event" if "created" in tool_result else "calendar.list_events",
            "provider": tool_result.get("provider", "mock"),
            "demo": bool(tool_result.get("demo", False)),
        }

    async def process_response(identity: str, *, turn_id: str, transcript: str) -> None:
        """Stream text by phrase and publish one Edge-TTS sentence at a time."""
        audio_failed = False
        audio_started = False
        full_response: list[str] = []
        sentence_buffer = SentenceBuffer()

        async def publish_sentence(sentence: str) -> None:
            nonlocal audio_failed, audio_started
            if audio_failed or tts is None or audio_source is None:
                return
            try:
                audio = await tts.synthesize(sentence)

                async def mark_audio_started() -> None:
                    nonlocal audio_started
                    if audio_started:
                        return
                    audio_started = True
                    mark_timing(turn_id, "tts_first_audio")
                    await send_event(
                        identity,
                        event_type="state",
                        turn_id=turn_id,
                        state=VoiceAgentState.SPEAKING,
                    )

                await publish_mp3(audio_source, audio, on_first_frame=mark_audio_started)
            except Exception:
                audio_failed = True
                logger.exception("Local Edge TTS failed; response remains available as text")

        try:
            await send_event(identity, event_type="state", turn_id=turn_id, state=VoiceAgentState.THINKING)
            tool_result: dict | None = None
            async for delta, streamed_tool_result in conversation.respond_stream(
                transcript, auth=controller_for(identity).current()
            ):
                if not full_response:
                    mark_timing(turn_id, "llm_first_token")
                full_response.append(delta)
                tool_result = streamed_tool_result
                response_so_far = "".join(full_response)
                await send_event(
                    identity,
                    event_type="assistant_response",
                    turn_id=turn_id,
                    text=response_so_far,
                    is_final=False,
                    tool=tool_view(tool_result),
                )
                for sentence in sentence_buffer.push(delta):
                    mark_timing(turn_id, "llm_first_phrase")
                    await publish_sentence(sentence)

            for sentence in sentence_buffer.flush():
                mark_timing(turn_id, "llm_first_phrase")
                await publish_sentence(sentence)

            response = "".join(full_response).strip()
            if not response:
                raise RuntimeError("The language model returned no text response.")
            await send_event(
                identity,
                event_type="assistant_response",
                turn_id=turn_id,
                text=response,
                is_final=True,
                tool=tool_view(tool_result),
            )
            if audio_failed:
                await send_event(
                    identity,
                    event_type="state",
                    turn_id=turn_id,
                    state=VoiceAgentState.ERROR,
                    message="Audio unavailable. The text response is still available.",
                )
            elif audio_source is not None and tts is not None:
                await audio_source.wait_for_playout()
                mark_timing(turn_id, "playback_completed")
                await send_event(identity, event_type="state", turn_id=turn_id, state=VoiceAgentState.COMPLETED)
            else:
                await send_event(identity, event_type="state", turn_id=turn_id, state=VoiceAgentState.COMPLETED)
        except asyncio.CancelledError:
            if audio_source is not None:
                audio_source.clear_queue()
            await send_event(identity, event_type="state", turn_id=turn_id, state=VoiceAgentState.INTERRUPTED)
            raise
        except Exception:
            logger.exception("Local LLM response failed")
            await send_event(
                identity,
                event_type="state",
                turn_id=turn_id,
                state=VoiceAgentState.ERROR,
                message="Assistant unavailable. Please retry this response.",
            )
        finally:
            processing_turns.discard(identity)
            turn_controller_for(identity).finish_processing()
            log_timing(turn_id)

    async def process_turn(identity: str, *, turn_id: str, recording) -> None:
        if conversation is None or transcriber is None:
            return
        try:
            await send_event(identity, event_type="state", turn_id=turn_id, state=VoiceAgentState.TRANSCRIBING)
            mark_timing(turn_id, "transcript_start")
            transcript = await asyncio.to_thread(transcriber.transcribe, recording)
            if not transcript:
                await send_event(
                    identity,
                    event_type="state",
                    turn_id=turn_id,
                    state=VoiceAgentState.ERROR,
                    message="No clear speech was detected. Please try again.",
                )
                return
            retryable_turns[(identity, turn_id)] = transcript
            mark_timing(turn_id, "transcript_final")
            await send_event(identity, event_type="transcript", turn_id=turn_id, text=transcript, is_final=True)
            await process_response(identity, turn_id=turn_id, transcript=transcript)
        except Exception:
            logger.exception("Local conversational turn failed")
            await send_event(
                identity,
                event_type="state",
                turn_id=turn_id,
                state=VoiceAgentState.ERROR,
                message="Could not process this voice turn. Please try again.",
            )
        finally:
            processing_turns.discard(identity)
            turn_controller_for(identity).finish_processing()
            log_timing(turn_id)

    async def process_audio(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        controller = controller_for(participant.identity)
        stream = rtc.AudioStream(track)
        try:
            async for event in stream:
                frame = event.frame
                pcm = np.frombuffer(frame.data, dtype=np.int16)
                if frame.num_channels > 1:
                    pcm = pcm.reshape((-1, frame.num_channels))
                normalized = np.ascontiguousarray(pcm, dtype=np.float32) / 32768.0
                phase_before = controller.challenge_status().phase
                capture_method = getattr(controller, "capture_pcm", None)
                if capture_method is None:
                    decision = await asyncio.to_thread(
                        controller.append_pcm, normalized, sample_rate=frame.sample_rate
                    )
                else:
                    captured = await asyncio.to_thread(
                        capture_method, normalized, sample_rate=frame.sample_rate
                    )
                    decision = None
                    if captured:
                        await send_status(participant.identity, controller.current(), controller=controller)
                        decision = await asyncio.to_thread(controller.verify_pending)
                if decision is not None:
                    await send_status(participant.identity, decision, controller=controller)
                elif phase_before is AuthChallengePhase.CAPTURING:
                    status = controller.challenge_status()
                    last_elapsed_ms = auth_progress_elapsed_ms.get(participant.identity, -1)
                    if status.elapsed_ms - last_elapsed_ms >= 250:
                        auth_progress_elapsed_ms[participant.identity] = status.elapsed_ms
                        await send_status(
                            participant.identity,
                            controller.current(),
                            controller=controller,
                        )
                phase_after = controller.challenge_status().phase
                # Capture, matcher processing, and the explicit resume wait
                # all consume audio. This includes the terminal frame and any
                # speech that continues past the five-second boundary.
                if phase_before in {
                    AuthChallengePhase.CAPTURING,
                    AuthChallengePhase.PROCESSING,
                    AuthChallengePhase.WAITING_FOR_RESUME,
                } or phase_after in {
                    AuthChallengePhase.PROCESSING,
                    AuthChallengePhase.WAITING_FOR_RESUME,
                }:
                    if phase_before is not AuthChallengePhase.CAPTURING:
                        logger.info(
                            "voice_auth_audio_dropped identity=%s phase=%s",
                            participant.identity,
                            phase_before.value,
                        )
                    continue
                if monotonic() < resume_guard_until.get(participant.identity, 0.0):
                    continue
                # The full challenge remains outside ASR/LLM, including its terminal frame.
                if phase_before is AuthChallengePhase.CAPTURING:
                    turn_controller_for(participant.identity).begin_authentication()
                    continue
                if settings.conversation_enabled and participant.identity not in processing_turns:
                    ingress = turn_controller_for(participant.identity).append_pcm(
                        normalized,
                        sample_rate=frame.sample_rate,
                        auth_state=controller.current().state,
                    )
                    if ingress.started and ingress.turn_id is not None:
                        mark_timing(ingress.turn_id, "speech_start")
                        await send_event(
                            participant.identity,
                            event_type="state",
                            turn_id=ingress.turn_id,
                            state=VoiceAgentState.LISTENING,
                        )
                    if ingress.recording is not None and ingress.turn_id is not None:
                        turn_id = ingress.turn_id
                        mark_timing(turn_id, "speech_end")
                        processing_turns.add(participant.identity)
                        turn_controller_for(participant.identity).begin_processing()
                        task = asyncio.create_task(process_turn(participant.identity, turn_id=turn_id, recording=ingress.recording))
                        turn_tasks[participant.identity] = task

                        def clear_completed_task(completed_task, *, identity=participant.identity) -> None:
                            if turn_tasks.get(identity) is completed_task:
                                turn_tasks.pop(identity, None)

                        task.add_done_callback(clear_completed_task)
        except Exception:
            # Fail closed: no failure can grant a private capability.
            decision = controller.cancel()
            turn_controller_for(participant.identity).reset()
            logger.exception("Voice-auth audio processing failed for participant %s", participant.identity)
            await send_status(participant.identity, decision, controller=controller)
        finally:
            await stream.aclose()

    @room.on("data_received")
    def on_data_received(packet: rtc.DataPacket) -> None:
        if packet.participant is None:
            return
        try:
            command = packet.data.decode("utf-8")
        except UnicodeDecodeError:
            return
        if packet.topic == AGENT_COMMAND_TOPIC:
            try:
                payload = json.loads(command)
            except json.JSONDecodeError:
                return
            turn_id = payload.get("turn_id")
            if payload.get("action") != "retry" or not isinstance(turn_id, str):
                return
            controller = controller_for(packet.participant.identity)
            if controller.phase not in {
                AuthChallengePhase.IDLE,
                AuthChallengePhase.CONVERSATION_READY,
            }:
                return
            transcript = retryable_turns.get((packet.participant.identity, turn_id))
            if transcript is None or packet.participant.identity in processing_turns:
                return
            processing_turns.add(packet.participant.identity)
            asyncio.create_task(process_response(packet.participant.identity, turn_id=turn_id, transcript=transcript))
            return
        if packet.topic != AUTH_COMMAND_TOPIC:
            return
        controller = controller_for(packet.participant.identity)

        async def handle_auth_command() -> None:
            identity = packet.participant.identity
            if command in {REQUEST_PRIVATE_MODE, RETRY_VOICE}:
                active_task = turn_tasks.get(identity)
                if active_task is not None and not active_task.done():
                    active_task.cancel()
                processing_turns.discard(identity)
                turn_controller_for(identity).begin_authentication()
                auth_progress_elapsed_ms[identity] = -1
                if audio_source is not None:
                    audio_source.clear_queue()
                decision = controller.request_private_mode()
            elif command == CANCEL_PRIVATE_MODE:
                turn_controller_for(identity).reset()
                decision = controller.cancel()
                auth_progress_elapsed_ms[identity] = -1
                resume_guard_until[identity] = monotonic() + settings.auth_resume_guard_seconds
            elif command in {RESUME_CONVERSATION, CONTINUE_AS_GUEST}:
                turn_controller_for(identity).reset()
                if audio_source is not None:
                    audio_source.clear_queue()
                try:
                    decision = controller.resume_conversation()
                except RuntimeError:
                    await send_event(
                        identity,
                        event_type="state",
                        state=VoiceAgentState.ERROR,
                        message="Hãy hoàn tất voice challenge trước khi bắt đầu nói.",
                    )
                    return
                resume_guard_until[identity] = monotonic() + settings.auth_resume_guard_seconds
            else:
                return
            await send_status(identity, decision, controller=controller)

        asyncio.create_task(handle_auth_command())

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, _publication, participant: rtc.RemoteParticipant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(process_audio(track, participant))

    async def initialize_participant(_ctx, participant) -> None:
        participant_controller = controller_for(participant.identity)
        await send_status(
            participant.identity,
            participant_controller.current(),
            controller=participant_controller,
        )

    async def close_voice_dependencies() -> None:
        await asyncio.to_thread(gate.close)

    # Register before connect so an already-present participant starts guest.
    ctx.add_participant_entrypoint(entrypoint_fnc=initialize_participant)
    ctx.add_shutdown_callback(close_voice_dependencies)
    if transcriber is not None:
        logger.info("warming local Whisper model before the first conversational turn")
        await asyncio.to_thread(transcriber.warmup)
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    if settings.conversation_enabled:
        audio_source = rtc.AudioSource(48000, 1, queue_size_ms=200)
        assistant_track = rtc.LocalAudioTrack.create_audio_track("assistant-audio", audio_source)
        await room.local_participant.publish_track(assistant_track)


def build_server():
    """Create an AgentServer using documented raw-audio LiveKit APIs."""
    require_livekit()
    from livekit import agents

    server = agents.AgentServer()
    server.rtc_session(
        auth_gate_entrypoint,
        agent_name="voice-auth-gate",
        on_request=accept_auth_gate_job,
    )
    return server


def main() -> None:
    from livekit.agents import cli

    cli.run_app(build_server())


if __name__ == "__main__":
    main()
