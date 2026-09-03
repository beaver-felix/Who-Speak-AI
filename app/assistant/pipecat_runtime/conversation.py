"""Pipecat frame bridge for the existing policy-first conversation supervisor."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.assistant.agents.conversation import ConversationSupervisor
from app.assistant.contracts import AuthDecision
from app.assistant.events import VoiceAgentState
from app.assistant.streaming import SentenceBuffer
from app.assistant.events_bridge import SessionEventSink
from app.assistant.pipecat_runtime.edge_tts_service import (
    VoiceTTSSpeakFrame,
    VoiceTTSTurnEndFrame,
)

logger = logging.getLogger(__name__)


class PipecatConversationProcessor(FrameProcessor):
    """Turn final local STT frames into safe events and TTS frames.

    The processor keeps the existing ``ConversationSupervisor`` as the
    trusted policy/LLM boundary. Pipecat owns audio/VAD/frame scheduling; this
    adapter owns per-room turn IDs and never forwards challenge audio or raw
    transcripts to the browser except as final UI text.
    """

    def __init__(
        self,
        *,
        supervisor: ConversationSupervisor,
        auth: Callable[[], AuthDecision],
        events: SessionEventSink,
        tts_enabled: bool,
        max_sentence_chars: int = 120,
        session_id: str = "",
        room_id_hash: str = "-",
        conversation_ingress_allowed: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(name="conversation-supervisor")
        if max_sentence_chars < 8:
            raise ValueError("max_sentence_chars must be at least 8.")
        self._supervisor = supervisor
        self._auth = auth
        self._events = events
        self._tts_enabled = tts_enabled
        self._max_sentence_chars = max_sentence_chars
        self._session_id = session_id
        self._room_id_hash = room_id_hash
        self._conversation_ingress_allowed = conversation_ingress_allowed or (lambda: True)
        self._active_turn: str | None = None
        self._speaking_turn: str | None = None
        self._response_task: asyncio.Task | None = None
        self._retryable: dict[str, str] = {}
        self._turn_started_at: dict[str, float] = {}
        self._timing_marks: dict[str, set[str]] = {}

    def _mark_timing(self, turn_id: str | None, stage: str) -> None:
        if not turn_id or stage in self._timing_marks.setdefault(turn_id, set()):
            return
        self._timing_marks[turn_id].add(stage)
        started = self._turn_started_at.setdefault(turn_id, time.monotonic())
        logger.info(
            "voice_turn_stage session_id=%s room_id_hash=%s turn_id=%s stage=%s elapsed_ms=%.1f",
            self._session_id,
            self._room_id_hash,
            turn_id,
            stage,
            (time.monotonic() - started) * 1000,
        )

    async def _send_state(
        self,
        state: VoiceAgentState,
        *,
        turn_id: str | None = None,
        stage: str | None = None,
        message: str | None = None,
    ) -> None:
        values: dict[str, object] = {"state": state}
        if stage is not None:
            values["stage"] = stage
        if message is not None:
            values["message"] = message
        await self._events.send_event("state", turn_id=turn_id, **values)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            await self._interrupt_active_turn()
            await self.push_frame(frame, direction)
            return
        if not self._conversation_ingress_allowed() and isinstance(
            frame,
            (
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                TranscriptionFrame,
                ErrorFrame,
            ),
        ):
            # AuthRouter is the first boundary, but VAD/STT frames already
            # queued before a private-mode request can still arrive here.
            # Drop only frames that could create a conversation turn. Pipeline
            # lifecycle frames (especially StartFrame) must keep flowing: the
            # output transport creates its TTS media sender on StartFrame.
            logger.debug("conversation_frame_dropped reason=auth_ingress_closed")
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            response_in_progress = (
                self._response_task is not None and not self._response_task.done()
            )
            if response_in_progress:
                # VAD detected a new request even if the prior answer has not
                # reached TTS yet. Cancel it before creating a new turn; do
                # not silently drop the later transcript.
                await self._interrupt_active_turn()
                self._active_turn = None
                self._speaking_turn = None
                await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            elif self._speaking_turn:
                self._mark_timing(self._speaking_turn, "interrupted")
                await self._send_state(
                    VoiceAgentState.INTERRUPTED,
                    turn_id=self._speaking_turn,
                    stage="transport",
                )
                self._speaking_turn = None
                await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            if self._active_turn is None:
                self._active_turn = str(uuid4())
                self._mark_timing(self._active_turn, "speech_started")
                await self._send_state(VoiceAgentState.LISTENING, turn_id=self._active_turn, stage="vad")
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._active_turn:
                self._mark_timing(self._active_turn, "speech_stopped")
                await self._send_state(VoiceAgentState.TRANSCRIBING, turn_id=self._active_turn, stage="stt")
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            logger.info(
                "pipecat_transcription_received session_id=%s turn_id=%s text_chars=%s",
                self._session_id,
                self._active_turn or "-",
                len(text),
            )
            if not text:
                self._mark_timing(self._active_turn, "error")
                await self._send_state(VoiceAgentState.ERROR, turn_id=self._active_turn, stage="stt", message="No clear speech was detected. Please try again.")
                self._active_turn = None
                return
            if self._active_turn is None:
                self._active_turn = str(uuid4())
                self._mark_timing(self._active_turn, "speech_started")
                await self._send_state(VoiceAgentState.LISTENING, turn_id=self._active_turn, stage="vad")
            if self._response_task is not None and not self._response_task.done():
                logger.warning("pipecat_transcript_dropped reason=response_in_progress")
                return
            turn_id = self._active_turn
            self._mark_timing(turn_id, "stt_final")
            self._retryable[turn_id] = text
            await self._events.send_event(
                "transcript", turn_id=turn_id, text=text, is_final=True
            )
            logger.info(
                "pipecat_transcript_final session_id=%s turn_id=%s text_chars=%s",
                self._session_id,
                turn_id,
                len(text),
            )
            self._response_task = self.create_task(
                self._respond(turn_id, text), name=f"respond-{turn_id}"
            )
            return
        if isinstance(frame, ErrorFrame):
            self._mark_timing(self._active_turn, "error")
            await self._send_state(VoiceAgentState.ERROR, turn_id=self._active_turn, stage="stt", message="Local speech recognition is unavailable for this turn.")
            self._active_turn = None
            return
        await self.push_frame(frame, direction)

    async def _respond(self, turn_id: str, transcript: str) -> None:
        sentence_buffer = SentenceBuffer(maximum_characters=self._max_sentence_chars)
        response_parts: list[str] = []
        tool_view: dict | None = None
        try:
            self._mark_timing(turn_id, "llm_started")
            await self._send_state(VoiceAgentState.THINKING, turn_id=turn_id, stage="llm")
            async for delta, result in self._supervisor.respond_stream(
                transcript, auth=self._auth()
            ):
                if delta:
                    self._mark_timing(turn_id, "llm_first_delta")
                    response_parts.append(delta)
                    tool_view = self._tool_view(result)
                    await self._events.send_event(
                        "assistant_response",
                        turn_id=turn_id,
                        text="".join(response_parts),
                        is_final=False,
                        tool=tool_view,
                    )
                    for sentence in sentence_buffer.push(delta):
                        if self._tts_enabled:
                            self._mark_timing(turn_id, "tts_started")
                            logger.info(
                                "tts_sentence_enqueued session_id=%s turn_id=%s text_chars=%s",
                                self._session_id,
                                turn_id,
                                len(sentence),
                            )
                            await self.push_frame(
                                VoiceTTSSpeakFrame(sentence, turn_id=turn_id)
                            )
            for sentence in sentence_buffer.flush():
                if self._tts_enabled:
                    self._mark_timing(turn_id, "tts_started")
                    logger.info(
                        "tts_sentence_enqueued session_id=%s turn_id=%s text_chars=%s",
                        self._session_id,
                        turn_id,
                        len(sentence),
                    )
                    await self.push_frame(VoiceTTSSpeakFrame(sentence, turn_id=turn_id))
            response = "".join(response_parts).strip()
            if not response:
                raise RuntimeError("The language model returned no text response.")
            self._mark_timing(turn_id, "llm_final")
            await self._events.send_event(
                "assistant_response",
                turn_id=turn_id,
                text=response,
                is_final=True,
                tool=tool_view,
            )
            logger.info(
                "pipecat_assistant_final session_id=%s turn_id=%s text_chars=%s tool=%s",
                self._session_id,
                turn_id,
                len(response),
                tool_view["name"] if tool_view else "none",
            )
            if self._tts_enabled:
                await self.push_frame(VoiceTTSTurnEndFrame(turn_id=turn_id))
            else:
                self._mark_timing(turn_id, "completed")
                await self._send_state(VoiceAgentState.COMPLETED, turn_id=turn_id, stage="llm")
        except asyncio.CancelledError:
            self._mark_timing(turn_id, "interrupted")
            await self._send_state(VoiceAgentState.INTERRUPTED, turn_id=turn_id, stage="transport")
            raise
        except Exception:
            logger.exception("pipecat_conversation_failed turn_id=%s", turn_id)
            self._mark_timing(turn_id, "error")
            await self._send_state(VoiceAgentState.ERROR, turn_id=turn_id, stage="llm", message="Assistant unavailable. Please retry this response.")
        finally:
            if self._active_turn == turn_id:
                self._active_turn = None
            self._response_task = None

    async def _interrupt_active_turn(self) -> None:
        task = self._response_task
        if task is not None and not task.done():
            await self.cancel_task(task)

    async def retry(self, turn_id: str) -> bool:
        """Retry only a transcript already produced in this room session."""

        transcript = self._retryable.get(turn_id)
        if not transcript or (self._response_task is not None and not self._response_task.done()):
            return False
        self._active_turn = turn_id
        self._response_task = self.create_task(
            self._respond(turn_id, transcript), name=f"retry-{turn_id}"
        )
        return True

    async def report_stt_error(self) -> None:
        if self._active_turn is None:
            return
        self._mark_timing(self._active_turn, "error")
        await self._send_state(
            VoiceAgentState.ERROR,
            turn_id=self._active_turn,
            stage="stt",
            message="Local speech recognition is unavailable for this turn.",
        )
        self._active_turn = None

    async def audio_started(self, turn_id: str) -> None:
        self._speaking_turn = turn_id
        self._mark_timing(turn_id, "tts_first_audio")
        await self._send_state(VoiceAgentState.SPEAKING, turn_id=turn_id, stage="tts")

    async def audio_completed(self, turn_id: str) -> None:
        if self._speaking_turn == turn_id:
            self._speaking_turn = None
        self._mark_timing(turn_id, "tts_finished")
        self._mark_timing(turn_id, "completed")
        await self._send_state(VoiceAgentState.COMPLETED, turn_id=turn_id, stage="tts")

    async def audio_error(self, turn_id: str, message: str) -> None:
        if self._speaking_turn == turn_id:
            self._speaking_turn = None
        self._mark_timing(turn_id, "error")
        await self._send_state(VoiceAgentState.ERROR, turn_id=turn_id, stage="tts", message=message)

    @staticmethod
    def _tool_view(tool_result: dict | None) -> dict | None:
        if tool_result is None:
            return None
        return {
            "name": "calendar.create_event" if "created" in tool_result else "calendar.list_events",
            "provider": tool_result.get("provider", "mock"),
            "demo": bool(tool_result.get("demo", False)),
        }

    async def cleanup(self):
        if self._response_task is not None and not self._response_task.done():
            await self.cancel_task(self._response_task)
        self._response_task = None
        await super().cleanup()
