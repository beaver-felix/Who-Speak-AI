from __future__ import annotations

import asyncio
import pickle

from livekit.agents import testing

from app.assistant.livekit_agent import (
    AGENT_PARTICIPANT_IDENTITY,
    AUTH_COMMAND_TOPIC,
    REQUEST_PRIVATE_MODE,
    auth_gate_entrypoint,
    build_server,
)


def test_livekit_auth_server_registers_an_rtc_session() -> None:
    server = build_server()

    assert server._entrypoint_fnc is not None
    assert server._agent_name == "voice-auth-gate"


def test_livekit_entrypoint_is_picklable_for_macos_spawn() -> None:
    pickle.dumps(auth_gate_entrypoint)


def test_livekit_test_harness_creates_an_in_process_job_context() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with testing.fake_job_context() as context:
            assert context.job.id == "fake-job"
            assert context.room is not None
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_private_mode_command_is_an_explicit_data_channel_operation() -> None:
    assert AUTH_COMMAND_TOPIC == "voice-auth"
    assert REQUEST_PRIVATE_MODE == "request_private_mode"
    assert AGENT_PARTICIPANT_IDENTITY == "voice-auth-gate"
