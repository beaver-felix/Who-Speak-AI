from __future__ import annotations

from pathlib import Path

from app.assistant.config import LocalAgentSettings
from voiceauth.config import VoiceSettings


def test_voice_settings_resolve_relative_paths_from_supplied_app_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOICE_MATCHER_TOKEN", "test-token")
    monkeypatch.setenv("VOICE_MODEL_CHECKPOINT", "models/best.pt")
    monkeypatch.setenv("VOICE_MODEL_CACHE_DIR", "cache")

    settings = VoiceSettings.from_environment(root=tmp_path)

    assert settings.model_checkpoint == tmp_path / "models/best.pt"
    assert settings.model_cache_dir == tmp_path / "cache"


def test_local_agent_settings_expose_streaming_turn_timing(monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("LIVEKIT_DEV_MODE", "true")
    monkeypatch.setenv("VOICE_VAD_MIN_SPEECH_SECONDS", "0.25")
    monkeypatch.setenv("VOICE_VAD_SILENCE_SECONDS", "0.4")
    monkeypatch.setenv("VOICE_VAD_MAXIMUM_SECONDS", "12")
    monkeypatch.setenv("VOICE_SMART_TURN_MODEL_PATH", "smart-turn.onnx")
    monkeypatch.setenv("VOICE_SMART_TURN_CPU_COUNT", "2")
    monkeypatch.setenv("VOICE_SMART_TURN_PRE_SPEECH_MS", "500")
    monkeypatch.setenv("VOICE_SMART_TURN_MAX_WAIT_SECONDS", "2")
    monkeypatch.setenv("VOICE_WHISPER_MODEL", "vudang449/PhoWhisper-small-ct2")
    monkeypatch.setenv("VOICE_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("VOICE_WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("VOICE_WHISPER_CPU_THREADS", "4")
    monkeypatch.setenv("VOICE_AGENT_JOIN_TIMEOUT_SECONDS", "25")
    monkeypatch.setenv("VOICE_AGENT_PIPELINE_SETUP_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("VOICE_AGENT_PIPELINE_START_TIMEOUT_SECONDS", "75")
    monkeypatch.setenv("VOICE_TTS_PROVIDER", "zerotts")
    monkeypatch.setenv("VOICE_TTS_FALLBACK_PROVIDER", "edge")
    monkeypatch.setenv("VOICE_ZEROTTS_MODEL", "test/model")
    monkeypatch.setenv("VOICE_ZEROTTS_VOICE", "maichi")
    monkeypatch.setenv("VOICE_ZEROTTS_CACHE_DIR", "tts-cache")
    monkeypatch.setenv("VOICE_ZEROTTS_WARMUP", "false")
    monkeypatch.setenv("VOICE_TTS_QUEUE_MAX_CHUNKS", "4")
    monkeypatch.setenv("VOICE_TTS_MAX_SENTENCE_CHARS", "120")

    settings = LocalAgentSettings.from_environment()

    assert settings.vad_min_speech_seconds == 0.25
    assert settings.vad_silence_seconds == 0.4
    assert settings.vad_maximum_seconds == 12
    assert settings.smart_turn_model_path == "smart-turn.onnx"
    assert settings.smart_turn_cpu_count == 2
    assert settings.smart_turn_pre_speech_ms == 500
    assert settings.smart_turn_max_wait_seconds == 2
    assert settings.whisper_model == "vudang449/PhoWhisper-small-ct2"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"
    assert settings.whisper_cpu_threads == 4
    assert settings.participant_join_timeout_seconds == 25
    assert settings.pipeline_setup_timeout_seconds == 90
    assert settings.pipeline_start_timeout_seconds == 75
    assert settings.tts_provider == "zerotts"
    assert settings.tts_fallback_provider == "edge"
    assert settings.zerotts_model == "test/model"
    assert settings.zerotts_voice == "maichi"
    assert settings.zerotts_cache_dir == "tts-cache"
    assert settings.zerotts_warmup is False
    assert settings.tts_queue_max_chunks == 4
    assert settings.tts_max_sentence_chars == 120


def test_local_agent_defaults_tolerate_natural_pauses(monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("LIVEKIT_DEV_MODE", "true")
    monkeypatch.delenv("VOICE_VAD_SILENCE_SECONDS", raising=False)
    monkeypatch.delenv("VOICE_VAD_MAXIMUM_SECONDS", raising=False)

    settings = LocalAgentSettings.from_environment()

    assert settings.vad_silence_seconds == 0.65
    assert settings.vad_maximum_seconds == 20
