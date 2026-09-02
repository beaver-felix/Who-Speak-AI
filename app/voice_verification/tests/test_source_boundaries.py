from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_matcher_does_not_import_raw_audio_or_rawnet3() -> None:
    matcher_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "apps" / "matcher_api").glob("*.py")
    )
    forbidden = ("soundfile", "RawNet3", "speaker_recognition", "waveform")
    assert not any(token in matcher_source for token in forbidden)


def test_livekit_scaffold_does_not_import_voice_private_context() -> None:
    source = (ROOT.parent / "assistant" / "livekit_agent.py").read_text(encoding="utf-8")
    assert "VoiceHEClient" not in source
    assert "RawNet3SpeakerEncoder" not in source
