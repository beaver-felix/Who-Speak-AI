"""Build the configured streaming TTS provider stack."""

from __future__ import annotations

from app.assistant.providers.edge_tts_local import EdgeTTSProvider
from app.assistant.providers.streaming_tts import BufferedEdgeTTSProvider, StreamingTTSProvider
from app.assistant.providers.zerotts_local import ZeroTTSLocalProvider


def build_tts_stack(
    *,
    provider_name: str,
    fallback_name: str | None,
    zerotts_model: str,
    zerotts_voice: str,
    zerotts_cache_dir: str | None,
    zerotts_startup_buffer_ms: float,
    zerotts_intra_op_threads: int,
    zerotts_codec_threads: int | None,
    queue_max_chunks: int,
    edge_voice: str,
) -> tuple[StreamingTTSProvider, StreamingTTSProvider | None]:
    """Create primary/fallback providers without loading model resources."""

    def build(name: str) -> StreamingTTSProvider:
        if name == "zerotts":
            return ZeroTTSLocalProvider(
                model_name=zerotts_model,
                voice=zerotts_voice,
                cache_dir=zerotts_cache_dir,
                queue_max_chunks=queue_max_chunks,
                startup_buffer_ms=zerotts_startup_buffer_ms,
                intra_op_num_threads=zerotts_intra_op_threads,
                codec_intra_op_num_threads=zerotts_codec_threads,
            )
        if name == "edge":
            return BufferedEdgeTTSProvider(EdgeTTSProvider(voice=edge_voice))
        raise ValueError(f"Unsupported TTS provider: {name}")

    primary = build(provider_name)
    fallback = build(fallback_name) if fallback_name else None
    return primary, fallback
