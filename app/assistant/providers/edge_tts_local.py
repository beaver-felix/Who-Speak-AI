"""Edge TTS adapter kept separate from LiveKit transport."""

from __future__ import annotations


class EdgeTTSProvider:
    def __init__(self, *, voice: str = "vi-VN-HoaiMyNeural") -> None:
        self._voice = voice

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        try:
            import edge_tts
        except ImportError as error:
            raise RuntimeError("Install the [providers] extra to enable Edge TTS.") from error
        output = bytearray()
        communicate = edge_tts.Communicate(text=text, voice=self._voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output.extend(chunk["data"])
        if not output:
            raise RuntimeError("Edge TTS returned no audio.")
        return bytes(output)
