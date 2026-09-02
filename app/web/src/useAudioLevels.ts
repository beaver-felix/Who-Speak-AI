import { useEffect, useState } from 'react'

export type AudioLevels = {
  microphone: number
  assistant: number
}

const SILENT: AudioLevels = { microphone: 0, assistant: 0 }

function rms(analyser: AnalyserNode, buffer: Uint8Array<ArrayBuffer>): number {
  analyser.getByteTimeDomainData(buffer)
  let sum = 0
  for (const sample of buffer) {
    const centered = (sample - 128) / 128
    sum += centered * centered
  }
  // RMS is intentionally boosted for a useful visual response at normal
  // speech volume. It is a display signal, never an authentication signal.
  return Math.min(1, Math.sqrt(sum / buffer.length) * 4)
}

function smooth(previous: number, next: number): number {
  const attack = 0.45
  const release = 0.14
  return previous + (next - previous) * (next > previous ? attack : release)
}

/** Read levels from the already-published LiveKit tracks without opening a second microphone. */
export function useAudioLevels(
  microphoneTrack: MediaStreamTrack | null,
  assistantTracks: MediaStreamTrack[],
): AudioLevels {
  const [levels, setLevels] = useState<AudioLevels>(SILENT)
  const assistantTrackIds = assistantTracks.map((track) => track.id).sort().join('|')

  useEffect(() => {
    if (!microphoneTrack && assistantTracks.length === 0) {
      setLevels(SILENT)
      return
    }

    const AudioContextConstructor = window.AudioContext ?? window.webkitAudioContext
    if (!AudioContextConstructor) {
      setLevels(SILENT)
      return
    }

    let context: AudioContext
    try {
      context = new AudioContextConstructor()
    } catch {
      setLevels(SILENT)
      return
    }

    const microphoneAnalyser = context.createAnalyser()
    const assistantAnalyser = context.createAnalyser()
    const silentSink = context.createGain()
    silentSink.gain.value = 0
    microphoneAnalyser.fftSize = 256
    assistantAnalyser.fftSize = 256
    const microphoneBuffer = new Uint8Array(new ArrayBuffer(microphoneAnalyser.fftSize))
    const assistantBuffer = new Uint8Array(new ArrayBuffer(assistantAnalyser.fftSize))
    const sources: MediaStreamAudioSourceNode[] = []

    const connectTrack = (track: MediaStreamTrack | null, analyser: AnalyserNode) => {
      if (!track || track.readyState === 'ended') return
      try {
        const source = context.createMediaStreamSource(new MediaStream([track]))
        source.connect(analyser)
        sources.push(source)
      } catch {
        // A track can end while LiveKit is replacing it. The next track event
        // will recreate the analyser graph.
      }
    }

    connectTrack(microphoneTrack, microphoneAnalyser)
    assistantTracks.forEach((track) => connectTrack(track, assistantAnalyser))
    // Keep the analyser graph alive without routing a second audible copy of
    // either microphone or assistant audio to the speakers.
    microphoneAnalyser.connect(silentSink)
    assistantAnalyser.connect(silentSink)
    silentSink.connect(context.destination)
    void context.resume().catch(() => undefined)

    let frame = 0
    let lastPaint = 0
    let microphoneLevel = 0
    let assistantLevel = 0

    const render = (timestamp: number) => {
      const nextMicrophone = rms(microphoneAnalyser, microphoneBuffer)
      const nextAssistant = rms(assistantAnalyser, assistantBuffer)
      microphoneLevel = smooth(microphoneLevel, nextMicrophone)
      assistantLevel = smooth(assistantLevel, nextAssistant)
      if (timestamp - lastPaint >= 33) {
        lastPaint = timestamp
        setLevels({ microphone: microphoneLevel, assistant: assistantLevel })
      }
      frame = window.requestAnimationFrame(render)
    }
    frame = window.requestAnimationFrame(render)

    return () => {
      window.cancelAnimationFrame(frame)
      sources.forEach((source) => source.disconnect())
      microphoneAnalyser.disconnect()
      assistantAnalyser.disconnect()
      silentSink.disconnect()
      void context.close().catch(() => undefined)
    }
  }, [microphoneTrack?.id, assistantTrackIds])

  return levels
}

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext
  }
}
