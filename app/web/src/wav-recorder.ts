function encodeWav(buffer: AudioBuffer): Blob {
  const channels = 1
  const samples = buffer.length
  const output = new ArrayBuffer(44 + samples * 2)
  const view = new DataView(output)
  const write = (offset: number, text: string) => [...text].forEach((value, index) => view.setUint8(offset + index, value.charCodeAt(0)))
  write(0, 'RIFF'); view.setUint32(4, 36 + samples * 2, true); write(8, 'WAVE'); write(12, 'fmt ')
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, channels, true)
  view.setUint32(24, buffer.sampleRate, true); view.setUint32(28, buffer.sampleRate * 2, true); view.setUint16(32, 2, true)
  view.setUint16(34, 16, true); write(36, 'data'); view.setUint32(40, samples * 2, true)
  const channel = buffer.getChannelData(0)
  for (let index = 0; index < samples; index += 1) {
    const value = Math.max(-1, Math.min(1, channel[index]))
    view.setInt16(44 + index * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true)
  }
  return new Blob([output], { type: 'audio/wav' })
}

export async function blobToWav(blob: Blob): Promise<Blob> {
  const context = new AudioContext()
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer())
    const mono = context.createBuffer(1, decoded.length, decoded.sampleRate)
    const target = mono.getChannelData(0)
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const source = decoded.getChannelData(channel)
      for (let sample = 0; sample < source.length; sample += 1) target[sample] += source[sample] / decoded.numberOfChannels
    }
    return encodeWav(mono)
  } finally { await context.close() }
}

export async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer())
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary)
}
