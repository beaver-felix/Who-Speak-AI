import { useEffect, useRef, useState } from 'react'
import { Room, RoomEvent, Track } from 'livekit-client'

import { api } from './api'
import { AUTH_COMMAND_TOPIC, AUTH_STATUS_TOPIC, CANCEL_PRIVATE_MODE, parseAuthStatus, REQUEST_PRIVATE_MODE, type AuthStatus } from './auth-status'
import { AGENT_COMMAND_TOPIC, AGENT_EVENT_TOPIC, initialConversationState, parseVoiceAgentEvent, reduceConversation, type ConversationState, type SessionStatus } from './conversation'
import type { AudioPlaybackState, ConnectionState } from './VoiceStage'

const initialAuth: AuthStatus = {
  state: 'guest',
  displayName: null,
  expiresAt: null,
  phase: 'idle',
  elapsedMs: 0,
  targetMs: 5000,
  canResume: false,
  message: null,
  sessionId: null,
  sequence: null,
}

function toError(value: unknown, fallback: string): string {
  return value instanceof Error ? value.message : fallback
}

export function useLiveKitVoiceSession() {
  const roomRef = useRef<Room | null>(null)
  const audioElementsRef = useRef<Map<string, HTMLMediaElement[]>>(new Map())
  const joiningRef = useRef(false)
  const rotatingRef = useRef(false)
  const authSessionIdRef = useRef<string | null>(null)
  const authSequenceRef = useRef(0)
  const [connection, setConnection] = useState<ConnectionState>('idle')
  const [auth, setAuth] = useState<AuthStatus>(initialAuth)
  const [conversation, dispatch] = useState<ConversationState>(initialConversationState)
  const [error, setError] = useState<string | null>(null)
  const [audioPlayback, setAudioPlayback] = useState<AudioPlaybackState>('unknown')
  const [microphoneEnabled, setMicrophoneEnabled] = useState(false)
  const [microphoneTrack, setMicrophoneTrack] = useState<MediaStreamTrack | null>(null)
  const [assistantTracks, setAssistantTracks] = useState<MediaStreamTrack[]>([])

  useEffect(() => () => {
    rotatingRef.current = false
    const room = roomRef.current
    roomRef.current = null
    void room?.disconnect()
    for (const elements of audioElementsRef.current.values()) elements.forEach((element) => element.remove())
    audioElementsRef.current.clear()
  }, [])

  // Keep reducer dispatch local to this hook so the room lifecycle and the
  // conversation history are reset together on a fresh session.
  const conversationDispatch = (event: Parameters<typeof reduceConversation>[1]) => {
    dispatch((current) => reduceConversation(current, event))
  }

  function cleanAudioElements() {
    for (const elements of audioElementsRef.current.values()) {
      elements.forEach((element) => element.remove())
    }
    audioElementsRef.current.clear()
    setAssistantTracks([])
  }

  function resetVisibleSession(status: SessionStatus = 'idle', message?: string) {
    setAuth(initialAuth)
    conversationDispatch({ type: 'reset' })
    conversationDispatch({ type: 'set_session_status', status, message })
  }

  async function join() {
    if (joiningRef.current || roomRef.current) return
    joiningRef.current = true
    authSessionIdRef.current = null
    authSequenceRef.current = 0
    setError(null)
    setConnection('connecting')
    resetVisibleSession('connecting', 'Đang lấy token local…')

    const room = new Room({ adaptiveStream: true, dynacast: true })
    let token: Awaited<ReturnType<typeof api.livekitToken>> | null = null

    const applyAuthStatus = (status: AuthStatus) => {
      if (status.sessionId && authSessionIdRef.current && status.sessionId !== authSessionIdRef.current) return
      if (status.sessionId) authSessionIdRef.current = status.sessionId
      if (status.sequence !== null && status.sequence <= authSequenceRef.current) return
      if (status.sequence !== null) authSequenceRef.current = status.sequence
      setAuth(status)
    }

    const handleEvent = (payload: string, participantIdentity = '') => {
      const event = parseVoiceAgentEvent(payload)
      if (event) conversationDispatch(event)
      const status = parseAuthStatus(payload, participantIdentity, token?.agent_identity)
      if (status) applyAuthStatus(status)
    }

    const rotateWithFreshToken = async () => {
      if (rotatingRef.current || roomRef.current !== room) return
      rotatingRef.current = true
      resetVisibleSession('reconnecting', 'Đang tạo phiên mới…')
      try {
        await room.disconnect()
      } catch {
        // The disconnected event still performs cleanup; a new token is safer
        // than attempting to reuse a possibly stale authenticated room.
      } finally {
        rotatingRef.current = false
        if (!roomRef.current) await join()
      }
    }

    room.registerTextStreamHandler(AUTH_STATUS_TOPIC, async (reader, participant) => {
      if (!token) return
      const status = parseAuthStatus(await reader.readAll(), participant.identity, token.agent_identity)
      if (status) applyAuthStatus(status)
    })
    room.registerTextStreamHandler(AGENT_EVENT_TOPIC, async (reader) => {
      handleEvent(await reader.readAll())
    })
    room.on(RoomEvent.DataReceived, (payload, participant, _kind, topic) => {
      if (!token || (topic && topic !== AGENT_EVENT_TOPIC && topic !== AUTH_STATUS_TOPIC) || token.runtime !== 'pipecat') return
      const text = new TextDecoder().decode(payload)
      const senderIdentity = participant?.identity ?? ''
      if (senderIdentity && senderIdentity !== token.agent_identity) return
      handleEvent(text, senderIdentity)
    })
    room.on(RoomEvent.TrackSubscribed, (track) => {
      if (track.kind !== Track.Kind.Audio) return
      const trackId = track.mediaStreamTrack.id
      if (!audioElementsRef.current.has(trackId)) {
        const element = track.attach()
        element.autoplay = true
        element.setAttribute('aria-label', 'Voice assistant response')
        document.body.appendChild(element)
        audioElementsRef.current.set(trackId, [element])
      }
      setAssistantTracks((current) => current.some((item) => item.id === trackId) ? current : [...current, track.mediaStreamTrack])
    })
    room.on(RoomEvent.TrackUnsubscribed, (track) => {
      if (track.kind !== Track.Kind.Audio) return
      track.detach().forEach((element) => element.remove())
      audioElementsRef.current.delete(track.mediaStreamTrack.id)
      setAssistantTracks((current) => current.filter((item) => item.id !== track.mediaStreamTrack.id))
    })
    room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
      setAudioPlayback(room.canPlaybackAudio ? 'allowed' : 'blocked')
    })
    room.on(RoomEvent.Reconnecting, () => {
      setConnection('reconnecting')
      resetVisibleSession('reconnecting', 'Kết nối bị gián đoạn, đang xác thực lại…')
    })
    room.on(RoomEvent.Reconnected, () => {
      setConnection('connected')
      resetVisibleSession('reconnecting', 'Đang tạo phiên voice mới…')
      void rotateWithFreshToken()
    })
    room.on(RoomEvent.Disconnected, () => {
      // A delayed event from a room replaced during retry/reconnect must not
      // reset the fresh room back to idle.
      if (roomRef.current !== room && !rotatingRef.current) return
      if (roomRef.current === room) roomRef.current = null
      cleanAudioElements()
      setMicrophoneEnabled(false)
      setMicrophoneTrack(null)
      if (rotatingRef.current) {
        setConnection('connecting')
        conversationDispatch({ type: 'set_session_status', status: 'starting', message: 'Đang kết nối phiên mới…' })
      } else {
        setConnection('idle')
        resetVisibleSession()
      }
    })
    room.on(RoomEvent.MediaDevicesError, (reason) => setError(`Microphone error: ${reason.message}`))

    try {
      token = await api.livekitToken()
      room.prepareConnection(token.server_url, token.participant_token)
      await room.connect(token.server_url, token.participant_token)
      roomRef.current = room
      setConnection('connected')
      conversationDispatch({ type: 'set_session_status', status: token.runtime === 'pipecat' ? 'starting' : 'ready', message: token.runtime === 'pipecat' ? 'Local voice engine đang khởi động…' : 'Sẵn sàng lắng nghe.' })
      try {
        await room.startAudio()
        setAudioPlayback('allowed')
      } catch {
        setAudioPlayback('blocked')
      }
      await room.localParticipant.setMicrophoneEnabled(true)
      setMicrophoneEnabled(true)
      setMicrophoneTrack(room.localParticipant.getTrackPublication(Track.Source.Microphone)?.track?.mediaStreamTrack ?? null)
    } catch (caught) {
      await room.disconnect().catch(() => undefined)
      roomRef.current = null
      setConnection('idle')
      resetVisibleSession('failed')
      setError(toError(caught, 'Không thể tham gia local room.'))
    } finally {
      joiningRef.current = false
    }
  }

  async function leave() {
    rotatingRef.current = false
    const room = roomRef.current
    if (room) await room.disconnect()
    else resetVisibleSession()
  }

  async function retrySession() {
    if (joiningRef.current) return
    const room = roomRef.current
    if (room) {
      rotatingRef.current = true
      try {
        await room.disconnect()
      } finally {
        if (roomRef.current === room) {
          roomRef.current = null
          cleanAudioElements()
          setMicrophoneEnabled(false)
          setMicrophoneTrack(null)
        }
        rotatingRef.current = false
      }
    }
    await join()
  }

  async function toggleMicrophone() {
    const room = roomRef.current
    if (!room) return
    try {
      const enabled = !microphoneEnabled
      await room.localParticipant.setMicrophoneEnabled(enabled)
      setMicrophoneEnabled(enabled)
      setMicrophoneTrack(enabled ? room.localParticipant.getTrackPublication(Track.Source.Microphone)?.track?.mediaStreamTrack ?? null : null)
    } catch (caught) {
      setError(toError(caught, 'Không thể thay đổi microphone.'))
    }
  }

  async function sendAuthCommand(command: string) {
    const room = roomRef.current
    if (!room) {
      setError('Hãy join local room trước.')
      return
    }
    try {
      if (command === REQUEST_PRIVATE_MODE) conversationDispatch({ type: 'clear_notice' })
      await room.localParticipant.publishData(new TextEncoder().encode(command), { reliable: true, topic: AUTH_COMMAND_TOPIC })
    } catch (caught) {
      setError(toError(caught, 'Không thể liên hệ Auth Gate.'))
    }
  }

  async function retry(turnId: string) {
    const room = roomRef.current
    if (!room) {
      setError('Hãy join local room trước khi retry.')
      return
    }
    try {
      await room.localParticipant.publishData(new TextEncoder().encode(JSON.stringify({ action: 'retry', turn_id: turnId })), { reliable: true, topic: AGENT_COMMAND_TOPIC })
    } catch (caught) {
      setError(toError(caught, 'Không thể retry response.'))
    }
  }

  async function enableAudio() {
    const room = roomRef.current
    if (!room) return
    try {
      await room.startAudio()
      setAudioPlayback('allowed')
      setError(null)
    } catch (caught) {
      setError(toError(caught, 'Browser vẫn đang chặn audio. Hãy bấm lại hoặc kiểm tra quyền âm thanh.'))
    }
  }

  return {
    connection,
    auth,
    conversation,
    error,
    audioPlayback,
    microphoneEnabled,
    microphoneTrack,
    assistantTracks,
    join,
    leave,
    retrySession,
    toggleMicrophone,
    sendAuthCommand,
    retry,
    enableAudio,
  }
}
