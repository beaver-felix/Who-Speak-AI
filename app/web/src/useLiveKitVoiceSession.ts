import { useEffect, useRef, useState } from 'react'
import { Room, RoomEvent, Track } from 'livekit-client'

import { api } from './api'
import { AUTH_COMMAND_TOPIC, AUTH_STATUS_TOPIC, parseAuthStatus, REQUEST_PRIVATE_MODE, type AuthStatus } from './auth-status'
import { AGENT_COMMAND_TOPIC, AGENT_EVENT_TOPIC, initialConversationState, parseVoiceAgentEvent, reduceConversation, type ConversationState, type SessionStatus } from './conversation'
import { decideRoomLifecycle, statusAfterTransientReconnect } from './room-lifecycle'
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
  const leavingRef = useRef(false)
  const roomGenerationRef = useRef(0)
  const stableSessionStatusRef = useRef<SessionStatus>('idle')
  const transientReconnectRef = useRef(false)
  const startupFailedRef = useRef(false)
  const intentionalDisconnectRef = useRef<Room | null>(null)
  const replacementPromiseRef = useRef<Promise<void> | null>(null)
  const agentRecoveryTimerRef = useRef<number | null>(null)
  const reconnectNoticeTimerRef = useRef<number | null>(null)
  const authSessionIdRef = useRef<string | null>(null)
  const agentIdentityRef = useRef<string | null>(null)
  const authSequenceRef = useRef(0)
  const [connection, setConnection] = useState<ConnectionState>('idle')
  const [auth, setAuth] = useState<AuthStatus>(initialAuth)
  const [conversation, dispatch] = useState<ConversationState>(initialConversationState)
  const [error, setError] = useState<string | null>(null)
  const [audioPlayback, setAudioPlayback] = useState<AudioPlaybackState>('unknown')
  const [microphoneEnabled, setMicrophoneEnabled] = useState(false)
  const [microphoneTrack, setMicrophoneTrack] = useState<MediaStreamTrack | null>(null)
  const [assistantTracks, setAssistantTracks] = useState<MediaStreamTrack[]>([])
  const authCommandTimerRef = useRef<number | null>(null)
  const [authCommandPending, setAuthCommandPending] = useState(false)

  function clearAuthCommandWait() {
    if (authCommandTimerRef.current !== null) {
      window.clearTimeout(authCommandTimerRef.current)
      authCommandTimerRef.current = null
    }
    setAuthCommandPending(false)
  }

  function clearAgentRecoveryTimer() {
    if (agentRecoveryTimerRef.current !== null) {
      window.clearTimeout(agentRecoveryTimerRef.current)
      agentRecoveryTimerRef.current = null
    }
  }

  function clearReconnectNoticeTimer() {
    if (reconnectNoticeTimerRef.current !== null) {
      window.clearTimeout(reconnectNoticeTimerRef.current)
      reconnectNoticeTimerRef.current = null
    }
  }

  useEffect(() => () => {
    roomGenerationRef.current += 1
    rotatingRef.current = false
    leavingRef.current = true
    clearAgentRecoveryTimer()
    clearReconnectNoticeTimer()
    clearAuthCommandWait()
    const room = roomRef.current
    intentionalDisconnectRef.current = room
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

  function setSessionStatus(status: SessionStatus, message?: string) {
    if (status === 'starting' || status === 'ready') stableSessionStatusRef.current = status
    conversationDispatch({ type: 'set_session_status', status, message })
  }

  function cleanAudioElements() {
    for (const elements of audioElementsRef.current.values()) {
      elements.forEach((element) => element.remove())
    }
    audioElementsRef.current.clear()
    setAssistantTracks([])
  }

  function resetVisibleSession(status: SessionStatus = 'idle', message?: string) {
    clearReconnectNoticeTimer()
    transientReconnectRef.current = false
    startupFailedRef.current = false
    stableSessionStatusRef.current = status === 'starting' || status === 'ready' ? status : 'idle'
    setAuth(initialAuth)
    conversationDispatch({ type: 'reset' })
    setSessionStatus(status, message)
  }

  async function replaceCurrentRoom(room: Room, message: string) {
    if (leavingRef.current || rotatingRef.current || roomRef.current !== room) return
    if (replacementPromiseRef.current) return replacementPromiseRef.current

    let replacement: Promise<void>
    replacement = (async () => {
      rotatingRef.current = true
      startupFailedRef.current = false
      clearAgentRecoveryTimer()
      clearReconnectNoticeTimer()
      setConnection('reconnecting')
      resetVisibleSession('reconnecting', message)
      intentionalDisconnectRef.current = room
      roomRef.current = null
      clearAuthCommandWait()
      cleanAudioElements()
      setMicrophoneEnabled(false)
      setMicrophoneTrack(null)

      try {
        await room.disconnect()
      } catch {
        // The old room is no longer reusable. Continue with a fresh token.
      } finally {
        if (intentionalDisconnectRef.current === room) intentionalDisconnectRef.current = null
        rotatingRef.current = false
        if (!leavingRef.current) await join()
        else {
          setConnection('idle')
          resetVisibleSession()
        }
      }
    })()
    replacementPromiseRef.current = replacement
    try {
      await replacement
    } finally {
      if (intentionalDisconnectRef.current === room) intentionalDisconnectRef.current = null
      if (replacementPromiseRef.current === replacement) replacementPromiseRef.current = null
    }
  }

  async function join() {
    if (joiningRef.current || roomRef.current || rotatingRef.current) return
    joiningRef.current = true
    leavingRef.current = false
    clearReconnectNoticeTimer()
    transientReconnectRef.current = false
    startupFailedRef.current = false
    const roomGeneration = roomGenerationRef.current + 1
    roomGenerationRef.current = roomGeneration
    authSessionIdRef.current = null
    agentIdentityRef.current = null
    authSequenceRef.current = 0
    setError(null)
    setConnection('connecting')
    resetVisibleSession('connecting', 'Đang lấy token local…')

    const room = new Room({ adaptiveStream: true, dynacast: true })
    let token: Awaited<ReturnType<typeof api.livekitToken>> | null = null
    let connectedOnce = false

    const isCurrentRoom = () => roomRef.current === room && roomGenerationRef.current === roomGeneration && !leavingRef.current

    const restoreAfterReconnect = () => {
      const status: SessionStatus = statusAfterTransientReconnect(stableSessionStatusRef.current)
      setSessionStatus(status, status === 'ready'
        ? 'Đã kết nối lại. Phiên voice hiện tại vẫn được giữ nguyên.'
        : 'Đã kết nối lại. Voice engine tiếp tục khởi động…')
    }

    const applyAuthStatus = (status: AuthStatus) => {
      if (status.sessionId && authSessionIdRef.current && status.sessionId !== authSessionIdRef.current) return
      if (status.sessionId) authSessionIdRef.current = status.sessionId
      if (status.sequence !== null && status.sequence <= authSequenceRef.current) return
      if (status.sequence !== null) authSequenceRef.current = status.sequence
      clearAuthCommandWait()
      setAuth(status)
    }

    const handleEvent = (payload: string, participantIdentity = '') => {
      if (!isCurrentRoom()) return
      const event = parseVoiceAgentEvent(payload)
      if (event?.type === 'session_status' && event.status === 'failed') {
        // A startup failure is terminal for this Agent participant. Keep the
        // error visible and let the user explicitly retry instead of treating
        // the participant exit as a transient reconnect.
        startupFailedRef.current = true
        transientReconnectRef.current = false
        clearReconnectNoticeTimer()
        setConnection('failed')
      } else if (event?.type === 'session_status' && (event.status === 'connecting' || event.status === 'starting')) {
        startupFailedRef.current = false
      }
      const suppressTransientStatus = transientReconnectRef.current
        && event?.type === 'session_status'
        && event.status !== 'failed'
      if (event && !suppressTransientStatus) {
        if (event.type === 'session_status' && (event.status === 'starting' || event.status === 'ready')) {
          stableSessionStatusRef.current = event.status
        }
        conversationDispatch(event)
      }
      const status = parseAuthStatus(payload, participantIdentity, token?.agent_identity)
      if (status) applyAuthStatus(status)
    }

    room.registerTextStreamHandler(AUTH_STATUS_TOPIC, async (reader, participant) => {
      if (!token || !isCurrentRoom()) return
      const payload = await reader.readAll()
      if (!isCurrentRoom()) return
      const status = parseAuthStatus(payload, participant.identity, token.agent_identity)
      if (status) applyAuthStatus(status)
    })
    room.registerTextStreamHandler(AGENT_EVENT_TOPIC, async (reader) => {
      if (!isCurrentRoom()) return
      const payload = await reader.readAll()
      if (!isCurrentRoom()) return
      handleEvent(payload)
    })
    room.on(RoomEvent.DataReceived, (payload, participant, _kind, topic) => {
      if (!token || !isCurrentRoom() || (topic && topic !== AGENT_EVENT_TOPIC && topic !== AUTH_STATUS_TOPIC) || token.runtime !== 'pipecat') return
      const text = new TextDecoder().decode(payload)
      const senderIdentity = participant?.identity ?? ''
      if (senderIdentity && senderIdentity !== token.agent_identity) return
      handleEvent(text, senderIdentity)
    })
    room.on(RoomEvent.TrackSubscribed, (track) => {
      if (!isCurrentRoom() || track.kind !== Track.Kind.Audio) return
      const trackId = track.mediaStreamTrack.id
      if (!audioElementsRef.current.has(trackId)) {
        const element = track.attach()
        element.autoplay = true
        element.setAttribute('aria-label', 'Voice assistant response')
        document.body.appendChild(element)
        audioElementsRef.current.set(trackId, [element])
        // `attach()` creates the correct media element, but a browser can
        // still require an explicit play attempt after an asynchronous room
        // join. Keep the failure visible so a user can use the existing
        // “Bật âm thanh Agent” recovery control instead of silently losing
        // every remote TTS track.
        void element.play().then(() => {
          if (isCurrentRoom()) setAudioPlayback('allowed')
        }).catch((error: unknown) => {
          if (!isCurrentRoom()) return
          console.warn('[voice-session] assistant audio playback blocked', {
            track_id: trackId,
            error: error instanceof Error ? error.name : 'unknown',
          })
          setAudioPlayback('blocked')
        })
      }
      setAssistantTracks((current) => current.some((item) => item.id === trackId) ? current : [...current, track.mediaStreamTrack])
    })
    room.on(RoomEvent.TrackUnsubscribed, (track) => {
      if (track.kind !== Track.Kind.Audio) return
      track.detach().forEach((element) => element.remove())
      if (!isCurrentRoom()) return
      audioElementsRef.current.delete(track.mediaStreamTrack.id)
      setAssistantTracks((current) => current.filter((item) => item.id !== track.mediaStreamTrack.id))
    })
    room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
      if (!isCurrentRoom()) return
      setAudioPlayback(room.canPlaybackAudio ? 'allowed' : 'blocked')
    })
    room.on(RoomEvent.Reconnecting, () => {
      const decision = decideRoomLifecycle({
        event: 'reconnecting',
        isCurrentRoom: isCurrentRoom(),
        intentionalDisconnect: intentionalDisconnectRef.current === room,
        leaving: leavingRef.current,
        replacementInFlight: rotatingRef.current,
      })
      if (decision !== 'preserve') return
      console.info('[voice-session] room lifecycle', {
        event: 'room_reconnecting',
        session_id: token?.session_id ?? null,
        active_room_generation: roomGeneration,
      })
      transientReconnectRef.current = true
      clearReconnectNoticeTimer()
      setConnection('reconnecting')
      // Keep the last stable status during a short ICE/signalling recovery.
      // Controls are disabled immediately through connection state, but the
      // status card is delayed to avoid a distracting reconnect flash.
      reconnectNoticeTimerRef.current = window.setTimeout(() => {
        reconnectNoticeTimerRef.current = null
        if (isCurrentRoom() && transientReconnectRef.current) {
          setSessionStatus('reconnecting', 'Kết nối tạm thời bị gián đoạn. Đang khôi phục phiên hiện tại…')
        }
      }, 1000)
    })
    room.on(RoomEvent.Reconnected, () => {
      const decision = decideRoomLifecycle({
        event: 'reconnected',
        isCurrentRoom: isCurrentRoom(),
        intentionalDisconnect: intentionalDisconnectRef.current === room,
        leaving: leavingRef.current,
        replacementInFlight: rotatingRef.current,
      })
      if (decision !== 'restore') return
      transientReconnectRef.current = false
      clearReconnectNoticeTimer()
      console.info('[voice-session] room lifecycle', {
        event: 'room_reconnected',
        session_id: token?.session_id ?? null,
        active_room_generation: roomGeneration,
      })
      setConnection('connected')
      setError(null)
      restoreAfterReconnect()
    })
    room.on(RoomEvent.ParticipantConnected, (participant) => {
      if (!isCurrentRoom() || participant.identity !== token?.agent_identity || agentRecoveryTimerRef.current === null) return
      clearAgentRecoveryTimer()
      clearReconnectNoticeTimer()
      setConnection('connected')
      setError(null)
      restoreAfterReconnect()
    })
    room.on(RoomEvent.ParticipantDisconnected, (participant) => {
      // The browser can remain connected after the one-room Pipecat worker
      // disappears. Without this recovery path the UI keeps the last auth
      // progress forever while audio is sent to a room with no Agent.
      if (!isCurrentRoom() || participant.identity !== token?.agent_identity || leavingRef.current || rotatingRef.current) return
      if (startupFailedRef.current) {
        clearAgentRecoveryTimer()
        clearReconnectNoticeTimer()
        // The transport may still be connected, but there is no Agent left
        // to process audio. Do not create a new room automatically after a
        // startup failure; the retry card is the explicit recovery action.
        setConnection('failed')
        return
      }
      console.info('[voice-session] room lifecycle', {
        event: 'agent_disconnected',
        session_id: token?.session_id ?? null,
        active_room_generation: roomGeneration,
      })
      setConnection('reconnecting')
      clearReconnectNoticeTimer()
      clearAgentRecoveryTimer()
      agentRecoveryTimerRef.current = window.setTimeout(() => {
        agentRecoveryTimerRef.current = null
        if (!isCurrentRoom()) return
        setError('Local Agent đã ngắt kết nối, đang tạo phiên mới…')
        setSessionStatus('reconnecting', 'Agent tạm thời bị gián đoạn. Đang kiểm tra kết nối…')
        void replaceCurrentRoom(room, 'Phiên voice đã kết thúc. Đang tạo phiên mới, bạn sẽ cần xác thực lại.')
      }, 1000)
    })
    room.on(RoomEvent.Disconnected, (reason) => {
      // A failed initial connect is handled by join()'s catch block. Do not
      // start a second replacement while that connect operation is unwinding.
      if (joiningRef.current && !connectedOnce) return
      const decision = decideRoomLifecycle({
        event: 'disconnected',
        isCurrentRoom: isCurrentRoom(),
        intentionalDisconnect: intentionalDisconnectRef.current === room,
        leaving: leavingRef.current,
        replacementInFlight: rotatingRef.current,
      })
      if (decision === 'ignore') return
      if (decision === 'preserve') {
        clearReconnectNoticeTimer()
        if (intentionalDisconnectRef.current === room) intentionalDisconnectRef.current = null
        if (roomRef.current === room) roomRef.current = null
        return
      }
      if (startupFailedRef.current) {
        clearReconnectNoticeTimer()
        setConnection('failed')
        return
      }
      console.info('[voice-session] room disconnected', {
        event: 'disconnected',
        session_id: token?.session_id ?? null,
        active_room_generation: roomGeneration,
        reason: reason ? String(reason) : 'unknown',
      })
      clearReconnectNoticeTimer()
      void replaceCurrentRoom(room, 'Phiên voice đã kết thúc. Đang tạo phiên mới, bạn sẽ cần xác thực lại.')
    })
    room.on(RoomEvent.MediaDevicesError, (reason) => {
      if (isCurrentRoom()) setError(`Microphone error: ${reason.message}`)
    })

    try {
      token = await api.livekitToken()
      if (roomGenerationRef.current !== roomGeneration || leavingRef.current) return
      agentIdentityRef.current = token.agent_identity
      roomRef.current = room
      room.prepareConnection(token.server_url, token.participant_token)
      await room.connect(token.server_url, token.participant_token)
      connectedOnce = true
      if (!isCurrentRoom()) return
      setConnection('connected')
      setSessionStatus(token.runtime === 'pipecat' ? 'starting' : 'ready', token.runtime === 'pipecat' ? 'Local voice engine đang khởi động…' : 'Sẵn sàng lắng nghe.')
      try {
        await room.startAudio()
        if (isCurrentRoom()) setAudioPlayback('allowed')
      } catch {
        if (isCurrentRoom()) setAudioPlayback('blocked')
      }
      await room.localParticipant.setMicrophoneEnabled(true)
      if (!isCurrentRoom()) return
      setMicrophoneEnabled(true)
      setMicrophoneTrack(room.localParticipant.getTrackPublication(Track.Source.Microphone)?.track?.mediaStreamTrack ?? null)
    } catch (caught) {
      intentionalDisconnectRef.current = room
      if (roomRef.current === room) roomRef.current = null
      await room.disconnect().catch(() => undefined)
      if (roomGenerationRef.current !== roomGeneration || leavingRef.current) return
      setConnection('idle')
      resetVisibleSession('failed')
      setError(toError(caught, 'Không thể tham gia local room.'))
    } finally {
      joiningRef.current = false
    }
  }

  async function leave() {
    leavingRef.current = true
    rotatingRef.current = false
    clearAgentRecoveryTimer()
    clearReconnectNoticeTimer()
    const room = roomRef.current
    if (room) {
      intentionalDisconnectRef.current = room
      roomRef.current = null
      await room.disconnect()
      if (intentionalDisconnectRef.current === room) intentionalDisconnectRef.current = null
    }
    cleanAudioElements()
    clearAuthCommandWait()
    setMicrophoneEnabled(false)
    setMicrophoneTrack(null)
    setConnection('idle')
    resetVisibleSession()
  }

  async function retrySession() {
    if (joiningRef.current || replacementPromiseRef.current) return
    leavingRef.current = false
    const room = roomRef.current
    if (room) {
      await replaceCurrentRoom(room, 'Đang tạo phiên voice mới…')
      return
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
      if (authCommandTimerRef.current !== null) window.clearTimeout(authCommandTimerRef.current)
      setAuthCommandPending(true)
      setError(null)
      if (command === REQUEST_PRIVATE_MODE) conversationDispatch({ type: 'clear_notice' })
      const wireCommand = JSON.stringify({ action: command })
      await room.localParticipant.publishData(new TextEncoder().encode(wireCommand), {
        reliable: true,
        topic: AUTH_COMMAND_TOPIC,
        destinationIdentities: agentIdentityRef.current ? [agentIdentityRef.current] : undefined,
      })
      authCommandTimerRef.current = window.setTimeout(() => {
        authCommandTimerRef.current = null
        setAuthCommandPending(false)
        setError('Agent không phản hồi yêu cầu voice. Hãy khởi động lại Pipecat supervisor rồi thử lại.')
      }, 3000)
    } catch (caught) {
      clearAuthCommandWait()
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
    authCommandPending,
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
