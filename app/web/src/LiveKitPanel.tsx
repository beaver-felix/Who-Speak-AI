import { useEffect, useReducer, useRef, useState } from 'react'
import { Room, RoomEvent, Track } from 'livekit-client'

import './conversation-status.css'
import { api } from './api'
import { AUTH_COMMAND_TOPIC, AUTH_STATUS_TOPIC, CANCEL_PRIVATE_MODE, parseAuthStatus, REQUEST_PRIVATE_MODE, type AuthStatus } from './auth-status'
import { AGENT_COMMAND_TOPIC, AGENT_EVENT_TOPIC, initialConversationState, parseVoiceAgentEvent, reduceConversation, type ChatMessage, type VoiceAgentState } from './conversation'

const initialAuth: AuthStatus = { state: 'guest', displayName: null, expiresAt: null }

function authLabel(state: AuthStatus['state']): string {
  return { guest: 'Guest', auth_pending: 'Voice challenge', authenticated: 'Authenticated', session_expired: 'Session expired' }[state]
}

function turnLabel(state: VoiceAgentState | undefined): string {
  return {
    listening: 'Listening',
    transcribing: 'Transcribing',
    thinking: 'Thinking',
    speaking: 'Speaking',
    completed: 'Played',
    interrupted: 'Interrupted',
    error: 'Needs attention',
  }[state ?? 'completed']
}

function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(timestamp)
}

function MessageBubble({ message, state, retry }: { message: ChatMessage; state?: { state: VoiceAgentState; message?: string }; retry: () => void }) {
  const canRetry = message.role === 'user' && state?.state === 'error'
  const audioState = message.role === 'assistant' && state?.state === 'speaking' ? 'Speaking' : message.role === 'assistant' && state?.state === 'completed' ? 'Played' : message.role === 'assistant' && state?.state === 'interrupted' ? 'Interrupted' : null
  return <article className={`message ${message.role}${message.provisional ? ' provisional' : ''}`} aria-label={message.role === 'user' ? 'Your transcript' : 'Agent response'}>
    <div className="message-meta"><span>{message.role === 'user' ? (message.provisional ? 'You · voice input' : 'You · ASR final') : 'Who Speak AI'}</span><time>{formatTime(message.createdAt)}</time></div>
    <p>{message.text}</p>
    {message.tool?.demo && <span className="tool-badge">Demo Calendar · Mock MCP</span>}
    {audioState && <span className={`audio-state ${state?.state}`}>{audioState}</span>}
    {state?.state === 'error' && <div className="turn-error" role="alert"><span>{state.message ?? 'This turn could not finish.'}</span>{canRetry && <button type="button" className="text-button" onClick={retry}>Retry response</button>}</div>}
  </article>
}

export function LiveKitPanel() {
  const roomRef = useRef<Room | null>(null)
  const audioElementsRef = useRef<Set<HTMLMediaElement>>(new Set())
  const threadRef = useRef<HTMLDivElement | null>(null)
  const [connection, setConnection] = useState<'idle' | 'connecting' | 'connected'>('idle')
  const [auth, setAuth] = useState<AuthStatus>(initialAuth)
  const [error, setError] = useState<string | null>(null)
  const [conversation, dispatch] = useReducer(reduceConversation, initialConversationState)
  const [showNewResponse, setShowNewResponse] = useState(false)

  useEffect(() => () => {
    void roomRef.current?.disconnect()
    audioElementsRef.current.forEach((element) => element.remove())
    audioElementsRef.current.clear()
  }, [])

  useEffect(() => {
    const thread = threadRef.current
    if (!thread) return
    const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 96
    if (nearBottom) {
      thread.scrollTo({ top: thread.scrollHeight, behavior: 'smooth' })
      setShowNewResponse(false)
    } else if (conversation.messages.length > 0) setShowNewResponse(true)
  }, [conversation.messages.length])

  function cleanAudioElements() {
    audioElementsRef.current.forEach((element) => element.remove())
    audioElementsRef.current.clear()
  }

  async function join() {
    setError(null)
    setConnection('connecting')
    dispatch({ type: 'reset' })
    try {
      const token = await api.livekitToken()
      const room = new Room({ adaptiveStream: true, dynacast: true })
      room.registerTextStreamHandler(AUTH_STATUS_TOPIC, async (reader, participant) => {
        const status = parseAuthStatus(await reader.readAll(), participant.identity)
        if (status) setAuth(status)
      })
      room.registerTextStreamHandler(AGENT_EVENT_TOPIC, async (reader) => {
        const event = parseVoiceAgentEvent(await reader.readAll())
        if (event) dispatch(event)
      })
      room.on(RoomEvent.DataReceived, (payload, participant, _kind, topic) => {
        // Pipecat 1.8.1 uses the public LiveKit data API without a topic. A
        // server-originated packet may have no participant object, so accept
        // only untagged packets in Pipecat mode or packets from this room's
        // expected agent identity. Legacy topic handlers above remain intact.
        if (topic || token.runtime !== 'pipecat') return
        const text = new TextDecoder().decode(payload)
        const senderIdentity = participant?.identity ?? ''
        if (senderIdentity && senderIdentity !== token.agent_identity) return
        const event = parseVoiceAgentEvent(text)
        if (event) dispatch(event)
        const status = parseAuthStatus(text, senderIdentity, token.agent_identity)
        if (status) setAuth(status)
      })
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return
        const element = track.attach()
        element.autoplay = true
        element.setAttribute('aria-label', 'Voice assistant response')
        document.body.appendChild(element)
        audioElementsRef.current.add(element)
      })
      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        if (track.kind !== Track.Kind.Audio) return
        track.detach().forEach((element) => {
          audioElementsRef.current.delete(element)
          element.remove()
        })
      })
      room.on(RoomEvent.Disconnected, () => {
        roomRef.current = null
        cleanAudioElements()
        setConnection('idle')
        setAuth(initialAuth)
        dispatch({ type: 'reset' })
      })
      room.on(RoomEvent.MediaDevicesError, (reason) => setError(`Microphone error: ${reason.message}`))
      await room.connect(token.server_url, token.participant_token)
      await room.startAudio()
      await room.localParticipant.setMicrophoneEnabled(true)
      roomRef.current = room
      setConnection('connected')
    } catch (caught) {
      roomRef.current?.disconnect()
      roomRef.current = null
      setConnection('idle')
      setError(caught instanceof Error ? caught.message : 'Could not join the local room.')
    }
  }

  async function sendAuthCommand(command: string) {
    try {
      if (command === REQUEST_PRIVATE_MODE) dispatch({ type: 'clear_notice' })
      await roomRef.current?.localParticipant.publishData(new TextEncoder().encode(command), { reliable: true, topic: AUTH_COMMAND_TOPIC })
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not contact the Auth Gate.') }
  }

  async function retry(turnId: string) {
    try {
      await roomRef.current?.localParticipant.publishData(
        new TextEncoder().encode(JSON.stringify({ action: 'retry', turn_id: turnId })),
        { reliable: true, topic: AGENT_COMMAND_TOPIC },
      )
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not retry this response.') }
  }

  const connected = connection === 'connected'
  const activeTurn = Object.values(conversation.turnStates).sort((left, right) => right.updatedAt - left.updatedAt)[0]
  const awaitingChallenge = auth.state === 'auth_pending'
  const visibleError = error ?? conversation.notice

  return <section className="assistant-layout" aria-label="Voice assistant">
    <aside className="session-panel">
      <div className="panel-heading"><p className="eyebrow">Voice session</p><h2>{connected ? 'Connected locally' : 'Ready to connect'}</h2></div>
      <div className="auth-summary"><span className={`status-dot ${auth.state}`} aria-hidden="true" /><div><strong>{authLabel(auth.state)}</strong><small>{auth.displayName ? `Signed in as ${auth.displayName}` : 'Private tools are locked'}</small></div></div>
      <div className="voice-meter" aria-label={connected ? 'Microphone active' : 'Microphone inactive'}><span /><span /><span /><span /><span /><span /><span /></div>
      <p className="session-state"><span className={`state-icon ${activeTurn?.state ?? 'completed'}`} aria-hidden="true" />{connected ? turnLabel(activeTurn?.state) : 'Microphone off'}</p>
      <div className="session-actions">
        {!connected ? <button className="primary" type="button" onClick={() => void join()} disabled={connection === 'connecting'}>{connection === 'connecting' ? 'Joining…' : 'Join local room'}</button> : <><button className="primary" type="button" disabled={awaitingChallenge || auth.state === 'authenticated'} onClick={() => void sendAuthCommand(REQUEST_PRIVATE_MODE)}>Start voice challenge</button>{awaitingChallenge && <button className="secondary" type="button" onClick={() => void sendAuthCommand(CANCEL_PRIVATE_MODE)}>Cancel challenge</button>}<button className="danger" type="button" onClick={() => roomRef.current?.disconnect()}>Leave room</button></>}
      </div>
      <p className="privacy-note">Audio stays in local LiveKit. The chat never receives your embedding, HE key, score, or API keys.</p>
    </aside>

    <section className="conversation-panel">
      <header className="conversation-header"><div><p className="eyebrow">Conversation</p><h2>Voice chat</h2></div><span className={`connection-chip ${connection}`}>{connection === 'connected' ? 'Microphone on' : connection === 'connecting' ? 'Connecting' : 'Offline'}</span></header>
      <div className="chat-thread" ref={threadRef} onScroll={() => setShowNewResponse(false)} aria-live="polite">
        {!conversation.messages.length && <div className="empty-chat"><strong>{connected ? 'Start speaking when you are ready.' : 'Join the local room to begin.'}</strong><p>You will see Listening, Transcribing, and the final transcript before the Agent reply. Calendar data is always demo-only in this phase.</p></div>}
        {conversation.messages.map((message) => <MessageBubble key={message.id} message={message} state={conversation.turnStates[message.turnId]} retry={() => void retry(message.turnId)} />)}
      </div>
      {showNewResponse && <button className="new-response" type="button" onClick={() => { threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: 'smooth' }); setShowNewResponse(false) }}>New response ↓</button>}
      {visibleError && <p className="error" role="alert">{visibleError}</p>}
      <footer className="chat-footer"><span className={`state-icon ${activeTurn?.state ?? 'completed'}`} aria-hidden="true" />{connected ? turnLabel(activeTurn?.state) : 'Waiting for connection'}<span className="footer-separator">·</span>Final ASR · Streaming reply</footer>
    </section>
  </section>
}
