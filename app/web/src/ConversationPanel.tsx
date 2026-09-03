import { useEffect, useMemo, useRef, useState } from 'react'

import type { AuthStatus } from './auth-status'
import type { ChatMessage, ConversationState, SessionStatus, TurnStatus } from './conversation'
import type { ConnectionState } from './VoiceStage'

type ConversationPanelProps = {
  conversation: ConversationState
  auth: AuthStatus
  connection: ConnectionState
  mobileOpen: boolean
  onCloseMobile: () => void
  onRetry: (turnId: string) => void
  onRetrySession: () => void
}

function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(timestamp)
}

function toolLabel(message: ChatMessage): string | null {
  if (!message.tool) return null
  if (message.tool.provider === 'google_mcp') {
    return message.tool.name === 'calendar.create_event' ? 'Google Calendar · create' : 'Google Calendar · MCP'
  }
  if (!message.tool.demo) return null
  return message.tool.name === 'calendar.create_event' ? 'Demo Calendar · create' : 'Demo Calendar · Mock MCP'
}

function audioLabel(status: TurnStatus | undefined, message: ChatMessage): string | null {
  if (message.role !== 'assistant' || !status) return null
  const labels: Partial<Record<TurnStatus['state'], string>> = {
    speaking: 'Speaking',
    completed: 'Played',
    interrupted: 'Interrupted',
    error: 'Audio unavailable',
  }
  return labels[status.state] ?? null
}

function turnStateLabel(state: TurnStatus['state'] | undefined): string | null {
  if (!state) return null
  return {
    listening: 'Listening',
    transcribing: 'Transcribing',
    thinking: 'Thinking',
    speaking: 'Speaking',
    completed: 'Played',
    interrupted: 'Interrupted',
    error: 'Needs attention',
  }[state]
}

type SessionStatusCardProps = {
  status: SessionStatus
  message: string | null
  onRetry: () => void
}

function sessionStatusCopy(status: SessionStatus, message: string | null): {
  title: string
  detail: string
  tone: 'loading' | 'ready' | 'error' | 'neutral'
  retry: boolean
} | null {
  if (status === 'idle') return null
  if (status === 'connecting') {
    return { title: 'Đang kết nối local room…', detail: 'Đang thiết lập phiên voice local.', tone: 'loading', retry: false }
  }
  if (status === 'starting') {
    return { title: 'Local voice engine đang khởi động…', detail: 'Whisper, VAD và TTS local đang được chuẩn bị. Vui lòng chờ đến khi hiển thị “Sẵn sàng lắng nghe”.', tone: 'loading', retry: false }
  }
  if (status === 'ready') {
    return { title: 'Sẵn sàng lắng nghe', detail: 'Bạn có thể bắt đầu nói.', tone: 'ready', retry: false }
  }
  if (status === 'reconnecting') {
    const freshSession = message?.includes('tạo phiên mới') || message?.includes('xác thực lại')
    return {
      title: freshSession ? 'Phiên voice đã kết thúc' : 'Kết nối tạm thời bị gián đoạn',
      detail: message ?? 'Đang khôi phục phiên voice hiện tại…',
      tone: 'loading',
      retry: false,
    }
  }
  if (status === 'stopping') {
    return { title: 'Đang kết thúc phiên…', detail: 'Audio conversation đang được dọn dẹp.', tone: 'neutral', retry: false }
  }
  return { title: 'Local voice engine không khởi động được', detail: message || 'Kiểm tra Agent, Whisper hoặc TTS local rồi thử lại.', tone: 'error', retry: true }
}

function SessionStatusCard({ status, message, onRetry }: SessionStatusCardProps) {
  const copy = sessionStatusCopy(status, message)
  if (!copy) return null
  const busy = copy.tone === 'loading'
  return <div className={`session-status-card ${copy.tone}`} role="status" aria-live="polite" aria-atomic="true">
    <span className="session-status-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" focusable="false">
        {busy && <circle className="session-status-spinner" cx="12" cy="12" r="8" />}
        {copy.tone === 'ready' && <path d="m7 12 3.2 3.2L17.5 8" />}
        {copy.tone === 'error' && <path d="M12 7v6m0 4h.01M4.7 19h14.6a1.2 1.2 0 0 0 1.04-1.8L13.3 4.6a1.5 1.5 0 0 0-2.6 0L3.66 17.2A1.2 1.2 0 0 0 4.7 19Z" />}
        {copy.tone === 'neutral' && <circle cx="12" cy="12" r="8" />}
      </svg>
    </span>
    <div className="session-status-copy">
      <strong>{copy.title}</strong>
      <p>{copy.detail}</p>
    </div>
    {copy.retry && <button className="secondary session-status-retry" type="button" onClick={onRetry}>Thử lại</button>}
  </div>
}

function AuthChallengeCard({ auth }: { auth: AuthStatus }) {
  if (auth.phase === 'idle' || auth.phase === 'conversation_ready') return null
  const capturing = auth.phase === 'capturing'
  const processing = auth.phase === 'processing'
  const success = auth.phase === 'waiting_for_resume' && auth.state === 'authenticated'
  const title = capturing
    ? `Đang thu giọng nói · ${(auth.elapsedMs / 1000).toFixed(1)}/${(auth.targetMs / 1000).toFixed(1)} giây`
    : processing
      ? 'Đã thu đủ 5 giây · Đang kiểm tra giọng nói…'
      : success
        ? 'Xác thực thành công'
        : 'Chưa xác thực được'
  const detail = capturing
    ? 'Hãy nói tự nhiên. Khi đủ thời gian, vui lòng dừng nói. Phần nói dư sẽ không đi vào chat.'
    : processing
      ? 'Vui lòng chờ, chưa bắt đầu nói với Agent.'
      : success
        ? 'Bấm “Bắt đầu nói với Agent” ở khung Voice session để trò chuyện.'
        : 'Bạn có thể thử lại voice hoặc tiếp tục ở chế độ Guest.'
  return <div className={`auth-challenge-card ${capturing ? 'capturing' : processing ? 'processing' : 'waiting'}`} role="status" aria-live="polite" aria-atomic="true">
    <span className="auth-challenge-icon" aria-hidden="true">{capturing || processing ? <span className="auth-spinner" /> : success ? '✓' : '!'}</span>
    <div className="auth-challenge-copy">
      <strong>{title}</strong>
      <p>{detail}</p>
      {capturing && <div className="challenge-progress-bar" role="progressbar" aria-label="Tiến trình voice challenge" aria-valuemin={0} aria-valuemax={auth.targetMs} aria-valuenow={Math.min(auth.elapsedMs, auth.targetMs)}><span style={{ width: `${Math.min(100, auth.targetMs ? auth.elapsedMs / auth.targetMs * 100 : 0)}%` }} /></div>}
    </div>
  </div>
}

function MessageBubble({ message, status }: { message: ChatMessage; status?: TurnStatus }) {
  const tool = toolLabel(message)
  const audio = audioLabel(status, message)
  return <article className={`message ${message.role}${message.provisional ? ' provisional' : ''}`} aria-label={message.role === 'user' ? 'Your final transcript' : 'Agent response'}>
    <div className="message-meta"><span>{message.role === 'user' ? 'You · ASR final' : 'Agent'}</span><time>{formatTime(message.createdAt)}</time></div>
    <p>{message.text}</p>
    <div className="message-badges">
      {tool && <span className="tool-badge">{tool}</span>}
      {audio && <span className={`audio-state ${status?.state}`}>{audio}</span>}
    </div>
  </article>
}

export function ConversationPanel({ conversation, auth, connection, mobileOpen, onCloseMobile, onRetry, onRetrySession }: ConversationPanelProps) {
  const threadRef = useRef<HTMLDivElement | null>(null)
  const threadContentRef = useRef<HTMLDivElement | null>(null)
  const statusCardRef = useRef<HTMLDivElement | null>(null)
  const scrollFrameRef = useRef<number | null>(null)
  const pinnedToBottom = useRef(true)
  const previousMessageCount = useRef(0)
  const sessionStatusRef = useRef(conversation.sessionStatus)
  const messageCountRef = useRef(conversation.messages.length)
  const [showNewResponse, setShowNewResponse] = useState(false)
  const connected = connection === 'connected'
  sessionStatusRef.current = conversation.sessionStatus
  messageCountRef.current = conversation.messages.length

  function scrollBehavior(): ScrollBehavior {
    return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
  }

  function scrollToBottomNow() {
    const thread = threadRef.current
    if (!thread || !pinnedToBottom.current) return
    thread.scrollTop = thread.scrollHeight
    setShowNewResponse(false)
  }

  function schedulePinnedScroll() {
    const startupWithoutMessages = messageCountRef.current === 0 && sessionStatusRef.current !== 'idle'
    if (startupWithoutMessages || !pinnedToBottom.current || scrollFrameRef.current !== null || typeof window === 'undefined') return
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      scrollFrameRef.current = null
      scrollToBottomNow()
    })
  }

  const turns = useMemo(() => {
    const order: string[] = []
    const groups = new Map<string, { user?: ChatMessage; assistant?: ChatMessage }>()
    for (const message of conversation.messages) {
      if (!groups.has(message.turnId)) {
        groups.set(message.turnId, {})
        order.push(message.turnId)
      }
      const group = groups.get(message.turnId)!
      group[message.role] = message
    }
    return order.map((turnId) => ({ turnId, ...groups.get(turnId)! }))
  }, [conversation.messages])

  const latestTurnState = conversation.turnStates[turns.at(-1)?.turnId ?? '']?.state
  const footerMessage = conversation.sessionStatus === 'connecting' || conversation.sessionStatus === 'starting' || conversation.sessionStatus === 'reconnecting'
    ? 'Agent chưa sẵn sàng nhận giọng nói'
    : turnStateLabel(latestTurnState) ?? conversation.sessionMessage ?? (connected ? 'Final transcript · response audio' : 'Waiting for connection')

  useEffect(() => {
    const isStarting = conversation.sessionStatus === 'connecting' || conversation.sessionStatus === 'starting'
    const isFreshSession = conversation.sessionStatus === 'connecting' || conversation.sessionStatus === 'reconnecting'
    if (isFreshSession && conversation.messages.length === 0) {
      pinnedToBottom.current = true
      setShowNewResponse(false)
    }
    if (!isStarting || !statusCardRef.current || !pinnedToBottom.current || conversation.messages.length > 0) return
    const thread = threadRef.current
    if (!thread) return
    thread.scrollTo({ top: Math.max(0, statusCardRef.current.offsetTop - 16), behavior: 'auto' })
  }, [conversation.sessionStatus, conversation.messages.length])

  useEffect(() => {
    const content = threadContentRef.current
    if (!content || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => schedulePinnedScroll())
    observer.observe(content)
    return () => {
      observer.disconnect()
      if (scrollFrameRef.current !== null && typeof window !== 'undefined') {
        window.cancelAnimationFrame(scrollFrameRef.current)
        scrollFrameRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    const hasNewMessage = conversation.messages.length > previousMessageCount.current
    previousMessageCount.current = conversation.messages.length
    if (pinnedToBottom.current) {
      schedulePinnedScroll()
    } else if (hasNewMessage) {
      setShowNewResponse(true)
    }
  }, [conversation.messages])

  function handleScroll() {
    const thread = threadRef.current
    if (!thread) return
    pinnedToBottom.current = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 72
    if (pinnedToBottom.current) setShowNewResponse(false)
  }

  function scrollToLatest() {
    const thread = threadRef.current
    if (!thread) return
    pinnedToBottom.current = true
    thread.scrollTo({ top: thread.scrollHeight, behavior: scrollBehavior() })
    setShowNewResponse(false)
  }

  return <section className={`conversation-panel ${mobileOpen ? 'mobile-open' : ''}`} aria-label="Conversation">
    <header className="conversation-header">
      <div>
        <p className="eyebrow">Conversation</p>
        <h2>Voice chat</h2>
      </div>
      <div className="conversation-header-actions">
        <span className={`connection-chip ${connection}`}>{connection === 'connected' ? 'Live' : connection === 'connecting' ? 'Connecting' : connection === 'reconnecting' ? 'Reconnecting' : connection === 'failed' ? 'Agent unavailable' : 'Offline'}</span>
        <button className="icon-button mobile-only" type="button" onClick={onCloseMobile} aria-label="Đóng conversation">Close</button>
      </div>
    </header>
    <div className="chat-thread" ref={threadRef} onScroll={handleScroll} role="log" aria-label="Voice conversation" aria-busy={conversation.sessionStatus === 'starting' || conversation.sessionStatus === 'connecting' || conversation.sessionStatus === 'reconnecting'}>
      <div className="chat-content" ref={threadContentRef}>
        {conversation.sessionStatus !== 'idle' && <div ref={statusCardRef}><SessionStatusCard status={conversation.sessionStatus} message={conversation.sessionMessage} onRetry={onRetrySession} /></div>}
        <AuthChallengeCard auth={auth} />
        {!turns.length && <div className="empty-chat"><span className="empty-orb" aria-hidden="true" /><strong>{connected ? 'Bạn có thể bắt đầu nói.' : 'Tham gia local room để bắt đầu.'}</strong><p>Transcript cuối cùng của bạn và câu trả lời Agent sẽ xuất hiện ở đây. Dữ liệu Calendar luôn là demo trong phase này.</p></div>}
        {turns.map((turn) => <div className="turn-row" key={turn.turnId}>
          {turn.user && <MessageBubble message={turn.user} status={conversation.turnStates[turn.turnId]} />}
          {turn.assistant && <MessageBubble message={turn.assistant} status={conversation.turnStates[turn.turnId]} />}
          {conversation.turnStates[turn.turnId]?.state === 'error' && <div className="turn-error" role="alert"><span>{conversation.turnStates[turn.turnId].message ?? 'Turn này không hoàn tất.'}</span>{turn.user && <button className="text-button" type="button" onClick={() => onRetry(turn.turnId)}>Retry response</button>}</div>}
        </div>)}
      </div>
    </div>
    {showNewResponse && <button className="new-response" type="button" onClick={scrollToLatest}>New response ↓</button>}
    {conversation.notice && <p className="error conversation-error" role="alert">{conversation.notice}</p>}
    <footer className="chat-footer"><span className={`state-icon ${conversation.turnStates[turns.at(-1)?.turnId ?? '']?.state ?? 'completed'}`} aria-hidden="true" />{footerMessage}</footer>
  </section>
}
