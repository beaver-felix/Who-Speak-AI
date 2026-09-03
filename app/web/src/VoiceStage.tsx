import type { AuthStatus } from './auth-status'
import type { SessionStatus, VoiceAgentState } from './conversation'
import type { AudioLevels } from './useAudioLevels'

export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'reconnecting'
export type AudioPlaybackState = 'unknown' | 'allowed' | 'blocked'

type VoiceStageProps = {
  connection: ConnectionState
  sessionStatus: SessionStatus
  sessionMessage: string | null
  auth: AuthStatus
  activeState?: VoiceAgentState
  levels: AudioLevels
  microphoneEnabled: boolean
  audioPlayback: AudioPlaybackState
  authCommandPending: boolean
  conversationCount: number
  error: string | null
  onJoin: () => void
  onLeave: () => void
  onToggleMicrophone: () => void
  onStartChallenge: () => void
  onCancelChallenge: () => void
  onResumeConversation: () => void
  onContinueAsGuest: () => void
  onRetryVoice: () => void
  onEnableAudio: () => void
  onOpenConversation: () => void
}

function sessionLabel(status: SessionStatus): string {
  return {
    idle: 'Chưa kết nối',
    connecting: 'Đang kết nối',
    starting: 'Đang khởi động voice engine',
    ready: 'Sẵn sàng lắng nghe',
    reconnecting: 'Đang kết nối lại',
    stopping: 'Đang dừng phiên',
    failed: 'Kết nối thất bại',
  }[status]
}

function authLabel(auth: AuthStatus): string {
  return {
    guest: 'Guest',
    auth_pending: 'Đang xác thực giọng nói',
    authenticated: 'Đã xác thực',
    session_expired: 'Phiên đã hết hạn',
  }[auth.state]
}

function challengeLabel(auth: AuthStatus): string {
  if (auth.phase === 'capturing') {
    return `Đang thu giọng nói · ${(auth.elapsedMs / 1000).toFixed(1)}/${(auth.targetMs / 1000).toFixed(1)} giây`
  }
  if (auth.phase === 'processing') return 'Đã thu đủ 5 giây · Đang kiểm tra…'
  if (auth.phase === 'waiting_for_resume' && auth.state === 'authenticated') return 'Đã xác thực · Chờ bạn bắt đầu nói'
  if (auth.phase === 'waiting_for_resume') return 'Chưa xác thực · Chờ lựa chọn của bạn'
  if (auth.phase === 'conversation_ready') return 'Sẵn sàng nói với Agent'
  return ''
}

function voiceLabel(state: VoiceAgentState | undefined, connected: boolean, sessionStatus: SessionStatus): string {
  if (!connected) return 'Chưa bật microphone'
  if (sessionStatus === 'connecting' || sessionStatus === 'starting' || sessionStatus === 'reconnecting') {
    return sessionLabel(sessionStatus)
  }
  if (sessionStatus === 'stopping') return 'Đang kết thúc phiên'
  if (sessionStatus === 'failed') return 'Voice engine chưa sẵn sàng'
  if (!state) return 'Sẵn sàng nói'
  return {
    listening: 'Đang nghe',
    transcribing: 'Đang nhận diện',
    thinking: 'Agent đang suy nghĩ',
    speaking: 'Agent đang nói',
    completed: 'Hoàn tất',
    interrupted: 'Đã ngắt lời',
    error: 'Cần thử lại',
  }[state ?? 'completed']
}

function orbState(sessionStatus: SessionStatus, activeState?: VoiceAgentState): string {
  if (sessionStatus === 'failed') return 'error'
  if (activeState === 'speaking') return 'speaking'
  if (activeState === 'thinking') return 'thinking'
  if (activeState === 'transcribing') return 'transcribing'
  if (activeState === 'listening') return 'listening'
  if (sessionStatus === 'starting' || sessionStatus === 'connecting' || sessionStatus === 'reconnecting') return 'starting'
  return 'idle'
}

export function VoiceStage({
  connection,
  sessionStatus,
  sessionMessage,
  auth,
  activeState,
  levels,
  microphoneEnabled,
  audioPlayback,
  authCommandPending,
  conversationCount,
  error,
  onJoin,
  onLeave,
  onToggleMicrophone,
  onStartChallenge,
  onCancelChallenge,
  onResumeConversation,
  onContinueAsGuest,
  onRetryVoice,
  onEnableAudio,
  onOpenConversation,
}: VoiceStageProps) {
  const connected = connection === 'connected'
  const sessionReady = connected && sessionStatus === 'ready'
  const challengeCapturing = auth.phase === 'capturing'
  const challengeProcessing = auth.phase === 'processing'
  const challengeWaiting = auth.phase === 'waiting_for_resume'
  const conversationBlocked = challengeCapturing || challengeProcessing || challengeWaiting
  const startChallenge = () => {
    if (sessionReady) onStartChallenge()
  }
  const orbStyle = {
    '--microphone-level': levels.microphone.toFixed(3),
    '--assistant-level': levels.assistant.toFixed(3),
  } as React.CSSProperties & Record<string, string>

  return <section className="voice-stage" aria-label="Voice session">
    <div className="stage-heading">
      <div>
        <p className="eyebrow">Voice session</p>
        <h2>{connected ? 'Nói với Agent' : 'Voice workspace'}</h2>
      </div>
      <span className={`auth-pill ${auth.state}`}><span className="status-dot" aria-hidden="true" />{authLabel(auth)}</span>
    </div>

    <div className="orb-wrap">
      <div className={`voice-orb ${orbState(sessionStatus, activeState)}`} style={orbStyle} role="img" aria-label={`Voice state: ${voiceLabel(activeState, connected, sessionStatus)}`}>
        <span className="orb-ring orb-ring-one" aria-hidden="true" />
        <span className="orb-ring orb-ring-two" aria-hidden="true" />
        <span className="orb-core" aria-hidden="true" />
      </div>
      <p className="voice-state" aria-live="polite">{voiceLabel(activeState, connected, sessionStatus)}</p>
      <p className={`session-status ${sessionStatus}`} role="status" aria-live="polite">{sessionMessage ?? sessionLabel(sessionStatus)}</p>
    </div>

    <div className="voice-controls" aria-label="Voice controls">
      {!connected ? <button className="primary" type="button" onClick={onJoin} disabled={connection === 'connecting' || connection === 'reconnecting'}>{connection === 'reconnecting' ? 'Đang kết nối lại…' : connection === 'connecting' ? 'Đang tham gia…' : 'Join local room'}</button> : <>
        <button className="secondary" type="button" onClick={onToggleMicrophone} aria-pressed={microphoneEnabled}>{microphoneEnabled ? 'Tắt microphone' : 'Bật microphone'}</button>
        {challengeCapturing || challengeProcessing ? <button className="primary" type="button" onClick={onCancelChallenge} disabled={!sessionReady || authCommandPending}>{challengeProcessing ? 'Hủy kiểm tra' : 'Hủy voice challenge'}</button> : challengeWaiting ? auth.state === 'authenticated' ? <button className="primary" type="button" onClick={onResumeConversation} disabled={!sessionReady || !auth.canResume || authCommandPending}>{authCommandPending ? 'Đang xử lý…' : 'Bắt đầu nói với Agent'}</button> : <><button className="primary" type="button" onClick={onContinueAsGuest} disabled={!sessionReady || !auth.canResume || authCommandPending}>{authCommandPending ? 'Đang xử lý…' : 'Tiếp tục ở Guest'}</button><button className="secondary" type="button" onClick={onRetryVoice} disabled={!sessionReady || authCommandPending}>{authCommandPending ? 'Đang xử lý…' : 'Thử lại voice'}</button></> : <button className="primary" type="button" onClick={startChallenge} disabled={!sessionReady || auth.state === 'authenticated' || authCommandPending}>{authCommandPending ? 'Đang xử lý…' : auth.state === 'authenticated' ? 'Đã mở private mode' : 'Xác thực voice'}</button>}
        <button className="danger" type="button" onClick={onLeave}>Leave room</button>
      </>}
      {audioPlayback === 'blocked' && <button className="audio-recovery" type="button" onClick={onEnableAudio}>Bật âm thanh Agent</button>}
      <button className="conversation-toggle" type="button" onClick={onOpenConversation} aria-label="Mở conversation">Conversation{conversationCount > 0 ? ` (${conversationCount})` : ''}</button>
    </div>

    {connected && !sessionReady && <p className="readiness-hint" role="status">Agent chưa sẵn sàng nhận giọng nói. Vui lòng chờ trạng thái “Sẵn sàng lắng nghe”.</p>}
    {conversationBlocked && <div className={`challenge-progress ${challengeProcessing ? 'processing' : challengeWaiting ? 'waiting' : ''}`} role="status" aria-live="polite">
      {challengeCapturing && <div className="challenge-progress-bar" role="progressbar" aria-label="Tiến trình voice challenge" aria-valuemin={0} aria-valuemax={auth.targetMs} aria-valuenow={Math.min(auth.elapsedMs, auth.targetMs)}><span style={{ width: `${Math.min(100, auth.targetMs ? auth.elapsedMs / auth.targetMs * 100 : 0)}%` }} /></div>}
      <strong>{challengeLabel(auth)}</strong>
      <span>{challengeCapturing ? 'Hãy nói tự nhiên. Khi đủ thời gian, vui lòng dừng nói.' : challengeProcessing ? 'Vui lòng chờ, chưa bắt đầu nói với Agent.' : auth.message ?? 'Chọn một hành động để tiếp tục.'}</span>
    </div>}
    {error && <p className="error" role="alert">{error}</p>}
    <p className="privacy-note">Audio, RawNet3, HE và voice template được xử lý local. Browser không nhận private key, embedding, score hoặc API key.</p>
  </section>
}
