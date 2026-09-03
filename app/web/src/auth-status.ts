export const AUTH_STATUS_TOPIC = 'voice-auth-status'
export const AGENT_TRANSCRIPT_TOPIC = 'voice-agent-transcript'
export const AGENT_RESPONSE_TOPIC = 'voice-agent-response'
export const AUTH_COMMAND_TOPIC = 'voice-auth'
export const REQUEST_PRIVATE_MODE = 'request_private_mode'
export const CANCEL_PRIVATE_MODE = 'cancel_private_mode'
export const RESUME_CONVERSATION = 'resume_conversation'
export const CONTINUE_AS_GUEST = 'continue_as_guest'
export const RETRY_VOICE = 'retry_voice'
export const AUTH_GATE_IDENTITY = 'voice-auth-gate'

export type AuthState = 'guest' | 'auth_pending' | 'authenticated' | 'session_expired'
export type AuthChallengePhase = 'idle' | 'capturing' | 'processing' | 'waiting_for_resume' | 'conversation_ready'
export type AuthStatus = {
  state: AuthState
  displayName: string | null
  expiresAt: string | null
  phase: AuthChallengePhase
  elapsedMs: number
  targetMs: number
  canResume: boolean
  message: string | null
  sessionId: string | null
  sequence: number | null
}
const states = new Set<AuthState>(['guest', 'auth_pending', 'authenticated', 'session_expired'])
const phases = new Set<AuthChallengePhase>(['idle', 'capturing', 'processing', 'waiting_for_resume', 'conversation_ready'])

function safeNonNegativeInteger(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.round(value) : fallback
}

export function parseAuthStatus(text: string, senderIdentity: string, expectedIdentity = AUTH_GATE_IDENTITY): AuthStatus | null {
  if (senderIdentity && senderIdentity !== expectedIdentity) return null
  try {
    const value: unknown = JSON.parse(text)
    if (typeof value !== 'object' || value === null) return null
    const payload = value as Record<string, unknown>
    if (typeof payload.state !== 'string' || !states.has(payload.state as AuthState)) return null
    const phase = typeof payload.phase === 'string' && phases.has(payload.phase as AuthChallengePhase)
      ? payload.phase as AuthChallengePhase
      : 'idle'
    return {
      state: payload.state as AuthState,
      displayName: typeof payload.display_name === 'string' ? payload.display_name : null,
      expiresAt: typeof payload.expires_at === 'string' ? payload.expires_at : null,
      phase,
      elapsedMs: safeNonNegativeInteger(payload.elapsed_ms, 0),
      targetMs: safeNonNegativeInteger(payload.target_ms, 5000),
      canResume: payload.can_resume === true,
      message: typeof payload.message === 'string' ? payload.message : null,
      sessionId: typeof payload.session_id === 'string' ? payload.session_id : null,
      sequence: typeof payload.sequence === 'number' && Number.isInteger(payload.sequence) && payload.sequence >= 0 ? payload.sequence : null,
    }
  } catch { return null }
}
