export const AUTH_STATUS_TOPIC = 'voice-auth-status'
export const AGENT_TRANSCRIPT_TOPIC = 'voice-agent-transcript'
export const AGENT_RESPONSE_TOPIC = 'voice-agent-response'
export const AUTH_COMMAND_TOPIC = 'voice-auth'
export const REQUEST_PRIVATE_MODE = 'request_private_mode'
export const CANCEL_PRIVATE_MODE = 'cancel_private_mode'
export const AUTH_GATE_IDENTITY = 'voice-auth-gate'

export type AuthState = 'guest' | 'auth_pending' | 'authenticated' | 'session_expired'
export type AuthStatus = { state: AuthState; displayName: string | null; expiresAt: string | null }
const states = new Set<AuthState>(['guest', 'auth_pending', 'authenticated', 'session_expired'])

export function parseAuthStatus(text: string, senderIdentity: string, expectedIdentity = AUTH_GATE_IDENTITY): AuthStatus | null {
  if (senderIdentity && senderIdentity !== expectedIdentity) return null
  try {
    const value: unknown = JSON.parse(text)
    if (typeof value !== 'object' || value === null) return null
    const payload = value as Record<string, unknown>
    if (typeof payload.state !== 'string' || !states.has(payload.state as AuthState)) return null
    return {
      state: payload.state as AuthState,
      displayName: typeof payload.display_name === 'string' ? payload.display_name : null,
      expiresAt: typeof payload.expires_at === 'string' ? payload.expires_at : null,
    }
  } catch { return null }
}
