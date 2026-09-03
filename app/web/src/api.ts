// In local Vite development the default is same-origin `/api`, proxied to the
// gateway. This avoids a browser treating `localhost` and `127.0.0.1` as
// different sites and dropping the HttpOnly session cookie.
const gateway = import.meta.env.VITE_GATEWAY_URL ?? ''

export type CurrentUser = { id: string; email: string; display_name: string; voice_enrolled: boolean }

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${gateway}${path}`, {
    ...init,
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
  })
  if (response.status === 204) return undefined as T
  const body = await response.json().catch(() => ({})) as T & { detail?: string }
  if (!response.ok) throw new Error(body.detail ?? `Request failed (${response.status}).`)
  return body
}

export const api = {
  me: () => request<CurrentUser>('/api/auth/me'),
  register: (email: string, password: string, displayName: string) => request<CurrentUser>('/api/auth/register', { method: 'POST', body: JSON.stringify({ email, password, display_name: displayName }) }),
  login: (email: string, password: string) => request<CurrentUser>('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  logout: () => request<void>('/api/auth/logout', { method: 'POST' }),
  enroll: (samples: string[], displayName?: string) => request<{ identity_id: string; display_name: string; enrolled: boolean }>('/api/voice/enroll', { method: 'POST', body: JSON.stringify({ samples_b64: samples, display_name: displayName }) }),
  livekitToken: () => request<{
    server_url: string
    participant_token: string
    room_name: string
    session_id: string
    participant_identity: string
    agent_identity: string
    runtime: 'livekit' | 'pipecat'
  }>('/api/livekit/token', { method: 'POST' }),
}
