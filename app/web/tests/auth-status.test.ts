import { describe, expect, it } from 'vitest'

import { AUTH_GATE_IDENTITY, parseAuthStatus } from '../src/auth-status'

describe('parseAuthStatus', () => {
  it('accepts a well-formed status from the fixed Auth Gate participant', () => {
    expect(
      parseAuthStatus(
        '{"state":"authenticated","display_name":"An","expires_at":"2026-08-24T12:00:00+00:00"}',
        AUTH_GATE_IDENTITY,
      ),
    ).toMatchObject({ state: 'authenticated', displayName: 'An', expiresAt: '2026-08-24T12:00:00+00:00', phase: 'idle', elapsedMs: 0, targetMs: 5000, canResume: false })
  })

  it('rejects a forged status from another participant', () => {
    expect(parseAuthStatus('{"state":"authenticated","display_name":"An","expires_at":null}', 'guest-user')).toBeNull()
  })

  it('rejects malformed or unknown states', () => {
    expect(parseAuthStatus('{"state":"owner"}', AUTH_GATE_IDENTITY)).toBeNull()
    expect(parseAuthStatus('not json', AUTH_GATE_IDENTITY)).toBeNull()
  })

  it('accepts an untagged local Pipecat server packet only with the room agent identity', () => {
    expect(
      parseAuthStatus('{"state":"guest"}', '', 'pipecat-agent-session'),
    ).toMatchObject({ state: 'guest', displayName: null, expiresAt: null, phase: 'idle', elapsedMs: 0, targetMs: 5000, canResume: false })
    expect(parseAuthStatus('{"state":"authenticated"}', 'other-agent', 'pipecat-agent-session')).toBeNull()
  })

  it('parses safe challenge progress and resume metadata', () => {
    expect(parseAuthStatus(
      '{"state":"authenticated","phase":"waiting_for_resume","elapsed_ms":5000,"target_ms":5000,"can_resume":true,"session_id":"s1","sequence":7}',
      AUTH_GATE_IDENTITY,
    )).toMatchObject({ phase: 'waiting_for_resume', elapsedMs: 5000, targetMs: 5000, canResume: true, sessionId: 's1', sequence: 7 })
  })
})
