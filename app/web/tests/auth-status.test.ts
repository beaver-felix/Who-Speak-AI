import { describe, expect, it } from 'vitest'

import { AUTH_GATE_IDENTITY, parseAuthStatus } from '../src/auth-status'

describe('parseAuthStatus', () => {
  it('accepts a well-formed status from the fixed Auth Gate participant', () => {
    expect(
      parseAuthStatus(
        '{"state":"authenticated","display_name":"An","expires_at":"2026-08-24T12:00:00+00:00"}',
        AUTH_GATE_IDENTITY,
      ),
    ).toEqual({ state: 'authenticated', displayName: 'An', expiresAt: '2026-08-24T12:00:00+00:00' })
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
    ).toEqual({ state: 'guest', displayName: null, expiresAt: null })
    expect(parseAuthStatus('{"state":"authenticated"}', 'other-agent', 'pipecat-agent-session')).toBeNull()
  })
})
