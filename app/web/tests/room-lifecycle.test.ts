import { describe, expect, it } from 'vitest'

import { decideRoomLifecycle, statusAfterTransientReconnect } from '../src/room-lifecycle'

describe('room lifecycle policy', () => {
  const active = {
    isCurrentRoom: true,
    intentionalDisconnect: false,
    leaving: false,
    replacementInFlight: false,
  }

  it('preserves the active session while LiveKit is reconnecting', () => {
    expect(decideRoomLifecycle({ ...active, event: 'reconnecting' })).toBe('preserve')
  })

  it('restores the active session after LiveKit reconnects', () => {
    expect(decideRoomLifecycle({ ...active, event: 'reconnected' })).toBe('restore')
  })

  it('replaces the room only after an unexpected active-room disconnect', () => {
    expect(decideRoomLifecycle({ ...active, event: 'disconnected' })).toBe('replace')
  })

  it('does not replace a room that is being left or intentionally closed', () => {
    expect(decideRoomLifecycle({ ...active, event: 'disconnected', leaving: true })).toBe('preserve')
    expect(decideRoomLifecycle({ ...active, event: 'disconnected', intentionalDisconnect: true })).toBe('preserve')
    expect(decideRoomLifecycle({ ...active, event: 'disconnected', replacementInFlight: true })).toBe('preserve')
  })

  it('ignores events from a stale room', () => {
    expect(decideRoomLifecycle({ ...active, event: 'disconnected', isCurrentRoom: false })).toBe('ignore')
    expect(decideRoomLifecycle({ ...active, event: 'reconnected', isCurrentRoom: false })).toBe('ignore')
  })

  it('restores ready sessions without replaying startup', () => {
    expect(statusAfterTransientReconnect('ready')).toBe('ready')
  })

  it('keeps warming sessions in starting state after a transient reconnect', () => {
    expect(statusAfterTransientReconnect('starting')).toBe('starting')
    expect(statusAfterTransientReconnect('connecting')).toBe('starting')
  })
})
