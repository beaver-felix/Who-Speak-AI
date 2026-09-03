import type { SessionStatus } from './conversation'

export type RoomLifecycleEvent = 'reconnecting' | 'reconnected' | 'disconnected'
export type RoomLifecycleDecision = 'ignore' | 'preserve' | 'restore' | 'replace'

type RoomLifecycleInput = {
  event: RoomLifecycleEvent
  isCurrentRoom: boolean
  intentionalDisconnect: boolean
  leaving: boolean
  replacementInFlight: boolean
}

/**
 * Keep transient LiveKit recovery separate from a true room replacement.
 * Reconnecting/Reconnected are continuity events; only an unexpected
 * Disconnected event from the active room can request a fresh session.
 */
export function decideRoomLifecycle(input: RoomLifecycleInput): RoomLifecycleDecision {
  if (!input.isCurrentRoom) return 'ignore'
  if (input.event === 'reconnecting') return 'preserve'
  if (input.event === 'reconnected') return 'restore'
  if (input.intentionalDisconnect || input.leaving || input.replacementInFlight) return 'preserve'
  return 'replace'
}

export function statusAfterTransientReconnect(status: SessionStatus): 'starting' | 'ready' {
  return status === 'ready' ? 'ready' : 'starting'
}
