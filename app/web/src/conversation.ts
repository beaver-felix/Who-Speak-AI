export const AGENT_EVENT_TOPIC = 'voice-agent-event'
export const AGENT_COMMAND_TOPIC = 'voice-agent-command'

export type VoiceAgentState = 'listening' | 'transcribing' | 'thinking' | 'speaking' | 'completed' | 'interrupted' | 'error'
export type ToolBadge = { name: 'calendar.list_events' | 'calendar.create_event'; provider: 'mock' | 'google_mcp'; demo: boolean }
export type SessionStatus = 'idle' | 'connecting' | 'starting' | 'ready' | 'reconnecting' | 'stopping' | 'failed'
export type VoiceAgentStage = 'auth' | 'vad' | 'stt' | 'llm' | 'tool' | 'tts' | 'transport'

type EventBase = { message_id: string; turn_id?: string; sequence?: number }

export type VoiceAgentEvent =
  | (EventBase & { type: 'transcript'; turn_id: string; text: string; is_final: boolean })
  | (EventBase & { type: 'assistant_response'; turn_id: string; text: string; is_final: boolean; tool?: ToolBadge | null })
  | (EventBase & { type: 'state'; state: VoiceAgentState; stage?: VoiceAgentStage; message?: string })
  | (EventBase & { type: 'session_status'; status: SessionStatus; message?: string })

export type ChatMessage = {
  id: string
  turnId: string
  role: 'user' | 'assistant'
  text: string
  createdAt: number
  provisional: boolean
  isFinal: boolean
  tool?: ToolBadge | null
}

export type TurnStatus = { state: VoiceAgentState; stage?: VoiceAgentStage; message?: string; sequence: number; updatedAt: number }

export type ConversationState = {
  messages: ChatMessage[]
  seenEventIds: Set<string>
  turnStates: Record<string, TurnStatus>
  notice: string | null
  sessionStatus: SessionStatus
  sessionMessage: string | null
}

export const initialConversationState: ConversationState = {
  messages: [],
  seenEventIds: new Set(),
  turnStates: {},
  notice: null,
  sessionStatus: 'idle',
  sessionMessage: null,
}

const states = new Set<VoiceAgentState>(['listening', 'transcribing', 'thinking', 'speaking', 'completed', 'interrupted', 'error'])
const sessionStatuses = new Set<SessionStatus>(['idle', 'connecting', 'starting', 'ready', 'reconnecting', 'stopping', 'failed'])
const stages = new Set<VoiceAgentStage>(['auth', 'vad', 'stt', 'llm', 'tool', 'tts', 'transport'])

function safeSequence(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : undefined
}

export function parseVoiceAgentEvent(value: string): VoiceAgentEvent | null {
  try {
    const event: unknown = JSON.parse(value)
    if (typeof event !== 'object' || event === null) return null
    const payload = event as Record<string, unknown>
    if (typeof payload.type !== 'string' || typeof payload.message_id !== 'string') return null
    const base: EventBase = { message_id: payload.message_id, turn_id: typeof payload.turn_id === 'string' ? payload.turn_id : undefined, sequence: safeSequence(payload.sequence) }
    if (payload.type === 'state') {
      if (typeof payload.state !== 'string' || !states.has(payload.state as VoiceAgentState)) return null
      return { ...base, type: 'state', state: payload.state as VoiceAgentState, stage: typeof payload.stage === 'string' && stages.has(payload.stage as VoiceAgentStage) ? payload.stage as VoiceAgentStage : undefined, message: typeof payload.message === 'string' ? payload.message : undefined }
    }
    if (payload.type === 'session_status') {
      if (typeof payload.status !== 'string' || !sessionStatuses.has(payload.status as SessionStatus)) return null
      return { ...base, type: 'session_status', status: payload.status as SessionStatus, message: typeof payload.message === 'string' ? payload.message : undefined }
    }
    if ((payload.type === 'transcript' || payload.type === 'assistant_response') && typeof payload.turn_id === 'string' && typeof payload.text === 'string') {
      if (payload.type === 'transcript') return { ...base, type: 'transcript', turn_id: payload.turn_id, text: payload.text, is_final: payload.is_final === true }
      const rawTool = payload.tool
      const rawToolRecord = typeof rawTool === 'object' && rawTool !== null ? rawTool as Record<string, unknown> : null
      const validToolMarker = rawToolRecord?.provider === 'mock' && rawToolRecord.demo === true || rawToolRecord?.provider === 'google_mcp' && rawToolRecord.demo === false
      const tool = rawToolRecord !== null && validToolMarker && (rawToolRecord.name === 'calendar.list_events' || rawToolRecord.name === 'calendar.create_event') ? rawToolRecord as ToolBadge : null
      return { ...base, type: 'assistant_response', turn_id: payload.turn_id, text: payload.text, is_final: payload.is_final !== false, tool }
    }
    return null
  } catch { return null }
}

function mayApply(state: ConversationState, event: VoiceAgentEvent): boolean {
  if (!event.turn_id || event.sequence === undefined) return true
  const current = state.turnStates[event.turn_id]
  return !current || event.sequence > current.sequence
}

function withSequence(state: ConversationState, event: VoiceAgentEvent): ConversationState {
  if (!event.turn_id || event.sequence === undefined) return state
  const current = state.turnStates[event.turn_id]
  if (!current) return state
  return { ...state, turnStates: { ...state.turnStates, [event.turn_id]: { ...current, sequence: event.sequence, updatedAt: Date.now() } } }
}

function isStaleSessionStatus(current: SessionStatus, next: SessionStatus): boolean {
  // The browser marks a Pipecat room as starting immediately after connect.
  // Its queued `connecting` event may arrive a moment later, so do not let a
  // late lifecycle event make a ready/starting session look less ready.
  if (current === 'starting' && next === 'connecting') return true
  if (current === 'ready' && (next === 'connecting' || next === 'starting')) return true
  return false
}

function upsertMessage(state: ConversationState, message: Omit<ChatMessage, 'createdAt'>): ConversationState {
  const existing = state.messages.findIndex((item) => item.turnId === message.turnId && item.role === message.role)
  if (existing < 0) return { ...state, messages: [...state.messages, { ...message, createdAt: Date.now() }] }
  const messages = state.messages.map((item, index) => index === existing ? { ...item, ...message, createdAt: item.createdAt } : item)
  return { ...state, messages }
}

function provisionalLabel(state: VoiceAgentState): string | null {
  // Restore the useful old Conversation behavior: after final ASR, reserve
  // the Agent bubble immediately so the user sees that an answer is being
  // generated. Do not create faux user bubbles for listening/transcribing;
  // only a final transcript is user-visible conversation content.
  return state === 'thinking' ? 'Thinking…' : null
}

export function reduceConversation(
  state: ConversationState,
  event: VoiceAgentEvent | { type: 'reset' } | { type: 'clear_notice' } | { type: 'set_session_status'; status: SessionStatus; message?: string },
): ConversationState {
  if (event.type === 'reset') return initialConversationState
  if (event.type === 'clear_notice') return state.notice ? { ...state, notice: null } : state
  if (event.type === 'set_session_status') return { ...state, sessionStatus: event.status, sessionMessage: event.message ?? null }
  if (state.seenEventIds.has(event.message_id) || !mayApply(state, event)) return state
  const seenEventIds = new Set(state.seenEventIds).add(event.message_id)
  let next: ConversationState = { ...state, seenEventIds }
  if (event.type === 'session_status') {
    if (isStaleSessionStatus(state.sessionStatus, event.status)) return state
    return { ...next, sessionStatus: event.status, sessionMessage: event.message ?? null }
  }
  if (event.type === 'state') {
    // Authentication failures have no conversation turn. Keep them visible
    // rather than silently dropping them in the chat reducer.
    if (!event.turn_id) {
      return event.state === 'error' && event.message
        ? { ...next, notice: event.message }
        : next
    }
    next = {
        ...next,
        turnStates: {
          ...next.turnStates,
          [event.turn_id]: { state: event.state, stage: event.stage, message: event.message, sequence: event.sequence ?? Number.MAX_SAFE_INTEGER, updatedAt: Date.now() },
        },
    }
    const label = provisionalLabel(event.state)
    if (label) {
      const existing = next.messages.find((item) => item.turnId === event.turn_id && item.role === 'assistant')
      if (!existing || existing.provisional) {
        next = upsertMessage(next, {
          id: event.message_id,
          turnId: event.turn_id,
          role: 'assistant',
          text: label,
          provisional: true,
          isFinal: false,
        })
      }
    }
    return next
  }
  if (event.type === 'transcript') {
    if (!event.is_final) return withSequence(next, event)
    next = upsertMessage(next, { id: event.message_id, turnId: event.turn_id, role: 'user', text: event.text, provisional: false, isFinal: true })
    // A final transcript supersedes the temporary STT state in the footer;
    // it does not create a second, synthetic message.
    return withSequence(next, event)
  }
  next = upsertMessage(next, {
    id: event.message_id,
    turnId: event.turn_id,
    role: 'assistant',
    text: event.text,
    provisional: false,
    isFinal: event.is_final,
    tool: event.tool,
  })
  return withSequence(next, event)
}
