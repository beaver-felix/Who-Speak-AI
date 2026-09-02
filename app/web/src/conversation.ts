export const AGENT_EVENT_TOPIC = 'voice-agent-event'
export const AGENT_COMMAND_TOPIC = 'voice-agent-command'

export type VoiceAgentState = 'listening' | 'transcribing' | 'thinking' | 'speaking' | 'completed' | 'interrupted' | 'error'
export type ToolBadge = { name: 'calendar.list_events' | 'calendar.create_event'; provider: 'mock'; demo: true }

type EventBase = { message_id: string; turn_id?: string; sequence?: number }

export type VoiceAgentEvent =
  | (EventBase & { type: 'transcript'; turn_id: string; text: string; is_final: boolean })
  | (EventBase & { type: 'assistant_response'; turn_id: string; text: string; is_final: boolean; tool?: ToolBadge | null })
  | (EventBase & { type: 'state'; state: VoiceAgentState; message?: string })

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

export type TurnStatus = { state: VoiceAgentState; message?: string; sequence: number; updatedAt: number }

export type ConversationState = {
  messages: ChatMessage[]
  seenEventIds: Set<string>
  turnStates: Record<string, TurnStatus>
  notice: string | null
}

export const initialConversationState: ConversationState = { messages: [], seenEventIds: new Set(), turnStates: {}, notice: null }

const states = new Set<VoiceAgentState>(['listening', 'transcribing', 'thinking', 'speaking', 'completed', 'interrupted', 'error'])

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
      return { ...base, type: 'state', state: payload.state as VoiceAgentState, message: typeof payload.message === 'string' ? payload.message : undefined }
    }
    if ((payload.type === 'transcript' || payload.type === 'assistant_response') && typeof payload.turn_id === 'string' && typeof payload.text === 'string') {
      if (payload.type === 'transcript') return { ...base, type: 'transcript', turn_id: payload.turn_id, text: payload.text, is_final: payload.is_final === true }
      const rawTool = payload.tool
      const tool = typeof rawTool === 'object' && rawTool !== null && (rawTool as Record<string, unknown>).provider === 'mock' && (rawTool as Record<string, unknown>).demo === true && ((rawTool as Record<string, unknown>).name === 'calendar.list_events' || (rawTool as Record<string, unknown>).name === 'calendar.create_event') ? rawTool as ToolBadge : null
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

function upsertMessage(state: ConversationState, message: Omit<ChatMessage, 'createdAt'>): ConversationState {
  const existing = state.messages.findIndex((item) => item.turnId === message.turnId && item.role === message.role)
  if (existing < 0) return { ...state, messages: [...state.messages, { ...message, createdAt: Date.now() }] }
  const messages = state.messages.map((item, index) => index === existing ? { ...item, ...message, createdAt: item.createdAt } : item)
  return { ...state, messages }
}

function provisionalLabel(state: VoiceAgentState): string | null {
  if (state === 'listening') return 'Listening…'
  if (state === 'transcribing') return 'Transcribing…'
  if (state === 'thinking') return 'Thinking…'
  return null
}

export function reduceConversation(state: ConversationState, event: VoiceAgentEvent | { type: 'reset' } | { type: 'clear_notice' }): ConversationState {
  if (event.type === 'reset') return initialConversationState
  if (event.type === 'clear_notice') return state.notice ? { ...state, notice: null } : state
  if (state.seenEventIds.has(event.message_id) || !mayApply(state, event)) return state
  const seenEventIds = new Set(state.seenEventIds).add(event.message_id)
  let next: ConversationState = { ...state, seenEventIds }
  if (event.type === 'state') {
    // Authentication failures have no conversation turn. Keep them visible
    // rather than silently dropping them in the chat reducer.
    if (!event.turn_id) {
      return event.state === 'error' && event.message
        ? { ...next, notice: event.message }
        : next
    }
    const label = provisionalLabel(event.state)
    const messageRole = event.state === 'thinking' ? 'assistant' : event.state === 'listening' || event.state === 'transcribing' ? 'user' : null
    next = {
      ...next,
      turnStates: {
        ...next.turnStates,
        [event.turn_id]: { state: event.state, message: event.message, sequence: event.sequence ?? Number.MAX_SAFE_INTEGER, updatedAt: Date.now() },
      },
    }
    if (messageRole && label) {
      const existing = next.messages.find((item) => item.turnId === event.turn_id && item.role === messageRole)
      if (!existing || existing.provisional) {
        next = upsertMessage(next, { id: event.message_id, turnId: event.turn_id, role: messageRole, text: label, provisional: true, isFinal: false })
      }
    }
    return next
  }
  if (event.type === 'transcript') {
    if (!event.is_final) return withSequence(next, event)
    next = upsertMessage(next, { id: event.message_id, turnId: event.turn_id, role: 'user', text: event.text, provisional: false, isFinal: true })
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
