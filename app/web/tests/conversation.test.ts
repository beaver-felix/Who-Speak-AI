import { describe, expect, it } from 'vitest'

import { initialConversationState, parseVoiceAgentEvent, reduceConversation } from '../src/conversation'

describe('voice conversation reducer', () => {
  it('keeps a final transcript and its assistant reply in one ordered history', () => {
    const transcript = parseVoiceAgentEvent('{"type":"transcript","message_id":"m1","turn_id":"t1","text":"Lịch hôm nay?","is_final":true}')!
    const response = parseVoiceAgentEvent('{"type":"assistant_response","message_id":"m2","turn_id":"t1","text":"Bạn có một cuộc họp demo.","tool":{"name":"calendar.list_events","provider":"mock","demo":true}}')!
    const state = reduceConversation(reduceConversation(initialConversationState, transcript), response)

    expect(state.messages.map((message) => [message.role, message.text])).toEqual([
      ['user', 'Lịch hôm nay?'], ['assistant', 'Bạn có một cuộc họp demo.'],
    ])
    expect(state.messages[1].tool?.demo).toBe(true)
  })

  it('ignores duplicate and malformed events', () => {
    const event = parseVoiceAgentEvent('{"type":"transcript","message_id":"m1","turn_id":"t1","text":"Xin chào","is_final":true}')!
    const first = reduceConversation(initialConversationState, event)

    expect(reduceConversation(first, event).messages).toHaveLength(1)
    expect(parseVoiceAgentEvent('{"type":"owner"}')).toBeNull()
  })

  it('records a visible per-turn error state', () => {
    const event = parseVoiceAgentEvent('{"type":"state","message_id":"m3","turn_id":"t1","state":"error","message":"Audio unavailable"}')!
    const state = reduceConversation(initialConversationState, event)

    expect(state.turnStates.t1).toMatchObject({ state: 'error', message: 'Audio unavailable' })
  })

  it('keeps an authentication error visible even though it has no chat turn', () => {
    const event = parseVoiceAgentEvent('{"type":"state","message_id":"auth-error","state":"error","message":"Voice verification could not be completed. Please try again."}')!
    const state = reduceConversation(initialConversationState, event)

    expect(state.notice).toBe('Voice verification could not be completed. Please try again.')
  })

  it('clears an old authentication notice when the user starts a new challenge', () => {
    const event = parseVoiceAgentEvent('{"type":"state","message_id":"auth-error","state":"error","message":"Try again."}')!
    const withError = reduceConversation(initialConversationState, event)

    expect(reduceConversation(withError, { type: 'clear_notice' }).notice).toBeNull()
  })

  it('shows an Agent thinking placeholder, then replaces it with streamed text', () => {
    const listening = parseVoiceAgentEvent('{"type":"state","message_id":"m1","turn_id":"t1","state":"listening","sequence":1}')!
    const transcript = parseVoiceAgentEvent('{"type":"transcript","message_id":"m2","turn_id":"t1","text":"Xin chào","is_final":true,"sequence":2}')!
    const thinking = parseVoiceAgentEvent('{"type":"state","message_id":"m3","turn_id":"t1","state":"thinking","sequence":3}')!
    const streamed = parseVoiceAgentEvent('{"type":"assistant_response","message_id":"m4","turn_id":"t1","text":"Chào bạn","is_final":false,"sequence":4}')!
    const final = parseVoiceAgentEvent('{"type":"assistant_response","message_id":"m5","turn_id":"t1","text":"Chào bạn!","is_final":true,"sequence":5}')!
    const waiting = [listening, transcript, thinking].reduce(reduceConversation, initialConversationState)
    const state = [streamed, final].reduce(reduceConversation, waiting)

    expect(waiting.messages.map((message) => [message.role, message.text, message.provisional])).toEqual([
      ['user', 'Xin chào', false], ['assistant', 'Thinking…', true],
    ])

    expect(state.messages.map((message) => [message.role, message.text, message.provisional])).toEqual([
      ['user', 'Xin chào', false],
      ['assistant', 'Chào bạn!', false],
    ])
  })

  it('never renders a transcribing state as a user transcript', () => {
    const stateEvent = parseVoiceAgentEvent('{"type":"state","message_id":"state-1","turn_id":"t1","state":"transcribing","sequence":1}')!
    const state = reduceConversation(initialConversationState, stateEvent)

    expect(state.messages).toHaveLength(0)
    expect(state.turnStates.t1.state).toBe('transcribing')
  })

  it('drops an older event sequence instead of regressing the visible turn', () => {
    const newer = parseVoiceAgentEvent('{"type":"state","message_id":"m2","turn_id":"t1","state":"thinking","sequence":4}')!
    const stale = parseVoiceAgentEvent('{"type":"state","message_id":"m1","turn_id":"t1","state":"listening","sequence":1}')!
    const state = reduceConversation(reduceConversation(initialConversationState, newer), stale)

    expect(state.turnStates.t1.state).toBe('thinking')
  })

  it('tracks session readiness separately from conversation turns', () => {
    const starting = parseVoiceAgentEvent('{"type":"session_status","message_id":"s1","status":"starting","message":"Warming up"}')!
    const ready = parseVoiceAgentEvent('{"type":"session_status","message_id":"s2","status":"ready","message":"Ready to listen"}')!
    const state = [starting, ready].reduce(reduceConversation, initialConversationState)

    expect(state.sessionStatus).toBe('ready')
    expect(state.sessionMessage).toBe('Ready to listen')
    expect(state.messages).toHaveLength(0)
  })

  it('does not regress startup when a queued connecting event arrives late', () => {
    const starting = parseVoiceAgentEvent('{"type":"session_status","message_id":"s-start","status":"starting"}')!
    const connecting = parseVoiceAgentEvent('{"type":"session_status","message_id":"s-connect","status":"connecting"}')!
    const state = reduceConversation(reduceConversation(initialConversationState, starting), connecting)

    expect(state.sessionStatus).toBe('starting')
    expect(state.messages).toHaveLength(0)
  })

  it('keeps a startup failure separate from chat history', () => {
    const failed = parseVoiceAgentEvent('{"type":"session_status","message_id":"s-failed","status":"failed","message":"Whisper unavailable"}')!
    const state = reduceConversation(initialConversationState, failed)

    expect(state.sessionStatus).toBe('failed')
    expect(state.sessionMessage).toBe('Whisper unavailable')
    expect(state.messages).toHaveLength(0)
  })

  it('keeps the processing stage alongside the visible turn state', () => {
    const event = parseVoiceAgentEvent('{"type":"state","message_id":"m-stage","turn_id":"t-stage","state":"thinking","stage":"llm","sequence":1}')!
    const state = reduceConversation(initialConversationState, event)

    expect(state.turnStates['t-stage']).toMatchObject({ state: 'thinking', stage: 'llm' })
  })

  it('rejects an unknown session status without affecting chat history', () => {
    expect(parseVoiceAgentEvent('{"type":"session_status","message_id":"s1","status":"online"}')).toBeNull()
  })
})
