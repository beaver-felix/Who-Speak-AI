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

  it('replaces listening and thinking placeholders in their original turn', () => {
    const listening = parseVoiceAgentEvent('{"type":"state","message_id":"m1","turn_id":"t1","state":"listening","sequence":1}')!
    const transcript = parseVoiceAgentEvent('{"type":"transcript","message_id":"m2","turn_id":"t1","text":"Xin chào","is_final":true,"sequence":2}')!
    const thinking = parseVoiceAgentEvent('{"type":"state","message_id":"m3","turn_id":"t1","state":"thinking","sequence":3}')!
    const response = parseVoiceAgentEvent('{"type":"assistant_response","message_id":"m4","turn_id":"t1","text":"Chào bạn!","is_final":true,"sequence":4}')!
    const state = [listening, transcript, thinking, response].reduce(reduceConversation, initialConversationState)

    expect(state.messages.map((message) => [message.role, message.text, message.provisional])).toEqual([
      ['user', 'Xin chào', false],
      ['assistant', 'Chào bạn!', false],
    ])
  })

  it('drops an older event sequence instead of regressing the visible turn', () => {
    const newer = parseVoiceAgentEvent('{"type":"state","message_id":"m2","turn_id":"t1","state":"thinking","sequence":4}')!
    const stale = parseVoiceAgentEvent('{"type":"state","message_id":"m1","turn_id":"t1","state":"listening","sequence":1}')!
    const state = reduceConversation(reduceConversation(initialConversationState, newer), stale)

    expect(state.turnStates.t1.state).toBe('thinking')
  })
})
