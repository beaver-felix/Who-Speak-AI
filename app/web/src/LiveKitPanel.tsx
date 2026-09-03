import { useMemo, useState } from 'react'

import { CANCEL_PRIVATE_MODE, CONTINUE_AS_GUEST, REQUEST_PRIVATE_MODE, RESUME_CONVERSATION, RETRY_VOICE } from './auth-status'
import { ConversationPanel } from './ConversationPanel'
import { VoiceStage } from './VoiceStage'
import { useAudioLevels } from './useAudioLevels'
import { useLiveKitVoiceSession } from './useLiveKitVoiceSession'

export function LiveKitPanel() {
  const session = useLiveKitVoiceSession()
  const [mobileConversationOpen, setMobileConversationOpen] = useState(false)
  const levels = useAudioLevels(session.microphoneTrack, session.assistantTracks)
  const activeTurn = useMemo(() => Object.values(session.conversation.turnStates).sort((left, right) => right.updatedAt - left.updatedAt)[0], [session.conversation.turnStates])

  return <section className="assistant-layout" aria-label="Voice assistant">
    <VoiceStage
      connection={session.connection}
      sessionStatus={session.conversation.sessionStatus}
      sessionMessage={session.conversation.sessionMessage}
      auth={session.auth}
      activeState={activeTurn?.state}
      levels={levels}
      microphoneEnabled={session.microphoneEnabled}
      audioPlayback={session.audioPlayback}
      authCommandPending={session.authCommandPending}
      conversationCount={session.conversation.messages.length}
      error={session.error}
      onJoin={() => void session.join()}
      onLeave={() => void session.leave()}
      onToggleMicrophone={() => void session.toggleMicrophone()}
      onStartChallenge={() => void session.sendAuthCommand(REQUEST_PRIVATE_MODE)}
      onCancelChallenge={() => void session.sendAuthCommand(CANCEL_PRIVATE_MODE)}
      onResumeConversation={() => void session.sendAuthCommand(RESUME_CONVERSATION)}
      onContinueAsGuest={() => void session.sendAuthCommand(CONTINUE_AS_GUEST)}
      onRetryVoice={() => void session.sendAuthCommand(RETRY_VOICE)}
      onRetrySession={() => void session.retrySession()}
      onEnableAudio={() => void session.enableAudio()}
      onOpenConversation={() => setMobileConversationOpen(true)}
    />
    <ConversationPanel
      conversation={session.conversation}
      auth={session.auth}
      connection={session.connection}
      mobileOpen={mobileConversationOpen}
      onCloseMobile={() => setMobileConversationOpen(false)}
      onRetry={(turnId) => void session.retry(turnId)}
      onRetrySession={() => void session.retrySession()}
    />
  </section>
}
