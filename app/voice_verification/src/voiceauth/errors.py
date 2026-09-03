"""Typed, safe errors exposed by the voice-verification boundary."""


class VoiceAuthError(RuntimeError):
    """Base exception with a user-safe message."""


class AudioValidationError(VoiceAuthError):
    pass


class ModelInitializationError(VoiceAuthError):
    pass


class ModelWorkerError(VoiceAuthError):
    pass


class HEContextError(VoiceAuthError):
    pass


class MatcherError(VoiceAuthError):
    pass


class EnrollmentError(VoiceAuthError):
    pass
