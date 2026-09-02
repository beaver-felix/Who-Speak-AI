"""Streamlit MVP for enrollment and 1:N speaker identification."""

from __future__ import annotations

import hashlib

import streamlit as st

from voiceauth.audio import decode_recording
from voiceauth.config import VoiceSettings
from voiceauth.errors import VoiceAuthError
from voiceauth.gate import VoiceAuthGate
from voiceauth.service import VoiceVerificationService
from voiceauth.session_factory import VoiceAuthSessionFactory


st.set_page_config(page_title="Who Speak AI", page_icon="🎙️", layout="centered")


def settings() -> VoiceSettings:
    return VoiceSettings.from_environment()


def claim_audio_submission(action: str, payloads: list[bytes]) -> str | None:
    """Return a one-time submission ID without retaining raw audio in state."""
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    submission_id = f"{action}:{digest.hexdigest()}"
    processed = st.session_state.setdefault("processed_audio_submissions", set())
    if submission_id in processed:
        return None
    processed.add(submission_id)
    return submission_id


def release_audio_submission(submission_id: str | None) -> None:
    """Allow a deliberate retry when processing failed before completion."""
    if submission_id is not None:
        st.session_state.get("processed_audio_submissions", set()).discard(submission_id)


def workflow() -> VoiceVerificationService:
    """Create one private HE context per Streamlit session, never globally."""
    active = settings()
    if "voice_workflow" not in st.session_state:
        with st.status("Preparing the local speaker model and HE context…", expanded=True) as status:
            service = VoiceAuthSessionFactory(active).create(
                persistent_context=active.he_context_mode == "keychain"
            )
            status.update(label="Voice verification is ready", state="complete", expanded=False)
        st.session_state.voice_workflow = service
    return st.session_state.voice_workflow


def auth_gate() -> VoiceAuthGate:
    return VoiceAuthGate(workflow())


def home() -> None:
    st.title("Who Speak AI")
    st.caption("Privacy-preserving Vietnamese speaker verification")
    try:
        active = settings()
        st.success("Configuration loaded")
        st.write(f"RawNet3 profile: `rawnet3-vimd-best-epoch-2`  ")
        st.write(f"SV threshold: `{active.threshold:.6f}`")
        service = st.session_state.get("voice_workflow")
        if service is not None:
            st.metric("Encrypted identities in this session", service.matcher.identity_count(service.context.context_id))
        else:
            st.caption("No active verification session yet.")
        st.info("Enroll three voice samples, then verify a new sample against all enrolled identities.")
    except (ValueError, VoiceAuthError) as error:
        st.error(str(error))


def enroll_voice() -> None:
    st.title("Enroll voice")
    st.caption("Record three separate samples of clear Vietnamese speech, each at least four seconds.")
    with st.form("enroll-voice"):
        name = st.text_input("Display name", max_chars=120)
        samples = [
            st.audio_input(f"Voice sample {index}", sample_rate=16_000, key=f"enrollment-sample-{index}")
            for index in range(1, 4)
        ]
        submitted = st.form_submit_button("Encrypt and enroll", type="primary")
    if not submitted:
        return
    if any(sample is None for sample in samples):
        st.error("Record all three voice samples before enrolling.")
        return
    sample_bytes = [sample.getvalue() for sample in samples]
    submission_id = claim_audio_submission("enrollment", sample_bytes)
    if submission_id is None:
        st.info("This exact enrollment recording was already processed in this session.")
        return
    try:
        recordings = [decode_recording(payload) for payload in sample_bytes]
        del sample_bytes
        service = workflow()
        with st.status("Creating three local voice embeddings…", expanded=True) as status:
            result = service.enroll(name, recordings)
            status.update(label="Encrypted template saved", state="complete", expanded=False)
        st.success(f"Enrolled {result.display_name}.")
        if settings().he_context_mode == "keychain":
            st.info("Copy this local owner identity into VOICE_OWNER_ID before starting the LiveKit Auth Gate.")
            st.code(result.identity_id, language=None)
    except (ValueError, VoiceAuthError) as error:
        release_audio_submission(submission_id)
        st.error(str(error))


def verify_voice() -> None:
    st.title("Verify voice")
    st.caption("Record one new voice sample. The matcher receives ciphertext only.")
    with st.form("verify-voice"):
        sample = st.audio_input("Verification sample", sample_rate=16_000, key="verification-sample")
        submitted = st.form_submit_button("Verify identity", type="primary")
    if not submitted:
        return
    if sample is None:
        st.error("Record a voice sample before verifying.")
        return
    sample_bytes = sample.getvalue()
    submission_id = claim_audio_submission("verification", [sample_bytes])
    if submission_id is None:
        st.info("This exact verification recording was already processed in this session.")
        return
    try:
        recording = decode_recording(sample_bytes)
        del sample_bytes
        with st.status("Encrypting and comparing the voice sample…", expanded=True) as status:
            result = auth_gate().identify(recording)
            status.update(label="Verification complete", state="complete", expanded=False)
        if result.matched:
            st.success(f"Verified: {result.display_name}")
            if settings().show_debug_score:
                st.caption(f"Cosine score: {result.score:.6f}; candidates: {result.candidate_count}")
        else:
            st.warning("Not registered")
    except (ValueError, VoiceAuthError) as error:
        release_audio_submission(submission_id)
        st.error(str(error))


def app_settings() -> None:
    st.title("Settings")
    try:
        active = settings()
        st.text_input("Matcher URL", active.matcher_url, disabled=True)
        st.text_input("Model checkpoint", str(active.model_checkpoint), disabled=True)
        st.text_input("Model device", active.model_device, disabled=True)
        st.text_input("SV threshold", f"{active.threshold:.6f}", disabled=True)
        if st.button("Reset this session", type="secondary"):
            service = st.session_state.pop("voice_workflow", None)
            if service is not None and settings().he_context_mode == "session":
                try:
                    service.matcher.delete_context(service.context.context_id)
                finally:
                    service.matcher.close()
            elif service is not None:
                service.matcher.close()
            st.session_state.pop("processed_audio_submissions", None)
            if active.he_context_mode == "session":
                st.success("The session HE context and encrypted identities were removed.")
            else:
                st.success("The Streamlit session was reset. Persistent Keychain context and enrolled identities were kept.")
    except (ValueError, VoiceAuthError) as error:
        st.error(str(error))


page = st.navigation(
    [
        st.Page(home, title="Home", icon="🏠", default=True),
        st.Page(enroll_voice, title="Enroll voice", icon="➕"),
        st.Page(verify_voice, title="Verify voice", icon="🎙️"),
        st.Page(app_settings, title="Settings", icon="⚙️"),
    ]
)
page.run()
