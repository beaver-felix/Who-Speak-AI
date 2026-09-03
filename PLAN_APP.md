# Final Application Concept: Secure Virtual Assistant with Speaker Recognition

## Core Features Architecture
We are building a lightweight Python-based Virtual Assistant using Streamlit/Gradio, free LLM APIs (Gemini/Groq), OpenAI Whisper (ASR), Edge-TTS (TTS), and a fine-tuned Speaker Recognition Model (RawNet3 / WavLM).

### 1. General Function (No Auth)
- General Q&A, weather, news, and small talk.
- Open to all users.

### 2. Personalization Feature (Speaker Identification - SID)
- *Feature:* Adaptive Voice Profile & Personalized Responses.
- *Workflow:*
  1. On every audio input, run 1-vs-N Speaker Identification against registered embeddings in the database.
  2. If identified (e.g., "Thanh"): Fetch user metadata (Preferred Name, Tone, Persona) and dynamically inject into LLM System Prompt.
  3. If unidentified: Fallback to guest mode ("Friend") and prompt optional voice enrollment.

### 3. Important Feature (Speaker Verification - SV)
- *Feature:* Personal Voice Diary (Read / Write / Delete entries).
- *Workflow:*
  1. User requests: "Open my diary" or "Write a diary entry".
  2. System prompts for voice passphrase authentication.
  3. Run 1-vs-1 Speaker Verification (Cosine similarity between current passphrase audio embedding and target user's stored embedding).
  4. If Similarity Score >= Threshold (e.g., 0.75): Grant access to diaries/{user_id}/.
  5. Else: Reject access with visual score feedback.

### 4. User Enrollment & Storage Procedure
- *Inputs:* Name, Age, Gender, Voice Sample / Passphrase (3 short recordings).
- *Processing:* Extract average speaker embedding vector using fine-tuned model.
- *Storage:* Save user profile and vector embedding into users_db.json (or SQLite). Create diaries/{user_id}/ directory.
