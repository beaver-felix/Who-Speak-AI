import { FormEvent, useEffect, useRef, useState } from 'react'

import { api, type AgentCapabilities, type CurrentUser, type GoogleCalendarStatus } from './api'
import { LiveKitPanel } from './LiveKitPanel'
import { blobToBase64, blobToWav } from './wav-recorder'

type Mode = 'sign-in' | 'register'

function AuthForm({ onUser }: { onUser: (user: CurrentUser) => void }) {
  const [mode, setMode] = useState<Mode>('sign-in')
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setPending(true)
    setError(null)
    const form = new FormData(event.currentTarget)
    try {
      const user = mode === 'register'
        ? await api.register(String(form.get('email')), String(form.get('password')), String(form.get('displayName')))
        : await api.login(String(form.get('email')), String(form.get('password')))
      onUser(user)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not sign in.')
    } finally { setPending(false) }
  }

  return <main className="auth-shell"><section className="auth-card">
    <p className="eyebrow">Who Speak AI · local-first</p>
    <h1>{mode === 'sign-in' ? 'Sign in to your voice workspace.' : 'Create a local account.'}</h1>
    <p className="lede">Voice verification—not a transcript—unlocks private assistant capabilities.</p>
    <form onSubmit={submit} className="form">
      {mode === 'register' && <label>Display name<input name="displayName" required maxLength={120} /></label>}
      <label>Email<input name="email" type="email" required autoComplete="email" /></label>
      <label>Password<input name="password" type="password" required minLength={mode === 'register' ? 10 : 1} autoComplete={mode === 'register' ? 'new-password' : 'current-password'} /></label>
      {error && <p className="error" role="alert">{error}</p>}
      <button className="primary" disabled={pending}>{pending ? 'Working…' : mode === 'sign-in' ? 'Sign in' : 'Create account'}</button>
    </form>
    <button className="link" onClick={() => setMode(mode === 'sign-in' ? 'register' : 'sign-in')}>{mode === 'sign-in' ? 'Need a local account? Register' : 'Already have an account? Sign in'}</button>
  </section></main>
}

function Enrollment({ user, onComplete }: { user: CurrentUser; onComplete: () => void }) {
  const [samples, setSamples] = useState<(string | null)[]>([null, null, null])
  const [recording, setRecording] = useState<number | null>(null)
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const recorder = useRef<MediaRecorder | null>(null)
  const media = useRef<MediaStream | null>(null)

  useEffect(() => () => { recorder.current?.stop(); media.current?.getTracks().forEach((track) => track.stop()) }, [])
  useEffect(() => {
    if (recording === null) return
    const timer = window.setInterval(() => setSeconds((value) => value + 1), 1000)
    return () => window.clearInterval(timer)
  }, [recording])

  async function start(index: number) {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      media.current = stream
      const chunks: BlobPart[] = []
      const active = new MediaRecorder(stream)
      active.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data) }
      active.onstop = async () => {
        try {
          const wav = await blobToWav(new Blob(chunks, { type: active.mimeType || 'audio/webm' }))
          const base64 = await blobToBase64(wav)
          setSamples((current) => current.map((value, position) => position === index ? base64 : value))
        } catch (caught) { setError(caught instanceof Error ? `Could not process sample: ${caught.message}` : 'Could not process the recording.')
        } finally { stream.getTracks().forEach((track) => track.stop()); media.current = null; recorder.current = null; setRecording(null) }
      }
      recorder.current = active
      setSeconds(0)
      setRecording(index)
      active.start()
    } catch (caught) { setError(caught instanceof Error ? `Microphone error: ${caught.message}` : 'Microphone access was denied.') }
  }

  async function enroll() {
    setPending(true)
    setError(null)
    try { await api.enroll(samples as string[], user.display_name); onComplete()
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Enrollment failed.')
    } finally { setPending(false) }
  }

  return <section className="card enrollment-card"><p className="eyebrow">Step 1 of 1</p><h2>Enroll your voice</h2><p>Record three short samples (4–8 seconds each) in a quiet room. Audio is converted to WAV in the browser and sent only to the local gateway.</p>
    <div className="sample-grid">{samples.map((sample, index) => <div className="sample" key={index}><strong>Sample {index + 1}</strong><span>{sample ? 'Ready' : recording === index ? `Recording ${seconds}s` : 'Not recorded'}</span>{recording === index ? <button className="danger" type="button" onClick={() => recorder.current?.stop()}>Stop</button> : <button className="secondary" type="button" disabled={recording !== null} onClick={() => void start(index)}>{sample ? 'Record again' : 'Record sample'}</button>}</div>)}</div>
    {error && <p className="error" role="alert">{error}</p>}<button className="primary" disabled={samples.some((sample) => !sample) || pending || recording !== null} onClick={() => void enroll()}>{pending ? 'Encrypting and enrolling locally…' : 'Encrypt and enroll voice'}</button>
  </section>
}

function GoogleCalendarCard() {
  const [status, setStatus] = useState<GoogleCalendarStatus | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    try { setStatus(await api.googleCalendarStatus()); setError(null) }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load Google Calendar status.') }
  }

  useEffect(() => { void refresh() }, [])

  async function disconnect() {
    setPending(true)
    try { await api.disconnectGoogleCalendar(); await refresh() }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not disconnect Google Calendar.') }
    finally { setPending(false) }
  }

  function connect() {
    setPending(true)
    const popup = window.open(api.googleCalendarStartUrl(), 'who-speak-google-calendar', 'popup,width=520,height=720')
    if (!popup) { setError('Cho phép popup để liên kết Google Calendar.'); setPending(false); return }
    const started = Date.now()
    const poll = window.setInterval(() => {
      if (popup.closed || Date.now() - started > 120_000) {
        window.clearInterval(poll)
        setPending(false)
        void refresh()
      }
    }, 1500)
  }

  if (!status) return <section className="card subtle"><h2>Google Calendar</h2><p>Đang kiểm tra trạng thái kết nối…</p></section>
  if (!status.configured) return <section className="card subtle"><h2>Google Calendar</h2><p>Google Calendar MCP chưa được cấu hình trên gateway. Mock MCP vẫn là provider đang hoạt động.</p></section>
  return <section className="card subtle integration-card">
    <h2>Google Calendar</h2>
    {status.connected
      ? <><p><strong>Đã kết nối Google Calendar.</strong> Account: {status.email}</p><p className="muted">VoiceAuth vẫn bắt buộc trước khi Agent đọc dữ liệu cá nhân.</p><button className="secondary" type="button" disabled={pending} onClick={() => void disconnect()}>{pending ? 'Đang ngắt kết nối…' : 'Disconnect'}</button></>
      : <><p><strong>Google Calendar chưa kết nối.</strong> Liên kết đúng tài khoản Google có email trùng Who Speak để dùng lịch cá nhân.</p>{status.status === 'needs_reconnect' && <p className="error" role="alert">Quyền Google Calendar đã hết hạn. Hãy kết nối lại.</p>}<button className="primary" type="button" disabled={pending} onClick={connect}>{pending ? 'Đang mở Google…' : 'Connect Google Calendar'}</button></>}
    {error && <p className="error" role="alert">{error}</p>}
  </section>
}

function Workspace({ user, setUser }: { user: CurrentUser; setUser: (value: CurrentUser | null) => void }) {
  const [enrollOpen, setEnrollOpen] = useState(!user.voice_enrolled)
  const [capabilities, setCapabilities] = useState<AgentCapabilities | null>(null)
  async function logout() { await api.logout(); setUser(null) }
  useEffect(() => { void api.capabilities().then(setCapabilities).catch(() => setCapabilities(null)) }, [])
  const usingGoogleMcp = capabilities?.mcp_provider === 'mcp' || capabilities?.mcp_provider === 'local'
  const usingLocalMcp = capabilities?.mcp_provider === 'local'
  return <main className="shell">
    <header className="topbar"><div><p className="eyebrow">Who Speak AI</p><h1>Local voice workspace</h1></div><div className="account"><span>{user.display_name}</span><button className="link" onClick={() => void logout()}>Sign out</button></div></header>
    <section className="intro"><p>Account: {user.email}</p><p className="demo">{usingGoogleMcp ? (usingLocalMcp ? 'Self-hosted Google Calendar MCP configured · tokens stay on the local MCP server.' : 'Google Calendar MCP configured · tokens stay on the gateway.') : 'Mock MCP active · no Google account is connected.'}</p></section>
    {!user.voice_enrolled && enrollOpen ? <Enrollment user={user} onComplete={() => { setEnrollOpen(false); setUser({ ...user, voice_enrolled: true }) }} /> : <><LiveKitPanel />{usingGoogleMcp ? <GoogleCalendarCard /> : <section className="card subtle"><h2>Mock Calendar Provider</h2><p><strong>Demo calendar data.</strong> The Agent may use it only after trusted voice authentication. It never accesses Google Calendar in this phase.</p>{user.voice_enrolled && <button className="secondary" type="button" onClick={() => setEnrollOpen(true)}>Re-enroll voice</button>}</section>}</>}
  </main>
}

export default function App() {
  const [user, setUser] = useState<CurrentUser | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => { void api.me().then(setUser).catch(() => setUser(null)).finally(() => setLoading(false)) }, [])
  if (loading) return <main className="auth-shell"><p>Loading local workspace…</p></main>
  return user ? <Workspace user={user} setUser={setUser} /> : <AuthForm onUser={setUser} />
}
