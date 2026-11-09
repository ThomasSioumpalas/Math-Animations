export type Folder = { id: string; name: string; created_at: number }

export type Video = {
  id: string
  title: string
  prompt: string
  folder_id: string
  quality: string
  created_at: number
  attempts: number
  model?: string
  layout_notes?: string[]
}

export type Thinking = 'standard' | 'high' | 'max'
export type ChatMessage = { role: 'user' | 'assistant'; content: string; at: number; model?: string; thinking?: Thinking }

export type JobStatus = 'queued' | 'writing' | 'rendering' | 'fixing' | 'done' | 'failed' | 'cancelled'

export type Job = {
  id: string
  prompt: string
  folder_id: string
  quality: string
  status: JobStatus
  message: string
  attempt: number
  max_attempts: number
  code: string
  error: string
  video_id: string | null
  title: string
  created_at: number
  finished_at: number | null
  position: number
}

export type SessionView = {
  session_id: string
  created_at: number
  folders: Folder[]
  videos: Video[]
  jobs: Job[]
  expires_at: number
  ttl_seconds: number
  now: number
}

export type PublicConfig = {
  model: string
  allow_hd: boolean
  max_prompt_chars: number
  ttl_seconds: number
  latex: boolean
}

export const ACTIVE: JobStatus[] = ['queued', 'writing', 'rendering', 'fixing']

// The session id normally travels in an HttpOnly cookie. We also keep it in localStorage and
// send it as a header, so sessions still work where third-party cookies are blocked
// (e.g. when the site is embedded in the Hugging Face Space page iframe).
const SID_KEY = 'mstudio.sid'
let sid = ''
try { sid = localStorage.getItem(SID_KEY) ?? '' } catch { /* storage disabled */ }

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {}
  if (body) headers['Content-Type'] = 'application/json'
  if (sid) headers['X-Session-Id'] = sid
  const res = await fetch(url, {
    method,
    credentials: 'same-origin',
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })
  const newSid = res.headers.get('X-Session-Id')
  if (newSid && newSid !== sid) {
    sid = newSid
    try { localStorage.setItem(SID_KEY, sid) } catch { /* ignore */ }
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const j = await res.json()
      if (typeof j.detail === 'string') detail = j.detail
      else if (Array.isArray(j.detail)) detail = j.detail.map((d: { msg: string }) => d.msg).join(', ')
    } catch {
      /* not json */
    }
    throw new Error(detail)
  }
  const type = res.headers.get('content-type') ?? ''
  return (type.includes('json') ? res.json() : res.text()) as Promise<T>
}

export const api = {
  config: () => req<PublicConfig>('GET', '/api/config'),
  session: () => req<SessionView>('GET', '/api/session'),
  createFolder: (name: string) => req<Folder>('POST', '/api/folders', { name }),
  renameFolder: (id: string, name: string) => req<Folder>('PATCH', `/api/folders/${id}`, { name }),
  deleteFolder: (id: string) => req('DELETE', `/api/folders/${id}`),
  updateVideo: (id: string, patch: { title?: string; folder_id?: string }) =>
    req<Video>('PATCH', `/api/videos/${id}`, patch),
  deleteVideo: (id: string) => req('DELETE', `/api/videos/${id}`),
  code: (id: string) => req<string>('GET', `/api/videos/${id}/code`),
  cancelJob: (id: string) => req('POST', `/api/jobs/${id}/cancel`),
  videoChat: (id: string) => req<{ messages: ChatMessage[] }>('GET', `/api/videos/${id}/chat`),
  askVideo: (id: string, message: string, thinking: Thinking) =>
    req<{ messages: ChatMessage[] }>('POST', `/api/videos/${id}/chat`, { message, thinking }),
  generate: (prompt: string, folder_id: string, quality: string) =>
    req<Job>('POST', '/api/generate', { prompt, folder_id, quality }),
  videoUrl: (id: string, download = false) =>
    `/api/videos/${id}/file?sid=${encodeURIComponent(sid)}${download ? '&download=1' : ''}`,
  codeUrl: (id: string) => `/api/videos/${id}/code?download=1&sid=${encodeURIComponent(sid)}`,
  restore: (body: { folders: Folder[]; videos: (Video & { code: string; chat: ChatMessage[] })[] }) =>
    req<{ need_files: string[] }>('POST', '/api/restore', body),
  restoreFile: async (id: string, file: Blob) => {
    const res = await fetch(`/api/restore/videos/${id}/file`, {
      method: 'PUT', credentials: 'same-origin', body: file,
      headers: { 'Content-Type': 'video/mp4', ...(sid ? { 'X-Session-Id': sid } : {}) },
    })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  },
}
