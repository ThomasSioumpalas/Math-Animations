/**
 * While the page is open it keeps its own copy of every finished video: the MP4, its code
 * and its chat. The server only deletes a session 1 h after the page was last seen, but two
 * things can still empty it while a page is open: a server restart (deploys, waking up from
 * sleep wipe its disk) and a phone freezing a background tab for over an hour. When the
 * server comes back without our videos, the page hands them back (POST /api/restore).
 * Nothing here outlives the tab: it is plain memory, gone on close or reload.
 */
import { api, type ChatMessage, type Folder, type SessionView, type Video } from './api'

type Kept = { meta: Video; blob?: Blob; url?: string; code?: string; chat?: ChatMessage[] }

const kept = new Map<string, Kept>()
let folders: Folder[] = []
let born = 0 // created_at of the server session this page last saw (0 = none yet)
let copying = false
let failedRestores = 0

/** Blob URL of the page's own copy, if it has one. */
export const localUrl = (id: string) => kept.get(id)?.url
export const localCode = (id: string) => kept.get(id)?.code

/** Called by the chat whenever it loads or gets an answer, so a restore brings it back too. */
export function keepChat(id: string, chat: ChatMessage[]) {
  const k = kept.get(id)
  if (k) k.chat = chat
}

function forget(id: string) {
  const k = kept.get(id)
  if (k?.url) URL.revokeObjectURL(k.url)
  kept.delete(id)
}

/** True when the server no longer has the session this page knew and we hold something to give back. */
export function isLost(s: SessionView) {
  return born !== 0 && s.created_at != null && s.created_at !== born && (kept.size > 0 || folders.length > 1)
}

/** Mirror the server's library (only call when !isLost). */
export function sync(s: SessionView) {
  born = s.created_at ?? 0
  folders = s.folders
  failedRestores = 0
  const ids = new Set(s.videos.map((v) => v.id))
  for (const id of [...kept.keys()]) if (!ids.has(id)) forget(id) // deleted
  for (const v of s.videos) {
    const k = kept.get(v.id)
    if (k) k.meta = v // renames and moves
    else kept.set(v.id, { meta: v })
  }
  void copyMissing()
}

/** Download, one at a time, every video the page doesn't hold yet. */
async function copyMissing() {
  if (copying) return
  copying = true
  try {
    for (const [id, k] of kept) {
      if (k.blob) continue
      try {
        const [blob, code, chat] = await Promise.all([
          fetch(api.videoUrl(id)).then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.blob() }),
          api.code(id),
          api.videoChat(id).then((r) => r.messages),
        ])
        if (kept.get(id) !== k) continue // deleted meanwhile
        k.blob = blob
        k.url = URL.createObjectURL(blob)
        k.code = code
        k.chat ??= chat
      } catch {
        /* server busy or restarting: the next sync tries again */
      }
    }
  } finally {
    copying = false
  }
}

let running: Promise<number> | null = null

/**
 * Give the server back everything the page holds. Resolves to how many videos were
 * restored; rejects if the server can't be reached (call again on the next poll).
 */
export function restore(): Promise<number> {
  running ??= (async () => {
    const ready = [...kept.values()].filter((k) => k.blob)
    try {
      const r = await api.restore({
        folders,
        videos: ready.map((k) => ({ ...k.meta, code: k.code ?? '', chat: k.chat ?? [] })),
      })
      let restored = 0
      for (const id of r.need_files) {
        const k = kept.get(id)
        if (!k?.blob) continue
        try { await api.restoreFile(id, k.blob); restored++ } catch { /* this one is lost */ }
      }
      born = 0 // accept the server's new session on the next sync
      return restored
    } catch (e) {
      if (++failedRestores >= 5) born = 0 // give up rather than freeze the page forever
      throw e
    }
  })().finally(() => { running = null })
  return running
}
