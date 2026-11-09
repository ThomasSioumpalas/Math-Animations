import { useCallback, useEffect, useRef, useState } from 'react'
import { Menu, X } from 'lucide-react'
import { ACTIVE, api, type Job, type PublicConfig, type SessionView } from './api'
import * as keeper from './keeper'
import Sidebar from './components/Sidebar'
import Composer from './components/Composer'
import JobView from './components/JobView'
import VideoView from './components/VideoView'

export type Selection = { kind: 'new' } | { kind: 'video'; id: string } | { kind: 'job'; id: string }

type Toast = { id: number; text: string; kind: 'info' | 'error' }

// Small UI state lives in localStorage so a closed tab comes back exactly as it was.
function useStored<T>(key: string, initial: T) {
  const [v, setV] = useState<T>(() => {
    try { const s = localStorage.getItem(key); return s ? (JSON.parse(s) as T) : initial } catch { return initial }
  })
  useEffect(() => { localStorage.setItem(key, JSON.stringify(v)) }, [key, v])
  return [v, setV] as const
}

export default function App() {
  const [data, setData] = useState<SessionView | null>(null)
  const [config, setConfig] = useState<PublicConfig | null>(null)
  const [offline, setOffline] = useState(false)
  const [selection, setSelection] = useStored<Selection>('mstudio.selection', { kind: 'new' })
  const [draft, setDraft] = useStored('mstudio.draft', '')
  const [folderId, setFolderId] = useStored('mstudio.folder', 'general')
  const [quality, setQuality] = useStored('mstudio.quality.v2', 'medium')
  const [submitting, setSubmitting] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  const prevJobs = useRef<Map<string, Job>>(new Map())
  const restoring = useRef(false)
  const selectionRef = useRef(selection)
  selectionRef.current = selection

  const toast = useCallback((text: string, kind: Toast['kind'] = 'info') => {
    const id = Date.now() + Math.random()
    setToasts((t) => [...t, { id, text, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000)
  }, [])

  const refresh = useCallback(async () => {
    try {
      let s = await api.session()
      setOffline(false)
      // The server lost our session (a restart wipes its disk): put back what this page holds.
      if (keeper.isLost(s)) {
        if (restoring.current) return
        restoring.current = true
        let n = 0
        try { n = await keeper.restore() } catch { return } finally { restoring.current = false }
        s = await api.session()
        if (n) toast(`The server restarted, so this page put your ${n === 1 ? 'video' : `${n} videos`} back.`)
      }
      keeper.sync(s)
      setData(s)
      // Active jobs only vanish when the server restarts (cancelled ones are removed from prevJobs).
      for (const [id, j] of prevJobs.current) {
        if (ACTIVE.includes(j.status) && !s.jobs.some((x) => x.id === id)) {
          toast(`The server restarted and stopped “${j.title || 'your animation'}”. Your prompt is back in the box: press Generate to try again.`, 'error')
          setDraft((d) => (d.trim() ? d : j.prompt))
        }
      }
      // Notice jobs that just finished.
      for (const j of s.jobs) {
        const before = prevJobs.current.get(j.id)?.status
        if (before && ACTIVE.includes(before) && j.status === 'done' && j.video_id) {
          const sel = selectionRef.current
          if (sel.kind === 'job' && sel.id === j.id) setSelection({ kind: 'video', id: j.video_id })
          else toast(`“${j.title}” is ready`)
        }
        if (before && ACTIVE.includes(before) && j.status === 'failed') toast(`An animation failed: ${j.message}`, 'error')
      }
      prevJobs.current = new Map(s.jobs.map((j) => [j.id, j]))
    } catch {
      setOffline(true)
    }
  }, [setSelection, setDraft, toast])

  useEffect(() => { void refresh(); api.config().then(setConfig).catch(() => {}) }, [refresh])

  const hasActive = !!data?.jobs.some((j) => ACTIVE.includes(j.status))
  useEffect(() => {
    // Poll fast while rendering; otherwise a heartbeat every minute, in background tabs too:
    // the server keeps a session until 1 h after the page was LAST SEEN, so an open page
    // never expires. Catch up at once when the tab comes back or the network returns.
    const t = setInterval(() => void refresh(), hasActive ? 2000 : 60000)
    const wake = () => { if (document.visibilityState === 'visible') void refresh() }
    document.addEventListener('visibilitychange', wake)
    window.addEventListener('online', wake)
    window.addEventListener('pageshow', wake)
    return () => {
      clearInterval(t)
      document.removeEventListener('visibilitychange', wake)
      window.removeEventListener('online', wake)
      window.removeEventListener('pageshow', wake)
    }
  }, [hasActive, refresh])

  // If the selected thing disappeared (expired, deleted), go back to the composer.
  useEffect(() => {
    if (!data) return
    if (selection.kind === 'video' && !data.videos.some((v) => v.id === selection.id)) setSelection({ kind: 'new' })
    if (selection.kind === 'job' && !data.jobs.some((j) => j.id === selection.id)) setSelection({ kind: 'new' })
    if (!data.folders.some((f) => f.id === folderId)) setFolderId('general')
  }, [data, selection, folderId, setSelection, setFolderId])

  const run = async (fn: () => Promise<unknown>) => {
    try { await fn() } catch (e) { toast((e as Error).message, 'error') }
    await refresh()
  }

  const generate = async (prompt: string, folder = folderId, q = quality) => {
    setSubmitting(true)
    try {
      const job = await api.generate(prompt, folder, q)
      prevJobs.current.set(job.id, job)
      setSelection({ kind: 'job', id: job.id })
      setDraft('')
      await refresh()
    } catch (e) {
      toast((e as Error).message, 'error')
    } finally {
      setSubmitting(false)
    }
  }

  const select = (s: Selection) => { setSelection(s); setSidebarOpen(false) }

  const video = selection.kind === 'video' ? data?.videos.find((v) => v.id === selection.id) : undefined
  const job = selection.kind === 'job' ? data?.jobs.find((j) => j.id === selection.id) : undefined

  return (
    <div className="app">
      <Sidebar
        folders={data?.folders ?? []}
        videos={data?.videos ?? []}
        jobs={data?.jobs ?? []}
        selection={selection}
        expiresAt={data?.expires_at ?? null}
        ttlSeconds={data?.ttl_seconds ?? 3600}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={select}
        onCreateFolder={(name) => run(async () => { const f = await api.createFolder(name); setFolderId(f.id) })}
        onRenameFolder={(id, name) => run(() => api.renameFolder(id, name))}
        onDeleteFolder={(id) => run(() => api.deleteFolder(id))}
        onMoveVideo={(vid, fid) => run(() => api.updateVideo(vid, { folder_id: fid }))}
      />

      <main className="main">
        <div className="topbar mobile-only">
          <button className="icon-btn" onClick={() => setSidebarOpen(true)} aria-label="Open library"><Menu size={20} /></button>
          <span>Manim Studio</span>
        </div>
        {offline && <div className="banner">Can't reach the server. Retrying… (free servers can take ~30 s to wake up)</div>}

        {video ? (
          <VideoView
            video={video}
            folders={data?.folders ?? []}
            onRename={(title) => run(() => api.updateVideo(video.id, { title }))}
            onMove={(fid) => run(async () => { await api.updateVideo(video.id, { folder_id: fid }); toast('Moved') })}
            onDelete={() => run(async () => { await api.deleteVideo(video.id); setSelection({ kind: 'new' }) })}
            onRemix={() => { setDraft(video.prompt); setFolderId(video.folder_id); setSelection({ kind: 'new' }) }}
            onAnimate={(prompt) => { setDraft(prompt); setFolderId(video.folder_id); setSelection({ kind: 'new' }); toast('Prompt ready: edit it if you like, then press Generate.') }}
          />
        ) : job ? (
          <JobView
            job={job}
            onRetry={() => void generate(job.prompt, job.folder_id, job.quality)}
            onEdit={() => { setDraft(job.prompt); setSelection({ kind: 'new' }) }}
            onCancel={async () => {
              try {
                await api.cancelJob(job.id)
                prevJobs.current.delete(job.id)
                setDraft(job.prompt)
                setFolderId(job.folder_id)
                setSelection({ kind: 'new' })
                toast('Cancelled. Your prompt is back in the box.')
              } catch (e) {
                toast((e as Error).message, 'error')
              }
              await refresh()
            }}
          />
        ) : (
          <Composer
            config={config}
            draft={draft}
            setDraft={setDraft}
            folders={data?.folders ?? []}
            folderId={folderId}
            setFolderId={setFolderId}
            quality={quality}
            setQuality={setQuality}
            submitting={submitting}
            onSubmit={() => void generate(draft.trim())}
          />
        )}
      </main>

      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>
            <span>{t.text}</span>
            <button className="icon-btn sm" onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}><X size={14} /></button>
          </div>
        ))}
      </div>
    </div>
  )
}
