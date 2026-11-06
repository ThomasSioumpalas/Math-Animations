import { useEffect, useState } from 'react'
import {
  ChevronDown, ChevronRight, Clock, Film, Folder, FolderOpen, FolderPlus, LoaderCircle, Pencil,
  Plus, Trash2, TriangleAlert, X,
} from 'lucide-react'
import { ACTIVE, type Folder as FolderT, type Job, type Video } from '../api'
import type { Selection } from '../App'
import { timeAgo } from '../util'

type Props = {
  folders: FolderT[]
  videos: Video[]
  jobs: Job[]
  selection: Selection
  expiresAt: number | null
  ttlSeconds: number
  open: boolean
  onClose: () => void
  onSelect: (s: Selection) => void
  onCreateFolder: (name: string) => Promise<void>
  onRenameFolder: (id: string, name: string) => Promise<void>
  onDeleteFolder: (id: string) => Promise<void>
  onMoveVideo: (videoId: string, folderId: string) => Promise<void>
}

const OPEN_KEY = 'mstudio.closedFolders'

export default function Sidebar(p: Props) {
  const [closed, setClosed] = useState<Set<string>>(() => {
    try { return new Set(JSON.parse(localStorage.getItem(OPEN_KEY) ?? '[]')) } catch { return new Set() }
  })
  const [renaming, setRenaming] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [dropTarget, setDropTarget] = useState<string | null>(null)

  useEffect(() => { localStorage.setItem(OPEN_KEY, JSON.stringify([...closed])) }, [closed])

  const toggle = (id: string) =>
    setClosed((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n })

  const pendingJobs = p.jobs.filter((j) => j.status !== 'done')
  const ttlLabel = p.ttlSeconds >= 3600 ? `${Math.round(p.ttlSeconds / 3600)} h` : `${Math.round(p.ttlSeconds / 60)} min`

  return (
    <>
      <div className={`scrim ${p.open ? 'show' : ''}`} onClick={p.onClose} />
      <aside className={`sidebar ${p.open ? 'open' : ''}`}>
        <div className="brand">
          <div className="logo" aria-hidden>
            <svg viewBox="0 0 64 64"><path d="M10 44c8-22 14-22 22 0s14 22 22 0" fill="none" stroke="currentColor" strokeWidth="6" strokeLinecap="round" /><circle cx="32" cy="22" r="7" fill="var(--accent-2)" /></svg>
          </div>
          <span>Manim Studio</span>
          <button className="icon-btn mobile-only" onClick={p.onClose} aria-label="Close sidebar"><X size={18} /></button>
        </div>

        <button className={`new-btn ${p.selection.kind === 'new' ? 'active' : ''}`} onClick={() => p.onSelect({ kind: 'new' })}>
          <Plus size={17} /> New animation
        </button>

        <div className="section-head">
          <span>Library</span>
          <button className="icon-btn" title="New folder" onClick={() => setCreating(true)}><FolderPlus size={16} /></button>
        </div>

        <nav className="tree">
          {p.folders.map((f) => {
            const vids = p.videos.filter((v) => v.folder_id === f.id)
            const jobs = pendingJobs.filter((j) => j.folder_id === f.id)
            const isOpen = !closed.has(f.id)
            return (
              <div
                key={f.id}
                className={`folder ${dropTarget === f.id ? 'drop' : ''}`}
                onDragOver={(e) => { if (e.dataTransfer.types.includes('text/x-video')) { e.preventDefault(); setDropTarget(f.id) } }}
                onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setDropTarget(null) }}
                onDrop={(e) => {
                  e.preventDefault(); setDropTarget(null)
                  const id = e.dataTransfer.getData('text/x-video')
                  if (id) { void p.onMoveVideo(id, f.id); setClosed((s) => { const n = new Set(s); n.delete(f.id); return n }) }
                }}
              >
                <div className="folder-row" onClick={() => renaming !== f.id && toggle(f.id)}>
                  {isOpen ? <ChevronDown size={14} className="chev" /> : <ChevronRight size={14} className="chev" />}
                  {isOpen ? <FolderOpen size={16} className="ficon" /> : <Folder size={16} className="ficon" />}
                  {renaming === f.id ? (
                    <InlineInput
                      initial={f.name}
                      onDone={async (name) => { setRenaming(null); if (name && name !== f.name) await p.onRenameFolder(f.id, name) }}
                    />
                  ) : (
                    <span className="fname">{f.name}</span>
                  )}
                  <span className="count">{vids.length + jobs.length || ''}</span>
                  {renaming !== f.id && (
                    <span className="row-actions" onClick={(e) => e.stopPropagation()}>
                      <button className="icon-btn sm" title="Rename" onClick={() => setRenaming(f.id)}><Pencil size={13} /></button>
                      {f.id !== 'general' && (
                        <button className="icon-btn sm" title="Delete folder (videos move to General)"
                          onClick={() => { if (confirm(`Delete folder "${f.name}"? Its videos move to General.`)) void p.onDeleteFolder(f.id) }}>
                          <Trash2 size={13} />
                        </button>
                      )}
                    </span>
                  )}
                </div>
                {isOpen && (
                  <ul className="items">
                    {jobs.map((j) => (
                      <li key={j.id}>
                        <button className={`item ${p.selection.kind === 'job' && p.selection.id === j.id ? 'active' : ''}`}
                          onClick={() => p.onSelect({ kind: 'job', id: j.id })}>
                          {ACTIVE.includes(j.status)
                            ? <LoaderCircle size={15} className="spin accent" />
                            : <TriangleAlert size={15} className="danger" />}
                          <span className="ititle">{j.title || j.prompt}</span>
                        </button>
                      </li>
                    ))}
                    {vids.map((v) => (
                      <li key={v.id}>
                        <button
                          draggable
                          onDragStart={(e) => { e.dataTransfer.setData('text/x-video', v.id); e.dataTransfer.effectAllowed = 'move' }}
                          className={`item ${p.selection.kind === 'video' && p.selection.id === v.id ? 'active' : ''}`}
                          onClick={() => p.onSelect({ kind: 'video', id: v.id })}
                          title={v.prompt}
                        >
                          <Film size={15} className="muted" />
                          <span className="ititle">{v.title}</span>
                          <span className="ago">{timeAgo(v.created_at)}</span>
                        </button>
                      </li>
                    ))}
                    {!vids.length && !jobs.length && <li className="empty">Empty. Drag videos here.</li>}
                  </ul>
                )}
              </div>
            )
          })}
          {creating && (
            <div className="folder-row creating">
              <FolderPlus size={16} className="ficon" />
              <InlineInput initial="" placeholder="Folder name"
                onDone={async (name) => { setCreating(false); if (name) await p.onCreateFolder(name) }} />
            </div>
          )}
        </nav>

        <div className="sidebar-foot">
          <Clock size={14} />
          <div>
            <strong>No account needed.</strong> Your videos are kept for {ttlLabel} after you leave, so
            you can close the tab and come back. Download anything you want to keep.
          </div>
        </div>
      </aside>
    </>
  )
}

function InlineInput({ initial, placeholder, onDone }: { initial: string; placeholder?: string; onDone: (v: string) => void }) {
  const [v, setV] = useState(initial)
  return (
    <input
      autoFocus
      className="inline-input"
      value={v}
      maxLength={60}
      placeholder={placeholder}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => setV(e.target.value)}
      onFocus={(e) => e.target.select()}
      onBlur={() => onDone(v.trim())}
      onKeyDown={(e) => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        if (e.key === 'Escape') { setV(initial); onDone(initial === '' ? '' : initial) }
      }}
    />
  )
}
