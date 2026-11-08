import { useEffect, useState } from 'react'
import { Code2, Download, FolderInput, Pencil, Trash2, WandSparkles } from 'lucide-react'
import { api, type Folder, type Video } from '../api'
import CodePanel from './CodePanel'

type Props = {
  video: Video
  folders: Folder[]
  onRename: (title: string) => Promise<void>
  onMove: (folderId: string) => Promise<void>
  onDelete: () => Promise<void>
  onRemix: () => void
}

export default function VideoView({ video, folders, onRename, onMove, onDelete, onRemix }: Props) {
  const [showCode, setShowCode] = useState(false)
  const [code, setCode] = useState('')
  const [editing, setEditing] = useState(false)
  const [title, setTitle] = useState(video.title)

  useEffect(() => { setShowCode(false); setCode(''); setEditing(false); setTitle(video.title) }, [video.id, video.title])
  useEffect(() => {
    if (showCode && !code) api.code(video.id).then(setCode).catch(() => setCode('# could not load code'))
  }, [showCode, code, video.id])

  const saveTitle = async () => {
    setEditing(false)
    const t = title.trim()
    if (t && t !== video.title) await onRename(t)
    else setTitle(video.title)
  }

  return (
    <div className="view">
      <header className="view-head">
        <div className="grow">
          <div className="eyebrow">{new Date(video.created_at * 1000).toLocaleString()} · {video.quality === 'medium' ? '720p' : '480p'}{video.model ? ` · made with ${video.model}` : ''}</div>
          {editing ? (
            <input className="title-input" autoFocus value={title} maxLength={80}
              onChange={(e) => setTitle(e.target.value)} onBlur={saveTitle}
              onKeyDown={(e) => { if (e.key === 'Enter') void saveTitle(); if (e.key === 'Escape') { setTitle(video.title); setEditing(false) } }} />
          ) : (
            <h2 className="editable" onClick={() => setEditing(true)} title="Click to rename">
              {video.title} <Pencil size={15} className="muted" />
            </h2>
          )}
        </div>
      </header>

      <div className="player">
        <video key={video.id} src={api.videoUrl(video.id)} controls autoPlay muted playsInline loop />
      </div>

      <div className="toolbar">
        <a className="primary" href={api.videoUrl(video.id, true)}><Download size={15} /> Download MP4</a>
        <button className={`ghost ${showCode ? 'on' : ''}`} onClick={() => setShowCode((s) => !s)}><Code2 size={15} /> {showCode ? 'Hide code' : 'View code'}</button>
        <button className="ghost" onClick={onRemix} title="Start a new animation from this prompt"><WandSparkles size={15} /> Remix prompt</button>
        <label className="ghost select-wrap" title="Move to folder">
          <FolderInput size={15} />
          <select value={video.folder_id} onChange={(e) => void onMove(e.target.value)}>
            {folders.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
          </select>
        </label>
        <span className="grow" />
        <button className="ghost danger" onClick={() => { if (confirm('Delete this video?')) void onDelete() }}><Trash2 size={15} /> Delete</button>
      </div>

      {!!video.layout_notes?.length && (
        <details className="details notes">
          <summary>Automatic check: {video.layout_notes.length} layout issue{video.layout_notes.length > 1 ? 's' : ''} the AI could not fully fix</summary>
          <ul>{video.layout_notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </details>
      )}

      {showCode && <CodePanel code={code} downloadUrl={api.codeUrl(video.id)} />}

      <section className="prompt-box">
        <div className="label">Prompt</div>
        <p>{video.prompt}</p>
      </section>
    </div>
  )
}
