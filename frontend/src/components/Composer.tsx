import { useEffect, useRef, useState } from 'react'
import { ArrowUp, LoaderCircle, Sparkles } from 'lucide-react'
import type { Folder, PublicConfig } from '../api'
import ExampleLibrary from './ExampleLibrary'

type Props = {
  config: PublicConfig | null
  draft: string
  setDraft: (v: string) => void
  folders: Folder[]
  folderId: string
  setFolderId: (v: string) => void
  quality: string
  setQuality: (v: string) => void
  submitting: boolean
  onSubmit: () => void
}

export default function Composer(p: Props) {
  const ref = useRef<HTMLTextAreaElement>(null)
  const [focused, setFocused] = useState(false)
  const max = p.config?.max_prompt_chars ?? 8000
  const tooLong = p.draft.length > max
  const canSend = p.draft.trim().length >= 3 && !tooLong && !p.submitting

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, window.innerHeight * 0.5)}px`
  }, [p.draft])

  useEffect(() => { ref.current?.focus() }, [])

  return (
    <div className="composer-wrap">
      <div className="hero">
        <div className="badge"><Sparkles size={14} /> Text to animation</div>
        <h1>What should we animate?</h1>
        <p>Ask a question or describe what you want to see. An AI teacher plans the explanation step by step, and every picture and number is drawn and computed exactly. Usually ready in 1–2 minutes.</p>
      </div>

      <form
        className={`composer ${focused ? 'focused' : ''}`}
        onSubmit={(e) => { e.preventDefault(); if (canSend) p.onSubmit() }}
      >
        <textarea
          ref={ref}
          value={p.draft}
          onChange={(e) => p.setDraft(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && canSend) { e.preventDefault(); p.onSubmit() }
          }}
          placeholder="Describe the animation you want: what appears, how it moves, colors, labels, order of steps…"
          rows={5}
        />
        <div className="composer-bar">
          <label className="field">
            <span>Save to</span>
            <select value={p.folderId} onChange={(e) => p.setFolderId(e.target.value)}>
              {p.folders.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
            </select>
          </label>
          <div className="segmented" role="radiogroup" aria-label="Quality">
            <button type="button" className={p.quality === 'low' ? 'on' : ''} onClick={() => p.setQuality('low')}>Fast · 480p</button>
            <button type="button" disabled={p.config?.allow_hd === false} className={p.quality === 'medium' ? 'on' : ''} onClick={() => p.setQuality('medium')}>HD · 720p</button>
          </div>
          <span className={`chars ${tooLong ? 'danger' : ''}`}>{p.draft.length > 200 || tooLong ? `${p.draft.length}/${max}` : ''}</span>
          <button className="send" type="submit" disabled={!canSend} title="Generate (Ctrl+Enter)">
            {p.submitting ? <LoaderCircle size={18} className="spin" /> : <ArrowUp size={18} />}
            <span>Generate</span>
          </button>
        </div>
      </form>
      <div className="hint">Tip: use your own numbers or data, and say what you want to understand. <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to send.{p.config && <> Teacher: <code>{p.config.model}</code> + free fallbacks</>}</div>

      <ExampleLibrary
        onPick={(prompt) => {
          p.setDraft(prompt)
          ref.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
          ref.current?.focus({ preventScroll: true })
        }}
      />
    </div>
  )
}
