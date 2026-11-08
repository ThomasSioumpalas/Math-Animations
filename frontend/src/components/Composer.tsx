import { useEffect, useRef, useState } from 'react'
import { ArrowUp, LoaderCircle, Sparkles } from 'lucide-react'
import type { Folder, PublicConfig } from '../api'

const EXAMPLES = [
  { label: 'Bouncing ball', prompt: 'A red ball drops from the top of the screen onto a floor line and bounces three times, each bounce lower than the last, then rolls to the right and stops.' },
  { label: 'Sorting algorithm', prompt: 'Visualize bubble sort on the bars [5, 2, 8, 1, 9, 3]. Highlight the two bars being compared in yellow, swap them with a short animation when needed, and color each bar green once it is in its final position.' },
  { label: 'Function graph', prompt: 'Draw axes from -4 to 4, plot y = sin(x) in blue, then morph it into y = 2·sin(x) and finally into y = sin(2x). Label each curve with its formula.' },
  { label: 'Solar system', prompt: 'A yellow sun in the center with three planets of different sizes and colors orbiting it on circular paths at different speeds. Show the orbit paths as faint dashed circles.' },
  { label: 'Pie chart', prompt: 'Build a pie chart of the shares 40% Rent, 25% Food, 20% Transport, 15% Other. Each slice grows in one after the other with its own color and label.' },
  { label: 'Shape morph', prompt: 'A blue square rotates once, morphs into a green circle, then into an orange triangle, and finally shrinks to a point and fades out.' },
]

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
        <p>Describe what you want to see, in as much detail as you like. The AI translates it into Manim code, renders the video, and fixes its own errors.</p>
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
      <div className="hint">Tip: be specific. Name the steps, the numbers and the colors. <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to send.{p.config && <> Model: <code>{p.config.model}</code> + free fallbacks</>}</div>

      <div className="examples">
        <div className="examples-title">Try an example</div>
        <div className="example-grid">
          {EXAMPLES.map((ex) => (
            <button key={ex.label} className="example" onClick={() => { p.setDraft(ex.prompt); ref.current?.focus() }}>
              <strong>{ex.label}</strong>
              <span>{ex.prompt}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
