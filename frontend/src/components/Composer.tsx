import { useEffect, useRef, useState } from 'react'
import { ArrowUp, LoaderCircle, Sparkles } from 'lucide-react'
import type { Folder, PublicConfig } from '../api'

const EXAMPLES = [
  { label: 'Matrix as a transformation', prompt: 'Show how the 2x2 matrix [[1, 1], [0, 1]] (a shear) transforms the plane: draw the grid and the basis vectors i-hat and j-hat, then smoothly apply the matrix and show where i-hat and j-hat land. Finish by writing the determinant and explaining that area is preserved.' },
  { label: 'Gradient descent', prompt: 'Explain gradient descent on f(x) = (x - 2)^2 + 1. Plot the curve, place a ball at x = -2, then take 8 steps with learning rate 0.3, drawing the tangent line at each step and a short caption with the current x and f(x). End with the ball at the minimum.' },
  { label: 'Pythagorean theorem', prompt: 'Prove the Pythagorean theorem visually: draw a right triangle with sides 3, 4, 5, build squares on each side colored blue, green and yellow, show their areas 9, 16 and 25, then animate the two small squares\' areas combining into the large one.' },
  { label: 'Fourier series', prompt: 'Animate a Fourier series approximating a square wave: add the first 1, 3, 5 and 10 odd harmonics one at a time on the same axes, showing the partial sum getting closer to the square wave. Label how many terms are used.' },
  { label: 'Eigenvectors', prompt: 'Explain eigenvectors with A = [[2, 1], [1, 2]]: show several vectors on a grid, apply A, and highlight that only the vectors along y = x and y = -x stay on their own line, stretched by 3 and 1. Caption each step.' },
  { label: 'Neural network forward pass', prompt: 'Visualize a forward pass through a tiny neural network with 3 inputs, a hidden layer of 4 neurons and 2 outputs. Draw neurons as circles and weights as lines, then animate the signal flowing layer by layer with neurons lighting up by their activation.' },
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
        <div className="badge"><Sparkles size={14} /> Text to math animation</div>
        <h1>What should we animate?</h1>
        <p>Describe a concept, a proof or an algorithm. The AI writes the Manim code, renders it, and fixes its own errors.</p>
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
          placeholder="e.g. Show how a 2×2 matrix stretches and rotates the plane, step by step, with the basis vectors highlighted…"
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
      <div className="hint">Tip: be specific. Name the steps, the numbers and the colors. <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to send.{p.config && <> Model: <code>{p.config.model}</code></>}</div>

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
