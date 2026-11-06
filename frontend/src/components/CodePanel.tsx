import { useState } from 'react'
import { Check, Copy, Download } from 'lucide-react'

export default function CodePanel({ code, downloadUrl, label = 'Manim code' }: { code: string; downloadUrl?: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  return (
    <div className="code-panel">
      <div className="code-head">
        <span>{label}</span>
        <div className="code-actions">
          <button className="ghost sm" onClick={copy}>{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? 'Copied' : 'Copy'}</button>
          {downloadUrl && <a className="ghost sm" href={downloadUrl}><Download size={14} />.py</a>}
        </div>
      </div>
      <pre><code>{code || 'Loading…'}</code></pre>
    </div>
  )
}
