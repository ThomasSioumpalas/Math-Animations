import { useEffect, useState } from 'react'
import { Check, LoaderCircle, Pencil, RotateCcw, TriangleAlert } from 'lucide-react'
import { ACTIVE, type Job } from '../api'
import CodePanel from './CodePanel'
import { duration } from '../util'

const STEPS = ['Queued', 'Writing code', 'Rendering', 'Ready']

function stepIndex(job: Job): number {
  switch (job.status) {
    case 'queued': return 0
    case 'writing': return 1
    case 'rendering':
    case 'fixing': return 2
    case 'done': return 3
    case 'failed': return job.code ? 2 : 1
  }
}

export default function JobView({ job, onRetry, onEdit }: { job: Job; onRetry: () => void; onEdit: () => void }) {
  const [now, setNow] = useState(Date.now() / 1000)
  const active = ACTIVE.includes(job.status)
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(t)
  }, [active])

  const idx = stepIndex(job)
  const failed = job.status === 'failed'
  const elapsed = (job.finished_at ?? now) - job.created_at

  return (
    <div className="view">
      <header className="view-head">
        <div>
          <div className="eyebrow">{failed ? 'Failed' : active ? 'In progress' : 'Finished'} · {duration(elapsed)}</div>
          <h2>{job.title || 'New animation'}</h2>
        </div>
      </header>

      <div className="stage-card">
        <ol className="steps">
          {STEPS.map((s, i) => {
            const state = i < idx ? 'done' : i === idx ? (failed ? 'error' : active ? 'current' : 'done') : 'todo'
            const label = i === 2 && job.attempt > 1 ? `${job.status === 'fixing' ? 'Fixing' : 'Rendering'} (try ${job.attempt}/${job.max_attempts})` : s
            return (
              <li key={s} className={`step ${state}`}>
                <span className="dot">
                  {state === 'done' ? <Check size={13} /> : state === 'current' ? <LoaderCircle size={13} className="spin" /> : state === 'error' ? <TriangleAlert size={13} /> : i + 1}
                </span>
                <span>{label}</span>
              </li>
            )
          })}
        </ol>
        <div className={`stage-msg ${failed ? 'danger' : ''}`}>
          {active && <div className="progress"><div /></div>}
          <p>{job.message}{job.status === 'queued' && job.position > 1 ? ` (position ${job.position})` : ''}</p>
          {active && <p className="muted small">This usually takes 30 to 90 seconds. You can start another animation or browse your library while you wait.</p>}
        </div>
        {failed && (
          <div className="row gap">
            <button className="primary" onClick={onRetry}><RotateCcw size={15} /> Try again</button>
            <button className="ghost" onClick={onEdit}><Pencil size={15} /> Edit prompt</button>
          </div>
        )}
      </div>

      <section className="prompt-box">
        <div className="label">Prompt</div>
        <p>{job.prompt}</p>
      </section>

      {failed && job.error && (
        <details className="details">
          <summary>Error details</summary>
          <pre>{job.error}</pre>
        </details>
      )}
      {job.code && (
        <details className="details" open={failed ? false : undefined}>
          <summary>Generated code{job.attempt > 1 ? ` (attempt ${job.attempt})` : ''}</summary>
          <CodePanel code={job.code} />
        </details>
      )}
    </div>
  )
}
