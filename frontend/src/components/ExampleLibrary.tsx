import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Clock } from 'lucide-react'
import { COLLECTIONS, type Level } from '../examples'

const TAB_KEY = 'mstudio.exampleTab'

const LEVEL_CLASS: Record<Level, string> = {
  Beginner: 'lv-beg',
  Intermediate: 'lv-int',
  Advanced: 'lv-adv',
}

export default function ExampleLibrary({ onPick }: { onPick: (prompt: string) => void }) {
  const [tab, setTab] = useState<string>(() => {
    try { return localStorage.getItem(TAB_KEY) ?? COLLECTIONS[0].id } catch { return COLLECTIONS[0].id }
  })
  const tabsRef = useRef<HTMLDivElement>(null)
  useEffect(() => { try { localStorage.setItem(TAB_KEY, tab) } catch { /* ignore */ } }, [tab])

  const collection = COLLECTIONS.find((c) => c.id === tab) ?? COLLECTIONS[0]
  const index = COLLECTIONS.indexOf(collection)
  const go = (step: number) => setTab(COLLECTIONS[(index + step + COLLECTIONS.length) % COLLECTIONS.length].id)

  return (
    <section className="examples">
      <div className="examples-head">
        <div className="examples-title">Try an example</div>
        <div className="tab-strip">
          <button className="icon-btn sm" onClick={() => go(-1)} aria-label="Previous collection"><ChevronLeft size={16} /></button>
          <div className="tabs" ref={tabsRef} role="tablist">
            {COLLECTIONS.map((c) => (
              <button key={c.id} role="tab" aria-selected={c.id === collection.id}
                className={`tab ${c.id === collection.id ? 'on' : ''}`} onClick={() => setTab(c.id)}>
                {c.name}
              </button>
            ))}
          </div>
          <button className="icon-btn sm" onClick={() => go(1)} aria-label="Next collection"><ChevronRight size={16} /></button>
        </div>
      </div>

      {collection.groups.map((g) => (
        <div key={g.name} className="example-group">
          <div className="group-head">
            <h3>{g.name}</h3>
            <span>{g.hint}</span>
          </div>
          <div className="example-grid">
            {g.items.map((ex) => (
              <button key={ex.title} className="example" onClick={() => onPick(ex.prompt)} title={ex.prompt}>
                <div className="example-meta">
                  <span className={`level ${LEVEL_CLASS[ex.level]}`}>{ex.level}</span>
                  <span className="mins"><Clock size={12} /> {ex.minutes}</span>
                </div>
                <strong>{ex.title}</strong>
                <span className="summary">{ex.summary}</span>
              </button>
            ))}
          </div>
        </div>
      ))}
      <p className="examples-foot">Click a card to load its prompt, then edit it as you like before generating.</p>
    </section>
  )
}
