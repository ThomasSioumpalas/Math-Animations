import { Fragment, useEffect, useRef, useState, type ReactNode } from 'react'
import { ArrowUp, LoaderCircle, MessageCircle, Sparkles, WandSparkles } from 'lucide-react'
import { api, type ChatMessage, type Thinking } from '../api'
import { keepChat } from '../keeper'

const THINKING: { id: Thinking; label: string; hint: string }[] = [
  { id: 'standard', label: 'Standard', hint: 'Medium reasoning: fastest answers' },
  { id: 'high', label: 'High', hint: 'High reasoning: careful math (default)' },
  { id: 'max', label: 'Max', hint: 'Highest reasoning plus a self-check that verifies every number a second way (slowest)' },
]
const THINKING_KEY = 'mstudio.chatThinking'

const SUGGESTIONS = [
  'Explain the hardest step again, more slowly',
  'Why does this work? Give me the intuition',
  'Give me a small exercise to check I understood',
  'What should I learn next after this?',
]

/** Minimal, safe formatting for answers: paragraphs, "- " bullet lists, **bold** and `code`. */
function renderText(text: string): ReactNode {
  const inline = (line: string) =>
    line.split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g).map((part, i) =>
      part.startsWith('**') && part.endsWith('**') ? <strong key={i}>{part.slice(2, -2)}</strong>
        : part.length > 2 && part.startsWith('*') && part.endsWith('*') ? <em key={i}>{part.slice(1, -1)}</em>
          : part.startsWith('`') && part.endsWith('`') ? <code key={i}>{part.slice(1, -1)}</code>
            : <Fragment key={i}>{part}</Fragment>)
  const blocks: ReactNode[] = []
  let list: string[] = []
  let code: string[] | null = null // inside a ``` fenced block (worked calculations)
  const flush = () => {
    if (list.length) blocks.push(<ul key={blocks.length}>{list.map((l, i) => <li key={i}>{inline(l)}</li>)}</ul>)
    list = []
  }
  for (const raw of text.split('\n')) {
    const line = raw.trimEnd()
    if (line.trim().startsWith('```')) {
      if (code === null) { flush(); code = [] } else { blocks.push(<pre key={blocks.length}>{code.join('\n')}</pre>); code = null }
      continue
    }
    if (code !== null) { code.push(raw); continue }
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/)
    if (heading) { flush(); blocks.push(<p key={blocks.length} className="heading">{inline(heading[1])}</p>); continue }
    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { flush(); blocks.push(<hr key={blocks.length} />); continue }
    const bullet = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/)
    if (bullet) { list.push(bullet[1]); continue }
    flush()
    if (line.trim()) blocks.push(<p key={blocks.length}>{inline(line)}</p>)
  }
  flush()
  if (code !== null && code.length) blocks.push(<pre key={blocks.length}>{code.join('\n')}</pre>)
  return blocks
}

type Props = { videoId: string; videoTitle: string; onAnimate: (prompt: string) => void }

export default function VideoChat({ videoId, videoTitle, onAnimate }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [thinking, setThinking] = useState<Thinking>(() => {
    try { return (localStorage.getItem(THINKING_KEY) as Thinking) || 'high' } catch { return 'high' }
  })
  useEffect(() => { try { localStorage.setItem(THINKING_KEY, thinking) } catch { /* ignore */ } }, [thinking])
  const listRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    api.videoChat(videoId).then((r) => { setMessages(r.messages); keepChat(videoId, r.messages) }).catch(() => setMessages([]))
  }, [videoId])

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, sending])

  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [draft])

  const send = async (text: string) => {
    const q = text.trim()
    if (!q || sending) return
    setError('')
    setSending(true)
    setDraft('')
    setMessages((m) => [...m, { role: 'user', content: q, at: Date.now() / 1000 }])
    try {
      const r = await api.askVideo(videoId, q, thinking)
      setMessages(r.messages)
      keepChat(videoId, r.messages)
    } catch (e) {
      setError((e as Error).message)
      setMessages((m) => m.slice(0, -1))
      setDraft(q)
    } finally {
      setSending(false)
    }
  }

  // The user question that led to a given assistant answer (for "Animate this").
  const questionBefore = (index: number) => {
    for (let i = index - 1; i >= 0; i--) if (messages[i].role === 'user') return messages[i].content
    return ''
  }

  return (
    <section className="vchat">
      <div className="vchat-head">
        <MessageCircle size={16} />
        <span>Ask about this video</span>
        <span className="muted small">Questions about any step, or anything related.</span>
        <div className="segmented thinking" role="radiogroup" aria-label="Thinking level">
          <span className="seg-label">Thinking</span>
          {THINKING.map((t) => (
            <button key={t.id} type="button" title={t.hint} className={thinking === t.id ? 'on' : ''}
              onClick={() => setThinking(t.id)}>{t.label}</button>
          ))}
        </div>
      </div>

      {(messages.length > 0 || sending) && (
        <div className="vchat-list" ref={listRef}>
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="bubble">{m.role === 'assistant' ? renderText(m.content) : m.content}</div>
              {m.role === 'assistant' && questionBefore(i) && (
                <button className="ghost sm animate-btn" title="Make a new video that explains this"
                  onClick={() => onAnimate(`Following up on the video "${videoTitle}": ${questionBefore(i)}`)}>
                  <WandSparkles size={13} /> Animate this
                </button>
              )}
            </div>
          ))}
          {sending && (
            <div className="msg assistant">
              <div className="bubble typing"><LoaderCircle size={15} className="spin" />
                {thinking === 'max' ? 'Thinking hard and double-checking…' : thinking === 'high' ? 'Thinking carefully…' : 'Thinking…'}</div>
            </div>
          )}
        </div>
      )}

      {messages.length === 0 && !sending && (
        <div className="vchat-suggest">
          {SUGGESTIONS.map((s) => (
            <button key={s} className="chip" onClick={() => void send(s)}><Sparkles size={12} /> {s}</button>
          ))}
        </div>
      )}

      {error && <div className="vchat-error">{error}</div>}

      <form className="vchat-input" onSubmit={(e) => { e.preventDefault(); void send(draft) }}>
        <textarea
          ref={inputRef}
          value={draft}
          rows={1}
          maxLength={3000}
          placeholder="Ask a question about this video…"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(draft) }
          }}
        />
        <button className="send" type="submit" disabled={!draft.trim() || sending} title="Send (Enter)">
          {sending ? <LoaderCircle size={17} className="spin" /> : <ArrowUp size={17} />}
        </button>
      </form>
      <div className="vchat-foot">Enter to send, Shift+Enter for a new line. The conversation is kept with the video.</div>
    </section>
  )
}
