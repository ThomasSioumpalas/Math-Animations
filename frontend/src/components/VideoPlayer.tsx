import { useEffect, useRef, useState } from 'react'
import { Maximize, Minimize, Pause, Play, Volume2, VolumeX } from 'lucide-react'
import { duration } from '../util'

/** Video with its controls BELOW the picture (native controls overlay the bottom of the
 *  frame, which is exactly where the animation's captions are). */
export default function VideoPlayer({ src }: { src: string }) {
  const wrap = useRef<HTMLDivElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(true)
  const [time, setTime] = useState(0)
  const [length, setLength] = useState(0)
  const [full, setFull] = useState(false)

  useEffect(() => {
    const onFs = () => setFull(document.fullscreenElement === wrap.current)
    document.addEventListener('fullscreenchange', onFs)
    return () => document.removeEventListener('fullscreenchange', onFs)
  }, [])

  const toggle = () => {
    const v = video.current
    if (!v) return
    if (v.paused) void v.play()
    else v.pause()
  }

  const seek = (t: number) => {
    if (video.current) video.current.currentTime = t
    setTime(t)
  }

  const toggleFull = () => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void wrap.current?.requestFullscreen()
  }

  return (
    <div
      ref={wrap}
      className={`vplayer ${full ? 'full' : ''}`}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === ' ' || e.key === 'k') { e.preventDefault(); toggle() }
        if (e.key === 'ArrowRight') seek(Math.min(length, time + 5))
        if (e.key === 'ArrowLeft') seek(Math.max(0, time - 5))
        if (e.key === 'f') toggleFull()
        if (e.key === 'm') setMuted((m) => !m)
      }}
    >
      <div className="vscreen" onClick={toggle}>
        <video
          ref={video}
          src={src}
          autoPlay
          muted={muted}
          playsInline
          loop
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => setLength(e.currentTarget.duration)}
        />
        {!playing && <div className="vbig"><Play size={34} /></div>}
      </div>
      <div className="vbar">
        <button className="icon-btn" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <span className="vtime">{duration(time)} / {duration(length)}</span>
        <input
          className="vseek"
          type="range"
          min={0}
          max={length || 0}
          step={0.05}
          value={time}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Seek"
          style={{ ['--p' as string]: `${length ? (time / length) * 100 : 0}%` }}
        />
        <button className="icon-btn" onClick={() => setMuted((m) => !m)} aria-label={muted ? 'Unmute' : 'Mute'}>
          {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
        </button>
        <button className="icon-btn" onClick={toggleFull} aria-label="Fullscreen">
          {full ? <Minimize size={18} /> : <Maximize size={18} />}
        </button>
      </div>
    </div>
  )
}
