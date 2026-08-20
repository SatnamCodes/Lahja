"use client"

import * as React from "react"
import { Loader2Icon, PlayIcon, SquareIcon } from "lucide-react"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Waveform } from "@/components/site/waveform"
import { MethodBadge } from "@/components/site/method-badge"
import { ApiError, speak, type SpeakResponse } from "@/lib/api"
import { SAMPLE_TRP } from "@/lib/kokborok"

export function FeatureTextToSpeech() {
  const [text, setText] = React.useState(SAMPLE_TRP)
  const [status, setStatus] = React.useState<"idle" | "loading" | "error">("idle")
  const [result, setResult] = React.useState<SpeakResponse | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [playing, setPlaying] = React.useState(false)
  const [spokenText, setSpokenText] = React.useState<string | null>(null)
  const [playHint, setPlayHint] = React.useState<string | null>(null)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)

  async function synthesize() {
    if (!text.trim() || status === "loading") return
    setStatus("loading")
    setError(null)
    try {
      const data = await speak(text.trim())
      setResult(data)
      setSpokenText(text.trim())
      setStatus("idle")
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Synthesis failed.")
      setStatus("error")
    }
  }

  // The button doubles as playback transport once audio exists (matching the
  // stop icon it shows while playing): re-synthesizing on every click would
  // silently re-hit the backend instead of just pausing.
  function handleClick() {
    if (result && audioRef.current && text.trim() === spokenText) {
      if (playing) {
        audioRef.current.pause()
      } else {
        audioRef.current.currentTime = 0
        // A click IS a user gesture, so this play() is never autoplay-blocked.
        audioRef.current.play().then(
          () => setPlayHint(null),
          () => setPlayHint("Couldn't play the audio. Check your system output device.")
        )
      }
      return
    }
    synthesize()
  }

  React.useEffect(() => {
    const audio = audioRef.current
    if (!result || !audio) return
    // Always start from the beginning: the backend hashes audio_url from
    // method+text, so re-speaking identical text yields the SAME url and the
    // element would otherwise sit at the end of the previous playback.
    audio.currentTime = 0
    audio.play().then(
      () => setPlayHint(null),
      // Never swallow this. Browsers reject play() when it isn't tied to a
      // user gesture, and silently ignoring it is indistinguishable from
      // "the audio is broken" — which is exactly how it reads to a user.
      () => setPlayHint("Autoplay was blocked by your browser — press ▶ below to listen.")
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result?.audio_url])

  return (
    <Card className="w-full max-w-md ring-foreground/10">
      <CardHeader className="gap-3">
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm font-medium">Kokborok text</p>
          <span className="font-mono text-xs text-muted-foreground">trp</span>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Write something in Kokborok…"
          className="min-h-20 resize-none bg-background"
        />

        <div className="flex items-center gap-3">
          <Button
            onClick={handleClick}
            disabled={!text.trim() || status === "loading"}
            size="lg"
            className="gap-2"
          >
            {status === "loading" ? (
              <Loader2Icon className="size-3.5 animate-spin" />
            ) : playing ? (
              <SquareIcon className="size-3.5" />
            ) : (
              <PlayIcon className="size-3.5" />
            )}
            {status === "loading" ? "Synthesizing…" : playing ? "Pause" : "Speak it"}
          </Button>
          <Waveform active={playing} className="flex-1" />
        </div>

        {status === "loading" && (
          <p className="text-xs text-muted-foreground">
            Cloning a Kokborok voice with XTTS v2 from ~30s of reference audio — first request
            after a cold start can take a minute while the model loads.
          </p>
        )}

        {result && status !== "loading" && (
          <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-muted/40 p-4">
            <MethodBadge method={result.method} confidence={result.confidence} />
            <audio
              ref={audioRef}
              src={result.audio_url}
              controls
              className="w-full"
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
            />
            {playHint && (
              <p className="text-xs text-muted-foreground">{playHint}</p>
            )}
          </div>
        )}

        {status === "error" && (
          <p className="text-xs text-destructive">
            {error}
            {error?.includes("reference") && " Drop a WAV clip into data/audio/ and retry."}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
