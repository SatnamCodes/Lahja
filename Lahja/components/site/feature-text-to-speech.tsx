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
  const audioRef = React.useRef<HTMLAudioElement | null>(null)

  async function handleSpeak() {
    if (!text.trim() || status === "loading") return
    setStatus("loading")
    setError(null)
    try {
      const data = await speak(text.trim())
      setResult(data)
      setStatus("idle")
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Synthesis failed.")
      setStatus("error")
    }
  }

  React.useEffect(() => {
    if (result && audioRef.current) {
      audioRef.current.play().catch(() => {})
    }
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
            onClick={handleSpeak}
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
            {status === "loading" ? "Synthesizing…" : "Speak it"}
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
