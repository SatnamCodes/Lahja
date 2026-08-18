"use client"

import * as React from "react"
import { Loader2Icon, MicIcon, SquareIcon, UploadIcon } from "lucide-react"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Waveform } from "@/components/site/waveform"
import { MethodBadge } from "@/components/site/method-badge"
import { useAudioRecorder } from "@/hooks/use-audio-recorder"
import { ApiError, transcribe, type TranscribeResponse } from "@/lib/api"

export function FeatureSpeechToText() {
  const { supported, recording, error: recError, start, stop } = useAudioRecorder()
  const [status, setStatus] = React.useState<"idle" | "loading" | "error">("idle")
  const [result, setResult] = React.useState<TranscribeResponse | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const fileInputRef = React.useRef<HTMLInputElement | null>(null)

  async function runTranscribe(blob: Blob, filename?: string) {
    setStatus("loading")
    setError(null)
    setResult(null)
    try {
      const data = await transcribe(blob, filename)
      setResult(data)
      setStatus("idle")
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Transcription failed.")
      setStatus("error")
    }
  }

  async function toggleRecording() {
    if (recording) {
      const blob = await stop()
      if (blob) await runTranscribe(blob, "recording.webm")
    } else {
      setResult(null)
      setError(null)
      await start()
    }
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (file) runTranscribe(file, file.name)
  }

  return (
    <Card className="w-full max-w-md ring-foreground/10">
      <CardHeader className="gap-3">
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm font-medium">Kokborok speech</p>
          <span className="font-mono text-xs text-muted-foreground">trp</span>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <Button
            onClick={toggleRecording}
            disabled={!supported || status === "loading"}
            size="lg"
            variant={recording ? "destructive" : "default"}
            className="gap-2"
          >
            {status === "loading" ? (
              <Loader2Icon className="size-3.5 animate-spin" />
            ) : recording ? (
              <SquareIcon className="size-3.5" />
            ) : (
              <MicIcon className="size-3.5" />
            )}
            {status === "loading" ? "Transcribing…" : recording ? "Stop" : "Record"}
          </Button>
          <Waveform active={recording} className="flex-1" />
          <Button
            variant="outline"
            size="icon"
            aria-label="Upload audio file"
            disabled={status === "loading"}
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadIcon className="size-3.5" />
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={handleFile}
          />
        </div>

        <div className="min-h-24 rounded-lg border border-border/70 bg-muted/40 p-4">
          <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
            Transcript
          </p>
          <p className="font-mono text-base leading-relaxed break-words">
            {result?.text || (
              <span className="font-sans text-muted-foreground">
                Record or upload a Kokborok clip and it will appear here…
              </span>
            )}
          </p>
        </div>

        {result && (
          <>
            <MethodBadge method={result.method} confidence={result.confidence} />
            {result.method === "phoneme_zero_shot_bridge" && (
              <p className="text-xs text-muted-foreground">
                No Kokborok ASR model exists yet, so this is the raw IPA phoneme stream — sound,
                not spelling. A fine-tuned model (see Lahja&apos;s ASR pipeline) will replace this.
              </p>
            )}
            {result.method === "whisper_zero_shot_bridge" && (
              <p className="text-xs text-destructive">
                No phoneme model was available, so this is generic Whisper guessing a language it
                doesn&apos;t recognize — treat the words as unreliable.
              </p>
            )}
          </>
        )}

        {(recError || (status === "error" && error)) && (
          <p className="text-xs text-destructive">{recError ?? error}</p>
        )}
        {!supported && (
          <p className="text-xs text-muted-foreground">
            This browser can&apos;t record audio — use the upload button instead.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
