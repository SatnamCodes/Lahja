"use client"

import * as React from "react"
import { motion } from "motion/react"
import { MicIcon, SquareIcon, Volume2Icon } from "lucide-react"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { LanguageSelect } from "@/components/site/language-select"
import { Waveform } from "@/components/site/waveform"
import { useSpeechRecognition } from "@/hooks/use-speech-recognition"
import { useSpeechSynthesis } from "@/hooks/use-speech-synthesis"
import { ENDANGERED_LANGUAGES, getLanguage } from "@/lib/languages"

export function FeatureSpeechToText({ aiAssisted = false }: { aiAssisted?: boolean }) {
  const [langCode, setLangCode] = React.useState(ENDANGERED_LANGUAGES[0].code)
  const language = getLanguage(langCode)

  const { supported, listening, transcript, interimTranscript, confidence, error, start, stop, reset } =
    useSpeechRecognition()
  const { speak, speaking, hasVoiceFor } = useSpeechSynthesis()
  const voiceAvailable = hasVoiceFor(langCode)
  const wasListening = React.useRef(false)

  React.useEffect(() => {
    if (aiAssisted && wasListening.current && !listening && transcript.trim() && voiceAvailable) {
      speak(transcript, langCode)
    }
    wasListening.current = listening
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [listening])

  function toggleListening() {
    if (listening) {
      stop()
    } else {
      reset()
      start(langCode)
    }
  }

  return (
    <Card className="w-full max-w-md ring-foreground/10">
      <CardHeader className="gap-3">
        <div className="flex items-center justify-between gap-3">
          <LanguageSelect
            value={langCode}
            onChange={(code) => {
              setLangCode(code)
              reset()
            }}
            label="Spoken language"
          />
          <Badge variant={supported ? "secondary" : "outline"}>
            {supported ? "Mic ready" : "No mic support"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <Button
            onClick={toggleListening}
            disabled={!supported}
            size="lg"
            variant={listening ? "destructive" : "default"}
            className="gap-2"
          >
            <motion.span
              animate={listening ? { scale: [1, 1.15, 1] } : { scale: 1 }}
              transition={{ duration: 1.1, repeat: listening ? Infinity : 0 }}
              className="inline-flex"
            >
              {listening ? <SquareIcon className="size-3.5" /> : <MicIcon className="size-3.5" />}
            </motion.span>
            {listening ? "Stop" : "Speak now"}
          </Button>
          <Waveform active={listening} className="flex-1" />
        </div>

        <div className="min-h-24 rounded-lg border border-border/70 bg-muted/40 p-4">
          <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
            Transcript
          </p>
          <p className="text-base leading-relaxed">
            {transcript || (
              <span className="text-muted-foreground">
                Speak some {language.name} and it will appear here…
              </span>
            )}
            <span className="text-muted-foreground italic">
              {interimTranscript ? ` ${interimTranscript}` : ""}
            </span>
          </p>
        </div>

        {confidence !== null && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Confidence</span>
            <Progress value={Math.round(confidence * 100)} className="h-1.5 flex-1" />
            <span className="font-mono text-xs text-muted-foreground">
              {Math.round(confidence * 100)}%
            </span>
          </div>
        )}

        <Button
          variant="outline"
          className="gap-2"
          disabled={!transcript.trim() || !voiceAvailable}
          onClick={() => speak(transcript, langCode)}
        >
          <Volume2Icon className="size-3.5" />
          {speaking ? "Speaking…" : "Speak it back"}
        </Button>
        {aiAssisted && (
          <p className="text-xs text-muted-foreground">
            Speaks your words back automatically as soon as you stop talking.
          </p>
        )}

        {error && <p className="text-xs text-destructive">{error}</p>}
        {!supported && (
          <p className="text-xs text-muted-foreground">
            Live transcription needs Chrome, Edge, or another browser with speech recognition
            support.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
