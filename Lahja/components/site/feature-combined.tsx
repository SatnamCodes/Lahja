"use client"

import * as React from "react"
import { MicIcon, PlayIcon, SquareIcon } from "lucide-react"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { LanguageSelect } from "@/components/site/language-select"
import { Waveform } from "@/components/site/waveform"
import { useSpeechRecognition } from "@/hooks/use-speech-recognition"
import { useSpeechSynthesis } from "@/hooks/use-speech-synthesis"
import { ENDANGERED_LANGUAGES, getLanguage } from "@/lib/languages"

export function FeatureCombined({ aiAssisted = false }: { aiAssisted?: boolean }) {
  const [langCode, setLangCode] = React.useState(ENDANGERED_LANGUAGES[0].code)
  const language = getLanguage(langCode)
  const [text, setText] = React.useState(language.sample)
  const [mode, setMode] = React.useState<"type" | "speak">("type")
  const [hasTyped, setHasTyped] = React.useState(false)

  const { supported: sttSupported, listening, transcript, start, stop, reset } =
    useSpeechRecognition()
  const { supported: ttsSupported, speaking, speak, cancel, hasVoiceFor } = useSpeechSynthesis()
  const voiceAvailable = ttsSupported && hasVoiceFor(langCode)
  const wasListening = React.useRef(false)
  const displayText = mode === "speak" && transcript ? transcript : text

  React.useEffect(() => {
    if (aiAssisted && wasListening.current && !listening && transcript.trim() && voiceAvailable) {
      speak(transcript, langCode)
    }
    wasListening.current = listening
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [listening])

  React.useEffect(() => {
    if (!aiAssisted || mode !== "type" || !hasTyped || !voiceAvailable || !text.trim()) return
    const timer = setTimeout(() => speak(text, langCode), 900)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, aiAssisted, mode, hasTyped, voiceAvailable, langCode])

  function handleLangChange(code: string) {
    setLangCode(code)
    setText(getLanguage(code).sample)
    reset()
  }

  function handleModeChange(next: string) {
    if (mode === "speak" && transcript.trim()) {
      setText(transcript)
      reset()
    }
    setMode(next as "type" | "speak")
  }

  return (
    <Card className="w-full max-w-md ring-foreground/10">
      <CardHeader className="gap-3">
        <div className="flex items-center justify-between gap-3">
          <LanguageSelect value={langCode} onChange={handleLangChange} label="Language" />
          <div className="flex gap-1.5">
            <Badge variant={voiceAvailable ? "secondary" : "outline"}>
              {voiceAvailable ? "Speech ✓" : "Speech —"}
            </Badge>
            <Badge variant={sttSupported ? "secondary" : "outline"}>
              {sttSupported ? "Mic ✓" : "Mic —"}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Tabs value={mode} onValueChange={handleModeChange}>
          <TabsList className="w-full">
            <TabsTrigger value="type" className="flex-1">
              Type
            </TabsTrigger>
            <TabsTrigger value="speak" className="flex-1">
              Speak
            </TabsTrigger>
          </TabsList>

          <TabsContent value="type" className="mt-3">
            <Textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value)
                setHasTyped(true)
              }}
              placeholder={`Write something in ${language.name}…`}
              className="min-h-20 resize-none bg-background"
            />
            {aiAssisted && (
              <p className="mt-2 text-xs text-muted-foreground">
                Speaks automatically a moment after you stop typing.
              </p>
            )}
          </TabsContent>

          <TabsContent value="speak" className="mt-3">
            <div className="flex items-center gap-3">
              <Button
                size="lg"
                variant={listening ? "destructive" : "default"}
                disabled={!sttSupported}
                onClick={() => (listening ? stop() : start(langCode))}
                className="gap-2"
              >
                {listening ? <SquareIcon className="size-3.5" /> : <MicIcon className="size-3.5" />}
                {listening ? "Stop" : "Speak now"}
              </Button>
              <Waveform active={listening} className="flex-1" />
            </div>
            {aiAssisted && (
              <p className="mt-2 text-xs text-muted-foreground">
                Speaks your words back automatically as soon as you stop talking.
              </p>
            )}
          </TabsContent>
        </Tabs>

        <div className="rounded-lg border border-border/70 bg-muted/40 p-4">
          <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
            Text & speech, together
          </p>
          <p className="min-h-6 text-base leading-relaxed">{displayText || "…"}</p>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            className="gap-2"
            disabled={!voiceAvailable || !displayText.trim()}
            onClick={() => (speaking ? cancel() : speak(displayText, langCode))}
          >
            {speaking ? <SquareIcon className="size-3.5" /> : <PlayIcon className="size-3.5" />}
            {speaking ? "Stop" : "Speak it"}
          </Button>
          <Waveform active={speaking} className="flex-1" />
        </div>
      </CardContent>
    </Card>
  )
}
