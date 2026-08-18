"use client"

import * as React from "react"
import { motion } from "motion/react"
import { PlayIcon, SquareIcon } from "lucide-react"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { LanguageSelect } from "@/components/site/language-select"
import { Waveform } from "@/components/site/waveform"
import { useSpeechSynthesis } from "@/hooks/use-speech-synthesis"
import { ENDANGERED_LANGUAGES, getLanguage } from "@/lib/languages"

export function FeatureTextToSpeech({ aiAssisted = false }: { aiAssisted?: boolean }) {
  const [langCode, setLangCode] = React.useState(ENDANGERED_LANGUAGES[0].code)
  const language = getLanguage(langCode)
  const [text, setText] = React.useState(language.sample)
  const autoTextRef = React.useRef(language.sample)
  const [hasTyped, setHasTyped] = React.useState(false)

  const { supported, speaking, speak, cancel, wordIndex, hasVoiceFor } = useSpeechSynthesis()
  const voiceAvailable = supported && hasVoiceFor(langCode)

  React.useEffect(() => {
    if (!aiAssisted || !hasTyped || !voiceAvailable || !text.trim()) return
    const timer = setTimeout(() => speak(text, langCode), 900)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, aiAssisted, hasTyped, voiceAvailable, langCode])

  function handleLangChange(code: string) {
    const next = getLanguage(code)
    setLangCode(code)
    if (text.trim() === "" || text === autoTextRef.current) {
      setText(next.sample)
      autoTextRef.current = next.sample
    }
  }

  const words = text.trim().length ? text.trim().split(/\s+/) : []

  return (
    <Card className="w-full max-w-md ring-foreground/10">
      <CardHeader className="gap-3">
        <div className="flex items-center justify-between gap-3">
          <LanguageSelect value={langCode} onChange={handleLangChange} label="Source language" />
          <Badge variant={voiceAvailable ? "secondary" : "outline"}>
            {voiceAvailable ? "Voice ready" : "No local voice"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
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
          <p className="-mt-2 text-xs text-muted-foreground">
            Speaks automatically a moment after you stop typing.
          </p>
        )}

        <div className="flex items-center gap-3">
          <Button
            onClick={() => (speaking ? cancel() : speak(text, langCode))}
            disabled={!voiceAvailable || !text.trim()}
            size="lg"
            className="gap-2"
          >
            {speaking ? <SquareIcon className="size-3.5" /> : <PlayIcon className="size-3.5" />}
            {speaking ? "Stop" : "Speak it"}
          </Button>
          <Waveform active={speaking} className="flex-1" />
        </div>

        <div className="rounded-lg border border-border/70 bg-muted/40 p-4">
          <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
            Reconstructed as it&apos;s spoken
          </p>
          <p className="text-base leading-relaxed">
            {words.map((word, i) => (
              <motion.span
                key={`${i}-${word}`}
                animate={{
                  opacity: i === wordIndex ? 1 : speaking ? 0.35 : 0.85,
                  color: i === wordIndex ? "var(--color-foreground)" : undefined,
                }}
                className="mr-1.5 inline-block"
              >
                {word}
              </motion.span>
            ))}
          </p>
        </div>

        {!supported && (
          <p className="text-xs text-muted-foreground">
            Speech synthesis isn&apos;t available in this browser.
          </p>
        )}
        {supported && !voiceAvailable && (
          <p className="text-xs text-muted-foreground">
            No {language.name} voice installed on this device — try Chrome/Edge, or add the
            language in your OS speech settings.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
