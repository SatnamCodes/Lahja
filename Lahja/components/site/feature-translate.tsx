"use client"

import * as React from "react"
import { ArrowRightIcon, Loader2Icon } from "lucide-react"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MethodBadge } from "@/components/site/method-badge"
import { ApiError, translate, type TranslateResponse } from "@/lib/api"
import { SAMPLE_ENG, SAMPLE_TRP } from "@/lib/kokborok"

type Direction = "trp-eng" | "eng-trp"

const DIRECTIONS: Record<Direction, { source: string; target: string; sample: string }> = {
  "trp-eng": { source: "trp", target: "eng", sample: SAMPLE_TRP },
  "eng-trp": { source: "eng", target: "trp", sample: SAMPLE_ENG },
}

export function FeatureTranslate() {
  const [direction, setDirection] = React.useState<Direction>("trp-eng")
  const [text, setText] = React.useState(DIRECTIONS["trp-eng"].sample)
  const [status, setStatus] = React.useState<"idle" | "loading" | "error">("idle")
  const [result, setResult] = React.useState<TranslateResponse | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  function handleDirectionChange(next: string) {
    const dir = next as Direction
    setDirection(dir)
    setText(DIRECTIONS[dir].sample)
    setResult(null)
    setStatus("idle")
  }

  async function handleTranslate() {
    if (!text.trim() || status === "loading") return
    setStatus("loading")
    setError(null)
    try {
      const { source, target } = DIRECTIONS[direction]
      const data = await translate(text.trim(), source, target)
      setResult(data)
      setStatus("idle")
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Translation failed.")
      setStatus("error")
    }
  }

  return (
    <Card className="w-full max-w-md ring-foreground/10">
      <CardHeader className="gap-3">
        <Tabs value={direction} onValueChange={handleDirectionChange}>
          <TabsList className="w-full">
            <TabsTrigger value="trp-eng" className="flex-1">
              Kokborok → English
            </TabsTrigger>
            <TabsTrigger value="eng-trp" className="flex-1">
              English → Kokborok
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type something to translate…"
          className="min-h-20 resize-none bg-background"
        />

        <Button
          onClick={handleTranslate}
          disabled={!text.trim() || status === "loading"}
          size="lg"
          className="w-fit gap-2"
        >
          {status === "loading" ? (
            <Loader2Icon className="size-3.5 animate-spin" />
          ) : (
            <ArrowRightIcon className="size-3.5" />
          )}
          {status === "loading" ? "Translating…" : "Translate"}
        </Button>

        {result && status !== "loading" && (
          <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-muted/40 p-4">
            <p className="text-base leading-relaxed">{result.translated_text}</p>
            <MethodBadge method={result.method} confidence={result.confidence} />
          </div>
        )}

        {status === "error" && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  )
}
