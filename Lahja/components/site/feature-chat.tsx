"use client"

import * as React from "react"
import { ChevronDownIcon, Loader2Icon, SendIcon } from "lucide-react"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { MethodBadge } from "@/components/site/method-badge"
import { ApiError, chat, type ChatResponse } from "@/lib/api"
import { SAMPLE_TRP } from "@/lib/kokborok"
import { cn } from "@/lib/utils"

export function FeatureChat() {
  const [text, setText] = React.useState(SAMPLE_TRP)
  const [status, setStatus] = React.useState<"idle" | "loading" | "error">("idle")
  const [result, setResult] = React.useState<ChatResponse | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [showBridge, setShowBridge] = React.useState(false)

  async function handleAsk() {
    if (!text.trim() || status === "loading") return
    setStatus("loading")
    setError(null)
    setResult(null)
    try {
      const data = await chat(text.trim())
      setResult(data)
      setStatus("idle")
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The chatbot didn't respond.")
      setStatus("error")
    }
  }

  return (
    <Card className="w-full max-w-md ring-foreground/10">
      <CardHeader className="gap-3">
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm font-medium">Ask in Kokborok</p>
          <span className="font-mono text-xs text-muted-foreground">trp → trp</span>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask a question in Kokborok…"
          className="min-h-20 resize-none bg-background"
        />

        <Button
          onClick={handleAsk}
          disabled={!text.trim() || status === "loading"}
          size="lg"
          className="w-fit gap-2"
        >
          {status === "loading" ? (
            <Loader2Icon className="size-3.5 animate-spin" />
          ) : (
            <SendIcon className="size-3.5" />
          )}
          {status === "loading" ? "Bridging trp → eng → LLM → trp…" : "Ask"}
        </Button>

        {result && status !== "loading" && (
          <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-muted/40 p-4">
            <p className="text-base leading-relaxed">{result.answer}</p>
            <MethodBadge method={result.method} confidence={result.confidence} />
            <button
              type="button"
              onClick={() => setShowBridge((v) => !v)}
              className="flex items-center gap-1 text-left text-xs text-muted-foreground hover:text-foreground"
            >
              <ChevronDownIcon
                className={cn("size-3 transition-transform", showBridge && "rotate-180")}
              />
              {showBridge ? "Hide" : "Show"} the English bridge answer
            </button>
            {showBridge && (
              <p className="rounded-md bg-background p-3 text-sm text-muted-foreground">
                {result.english_bridge}
              </p>
            )}
          </div>
        )}

        {status === "error" && (
          <p className="text-xs text-destructive">
            {error}
            {error?.includes("GROQ_API_KEY") &&
              " Get a free key at console.groq.com/keys and export GROQ_API_KEY on the server."}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
