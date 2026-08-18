"use client"

import { motion } from "motion/react"
import { SparklesIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { confidenceTier } from "@/lib/kokborok"

type Tier = { method: string; confidence: number; note: string; available: boolean }
type Pipeline = { title: string; blurb: string; tiers: Tier[] }

// Mirrors the real fallback order and confidence values in
// service/{tts,mt,chat,asr}_engine.py — kept here as static copy (not
// fetched) because it describes engineering intent, not live state; the
// panels above show what actually ran for a given request.
const PIPELINES: Pipeline[] = [
  {
    title: "Speak — text → speech",
    blurb: "No native Kokborok TTS model exists.",
    tiers: [
      { method: "XTTS v2 zero-shot voice clone", confidence: 0.45, note: "primary", available: true },
      { method: "MMS-TTS Bengali bridge, zero-shot", confidence: 0.35, note: "fallback", available: true },
      { method: "MMS-TTS Bengali bridge, fine-tuned", confidence: 0.65, note: "stretch goal", available: false },
    ],
  },
  {
    title: "Listen — speech → text",
    blurb: "No native or bridge-language Kokborok ASR model exists.",
    tiers: [
      { method: "Whisper fine-tuned on Kokborok", confidence: 0.6, note: "in progress", available: false },
      { method: "IPA phoneme recognizer, zero-shot", confidence: 0.3, note: "current default", available: true },
      { method: "Whisper zero-shot, wrong-language guess", confidence: 0.15, note: "last resort", available: true },
    ],
  },
  {
    title: "Translate — Kokborok ↔ English",
    blurb: "Standard NLLB-200 has no Kokborok entry either.",
    tiers: [
      { method: "NLLB-200 fine-tuned on ~36k Kokborok pairs", confidence: 0.7, note: "only tier", available: true },
    ],
  },
  {
    title: "Ask — Kokborok question → Kokborok answer",
    blurb: "No LLM has native Kokborok fluency.",
    tiers: [
      { method: "MT-bridged LLM (trp→eng→LLM→trp)", confidence: 0.5, note: "only tier", available: true },
    ],
  },
]

export function HowItWorks() {
  return (
    <section id="how-it-works" className="border-t border-border/60 bg-muted/20">
      <div className="mx-auto max-w-3xl px-6 pt-28 pb-16 text-center sm:px-10">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.6 }}
          transition={{ duration: 0.6 }}
        >
          <Badge variant="secondary" className="gap-1">
            <SparklesIcon className="size-3" />
            Under the hood
          </Badge>
          <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            No pretending, tiered instead
          </h2>
          <p className="mt-4 text-lg text-muted-foreground">
            Kokborok has no first-party model for any of these four tasks. Lahja tries the closest
            real approximation first and falls back honestly — every response above names exactly
            which tier answered and how confident it is, never a single opaque &quot;AI says&quot;.
          </p>
        </motion.div>
      </div>

      <div className="mx-auto grid max-w-5xl gap-6 px-6 pb-28 sm:px-10 md:grid-cols-2">
        {PIPELINES.map((pipeline, i) => (
          <motion.div
            key={pipeline.title}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6, delay: i * 0.08 }}
          >
            <Card className="h-full p-6">
              <h3 className="text-lg font-semibold">{pipeline.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{pipeline.blurb}</p>
              <ul className="mt-5 flex flex-col gap-3">
                {pipeline.tiers.map((tier) => (
                  <li key={tier.method} className="flex flex-col gap-1.5">
                    <div className="flex items-center justify-between gap-3 text-sm">
                      <span className={tier.available ? "" : "text-muted-foreground italic"}>
                        {tier.method}
                        {!tier.available && " (not yet trained)"}
                      </span>
                      <span className="shrink-0 font-mono text-xs text-muted-foreground">
                        {tier.note}
                      </span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-border/70">
                      <div
                        className={
                          "h-full rounded-full " +
                          (confidenceTier(tier.confidence) === "high"
                            ? "bg-foreground/70"
                            : confidenceTier(tier.confidence) === "mid"
                              ? "bg-foreground/45"
                              : "bg-foreground/25")
                        }
                        style={{ width: `${Math.round(tier.confidence * 100)}%` }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            </Card>
          </motion.div>
        ))}
      </div>
    </section>
  )
}
