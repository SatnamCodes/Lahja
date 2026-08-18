"use client"

import { motion } from "motion/react"
import { SparklesIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { FeatureSectionShell } from "@/components/site/feature-section-shell"
import { FeatureTextToSpeech } from "@/components/site/feature-text-to-speech"
import { FeatureSpeechToText } from "@/components/site/feature-speech-to-text"
import { FeatureCombined } from "@/components/site/feature-combined"

export function AiTier() {
  return (
    <section id="ai-assisted" className="border-t border-border/60 bg-muted/20">
      <div className="mx-auto max-w-3xl px-6 pt-28 pb-4 text-center sm:px-10">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.6 }}
          transition={{ duration: 0.6 }}
        >
          <Badge variant="secondary" className="gap-1">
            <SparklesIcon className="size-3" />
            Preview
          </Badge>
          <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">AI-assisted</h2>
          <p className="mt-4 text-lg text-muted-foreground">
            The same three pipelines — now hands-free. Lahja listens for pauses and silence and
            responds on its own, so the round trip between text and speech happens without
            reaching for a button.
          </p>
        </motion.div>
      </div>

      <FeatureSectionShell
        index="01 · AI-assisted"
        eyebrow="Text → Speech → Text"
        title="Speaks as you write"
        description="Pause for under a second and Lahja reads your words aloud automatically — no click required."
      >
        <FeatureTextToSpeech aiAssisted />
      </FeatureSectionShell>

      <FeatureSectionShell
        index="02 · AI-assisted"
        eyebrow="Speech → Text → Speech"
        title="Answers back automatically"
        description="Stop talking, and Lahja transcribes and speaks your words back the moment you go quiet."
        reverse
      >
        <FeatureSpeechToText aiAssisted />
      </FeatureSectionShell>

      <FeatureSectionShell
        index="03 · AI-assisted"
        eyebrow="Text & Speech, combined"
        title="One panel, no switching"
        description="Type or talk — Lahja figures out which and keeps text and voice in sync without you managing the handoff."
      >
        <FeatureCombined aiAssisted />
      </FeatureSectionShell>
    </section>
  )
}
