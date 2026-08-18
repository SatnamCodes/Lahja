"use client"

import { motion } from "motion/react"
import { MapIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { IndiaMap } from "@/components/site/india-map"

export function TribalLanguageMapSection() {
  return (
    <section className="border-t border-border/60">
      <div className="mx-auto max-w-3xl px-6 pt-28 pb-10 text-center sm:px-10">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.6 }}
          transition={{ duration: 0.6 }}
        >
          <Badge variant="secondary" className="gap-1">
            <MapIcon className="size-3" />
            Kokborok, in context
          </Badge>
          <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            One of hundreds of tongues
          </h2>
          <p className="mt-4 text-lg text-muted-foreground">
            Kokborok is one of hundreds of tribal and indigenous languages spoken across India —
            most with little to no digital or AI support. Hover a highlighted state below for a
            sample of what&apos;s spoken there.
          </p>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.6 }}
        className="px-6 pb-28 sm:px-10"
      >
        <IndiaMap />
        <p className="mx-auto mt-4 max-w-md text-center text-xs text-muted-foreground">
          A representative sample, not an exhaustive list — India recognizes over 700 tribal
          communities across its states and union territories.
        </p>
      </motion.div>
    </section>
  )
}
