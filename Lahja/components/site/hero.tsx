"use client"

import { motion } from "motion/react"
import { ArrowDownIcon } from "lucide-react"

const WORDMARK = "Lahja".split("")

export function Hero() {
  return (
    <section
      id="top"
      className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 text-center"
    >
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 right-1/2 h-[32rem] w-[32rem] translate-x-1/3 rounded-full bg-amber-200/40 blur-3xl dark:bg-amber-500/10"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -bottom-40 left-1/2 h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-rose-200/30 blur-3xl dark:bg-rose-500/10"
      />

      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="mb-5 text-sm font-medium tracking-widest text-muted-foreground uppercase"
      >
        A digital language layer for Kokborok (trp)
      </motion.p>

      <h1 className="flex text-7xl font-semibold tracking-tight sm:text-8xl md:text-9xl">
        {WORDMARK.map((letter, i) => (
          <motion.span
            key={i}
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.15 + i * 0.06, ease: [0.16, 1, 0.3, 1] }}
          >
            {letter}
          </motion.span>
        ))}
      </h1>

      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.7 }}
        className="mt-6 max-w-lg text-lg text-muted-foreground"
      >
        Spoken by over a million people in Tripura, with no major speech or
        language model behind it — until now. Speech, text, translation, and
        Q&amp;A, in Kokborok.
      </motion.p>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1, y: [0, 8, 0] }}
        transition={{
          opacity: { duration: 0.6, delay: 1.1 },
          y: { duration: 1.8, repeat: Infinity, ease: "easeInOut", delay: 1.1 },
        }}
        className="absolute bottom-10 flex flex-col items-center gap-2 text-muted-foreground"
      >
        <span className="text-xs tracking-widest uppercase">Scroll</span>
        <ArrowDownIcon className="size-4" />
      </motion.div>
    </section>
  )
}
