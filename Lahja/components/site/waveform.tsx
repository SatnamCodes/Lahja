"use client"

import { motion } from "motion/react"

import { cn } from "@/lib/utils"

const BAR_COUNT = 24

export function Waveform({ active, className }: { active: boolean; className?: string }) {
  return (
    <div className={cn("flex h-10 items-center justify-center gap-[3px]", className)}>
      {Array.from({ length: BAR_COUNT }).map((_, i) => (
        <motion.span
          key={i}
          className="w-[3px] rounded-full bg-foreground/70"
          initial={{ height: 4 }}
          animate={
            active
              ? { height: [4, 8 + ((i * 37) % 28), 4, 6 + ((i * 19) % 20), 4] }
              : { height: 4 }
          }
          transition={
            active
              ? {
                  duration: 0.9 + (i % 5) * 0.08,
                  repeat: Infinity,
                  ease: "easeInOut",
                  delay: i * 0.02,
                }
              : { duration: 0.3 }
          }
        />
      ))}
    </div>
  )
}
