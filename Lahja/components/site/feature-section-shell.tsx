"use client"

import { motion } from "motion/react"

import { cn } from "@/lib/utils"

export function FeatureSectionShell({
  index,
  eyebrow,
  title,
  description,
  children,
  reverse = false,
  id,
}: {
  index: string
  eyebrow: string
  title: string
  description: string
  children: React.ReactNode
  reverse?: boolean
  id?: string
}) {
  return (
    <section id={id} className="mx-auto grid max-w-6xl gap-10 px-6 py-28 sm:px-10 lg:grid-cols-2 lg:gap-16">
      <motion.div
        initial={{ opacity: 0, x: reverse ? 32 : -32 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true, amount: 0.4 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className={cn("flex flex-col justify-center", reverse && "lg:order-2")}
      >
        <span className="font-mono text-sm text-muted-foreground">{index}</span>
        <p className="mt-3 text-sm font-medium tracking-wide text-muted-foreground uppercase">
          {eyebrow}
        </p>
        <h3 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h3>
        <p className="mt-4 max-w-md text-base text-muted-foreground">{description}</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, x: reverse ? -32 : 32, scale: 0.97 }}
        whileInView={{ opacity: 1, x: 0, scale: 1 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.7, ease: "easeOut", delay: 0.1 }}
        className={cn("flex items-center", reverse && "lg:order-1")}
      >
        {children}
      </motion.div>
    </section>
  )
}
