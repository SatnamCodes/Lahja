"use client"

import { useTheme } from "next-themes"
import { motion } from "motion/react"
import { MoonIcon, SunIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useIsClient } from "@/hooks/use-is-client"

export function Navbar() {
  const { resolvedTheme, setTheme } = useTheme()
  const mounted = useIsClient()

  return (
    <motion.header
      initial={{ y: -24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="fixed inset-x-0 top-0 z-40 flex items-center justify-between border-b border-border/60 bg-background/70 px-6 py-4 backdrop-blur-md sm:px-10"
    >
      <a href="#top" className="text-lg font-semibold tracking-tight">
        Lahja
      </a>
      <nav className="hidden items-center gap-6 text-sm text-muted-foreground sm:flex">
        <a href="#features" className="transition-colors hover:text-foreground">
          Features
        </a>
        <a href="#ai-assisted" className="transition-colors hover:text-foreground">
          AI-assisted
        </a>
      </nav>
      <Button
        variant="ghost"
        size="icon"
        aria-label="Toggle theme"
        onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
      >
        {mounted && resolvedTheme === "dark" ? (
          <SunIcon className="size-4" />
        ) : (
          <MoonIcon className="size-4" />
        )}
      </Button>
    </motion.header>
  )
}
