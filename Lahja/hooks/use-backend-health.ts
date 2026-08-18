"use client"

import * as React from "react"

import { checkHealth } from "@/lib/api"

export type BackendStatus = "checking" | "online" | "offline"

// Polled instead of fetched once: the backend can go from cold (still
// loading XTTS/NLLB the first time a model is touched) to warm mid-demo, and
// the nav dot should reflect that without a page reload.
export function useBackendHealth(intervalMs = 15000) {
  const [status, setStatus] = React.useState<BackendStatus>("checking")
  const [device, setDevice] = React.useState<string | null>(null)

  React.useEffect(() => {
    let cancelled = false
    async function poll() {
      const health = await checkHealth()
      if (cancelled) return
      setStatus(health ? "online" : "offline")
      setDevice(health?.device ?? null)
    }
    poll()
    const id = setInterval(poll, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [intervalMs])

  return { status, device }
}
