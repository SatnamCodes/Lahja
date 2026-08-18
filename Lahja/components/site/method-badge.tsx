"use client"

import { Badge } from "@/components/ui/badge"
import { confidenceTier, methodLabel } from "@/lib/kokborok"

const TIER_VARIANT = {
  high: "secondary",
  mid: "outline",
  low: "destructive",
} as const

// Lahja never has a single confident model for Kokborok, so every AI result
// names the tier that actually produced it and how sure the backend is —
// see the tiered fallback design in the root README. This badge is the one
// place that honesty surfaces in the UI.
export function MethodBadge({ method, confidence }: { method: string; confidence: number }) {
  const tier = confidenceTier(confidence)
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge variant={TIER_VARIANT[tier]}>{methodLabel(method)}</Badge>
      <Badge variant="outline" className="font-mono">
        confidence {confidence.toFixed(2)}
      </Badge>
    </div>
  )
}
