"use client"

// State boundary geometry in lib/india-states.json is derived from
// udit-001/india-maps-data's state-level topojson (github.com/udit-001/
// india-maps-data), projected to this file's viewBox with d3-geo. Reflects
// current (post-2019) internal boundaries: Telangana and Ladakh are
// separate from Andhra Pradesh and Jammu & Kashmir respectively.
import * as React from "react"
import { motion, AnimatePresence } from "motion/react"

import { Badge } from "@/components/ui/badge"
import indiaStatesData from "@/lib/india-states.json"
import { TRIBAL_LANGUAGE_BY_STATE } from "@/lib/tribal-languages"
import { cn } from "@/lib/utils"

type StateShape = { name: string; code: string; d: string; cx: number; cy: number }
const { width, height, states } = indiaStatesData as {
  width: number
  height: number
  states: StateShape[]
}

const TOOLTIP_WIDTH = 224 // matches the w-56 tooltip below
const TOOLTIP_EST_HEIGHT = 130 // only used to decide whether to flip above the pointer
const EDGE_MARGIN = 8

export function IndiaMap() {
  const [active, setActive] = React.useState<string | null>(null)
  // Pixel position within the map container (not the SVG's viewBox units),
  // so the tooltip can be clamped to the container's actual rendered size —
  // northeast states sit near the map's right edge, and a percentage-based
  // position pushed the tooltip off-screen on narrow viewports.
  const [pointer, setPointer] = React.useState<{
    x: number
    y: number
    containerWidth: number
    containerHeight: number
  } | null>(null)
  const svgRef = React.useRef<SVGSVGElement | null>(null)

  const activeEntry = active ? TRIBAL_LANGUAGE_BY_STATE.get(active) : undefined

  function handleMove(e: React.MouseEvent<SVGSVGElement>) {
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    setPointer({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      containerWidth: rect.width,
      containerHeight: rect.height,
    })
  }

  return (
    <div className="relative mx-auto w-full max-w-md">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full touch-manipulation"
        role="img"
        aria-label="Map of India with tribal-language hotspots"
        onMouseMove={handleMove}
        onMouseLeave={() => setActive(null)}
      >
        {states.map((s) => {
          const hasLanguage = TRIBAL_LANGUAGE_BY_STATE.has(s.name)
          const isActive = active === s.name
          return (
            <path
              key={s.code}
              d={s.d}
              className={cn(
                "cursor-default stroke-background transition-colors duration-150",
                isActive
                  ? "fill-foreground"
                  : hasLanguage
                    ? "fill-foreground/35"
                    : "fill-foreground/10"
              )}
              strokeWidth={0.75}
              onMouseEnter={() => setActive(s.name)}
              tabIndex={-1}
              aria-hidden="true"
            />
          )
        })}

        {states
          .filter((s) => TRIBAL_LANGUAGE_BY_STATE.has(s.name))
          .map((s) => (
            <g key={`dot-${s.code}`}>
              <circle
                cx={s.cx}
                cy={s.cy}
                r={active === s.name ? 5 : 3.5}
                className={cn(
                  "pointer-events-none transition-all duration-150",
                  active === s.name ? "fill-primary" : "fill-primary/70"
                )}
              />
              {active !== s.name && (
                <circle
                  cx={s.cx}
                  cy={s.cy}
                  r={3.5}
                  className="pointer-events-none fill-primary/40"
                >
                  <animate
                    attributeName="r"
                    values="3.5;8;3.5"
                    dur="2.4s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="opacity"
                    values="0.5;0;0.5"
                    dur="2.4s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
              {/* Larger invisible hit-target — some states (e.g. scattered
                  islands) are too small or fragmented to hover reliably. */}
              <circle
                cx={s.cx}
                cy={s.cy}
                r={10}
                className="cursor-default fill-transparent"
                onMouseEnter={() => setActive(s.name)}
                onFocus={() => setActive(s.name)}
                tabIndex={0}
                aria-label={`${s.name}: ${TRIBAL_LANGUAGE_BY_STATE.get(s.name)!.languages.join(", ")}`}
              />
            </g>
          ))}
      </svg>

      <AnimatePresence>
        {activeEntry && pointer && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            // Positioned with resolved left/top pixels rather than
            // left:50% + a CSS translate — motion/react owns the `transform`
            // property for its own y-animation and silently drops any
            // transform passed through `style`, which left the tooltip
            // anchored at its raw (un-centered) left edge and overflowing
            // narrow viewports for states near the map's right edge.
            className="pointer-events-none absolute z-10 w-56 rounded-lg border border-border/70 bg-popover p-3 text-left shadow-lg"
            style={{
              left: Math.min(
                Math.max(pointer.x, TOOLTIP_WIDTH / 2 + EDGE_MARGIN),
                pointer.containerWidth - TOOLTIP_WIDTH / 2 - EDGE_MARGIN
              ) - TOOLTIP_WIDTH / 2,
              top:
                pointer.y / pointer.containerHeight > 0.55
                  ? pointer.y - 14 - TOOLTIP_EST_HEIGHT
                  : pointer.y + 14,
            }}
          >
            <p className="text-sm font-semibold">{activeEntry.state}</p>
            <div className="mt-1.5 flex flex-wrap gap-1">
              {activeEntry.languages.map((lang) => (
                <Badge key={lang} variant="secondary">
                  {lang}
                </Badge>
              ))}
            </div>
            <p className="mt-2 text-xs text-muted-foreground">{activeEntry.note}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
