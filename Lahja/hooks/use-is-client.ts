"use client"

import * as React from "react"

function subscribe() {
  return () => {}
}

export function useIsClient() {
  return React.useSyncExternalStore(
    subscribe,
    () => true,
    () => false
  )
}
