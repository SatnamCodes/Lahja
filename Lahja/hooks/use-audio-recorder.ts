"use client"

import * as React from "react"

// Captures mic audio with MediaRecorder and hands back a Blob for
// /api/transcribe. There is no browser speech-recognition fallback here on
// purpose — no browser ships Kokborok ("trp") STT, so pretending otherwise
// would silently return wrong-language text. This always goes to the real
// backend ASR tiers instead.
function subscribeNoop() {
  return () => {}
}

export function useAudioRecorder() {
  const supported = React.useSyncExternalStore(
    subscribeNoop,
    () =>
      typeof navigator !== "undefined" &&
      Boolean(navigator.mediaDevices?.getUserMedia) &&
      typeof MediaRecorder !== "undefined",
    () => false
  )
  const [recording, setRecording] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const recorderRef = React.useRef<MediaRecorder | null>(null)
  const chunksRef = React.useRef<Blob[]>([])
  const streamRef = React.useRef<MediaStream | null>(null)

  const start = React.useCallback(async () => {
    if (!supported) {
      setError("This browser can't record audio. Try Chrome, Edge, or Firefox.")
      return
    }
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []
      const recorder = new MediaRecorder(stream)
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.start()
      recorderRef.current = recorder
      setRecording(true)
    } catch {
      setError("Microphone access was denied or unavailable.")
    }
  }, [supported])

  const stop = React.useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = recorderRef.current
      if (!recorder) {
        resolve(null)
        return
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" })
        streamRef.current?.getTracks().forEach((t) => t.stop())
        streamRef.current = null
        recorderRef.current = null
        setRecording(false)
        resolve(chunksRef.current.length ? blob : null)
      }
      recorder.stop()
    })
  }, [])

  React.useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  return { supported, recording, error, start, stop }
}
