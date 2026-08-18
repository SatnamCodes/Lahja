"use client"

import * as React from "react"

function getRecognitionCtor(): (new () => SpeechRecognition) | null {
  if (typeof window === "undefined") return null
  return window.SpeechRecognition ?? window.webkitSpeechRecognition ?? null
}

function subscribeNoop() {
  return () => {}
}

export function useSpeechRecognition() {
  const supported = React.useSyncExternalStore(
    subscribeNoop,
    () => getRecognitionCtor() !== null,
    () => false
  )
  const [listening, setListening] = React.useState(false)
  const [transcript, setTranscript] = React.useState("")
  const [interimTranscript, setInterimTranscript] = React.useState("")
  const [confidence, setConfidence] = React.useState<number | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const recognitionRef = React.useRef<SpeechRecognition | null>(null)

  const stop = React.useCallback(() => {
    recognitionRef.current?.stop()
  }, [])

  const start = React.useCallback((lang: string) => {
    const Ctor = getRecognitionCtor()
    if (!Ctor) {
      setError("Speech recognition isn't supported in this browser. Try Chrome or Edge.")
      return
    }

    setError(null)
    setTranscript("")
    setInterimTranscript("")
    setConfidence(null)

    const recognition = new Ctor()
    recognition.lang = lang
    recognition.continuous = true
    recognition.interimResults = true

    recognition.onstart = () => setListening(true)
    recognition.onend = () => setListening(false)
    recognition.onerror = (event) => {
      setError(
        event.error === "no-speech"
          ? "Didn't catch that — try speaking again."
          : event.error === "language-not-supported"
            ? "This language isn't supported for recognition in your browser yet."
            : `Recognition error: ${event.error}`
      )
      setListening(false)
    }
    recognition.onresult = (event) => {
      let finalText = ""
      let interimText = ""
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          finalText += result[0].transcript
          setConfidence(result[0].confidence)
        } else {
          interimText += result[0].transcript
        }
      }
      if (finalText) setTranscript((prev) => (prev + " " + finalText).trim())
      setInterimTranscript(interimText)
    }

    recognitionRef.current = recognition
    recognition.start()
  }, [])

  React.useEffect(() => () => recognitionRef.current?.stop(), [])

  const reset = React.useCallback(() => {
    setTranscript("")
    setInterimTranscript("")
    setConfidence(null)
    setError(null)
  }, [])

  return { supported, listening, transcript, interimTranscript, confidence, error, start, stop, reset }
}
