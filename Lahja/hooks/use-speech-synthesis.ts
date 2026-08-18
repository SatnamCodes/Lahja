"use client"

import * as React from "react"

const emptyVoices: SpeechSynthesisVoice[] = []
let cachedVoices: SpeechSynthesisVoice[] = emptyVoices
let cachedKey = ""

function getVoicesSnapshot(): SpeechSynthesisVoice[] {
  if (typeof window === "undefined" || !window.speechSynthesis) return emptyVoices
  const voices = window.speechSynthesis.getVoices()
  const key = voices.map((v) => v.voiceURI).join("|")
  if (key !== cachedKey) {
    cachedVoices = voices
    cachedKey = key
  }
  return cachedVoices
}

function getServerVoicesSnapshot(): SpeechSynthesisVoice[] {
  return emptyVoices
}

function subscribeVoices(callback: () => void) {
  if (typeof window === "undefined" || !window.speechSynthesis) return () => {}
  window.speechSynthesis.addEventListener("voiceschanged", callback)
  return () => window.speechSynthesis.removeEventListener("voiceschanged", callback)
}

function subscribeNoop() {
  return () => {}
}

export function useSpeechSynthesis() {
  const supported = React.useSyncExternalStore(
    subscribeNoop,
    () => typeof window !== "undefined" && Boolean(window.speechSynthesis),
    () => false
  )
  const voices = React.useSyncExternalStore(subscribeVoices, getVoicesSnapshot, getServerVoicesSnapshot)
  const [speaking, setSpeaking] = React.useState(false)
  const [wordIndex, setWordIndex] = React.useState(-1)

  React.useEffect(() => {
    if (!supported) return
    return () => {
      window.speechSynthesis.cancel()
    }
  }, [supported])

  const voiceForLang = React.useCallback(
    (lang: string) => {
      const prefix = lang.split("-")[0]
      return (
        voices.find((v) => v.lang.toLowerCase() === lang.toLowerCase()) ??
        voices.find((v) => v.lang.toLowerCase().startsWith(prefix.toLowerCase()))
      )
    },
    [voices]
  )

  const speak = React.useCallback(
    (text: string, lang: string) => {
      if (!supported || !text.trim()) return
      window.speechSynthesis.cancel()

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = lang
      const voice = voiceForLang(lang)
      if (voice) utterance.voice = voice

      utterance.onstart = () => setSpeaking(true)
      utterance.onend = () => {
        setSpeaking(false)
        setWordIndex(-1)
      }
      utterance.onerror = () => {
        setSpeaking(false)
        setWordIndex(-1)
      }
      utterance.onboundary = (event) => {
        if (event.name !== "word") return
        const upToChar = text.slice(0, event.charIndex)
        const index = upToChar.trim().length ? upToChar.trim().split(/\s+/).length : 0
        setWordIndex(index)
      }

      window.speechSynthesis.speak(utterance)
    },
    [supported, voiceForLang]
  )

  const cancel = React.useCallback(() => {
    if (!supported) return
    window.speechSynthesis.cancel()
    setSpeaking(false)
    setWordIndex(-1)
  }, [supported])

  const hasVoiceFor = React.useCallback((lang: string) => Boolean(voiceForLang(lang)), [voiceForLang])

  return { supported, speaking, speak, cancel, wordIndex, hasVoiceFor }
}
