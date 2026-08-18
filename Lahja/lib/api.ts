// Thin client for the FastAPI backend in ../service. Every mutating call
// returns the same shape the backend does: a `method` naming which model
// tier actually answered, and a heuristic `confidence` for it — see
// ../../README.md for what each method means. Nothing here ever invents a
// result: a failed or unavailable backend surfaces as a thrown ApiError.

export type SpeakResponse = { audio_url: string; confidence: number; method: string }
export type TranslateResponse = { translated_text: string; confidence: number; method: string }
export type ChatResponse = {
  answer: string
  english_bridge: string
  confidence: number
  method: string
}
export type TranscribeResponse = { text: string; confidence: number; method: string }
export type HealthResponse = { status: string; device: string }

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

const UNREACHABLE =
  "Can't reach the Lahja backend. Start it with ./scripts/run.sh (port 8000) alongside this app."

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new ApiError(UNREACHABLE, 0)
  }
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new ApiError(data.detail ?? `Request failed (${res.status})`, res.status)
  }
  return data as T
}

export function speak(text: string, language = "trp") {
  return request<SpeakResponse>("/api/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language }),
  })
}

export function translate(text: string, sourceLanguage: string, targetLanguage: string) {
  return request<TranslateResponse>("/api/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      source_language: sourceLanguage,
      target_language: targetLanguage,
    }),
  })
}

export function chat(text: string) {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
}

export function transcribe(audio: Blob, filename = "recording.webm") {
  const form = new FormData()
  form.append("audio", audio, filename)
  return request<TranscribeResponse>("/api/transcribe", { method: "POST", body: form })
}

export async function checkHealth(): Promise<HealthResponse | null> {
  try {
    const res = await fetch("/health", { cache: "no-store" })
    if (!res.ok) return null
    return (await res.json()) as HealthResponse
  } catch {
    return null
  }
}
