export const KOKBOROK = {
  iso: "trp",
  name: "Kokborok",
  region: "Tripura, India & parts of Bangladesh",
}

// The only Kokborok sentence with a verified English gloss anywhere in this
// repo (see ../../README.md and ../../scripts/test_*.sh) — used as the
// default across every panel instead of inventing new phrases no one here
// can actually verify.
export const SAMPLE_TRP = "Nwng bubagra tamwi?"
export const SAMPLE_ENG = "How are you?"

export type ConfidenceTier = "high" | "mid" | "low"

export function confidenceTier(confidence: number): ConfidenceTier {
  if (confidence >= 0.6) return "high"
  if (confidence >= 0.3) return "mid"
  return "low"
}

const METHOD_LABELS: Record<string, string> = {
  xtts_zero_shot: "XTTS v2 · zero-shot voice clone",
  mms_bridge_zero_shot: "MMS-TTS (Bengali bridge) · zero-shot",
  mms_fine_tuned: "MMS-TTS (Bengali bridge) · fine-tuned",
  kokborok_mt_nllb: "NLLB-200 · fine-tuned on Kokborok",
  mt_bridge_llm: "LLM answer, MT-bridged both ways",
  whisper_fine_tuned: "Whisper · fine-tuned on Kokborok",
  phoneme_zero_shot_bridge: "IPA phoneme recognizer · zero-shot",
  whisper_zero_shot_bridge: "Whisper · zero-shot (wrong-language guess)",
}

export function methodLabel(method: string): string {
  return METHOD_LABELS[method] ?? method
}
