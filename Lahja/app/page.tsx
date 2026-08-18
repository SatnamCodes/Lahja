import { Hero } from "@/components/site/hero"
import { FeatureSectionShell } from "@/components/site/feature-section-shell"
import { FeatureTextToSpeech } from "@/components/site/feature-text-to-speech"
import { FeatureSpeechToText } from "@/components/site/feature-speech-to-text"
import { FeatureTranslate } from "@/components/site/feature-translate"
import { FeatureChat } from "@/components/site/feature-chat"
import { HowItWorks } from "@/components/site/how-it-works"
import { Footer } from "@/components/site/footer"

export default function Page() {
  return (
    <main className="bg-background">
      <Hero />

      <div id="features">
        <FeatureSectionShell
          index="01"
          eyebrow="Text → Speech"
          title="Give Kokborok a voice"
          description="No Kokborok TTS model exists, so this clones a real Kokborok speaker's voice from a short reference clip and drives it through the closest phonetic path available — see how, below."
        >
          <FeatureTextToSpeech />
        </FeatureSectionShell>

        <FeatureSectionShell
          index="02"
          eyebrow="Speech → Text"
          title="Turn Kokborok speech into text"
          description="Record or upload a Kokborok clip. No Kokborok ASR model exists either, so this reports the closest honest tier it has — down to raw IPA phonemes if nothing better is available."
          reverse
        >
          <FeatureSpeechToText />
        </FeatureSectionShell>

        <FeatureSectionShell
          index="03"
          eyebrow="Kokborok ↔ English"
          title="Translate both directions"
          description="A version of NLLB-200 fine-tuned specifically on Kokborok — the only entry point most translation tools don't have for this language at all."
        >
          <FeatureTranslate />
        </FeatureSectionShell>

        <FeatureSectionShell
          index="04"
          eyebrow="Kokborok question → Kokborok answer"
          title="Ask Lahja anything"
          description="Your question is translated to English, answered by an LLM, and translated back — with the English bridge answer one click away, so the trip is never hidden."
          reverse
        >
          <FeatureChat />
        </FeatureSectionShell>
      </div>

      <HowItWorks />
      <Footer />
    </main>
  )
}
