import { Hero } from "@/components/site/hero"
import { FeatureSectionShell } from "@/components/site/feature-section-shell"
import { FeatureTextToSpeech } from "@/components/site/feature-text-to-speech"
import { FeatureSpeechToText } from "@/components/site/feature-speech-to-text"
import { FeatureCombined } from "@/components/site/feature-combined"
import { AiTier } from "@/components/site/ai-tier"
import { Footer } from "@/components/site/footer"

export default function Page() {
  return (
    <main className="bg-background">
      <Hero />

      <div id="features">
        <FeatureSectionShell
          index="01"
          eyebrow="Text → Speech → Text"
          title="Give written words a voice"
          description="Write in an endangered language and hear it spoken aloud — each word highlighted as it's read, closing the loop back into text."
        >
          <FeatureTextToSpeech />
        </FeatureSectionShell>

        <FeatureSectionShell
          index="02"
          eyebrow="Speech → Text → Speech"
          title="Turn spoken words into text"
          description="Speak into the mic and watch a live transcript appear, then play it back to check how it sounds."
          reverse
        >
          <FeatureSpeechToText />
        </FeatureSectionShell>

        <FeatureSectionShell
          index="03"
          eyebrow="Text & Speech, combined"
          title="One panel, both directions"
          description="Type or talk in the same place — Feature 1 and Feature 2, unified into a single translator."
        >
          <FeatureCombined />
        </FeatureSectionShell>
      </div>

      <AiTier />
      <Footer />
    </main>
  )
}
