export type EndangeredLanguage = {
  code: string
  name: string
  nativeName: string
  region: string
  status: "Vulnerable" | "Endangered" | "Severely Endangered" | "Critically Endangered"
  sample: string
  sampleMeaning: string
}

export const ENDANGERED_LANGUAGES: EndangeredLanguage[] = [
  { code: "cy-GB", name: "Welsh", nativeName: "Cymraeg", region: "Wales", status: "Vulnerable", sample: "Bore da", sampleMeaning: "“Good morning”" },
  { code: "ga-IE", name: "Irish", nativeName: "Gaeilge", region: "Ireland", status: "Endangered", sample: "Dia duit", sampleMeaning: "“Hello”" },
  { code: "gd-GB", name: "Scottish Gaelic", nativeName: "Gàidhlig", region: "Scotland", status: "Severely Endangered", sample: "Halò", sampleMeaning: "“Hello”" },
  { code: "eu-ES", name: "Basque", nativeName: "Euskara", region: "Basque Country", status: "Vulnerable", sample: "Kaixo", sampleMeaning: "“Hello”" },
  { code: "mi-NZ", name: "Māori", nativeName: "Te Reo Māori", region: "Aotearoa New Zealand", status: "Vulnerable", sample: "Kia ora", sampleMeaning: "“Hello / be well”" },
  { code: "yi", name: "Yiddish", nativeName: "ייִדיש", region: "Eastern Europe / diaspora", status: "Endangered", sample: "שלום עליכם", sampleMeaning: "“Peace be upon you”" },
  { code: "br-FR", name: "Breton", nativeName: "Brezhoneg", region: "Brittany, France", status: "Severely Endangered", sample: "Demat", sampleMeaning: "“Hello”" },
  { code: "oc-FR", name: "Occitan", nativeName: "Occitan", region: "Southern France", status: "Severely Endangered", sample: "Adiu", sampleMeaning: "“Hello”" },
  { code: "haw-US", name: "Hawaiian", nativeName: "ʻŌlelo Hawaiʻi", region: "Hawaiʻi", status: "Critically Endangered", sample: "Aloha", sampleMeaning: "“Hello / love”" },
  { code: "chr-US", name: "Cherokee", nativeName: "ᏣᎳᎩ", region: "Southeastern USA", status: "Critically Endangered", sample: "ᎣᏏᏲ", sampleMeaning: "“Hello” (Osiyo)" },
  { code: "nv-US", name: "Navajo", nativeName: "Diné bizaad", region: "Southwestern USA", status: "Vulnerable", sample: "Yá'át'ééh", sampleMeaning: "“Hello”" },
  { code: "se-NO", name: "Northern Sámi", nativeName: "Davvisámegiella", region: "Sápmi", status: "Vulnerable", sample: "Bures", sampleMeaning: "“Hello”" },
]

export function getLanguage(code: string) {
  return ENDANGERED_LANGUAGES.find((l) => l.code === code) ?? ENDANGERED_LANGUAGES[0]
}

export const STATUS_TONE: Record<EndangeredLanguage["status"], string> = {
  Vulnerable: "bg-amber-100 text-amber-900 dark:bg-amber-500/15 dark:text-amber-300",
  Endangered: "bg-orange-100 text-orange-900 dark:bg-orange-500/15 dark:text-orange-300",
  "Severely Endangered": "bg-rose-100 text-rose-900 dark:bg-rose-500/15 dark:text-rose-300",
  "Critically Endangered": "bg-red-100 text-red-900 dark:bg-red-500/15 dark:text-red-300",
}
