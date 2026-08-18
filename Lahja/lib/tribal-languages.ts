// A representative (not exhaustive) selection of Scheduled/tribal languages
// of India, positioned by the state most associated with them. India has
// hundreds of tribal languages across all its states — this highlights a
// well-documented sample per region rather than claiming completeness.
// State geometry: lib/india-states.json (see attribution there). Keys must
// match the `name` field in that file exactly.
export type TribalLanguageEntry = {
  state: string
  languages: string[]
  note: string
}

export const TRIBAL_LANGUAGES: TribalLanguageEntry[] = [
  {
    state: "Tripura",
    languages: ["Kokborok"],
    note: "The language this project is built for — no native TTS, ASR, or LLM support before now.",
  },
  {
    state: "Nagaland",
    languages: ["Ao", "Angami", "Konyak", "Sümi"],
    note: "Among India's most linguistically diverse states — dozens of distinct Naga languages.",
  },
  {
    state: "Mizoram",
    languages: ["Mizo"],
    note: "Also called Duhlian; the state's lingua franca.",
  },
  {
    state: "Meghalaya",
    languages: ["Khasi", "Garo"],
    note: "Two of the few Austroasiatic languages with an official role in an Indian state.",
  },
  {
    state: "Assam",
    languages: ["Bodo", "Karbi"],
    note: "Bodo is one of India's 22 scheduled languages.",
  },
  {
    state: "Arunachal Pradesh",
    languages: ["Nyishi", "Adi", "Apatani"],
    note: "Home to 60+ distinct languages across a small population.",
  },
  {
    state: "Manipur",
    languages: ["Tangkhul", "Thadou"],
    note: "Tibeto-Burman languages spoken across Manipur's hill districts.",
  },
  {
    state: "Sikkim",
    languages: ["Lepcha", "Bhutia"],
    note: "Lepcha is considered indigenous to Sikkim, predating Tibetan and Nepali migration.",
  },
  {
    state: "Jharkhand",
    languages: ["Santali", "Mundari", "Ho", "Kurukh"],
    note: "Santali has its own script (Ol Chiki) and is a scheduled language.",
  },
  {
    state: "Odisha",
    languages: ["Kui", "Santali", "Saora"],
    note: "Odisha alone is home to over 20 recognized tribal languages.",
  },
  {
    state: "Chhattisgarh",
    languages: ["Gondi", "Halbi"],
    note: "Gondi is the largest Dravidian language spoken outside South India.",
  },
  {
    state: "Madhya Pradesh",
    languages: ["Gondi", "Korku", "Bhili"],
    note: "Central India's tribal belt — home to some of the largest Scheduled Tribe populations.",
  },
  {
    state: "Rajasthan",
    languages: ["Bhili"],
    note: "Bhili (and its dialects) is spoken by one of India's largest tribal communities.",
  },
  {
    state: "Tamil Nadu",
    languages: ["Toda", "Irula"],
    note: "Toda, spoken in the Nilgiris, has only a few thousand speakers left.",
  },
  {
    state: "Andaman and Nicobar Islands",
    languages: ["Great Andamanese", "Onge", "Jarawa"],
    note: "Among the most endangered language families on Earth — some down to single-digit speakers.",
  },
]

export const TRIBAL_LANGUAGE_BY_STATE = new Map(
  TRIBAL_LANGUAGES.map((entry) => [entry.state, entry])
)
