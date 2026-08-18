"use client"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ENDANGERED_LANGUAGES } from "@/lib/languages"

export function LanguageSelect({
  value,
  onChange,
  label,
}: {
  value: string
  onChange: (code: string) => void
  label?: string
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger aria-label={label ?? "Language"} className="w-[200px] bg-background">
        <SelectValue placeholder="Choose a language" />
      </SelectTrigger>
      <SelectContent>
        {ENDANGERED_LANGUAGES.map((lang) => (
          <SelectItem key={lang.code} value={lang.code}>
            <span className="font-medium">{lang.name}</span>
            <span className="ml-1.5 text-muted-foreground">{lang.nativeName}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
