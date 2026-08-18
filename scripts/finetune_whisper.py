"""
Fine-tune Whisper on labeled Kokborok audio to produce the ASR checkpoint
that service/asr_engine.py picks up automatically once present.

Expects a manifest at data/asr_manifest.jsonl, one JSON object per line:
    {"audio": "data/audio/clip001.wav", "text": "Nwng bubagra tamwi?"}
`audio` paths are resolved relative to the repo root if not absolute.

Usage:
    python3 scripts/finetune_whisper.py

Writes the fine-tuned checkpoint to models/whisper_finetuned/trp/, where
ASREngine will find and prefer it over the zero-shot bridge fallback.
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import config  # noqa: E402

BASE_MODEL = config.WHISPER_MODEL_NAME
OUTPUT_DIR = config.WHISPER_FINETUNED_DIR


def load_manifest() -> list[dict]:
    if not config.ASR_MANIFEST_PATH.exists():
        raise SystemExit(
            f"No manifest at {config.ASR_MANIFEST_PATH}. Create it with lines of "
            '{"audio": "path/to/clip.wav", "text": "transcript"} before running this.'
        )
    rows = []
    with open(config.ASR_MANIFEST_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_path = Path(row["audio"])
            if not audio_path.is_absolute():
                audio_path = config.BASE_DIR / audio_path
            rows.append({"audio": str(audio_path), "text": row["text"]})
    if not rows:
        raise SystemExit(f"Manifest {config.ASR_MANIFEST_PATH} is empty.")
    return rows


def main():
    rows = load_manifest()
    print(f"Loaded {len(rows)} labeled Kokborok clips from {config.ASR_MANIFEST_PATH}")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.generation_config.forced_decoder_ids = None

    def prepare(row):
        audio, _ = librosa.load(row["audio"], sr=16000, mono=True)
        features = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        labels = processor.tokenizer(row["text"]).input_ids
        return {"input_features": features, "labels": labels}

    dataset = [prepare(row) for row in rows]

    def collate(batch):
        input_features = torch.tensor(np.array([b["input_features"] for b in batch]))
        label_batch = processor.tokenizer.pad(
            [{"input_ids": b["labels"]} for b in batch], return_tensors="pt"
        )
        labels = label_batch["input_ids"].masked_fill(label_batch.attention_mask.ne(1), -100)
        return {"input_features": input_features, "labels": labels}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=4,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned Whisper checkpoint to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
