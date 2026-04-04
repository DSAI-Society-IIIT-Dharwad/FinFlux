"""ASR training entrypoint for Whisper-small LoRA fine-tuning.
OPTIMIZED: RTX 4060 8GB VRAM Safe + The Final kwargs Bouncer Patch
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.data_manifest import (
    assert_multilingual_coverage,
    compute_manifest_coverage,
    validate_manifest_schema,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def prepare_dataset(csv_paths: str | list[str], eval_split_pct: float = 0.05):
    from datasets import load_dataset, concatenate_datasets

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    valid_paths = [str(Path(p)) for p in csv_paths if Path(p).exists()]
    if not valid_paths:
        return None, None
    
    datasets = []
    for path in valid_paths:
        ds = load_dataset("csv", data_files=path, split="train")
        datasets.append(ds)
        
    if not datasets:
        return None, None
        
    dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    split = dataset.train_test_split(test_size=eval_split_pct, seed=42)
    return split["train"], split["test"]


def build_model_and_processor(model_name: str):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers import BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(model_name)
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model.config.use_cache = False
    model.enable_input_require_grads()
    
    return model, processor


def apply_lora(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q_proj", "v_proj"],
    )
    return get_peft_model(model, config)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Lazy audio loader — reads WAV files on-the-fly. Zero disk cache needed."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        import soundfile as sf
        import numpy as np

        processed_inputs = []
        processed_labels = []

        for f in features:
            audio_path = f["audio_path"]
            text = f["text"]
            try:
                audio_array, sr = sf.read(audio_path)
                # Resample to 16kHz if needed
                if sr != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=16000)
                # Ensure mono float32
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                audio_array = audio_array.astype(np.float32)
            except Exception as e:
                # Skip bad files — use silence
                audio_array = np.zeros(16000, dtype=np.float32)

            sample = self.processor(
                audio_array, sampling_rate=16000, text=text, return_tensors=None
            )
            processed_inputs.append({"input_features": sample["input_features"][0]})
            processed_labels.append({"input_ids": sample["labels"]})

        batch = self.processor.feature_extractor.pad(processed_inputs, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(processed_labels, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small with LoRA.")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config YAML")
    args = parser.parse_args()

    config = load_yaml(args.config)
    training = config.get("training", {})
    data = config.get("data", {})
    coverage_cfg = training.get("coverage", {})
    model_name = training.get("model_name", "openai/whisper-small")
    output_dir = Path(training.get("output_dir", "outputs/whisper-small-lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data.get("train_csv", "data/processed/financial_asr_manifest.csv")
    rows = validate_manifest_schema(manifest_path)
    coverage = compute_manifest_coverage(rows)
    min_total_rows = int(coverage_cfg.get("min_total_rows", 1000))
    min_rows_by_language = coverage_cfg.get(
        "min_rows_by_language",
        {"en": 100, "hi": 100, "hinglish": 100},
    )
    assert_multilingual_coverage(
        coverage,
        min_total_rows=min_total_rows,
        min_rows_by_language={str(k): int(v) for k, v in min_rows_by_language.items()},
    )

    print("Manifest validation passed.")
    print(f"Coverage total_rows: {coverage.total_rows}")
    print(f"Coverage by language: {coverage.by_language}")
    print(f"Coverage by source: {coverage.by_source}")

    # ==========================================================
    # THE FINAL BUG FIX: Whisper kwargs Bouncer
    # Peft leaks 'input_ids' and 'inputs_embeds' into kwargs.
    # WhisperModel blindly passes kwargs to the decoder, causing
    # "multiple values for keyword argument" crash.
    # We pop them here. WhisperModel doesn't use them anyway!
    # ==========================================================
    import transformers.models.whisper.modeling_whisper as whisper_mod
    original_forward = whisper_mod.WhisperModel.forward

    def safe_forward(self, *args, **kwargs):
        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)
        return original_forward(self, *args, **kwargs)

    whisper_mod.WhisperModel.forward = safe_forward
    # ==========================================================

    print("Loading model in 4-bit...")
    model, processor = build_model_and_processor(model_name)
    
    print("Applying LoRA adapters...")
    model = apply_lora(
        model,
        r=int(training.get("lora_r", 16)),
        alpha=int(training.get("lora_alpha", 32)),
        dropout=float(training.get("lora_dropout", 0.05)),
    )
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_dataset, eval_dataset = prepare_dataset(manifest_path)

    if train_dataset is None or len(train_dataset) == 0:
        model.save_pretrained(str(output_dir))
        processor.save_pretrained(str(output_dir))
        print("No training data found. Exiting.")
        return 0

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": int(training.get("num_train_epochs", 3)),
        "learning_rate": float(training.get("learning_rate", 1e-4)),
        "bf16": True,
        "gradient_checkpointing": True,
        "save_steps": 500,
        "logging_steps": 25,
        "save_total_limit": 2,
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "predict_with_generate": False,
        "generation_max_length": 225,
        "dataloader_num_workers": 0,
    }
    
    training_args_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_args_params:
        training_kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in training_args_params:
        training_kwargs["eval_strategy"] = "no"
        
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    # Keep raw CSV rows — audio is loaded lazily inside the DataCollator.
    # This avoids writing any Arrow cache to disk (critical when C: is near full).
    print(f"Dataset ready: {len(train_dataset)} training samples (lazy audio loading)")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=getattr(model.config, "decoder_start_token_id", 50258),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }
    
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = processor
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print("Starting training! Expect ~3-4 hours on RTX 4060...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"status": "training_complete"}, indent=2), encoding="utf-8")
    print(f"SUCCESS! Checkpoint saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())