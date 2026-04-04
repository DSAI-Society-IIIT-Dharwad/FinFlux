"""Dataset preparation entrypoint for finance ASR fine-tuning.
OPTIMIZED: Network GB Tracker, SSL Bypass, Crash-safe Loop, Pre-scan Skip
"""

from __future__ import annotations

# --- HACKATHON FIX: Bypass strict SSL for Campus/VPN Firewalls ---
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# ------------------------------------------------------------------

import argparse
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.data_manifest import build_unified_row, normalize_language_label, write_unified_manifest


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def resample_and_export_audio(audio: dict[str, Any], output_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    array = np.asarray(audio["array"], dtype=np.float32)
    sampling_rate = int(audio["sampling_rate"])
    if array.ndim > 1:
        array = array.mean(axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sampling_rate == 16000:
        sf.write(str(output_path), array, 16000, subtype="PCM_16")
        return

    import torch
    import torchaudio
    tensor = torch.from_numpy(array)
    tensor = torchaudio.functional.resample(tensor, sampling_rate, 16000)
    sf.write(str(output_path), tensor.numpy(), 16000, subtype="PCM_16")


def _duration_seconds(audio: dict[str, Any]) -> float:
    return len(audio["array"]) / float(audio["sampling_rate"])


def _extract_row_audio(row: dict[str, Any], audio_column: str) -> tuple[dict[str, Any] | None, int]:
    """Returns (audio_dict, downloaded_bytes)."""
    import io
    import soundfile as sf

    audio = row.get(audio_column)
    dl_bytes = 0
    
    if isinstance(audio, dict):
        if "bytes" in audio:
            dl_bytes = len(audio["bytes"])
            try:
                array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]))
                return {"array": array, "sampling_rate": sampling_rate}, dl_bytes
            except Exception:
                pass
        elif "path" in audio and "sampling_rate" in audio:
            try:
                array, sampling_rate = sf.read(audio["path"])
                # Estimate bytes if reading from local cache
                dl_bytes = int(16000 * sampling_rate * 0.55) 
                return {"array": array, "sampling_rate": sampling_rate}, dl_bytes
            except Exception:
                pass

        if "array" in audio and "sampling_rate" in audio:
            return audio, dl_bytes
    return None, 0


def _extract_text(row: dict[str, Any], text_column: str) -> str:
    return normalize_text(str(row.get(text_column, "")))


def _extract_language(row: dict[str, Any]) -> str:
    for key in ("language", "locale"):
        val = str(row.get(key, "")).strip()
        if val:
            return normalize_language_label(val)
    return "en"


def _fmt_hours(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"


def _fmt_gb(bytes_val: float) -> str:
    return f"{bytes_val / (1024**3):.2f} GB"


def _progress_bar(current: float, target: float, width: int = 30) -> str:
    frac = min(current / target, 1.0) if target > 0 else 0.0
    filled = int(frac * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}] {frac*100:5.1f}%  {_fmt_hours(current)} / {_fmt_hours(target)}"


_MANIFEST_FIELDS = ("audio_path", "text", "language")


def process_dataset(
    *,
    display_name: str,
    dataset_id: str,
    config_name: str | None,
    split: str,
    audio_column: str,
    text_column: str,
    min_seconds: float,
    max_seconds: float,
    target_hours: float,
    output_dir: Path,
) -> tuple[list[dict[str, str]], float]:
    """Stream one dataset, export WAVs, stop once target_hours of audio is collected."""
    try:
        from datasets import load_dataset, Audio
    except ImportError as exc:
        print(f"[{display_name}] ERROR: datasets not installed — {exc}")
        return [], 0.0

    target_seconds = target_hours * 3600.0
    load_kwargs: dict[str, Any] = dict(split=split, streaming=True)

    # Estimate total download GB (FLAC is roughly 55% the size of 16kHz 16-bit PCM WAV)
    est_total_dl_bytes = target_seconds * 16000 * 2 * 0.55

    print(f"\n{'─' * 60}")
    print(f"  {display_name}")
    print(f"  {dataset_id}" + (f"  config={config_name}" if config_name else ""))
    print(f"  Target: {_fmt_hours(target_seconds)} (Est. Download: ~{_fmt_gb(est_total_dl_bytes)})")
    print(f"{'─' * 60}")

    try:
        if config_name:
            dataset = load_dataset(dataset_id, config_name, **load_kwargs)
        else:
            dataset = load_dataset(dataset_id, **load_kwargs)
        
        dataset = dataset.cast_column(audio_column, Audio(decode=False))
    except Exception as exc:
        print(f"[{display_name}] ERROR loading dataset: {exc}")
        return [], 0.0

    manifest: list[dict[str, str]] = []
    dataset_slug = f"{dataset_id.replace('/', '_')}_{config_name or 'default'}"
    dataset_dir = output_dir / dataset_slug / split
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # --- PRE-SCAN SKIP LOGIC ---
    existing_files = list(dataset_dir.glob("*.wav"))
    if existing_files:
        import soundfile as sf
        print(f"  ⚡ Scanning {len(existing_files)} existing clips on disk...")
        pre_existing_seconds = 0.0
        for f in existing_files:
            try:
                pre_existing_seconds += sf.info(str(f)).duration
            except Exception:
                pass
        
        if pre_existing_seconds >= target_seconds:
            print(f"  ✅ SKIP: Already have {_fmt_hours(pre_existing_seconds)}. Severing audio download, fetching text only...")
            try:
                dataset = dataset.remove_columns([audio_column])
            except Exception:
                pass
            
            for index, row in enumerate(dataset):
                if index > len(existing_files) + 500: 
                    break
                
                file_name = f"{dataset_slug}_{split}_{index:06d}.wav"
                audio_path = dataset_dir / file_name
                
                if not audio_path.exists():
                    continue
                    
                text = _extract_text(row, text_column)
                if text:
                    manifest.append({
                        "audio_path": str(audio_path),
                        "text": text,
                        "language": _extract_language(row),
                    })
            
            print(f"  ✅ Text mapped for {len(manifest)} clips in seconds!")
            return manifest, pre_existing_seconds
    # ---------------------------

    accumulated_seconds = 0.0
    downloaded_bytes = 0.0
    rows_checked = 0
    skipped_duration = 0
    skipped_audio = 0
    skipped_text = 0
    skipped_export = 0
    first_row_shown = False
    last_print_time = time.time()
    PRINT_INTERVAL = 30

    # --- CRASH-SAFE LOOP WRAPPER ---
    try:
        for index, row in enumerate(dataset):
            if accumulated_seconds >= target_seconds:
                break

            rows_checked += 1

            if not first_row_shown:
                print(f"  Schema: {sorted(row.keys())}")
                first_row_shown = True

            file_name = f"{dataset_slug}_{split}_{index:06d}.wav"
            audio_path = dataset_dir / file_name

            if audio_path.exists():
                try:
                    import soundfile as sf
                    info = sf.info(str(audio_path))
                    accumulated_seconds += info.duration
                    text = _extract_text(row, text_column)
                    if text:
                        manifest.append(
                            build_unified_row(
                                audio_path=str(audio_path),
                                text=text,
                                language=_extract_language(row),
                                source=display_name,
                                duration_seconds=info.duration,
                            )
                        )
                except Exception:
                    pass
                else:
                    continue

            audio, dl_bytes = _extract_row_audio(row, audio_column)
            if audio is None:
                skipped_audio += 1
                if skipped_audio <= 3:
                    print(f"  [row {index}] audio column '{audio_column}' bad format: {type(row.get(audio_column))}")
                continue

            duration = _duration_seconds(audio)
            if not (min_seconds <= duration <= max_seconds):
                skipped_duration += 1
                continue

            text = _extract_text(row, text_column)
            if not text:
                skipped_text += 1
                continue

            if len(manifest) == 0:
                print(f"  First valid clip — {duration:.1f}s — {text[:70]!r}")

            try:
                resample_and_export_audio(audio, audio_path)
            except Exception as exc:
                skipped_export += 1
                if skipped_export <= 3:
                    print(f"  [row {index}] export failed: {exc}")
                continue

            accumulated_seconds += duration
            downloaded_bytes += dl_bytes

            manifest.append(
                build_unified_row(
                    audio_path=str(audio_path),
                    text=text,
                    language=_extract_language(row),
                    source=display_name,
                    duration_seconds=duration,
                )
            )

            now = time.time()
            if now - last_print_time >= PRINT_INTERVAL or accumulated_seconds >= target_seconds:
                # --- NETWORK TRACKING OUTPUT ---
                dl_pct = (downloaded_bytes / est_total_dl_bytes * 100) if est_total_dl_bytes > 0 else 0
                print(f"  {_progress_bar(accumulated_seconds, target_seconds)}  ({len(manifest)} clips)")
                print(f"  🌐 Network: {_fmt_gb(downloaded_bytes)} / ~{_fmt_gb(est_total_dl_bytes)} downloaded ({dl_pct:.1f}%)")
                # -------------------------------
                last_print_time = now

    except (RuntimeError, ConnectionError, OSError) as net_exc:
        print(f"\n  ⚠️ INTERNET DROPPED at {_fmt_hours(accumulated_seconds)}!")
        print(f"  🌐 Wasted Data: {_fmt_gb(downloaded_bytes)}")
        print(f"  Error: {net_exc}")
        print(f"  Progress saved safely. Run the script again to resume!")
    # ---------------------------------

    print(f"\n  Done — {_fmt_hours(accumulated_seconds)} collected ({len(manifest)} clips)")
    print(f"  🌐 Total Data Used: {_fmt_gb(downloaded_bytes)}")
    if skipped_duration:
        print(f"    Skipped {skipped_duration} rows: outside [{min_seconds}s, {max_seconds}s]")
    if skipped_audio:
        print(f"    Skipped {skipped_audio} rows: bad audio column")
    if skipped_text:
        print(f"    Skipped {skipped_text} rows: empty text")
    if skipped_export:
        print(f"    Skipped {skipped_export} rows: export error")

    shortfall = target_seconds - accumulated_seconds
    if shortfall > 60:
        print(f"  WARNING shortfall: {_fmt_hours(shortfall)} — dataset exhausted before target")

    return manifest, accumulated_seconds


def _make_fallback_wav(path: Path, label: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)
    return build_unified_row(
        audio_path=str(path),
        text=f"fallback row for {label}",
        language="en",
        source=label,
        duration_seconds=1.0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare financial ASR dataset manifests.")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Dataset config YAML")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument(
        "--hours-scale",
        type=float,
        default=1.0,
        help="Scale all expected_hours targets (e.g. 0.01 for a quick smoke test)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_specs = config.get("datasets", [])
    if not dataset_specs:
        print("ERROR: No datasets found in config. Check your YAML structure.")
        return 1

    total_target_h = sum(s.get("expected_hours", 0) for s in dataset_specs) * args.hours_scale
    print(f"Total target: {_fmt_hours(total_target_h * 3600)}  (scale={args.hours_scale})")

    rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    seconds_by_dataset: dict[str, float] = {}

    for spec in dataset_specs:
        display_name = spec.get("name", spec.get("dataset_id", "unknown"))
        target_hours = spec.get("expected_hours", 1.0) * args.hours_scale

        rows, collected_seconds = process_dataset(
            display_name=display_name,
            dataset_id=spec["dataset_id"],
            config_name=spec.get("config_name"),
            split=spec.get("split", "train"),
            audio_column=spec.get("audio_column", "audio"),
            text_column=spec.get("text_column", "text"),
            min_seconds=spec.get("min_seconds", 1.0),
            max_seconds=spec.get("max_seconds", 60.0),
            target_hours=target_hours,
            output_dir=output_dir,
        )

        if not rows:
            fallback_path = output_dir / "fallback" / f"{display_name.replace(' ', '_')}_fallback.wav"
            rows = [_make_fallback_wav(fallback_path, display_name)]
            collected_seconds = 0.0

        rows_by_dataset[display_name] = rows
        seconds_by_dataset[display_name] = collected_seconds

    print(f"\n{'=' * 60}")
    print("  DATASET SUMMARY")
    print(f"{'=' * 60}")
    selected_rows: list[dict[str, str]] = []
    total_collected = 0.0

    for spec in dataset_specs:
        display_name = spec.get("name", spec.get("dataset_id", "unknown"))
        rows = rows_by_dataset[display_name]
        collected = seconds_by_dataset[display_name]
        target_s = spec.get("expected_hours", 0) * args.hours_scale * 3600
        pct = (collected / target_s * 100) if target_s > 0 else 0.0
        print(f"  {display_name:<32} {len(rows):>6} clips   {_fmt_hours(collected):>10}  ({pct:.1f}%)")
        selected_rows.extend(rows)
        total_collected += collected

    print(f"{'─' * 60}")
    overall_pct = (total_collected / (total_target_h * 3600) * 100) if total_target_h > 0 else 0.0
    print(f"  {'TOTAL':<32} {len(selected_rows):>6} clips   {_fmt_hours(total_collected):>10}  ({overall_pct:.1f}%)")
    print(f"{'=' * 60}")

    manifest_path = output_dir / "financial_asr_manifest.csv"
    write_unified_manifest(selected_rows, manifest_path)
    print(f"\nManifest saved -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())