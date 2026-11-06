import argparse
import io
import json
import random
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from .config import load_config
from .features import extract_features, save_features
from .manifest import write_jsonl
from .text import build_vocabulary, normalize_text, save_vocabulary, tokenize


def get_text_from_record(record: dict[str, Any]) -> Optional[str]:
    """Extract and normalize text from a dataset record."""
    candidates = ["text", "sentence", "verbatim", "transcript", "transcription", "normalized_text"]
    for field in candidates:
        if field in record and record[field]:
            value = str(record[field]).strip()
            if value:
                return normalize_text(value)
    return None


def make_record_id(record: dict[str, Any], index: int) -> str:
    """Generate a clean, filesystem-safe ID for the record."""
    raw_id = record.get("id")
    if raw_id is not None and isinstance(raw_id, (str, int)):
        clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(raw_id)).strip("_")
        if clean:
            return clean

    for key in ["audio_filepath", "audio"]:
        val = record.get(key)
        if isinstance(val, dict) and "path" in val and isinstance(val["path"], str):
            stem = Path(val["path"]).stem
            clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_")
            if clean:
                return clean
        elif isinstance(val, str):
            stem = Path(val).stem
            clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_")
            if clean:
                return clean

    return f"sample_{index:06d}"


def get_raw_audio_from_record(record: dict[str, Any]) -> Optional[tuple[bytes, float, int]]:
    """Extract raw audio bytes, duration, and sample rate from record using fast soundfile info.

    Returns (audio_bytes, duration_seconds, sample_rate) or None.
    """
    for field in ["audio", "audio_filepath"]:
        if field not in record or not record[field]:
            continue
        data = record[field]

        # Case 1: Audio feature with bytes
        if isinstance(data, dict) and "bytes" in data and data["bytes"]:
            raw_bytes = data["bytes"]
            try:
                info = sf.info(io.BytesIO(raw_bytes))
                return raw_bytes, float(info.duration), int(info.samplerate)
            except Exception:
                continue

        # Case 2: Audio feature with array (already decoded)
        if isinstance(data, dict) and "array" in data and data["array"] is not None:
            array = np.array(data["array"], dtype=np.float32)
            sr = int(data.get("sampling_rate", 22050))
            duration = float(len(array) / sr)
            buf = io.BytesIO()
            sf.write(buf, array, sr, format="WAV")
            return buf.getvalue(), duration, sr

        # Case 3: Raw bytes directly
        if isinstance(data, (bytes, bytearray)):
            try:
                info = sf.info(io.BytesIO(data))
                return bytes(data), float(info.duration), int(info.samplerate)
            except Exception:
                continue

    return None


def decode_audio_bytes(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode audio bytes to float32 mono numpy array resampled to target_sr."""
    array, sr = sf.read(io.BytesIO(raw_bytes))
    array = np.array(array, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)  # convert to mono
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    return array


def compute_uniform_durations(num_frames: int, num_tokens: int) -> list[int]:
    """Distribute mel frames uniformly across tokens, each duration >= 1, summing to num_frames."""
    if num_frames < num_tokens:
        raise ValueError(f"num_frames ({num_frames}) must be >= num_tokens ({num_tokens})")
    base_duration = num_frames // num_tokens
    remainder = num_frames % num_tokens
    return [max(1, base_duration + (1 if i < remainder else 0)) for i in range(num_tokens)]


def inspect_dataset(dataset_name: str, split: str, num_samples: int, target_sr: int) -> None:
    """Fast inspection of N samples using streaming mode without downloading the whole dataset."""
    print(f"\n{'='*60}")
    print(f"Inspecting {num_samples} samples from {dataset_name} (split: {split})")
    print(f"{'='*60}\n")

    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
        # Try casting audio column
        for col in ["audio", "audio_filepath"]:
            try:
                ds = ds.cast_column(col, Audio(decode=False))
            except Exception:
                pass
    except Exception as exc:
        print(f"Error loading streaming dataset: {exc}")
        return

    count = 0
    for record in ds:
        if count >= num_samples:
            break
        print(f"Sample #{count + 1}:")
        print(f"  Available fields: {list(record.keys())}")

        text = get_text_from_record(record)
        print(f"  Text: {text[:100] if text else '<none>'}{'...' if text and len(text) > 100 else ''}")

        audio_info = get_raw_audio_from_record(record)
        if audio_info:
            raw_bytes, duration, sr = audio_info
            print(f"  Audio: {len(raw_bytes)} bytes, {duration:.2f}s, original SR: {sr} Hz (target: {target_sr} Hz)")
        else:
            print("  Audio: <could not extract>")

        print()
        count += 1


def preprocess_dataset(
    config_path: str | Path,
    dataset_override: Optional[str] = None,
    inspect_count: Optional[int] = None,
    smoke_test: bool = False,
    use_streaming: bool = True,
    buffer_multiplier: float = 1.3,
) -> None:
    """Full preprocessing pipeline for FastSpeech 2 dataset."""
    config = load_config(config_path)
    root = Path(config_path).resolve().parent
    language = config["language"]
    dataset_name = dataset_override or config["dataset"]
    policy = config.get("dataset_policy", {})
    target_sr = config["audio"]["sample_rate"]
    seed = policy.get("seed", config.get("seed", 1234))

    # Determine split
    source_split = policy.get("source_split", "train")
    if source_split == "auto":
        from datasets import get_dataset_split_names
        splits = get_dataset_split_names(dataset_name)
        source_split = splits[0] if splits else "train"
        print(f"Auto-selected split: {source_split} (available: {splits})")

    # Fast inspection mode
    if inspect_count:
        inspect_dataset(dataset_name, source_split, inspect_count, target_sr)
        return

    # Determine duration targets
    if smoke_test:
        train_target = 60.0       # 1 minute smoke test
        val_target = 20.0         # 20 seconds
        print("Running in SMOKE-TEST mode (short duration targets for quick verification)")
    else:
        train_target = float(policy.get("train_seconds_target", config.get("targets", {}).get("train_seconds", 18000)))
        val_target = float(policy.get("validation_seconds_target", config.get("targets", {}).get("validation_seconds", 3600)))

    total_target = train_target + val_target
    print(f"\n{'='*60}")
    print(f"FastSpeech 2 Preprocessing — {language}")
    print(f"{'='*60}")
    print(f"Dataset:            {dataset_name} (split: {source_split})")
    print(f"Target duration:    Train: {train_target:.0f}s ({train_target/3600:.2f}h) | Val: {val_target:.0f}s ({val_target/3600:.2f}h)")
    print(f"Sample rate:        {target_sr} Hz")
    print(f"Random seed:        {seed}")
    print(f"Streaming mode:     {use_streaming}")

    # Phase 1: Fast scan & validation of records
    print("\n[Phase 1/2] Scanning dataset records and measuring durations...")
    collected_records = []
    accumulated_duration = 0.0

    try:
        ds = load_dataset(dataset_name, split=source_split, streaming=use_streaming)
        for col in ["audio", "audio_filepath"]:
            try:
                ds = ds.cast_column(col, Audio(decode=False))
            except Exception:
                pass
    except Exception as exc:
        print(f"Falling back to non-streaming load due to: {exc}")
        ds = load_dataset(dataset_name, split=source_split)
        for col in ["audio", "audio_filepath"]:
            try:
                ds = ds.cast_column(col, Audio(decode=False))
            except Exception:
                pass

    pbar = tqdm(desc="Scanning records", unit="rec")
    for idx, record in enumerate(ds):
        pbar.update(1)

        # Check optional filter
        if "dataset_filter" in config:
            field = config["dataset_filter"].get("field")
            val = config["dataset_filter"].get("value")
            if field and field in record and record[field] != val:
                continue

        text = get_text_from_record(record)
        if not text:
            continue

        audio_info = get_raw_audio_from_record(record)
        if not audio_info:
            continue

        raw_bytes, duration, _ = audio_info
        if duration < 0.3:  # Skip audio shorter than 300ms
            continue

        record_id = make_record_id(record, idx)

        collected_records.append({
            "id": record_id,
            "text": text,
            "duration": duration,
            "raw_bytes": raw_bytes,
        })
        accumulated_duration += duration

        # Stop scanning if we have enough buffer in streaming mode
        if use_streaming and accumulated_duration >= total_target * buffer_multiplier:
            print(f"\nReached buffer threshold: {accumulated_duration:.1f}s collected across {len(collected_records)} records")
            break

    pbar.close()
    print(f"Total valid candidate records: {len(collected_records)} ({accumulated_duration:.1f}s / {accumulated_duration/3600:.2f}h)")

    if not collected_records:
        raise RuntimeError("No valid records with text and decodable audio were found in the dataset.")

    # Phase 2: Deterministic selection
    print("\n[Phase 2/2] Deterministically selecting splits and extracting features...")

    # Sort by ID for stable baseline before shuffling
    collected_records.sort(key=lambda r: r["id"])

    # Shuffle with configured seed
    rng = random.Random(seed)
    rng.shuffle(collected_records)

    # Accumulate training set
    train_records = []
    train_seconds = 0.0
    for r in collected_records:
        if train_seconds >= train_target:
            break
        train_records.append(r)
        train_seconds += r["duration"]

    # Accumulate validation set from remaining records
    remaining_records = [r for r in collected_records if r["id"] not in {x["id"] for x in train_records}]
    val_records = []
    val_seconds = 0.0
    for r in remaining_records:
        if val_seconds >= val_target:
            break
        val_records.append(r)
        val_seconds += r["duration"]

    print(f"Split selection complete:")
    print(f"  Train:      {len(train_records)} records, {train_seconds:.1f}s ({train_seconds/3600:.2f}h) [Target: {train_target:.0f}s]")
    print(f"  Validation: {len(val_records)} records, {val_seconds:.1f}s ({val_seconds/3600:.2f}h) [Target: {val_target:.0f}s]")

    if train_seconds < train_target:
        print(f"  NOTICE: Available train duration is {train_seconds:.1f}s (< {train_target:.0f}s target). Using all available data.")
    if val_seconds < val_target:
        print(f"  NOTICE: Available validation duration is {val_seconds:.1f}s (< {val_target:.0f}s target). Using all available data.")

    # Setup directories
    manifest_dir = root / "data"
    features_dir = manifest_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Extract features for selected records only
    def process_split(records_list: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
        manifest = []
        for r in tqdm(records_list, desc=f"Extracting {split_name} features"):
            try:
                # Decode audio
                audio_array = decode_audio_bytes(r["raw_bytes"], target_sr)

                # Extract mel, pitch (F0), energy
                features = extract_features(
                    audio_array,
                    target_sr,
                    config["audio"]["n_fft"],
                    config["audio"]["hop_length"],
                    config["audio"]["n_mels"],
                    config["audio"]["f_min"],
                    config["audio"]["f_max"],
                )

                # Compute frame durations per token
                tokens = tokenize(r["text"])
                num_tokens = len(tokens) + 2  # +2 for BOS/EOS
                num_frames = len(features["mel"])

                if num_frames < num_tokens:
                    continue

                durations = compute_uniform_durations(num_frames, num_tokens)

                # Save feature npz
                feat_path = features_dir / f"{r['id']}.npz"
                save_features(features, feat_path)

                manifest.append({
                    "id": r["id"],
                    "text": r["text"],
                    "feature_path": str(feat_path),
                    "duration": durations,
                })
            except Exception as exc:
                print(f"Warning: error extracting {r['id']}: {exc}")
                continue

        return manifest

    train_manifest = process_split(train_records, "train")
    val_manifest = process_split(val_records, "validation")

    print(f"\nSuccessfully extracted features:")
    print(f"  Train:      {len(train_manifest)} items")
    print(f"  Validation: {len(val_manifest)} items")

    # Build vocabulary from training text
    print("Building vocabulary from training texts...")
    train_texts = [item["text"] for item in train_manifest]
    vocab = build_vocabulary(train_texts)
    print(f"Vocabulary size: {len(vocab)} symbols")

    # Save manifests and vocabulary
    write_jsonl(train_manifest, manifest_dir / "train.jsonl")
    write_jsonl(val_manifest, manifest_dir / "validation.jsonl")
    save_vocabulary(vocab, manifest_dir / "vocab.json")

    print(f"\n{'='*60}")
    print(f"Preprocessing Complete for {language}!")
    print(f"{'='*60}")
    print(f"Manifests saved to:  {manifest_dir}")
    print(f"Features saved to:   {features_dir}")
    print(f"Vocabulary saved to: {manifest_dir / 'vocab.json'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="FastSpeech 2 Dataset Preprocessing")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--dataset", help="Override dataset name from config")
    parser.add_argument("--inspect", type=int, metavar="N", help="Inspect N samples and exit")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test (short duration)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming, download full dataset")
    parser.add_argument("--buffer-multiplier", type=float, default=1.3, help="Buffer multiplier for streaming pool (default: 1.3)")
    args = parser.parse_args()

    preprocess_dataset(
        config_path=args.config,
        dataset_override=args.dataset,
        inspect_count=args.inspect,
        smoke_test=args.smoke_test,
        use_streaming=not args.no_stream,
        buffer_multiplier=args.buffer_multiplier,
    )


if __name__ == "__main__":
    main()
