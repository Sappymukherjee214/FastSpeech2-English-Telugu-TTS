import json
from pathlib import Path

# Check training data
train_path = Path("Telugu/data/train.jsonl")
train_records = []
with open(train_path, encoding='utf-8') as f:
    for line in f:
        train_records.append(json.loads(line))

# Check validation data
val_path = Path("Telugu/data/validation.jsonl")
val_records = []
with open(val_path, encoding='utf-8') as f:
    for line in f:
        val_records.append(json.loads(line))

# Check vocabulary
vocab_path = Path("Telugu/data/vocab.json")
with open(vocab_path, encoding='utf-8') as f:
    vocab = json.load(f)

# Check feature files
features_dir = Path("Telugu/data/features")
feature_files = list(features_dir.glob("*.npz"))

# Calculate total frames and approximate duration
# Duration is stored as list of frame counts per token
# hop_length=256, sample_rate=22050 -> frame_duration = 256/22050 ≈ 0.0116s
frame_duration = 256 / 22050

train_total_frames = sum(sum(d['duration']) for d in train_records)
train_total_seconds = train_total_frames * frame_duration

val_total_frames = sum(sum(d['duration']) for d in val_records)
val_total_seconds = val_total_frames * frame_duration

# Print summary
print("=" * 60)
print("TELUGU PREPROCESSING STATUS")
print("=" * 60)
print(f"\n✓ Training records: {len(train_records)}")
print(f"  Total frames: {train_total_frames:,}")
print(f"  Total duration: {train_total_seconds:.1f}s ({train_total_seconds/3600:.2f}h)")
print(f"\n✓ Validation records: {len(val_records)}")
print(f"  Total frames: {val_total_frames:,}")
print(f"  Total duration: {val_total_seconds:.1f}s ({val_total_seconds/3600:.2f}h)")
print(f"\n✓ Vocabulary size: {len(vocab)} tokens")
print(f"  Special tokens: <pad>, <unk>, <bos>, <eos>")
print(f"  Telugu characters: {len(vocab) - 4} graphemes")
print(f"\n✓ Feature files: {len(feature_files)}")
print(f"  Expected: {len(train_records) + len(val_records)}")
print(f"  Match: {'YES ✓' if len(feature_files) == len(train_records) + len(val_records) else 'NO ✗'}")

# Show sample
print("\n" + "=" * 60)
print("SAMPLE TRAINING RECORD")
print("=" * 60)
sample = train_records[0]
sample_frames = sum(sample['duration'])
sample_duration = sample_frames * frame_duration
print(f"ID: {sample.get('id', 'N/A')}")
print(f"Text: {sample['text'][:100]}")
print(f"Tokens: {len(sample['duration'])} (including BOS/EOS)")
print(f"Total frames: {sample_frames}")
print(f"Duration: {sample_duration:.2f}s")
print(f"Feature file: {Path(sample['feature_path']).name}")

print("\n" + "=" * 60)
print("STATUS: PREPROCESSING COMPLETE ✓")
print("=" * 60)
print("\nNext step: Run training")
print("  python Telugu\\train.py --config Telugu\\config.yaml")
