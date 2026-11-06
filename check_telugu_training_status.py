import json
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("TELUGU MODEL TRAINING STATUS CHECK")
print("=" * 70)
print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check preprocessing
print("1. PREPROCESSING")
print("-" * 70)
train_manifest = Path("Telugu/data/train.jsonl")
val_manifest = Path("Telugu/data/validation.jsonl")
vocab_file = Path("Telugu/data/vocab.json")
features_dir = Path("Telugu/data/features")

if train_manifest.exists() and val_manifest.exists() and vocab_file.exists():
    train_count = sum(1 for _ in open(train_manifest, encoding='utf-8'))
    val_count = sum(1 for _ in open(val_manifest, encoding='utf-8'))
    feature_count = len(list(features_dir.glob("*.npz"))) if features_dir.exists() else 0

    print(f"   ✓ Training manifest: {train_count} records")
    print(f"   ✓ Validation manifest: {val_count} records")
    print(f"   ✓ Feature files: {feature_count} files")
    print(f"   ✓ Vocabulary: exists")
    print(f"   STATUS: COMPLETE ✓")
else:
    print("   ✗ STATUS: INCOMPLETE - Run preprocessing first")

print()

# Check training
print("2. TRAINING")
print("-" * 70)
checkpoints_dir = Path("Telugu/checkpoints")
results_dir = Path("Results/Telugu")

# Check for checkpoint files
checkpoint_files = []
if checkpoints_dir.exists():
    checkpoint_files = [f for f in checkpoints_dir.iterdir()
                       if f.suffix == '.pt' and f.name != '.gitkeep']

# Check for training history
history_file = results_dir / "history.json"
training_loss_plot = results_dir / "training_loss.png"
validation_loss_plot = results_dir / "validation_loss.png"

if checkpoint_files:
    print(f"   ✓ Checkpoints found: {len(checkpoint_files)}")
    for ckpt in sorted(checkpoint_files):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
        print(f"     - {ckpt.name}: {size_mb:.1f} MB (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")

    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        epochs_completed = len(history.get('train_loss', []))
        print(f"   ✓ Training history: {epochs_completed} epochs completed")

        if epochs_completed > 0:
            print(f"     - Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"     - Final validation loss: {history['validation_loss'][-1]:.4f}")
            if 'best_epoch' in history:
                print(f"     - Best epoch: {history['best_epoch']} (val_loss: {history['best_validation_loss']:.4f})")
    else:
        print(f"   ⚠ No training history found")

    if training_loss_plot.exists() and validation_loss_plot.exists():
        print(f"   ✓ Loss plots generated")

    print(f"   STATUS: COMPLETE ✓")
else:
    print(f"   ✗ No checkpoint files found in Telugu/checkpoints/")
    print(f"   STATUS: NOT STARTED")
    print()
    print("   To start training, run:")
    print("   python Telugu\\train.py --config Telugu\\config.yaml")

print()

# Check inference/results
print("3. INFERENCE & RESULTS")
print("-" * 70)
test_audio_dir = results_dir / "test_generated_audios"
variance_plots_dir = results_dir / "variance_adaptor_plots"

if test_audio_dir.exists():
    audio_files = list(test_audio_dir.glob("T*.wav"))
    if audio_files:
        print(f"   ✓ Test audio files: {len(audio_files)} generated")
        for audio in sorted(audio_files)[:5]:  # Show first 5
            print(f"     - {audio.name}")
        print(f"   STATUS: COMPLETE ✓")
    else:
        print(f"   ✗ No test audio files (T1.wav, T2a.wav, etc.)")
        print(f"   STATUS: NOT STARTED")
        print()
        print("   To generate test audio, run:")
        print("   python Telugu\\inference.py --config Telugu\\config.yaml --checkpoint Telugu\\checkpoints\\best.pt")
else:
    print(f"   ✗ Test audio directory not found")
    print(f"   STATUS: NOT STARTED")

print()
print("=" * 70)
print("OVERALL STATUS SUMMARY")
print("=" * 70)

status_items = [
    ("Preprocessing", train_manifest.exists() and feature_count > 0),
    ("Training", len(checkpoint_files) > 0),
    ("Inference", test_audio_dir.exists() and len(list(test_audio_dir.glob("T*.wav"))) > 0 if test_audio_dir.exists() else False)
]

for item, status in status_items:
    print(f"   {item:20s} {'✓ Complete' if status else '✗ Pending'}")

print("=" * 70)
