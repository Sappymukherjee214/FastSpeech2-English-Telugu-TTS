import json
from pathlib import Path
from datetime import datetime

import torch

print("=" * 70)
print("ENGLISH MODEL TRAINING STATUS CHECK")
print("=" * 70)
print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check preprocessing
print("1. PREPROCESSING")
print("-" * 70)
train_manifest = Path("English/data/train.jsonl")
val_manifest = Path("English/data/validation.jsonl")
vocab_file = Path("English/data/vocab.json")
features_dir = Path("English/data/features")

feature_count = len(list(features_dir.glob("*.npz"))) if features_dir.exists() else 0

if train_manifest.exists() and val_manifest.exists() and vocab_file.exists():
    train_count = sum(1 for _ in open(train_manifest, encoding='utf-8'))
    val_count = sum(1 for _ in open(val_manifest, encoding='utf-8'))

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
checkpoints_dir = Path("English/checkpoints")
results_dir = Path("Results/English")

# Check for checkpoint files
checkpoint_files = []
if checkpoints_dir.exists():
    checkpoint_files = [f for f in checkpoints_dir.iterdir()
                       if f.suffix == '.pt' and f.name != '.gitkeep']

history_file = results_dir / "history.json"
training_loss_plot = results_dir / "training_loss.png"
validation_loss_plot = results_dir / "validation_loss.png"

if checkpoint_files:
    print(f"   ✓ Checkpoints found: {len(checkpoint_files)}")
    for ckpt in sorted(checkpoint_files):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
        print(f"     - {ckpt.name}: {size_mb:.1f} MB (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")

    history = []
    if history_file.exists():
        with open(history_file, encoding='utf-8') as f:
            history = json.load(f)
        history_epochs = {entry.get('epoch') for entry in history if entry.get('epoch') is not None}
        print(f"   ✓ Training history: {len(history_epochs)} recorded epochs ({sorted(history_epochs)})")

        if history:
            last_entry = max(history, key=lambda entry: entry.get('epoch', 0))
            print(f"     - Final train loss: {last_entry.get('train_loss', 0.0):.4f}")
            print(f"     - Final validation loss: {last_entry.get('validation_loss', 0.0):.4f}")
            val_losses = [(h['epoch'], h['validation_loss']) for h in history if 'epoch' in h and 'validation_loss' in h]
            if val_losses:
                best_ep, best_val = min(val_losses, key=lambda x: x[1])
                print(f"     - Best epoch: {best_ep} (val_loss: {best_val:.4f})")
    else:
        history_epochs = set()
        print(f"   ⚠ No training history found yet")

    # The checkpoint stores the authoritative latest completed epoch. This is
    # important when history.json was written incrementally after a resumed run.
    checkpoint_epochs = []
    for checkpoint in checkpoint_files:
        try:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if isinstance(state, dict) and isinstance(state.get('epoch'), int):
                checkpoint_epochs.append(state['epoch'])
        except Exception as exc:
            print(f"   ⚠ Could not inspect {checkpoint.name}: {exc}")
    latest_checkpoint_epoch = max(checkpoint_epochs, default=0)
    if latest_checkpoint_epoch:
        print(f"   ✓ Latest checkpoint epoch: {latest_checkpoint_epoch}")

    max_epoch = 10
    training_complete = latest_checkpoint_epoch >= max_epoch
    if training_complete and len(history_epochs) < max_epoch:
        print("   NOTE: Checkpoint confirms all 10 epochs completed; history.json contains only the epochs recorded in this resumed run.")

    if training_loss_plot.exists() and validation_loss_plot.exists():
        print(f"   ✓ Loss plots generated")

    print(f"   STATUS: {'COMPLETE ✓' if training_complete else 'IN PROGRESS / PARTIAL'}")
else:
    print(f"   ✗ No checkpoint files found in English/checkpoints/")
    print(f"   STATUS: NOT STARTED")
    print()
    print("   To start training, run:")
    print("   python English\\train.py --config English\\config.yaml")

print()

# Check inference/results
print("3. INFERENCE & RESULTS")
print("-" * 70)
test_audio_dir = results_dir / "test_generated_audios"
variance_plots_dir = results_dir / "variance_adaptor_plots"

if test_audio_dir.exists():
    audio_files = list(test_audio_dir.glob("E*.wav"))
    if audio_files:
        print(f"   ✓ Test audio files: {len(audio_files)} generated")
        for audio in sorted(audio_files)[:5]:
            print(f"     - {audio.name}")
        print(f"   STATUS: COMPLETE ✓")
    else:
        print(f"   ✗ No test audio files (E1.wav, E2.wav, etc.)")
        print(f"   STATUS: NOT STARTED")
        print()
        print("   To generate test audio, run:")
        print("   python English\\inference.py --config English\\config.yaml --checkpoint English\\checkpoints\\best.pt")
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
    ("Inference", test_audio_dir.exists() and len(list(test_audio_dir.glob("E*.wav"))) > 0 if test_audio_dir.exists() else False)
]

for item, status in status_items:
    print(f"   {item:20s} {'✓ Complete' if status else '✗ Pending'}")

print("=" * 70)
