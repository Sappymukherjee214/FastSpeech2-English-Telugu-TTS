import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .model import FastSpeech2
from .text import encode, load_vocabulary, tokenize


SAMPLES = {
    "English": [
        ("E1", "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."),
        ("E2a", "The orchestra's conduct during the ceremony was praised by the audience."),
        ("E2b", "The researchers will conduct an experiment to evaluate the proposed method."),
        ("E3", "The rhyth m of the orchestra changed unexpectedly."),
        ("E4", "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."),
    ],
    "Telugu": [
        ("T1", "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."),
        ("T2a", "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."),
        ("T2b", "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."),
        ("T3", "పులి అడవిలో నిశ్శబ్దంగా నడిచింది."),
        ("T4", "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."),
    ],
}


def main():
    parser = argparse.ArgumentParser(description="FastSpeech 2 inference with audio and plot generation")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--id", help="Generate only for this sample ID")
    parser.add_argument("--text", help="Custom text to synthesize")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio generation")
    parser.add_argument("--no-plots", action="store_true", help="Skip variance adaptor plots")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.config).resolve().parent
    language = config["language"]

    # Determine samples to process
    if args.text:
        samples = [(args.id or "custom", args.text)]
    else:
        samples = list(SAMPLES.get(language, []))

    # Load vocabulary and model
    vocabulary = load_vocabulary(root / "data" / "vocab.json")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FastSpeech2(len(vocabulary), config["audio"]["n_mels"], **config["model"])
    model.load_state_dict(state["model"])
    model.eval()

    # Setup output directories
    results_dir = root.parent / config["paths"]["results"]
    audio_dir = results_dir / "test_generated_audios"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Import optional modules (graceful fallback)
    try:
        from .vocoder import synthesize
        has_vocoder = True
    except ImportError:
        has_vocoder = False
        print("Warning: vocoder module not available, skipping audio generation")

    try:
        from .plots import plot_variance_adaptor
        has_plots = True
    except ImportError:
        has_plots = False
        print("Warning: plots module not available, skipping plot generation")

    print(f"\n{'='*60}")
    print(f"FastSpeech 2 Inference — {language}")
    print(f"{'='*60}")
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Vocabulary size:  {len(vocabulary)}")
    print(f"Samples:          {len(samples)}")
    print(f"Results dir:      {results_dir}")
    print()

    for sample_id, text in samples:
        if args.id and sample_id != args.id:
            continue

        print(f"[{sample_id}] {text[:80]}{'...' if len(text) > 80 else ''}")

        # Encode text
        tokens = torch.tensor([encode(text, vocabulary)])
        text_length = torch.tensor([tokens.size(1)])

        # Run model
        with torch.no_grad():
            output = model(tokens, text_length)

        mel = output["mel"][0].numpy()  # [time, n_mels]
        pitch = output["pitch"][0].numpy()  # [seq_len]
        energy = output["energy"][0].numpy()  # [seq_len]
        log_duration = output["log_duration"][0].numpy()  # [seq_len]
        duration = output["duration"][0].numpy()  # [seq_len]

        # Save mel spectrogram
        mel_path = results_dir / f"{sample_id}_mel.npy"
        np.save(mel_path, mel)
        print(f"  Mel spectrogram: {mel.shape} → {mel_path.name}")

        # Save variance adaptor predictions as CSV
        csv_path = results_dir / f"{sample_id}_variance.csv"
        tokens_chars = tokenize(text)
        _save_variance_csv(csv_path, tokens_chars, duration, pitch, energy, log_duration)
        print(f"  Variance CSV:    {csv_path.name}")

        # Generate audio
        if has_vocoder and not args.no_audio:
            try:
                audio_path = audio_dir / f"{sample_id}.wav"
                audio = synthesize(mel, config, str(audio_path))
                duration_sec = len(audio) / config["audio"]["sample_rate"]
                print(f"  Audio:           {duration_sec:.2f}s → {audio_path.name}")
            except Exception as exc:
                print(f"  Audio error:     {exc}")

        # Generate variance adaptor plots
        if has_plots and not args.no_plots:
            try:
                plot_variance_adaptor(sample_id, text, output, config, str(results_dir))
                print(f"  Plots:           duration, pitch, energy → variance_adaptor_plots/")
            except Exception as exc:
                print(f"  Plot error:      {exc}")

        print()

    # Generate loss curves if history exists
    if has_plots:
        history_path = results_dir / "history.json"
        if history_path.exists():
            try:
                from .plots import plot_loss_curves
                plot_loss_curves(str(history_path), str(results_dir))
                print(f"Loss curves saved to {results_dir}")
            except Exception as exc:
                print(f"Loss curve error: {exc}")

    print("Inference complete.")


def _save_variance_csv(path, tokens, duration, pitch, energy, log_duration):
    """Save variance adaptor predictions to CSV for analysis."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "token", "duration_frames", "log_duration", "pitch", "energy"])
        # Add BOS/EOS markers to tokens list
        display_tokens = ["<bos>"] + list(tokens) + ["<eos>"]
        for i in range(min(len(duration), len(display_tokens))):
            writer.writerow([
                i,
                display_tokens[i] if i < len(display_tokens) else f"[{i}]",
                int(duration[i]) if i < len(duration) else "",
                f"{log_duration[i]:.4f}" if i < len(log_duration) else "",
                f"{pitch[i]:.4f}" if i < len(pitch) else "",
                f"{energy[i]:.4f}" if i < len(energy) else "",
            ])


if __name__ == "__main__":
    main()
