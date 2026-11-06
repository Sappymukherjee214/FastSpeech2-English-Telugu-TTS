# FastSpeech 2-Based English and Telugu Text-to-Speech Synthesis

> **A reproducible, bilingual FastSpeech 2 implementation for English and Telugu speech synthesis**

This repository contains two independent FastSpeech 2 text-to-speech pipelines:

- **English**, trained with the [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) dataset.
- **Telugu**, trained with the [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) dataset.

The project is organized as a complete assignment submission: preprocessing, training, inference, trained checkpoints, generated audio, loss curves, variance-adaptor visualizations, automated tests, and the [technical report](Report/Saptarshi_Mukherjee_FastSpeech2_Technical_Report.pdf) are all included.

## What This Project Demonstrates

- Separate language-specific FastSpeech 2 models with independent configurations, vocabularies, manifests, checkpoints, and result directories.
- Grapheme-level tokenization for both languages, including Unicode Telugu text without phonemization.
- Deterministic, duration-based dataset selection with leakage-safe train/validation splitting.
- Mel-spectrogram, pitch (F0), energy, and duration feature extraction.
- Automatic generation of speech samples and duration/pitch/energy plots for every test sentence.
- A self-contained Griffin-Lim vocoder that does not require a separate neural-vocoder checkpoint.
- Reproducible commands and automated pipeline tests.

## Submission Contents

| Deliverable | Location |
|---|---|
| English preprocessing, training, and inference | `English/` |
| Telugu preprocessing, training, and inference | `Telugu/` |
| Shared model and pipeline implementation | `common/` |
| English and Telugu trained checkpoints | `English/checkpoints/`, `Telugu/checkpoints/` |
| Generated audio and analysis artifacts | `Results/English/`, `Results/Telugu/` |
| Technical report | `Report/Saptarshi_Mukherjee_FastSpeech2_Technical_Report.pdf` |
| Automated tests | `tests/` |
| Installation and runtime dependencies | `requirements.txt` |

## Repository Layout

```text
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Saptarshi_Mukherjee_FastSpeech2_Technical_Report.pdf
├── README.md
└── requirements.txt
```

The generated feature caches under `English/data/features/` and `Telugu/data/features/` are intentionally excluded from version control because they are large, reproducible preprocessing outputs. They can be regenerated with the commands below.

## Quick Start

### 1. Create the environment

Python 3.10–3.13 is supported by the implementation. A CUDA-capable GPU is recommended for training; CPU execution is supported but slower.

```powershell
python -m pip install -r requirements.txt
```

### 2. Run the tests

```powershell
python -m pytest -q
```

The current pipeline test suite contains 11 tests covering text processing, feature handling, model shapes, and core pipeline behavior.

### 3. Prepare the English data

```powershell
python English\preprocess.py --config English\config.yaml --inspect 8
python English\preprocess.py --config English\config.yaml
```

### 4. Train and synthesize English speech

```powershell
python English\train.py --config English\config.yaml
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt
```

### 5. Prepare, train, and synthesize Telugu speech

```powershell
python Telugu\preprocess.py --config Telugu\config.yaml --inspect 8
python Telugu\preprocess.py --config Telugu\config.yaml
python Telugu\train.py --config Telugu\config.yaml
python Telugu\inference.py --config Telugu\config.yaml --checkpoint Telugu\checkpoints\best.pt
```

To generate one sentence or custom text:

```powershell
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt --id E1
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt --text "Your custom sentence here."
```

## Dataset Selection and Preprocessing

The English and Telugu pipelines are fully independent. Each uses its own dataset configuration, source split, manifests, vocabulary, features, checkpoints, and results.

| Language | Dataset | Source split | Training target | Validation target |
|---|---|---|---:|---:|
| English | `ai4bharat/Svarah` | Automatically selected | 18,000 s (~5 h) | 3,600 s (~1 h) |
| Telugu | `SPRINGLab/IndicTTS_Telugu` | `train` | 18,000 s (~5 h) | 3,600 s (~1 h) |

Preprocessing is deterministic:

1. Load the configured dataset and source split.
2. Require non-empty transcript text and decodable audio of at least 0.3 seconds.
3. Measure actual audio duration rather than relying on metadata estimates.
4. Create stable record identifiers, sort records, and shuffle with seed `1234`.
5. Accumulate complete utterances into the training split until the 18,000-second target is reached.
6. Build validation only from records not selected for training, up to 3,600 seconds.
7. Extract features and build each vocabulary from training text only.

The saved manifests were verified with zero train/validation ID overlap:

| Language | Train records | Train audio | Validation records | Validation audio | Overlap |
|---|---:|---:|---:|---:|---:|
| English | 3,392 | 18,021.95 s | 721 | 3,609.39 s | 0 |
| Telugu | 891 | 18,013.36 s | 162 | 3,614.00 s | 0 |

Audio is converted to mono, resampled to 22,050 Hz, and represented using 80-bin mel spectrograms with `n_fft=1024`, `hop_length=256`, `f_min=0`, and `f_max=8000`. Pitch and energy are extracted for the variance adaptor. Uniform frame durations are used as a reproducible baseline alignment, with BOS/EOS tokens included.

## Model and Implementation

The implementation follows the FastSpeech 2 design described by [Ren et al.](https://arxiv.org/abs/2006.04558) and is organized around the following components:

1. **Grapheme encoder** — four Transformer layers, hidden size 192, two attention heads.
2. **Variance adaptor** — predicts duration, pitch/F0, and energy.
3. **Length regulator** — expands encoder states using duration values.
4. **Mel decoder** — four Transformer layers with hidden size 192 and two attention heads.
5. **Mel projection** — maps decoder states to 80-bin mel spectrograms.
6. **Griffin-Lim vocoder** — reconstructs 22,050 Hz waveforms from mel features using 32 iterations.

### Modifications from the reference implementation

- Character/grapheme tokenization replaces phoneme-dependent input processing.
- Telugu Unicode graphemes are preserved without an English phonemizer.
- Uniform duration initialization provides a deterministic baseline without MFA.
- English and Telugu use separate configurations, vocabularies, manifests, checkpoints, and outputs.
- Loss curves and variance-adaptor plots are generated automatically.
- Griffin-Lim provides an accessible waveform-generation path without an external neural vocoder.

The training objective combines masked mel reconstruction, duration, pitch, and energy losses. Variable-length sequences are handled with masks.

## Training Configuration

Both languages use the same model and optimization settings unless changed in their language-specific YAML file:

| Parameter | Value |
|---|---:|
| Batch size | 4 |
| Epochs | 10 |
| Learning rate | 0.0001 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Hidden size | 192 |
| Transformer layers | 4 |
| Attention heads | 2 |
| Random seed | 1234 |
| Sample rate | 22,050 Hz |
| Mel bins | 80 |

Training writes the latest checkpoint to `last.pt`, the best validation checkpoint to `best.pt`, and epoch metrics to `Results/{Language}/history.json`.

## Evaluation Artifacts

### Loss curves

- [English training loss](Results/English/training_loss.png)
- [English validation loss](Results/English/validation_loss.png)
- [Telugu training loss](Results/Telugu/training_loss.png)
- [Telugu validation loss](Results/Telugu/validation_loss.png)

### Generated audio

The repository includes five test cases per language:

- English: `E1.wav`, `E2a.wav`, `E2b.wav`, `E3.wav`, `E4.wav`
- Telugu: `T1.wav`, `T2a.wav`, `T2b.wav`, `T3.wav`, `T4.wav`

They are available under `Results/English/test_generated_audios/` and `Results/Telugu/test_generated_audios/`.

### Variance-adaptor analysis

For every test case in both languages, the repository includes:

- predicted duration plots;
- predicted pitch/F0 plots;
- predicted energy plots;
- mel arrays and CSV summaries.

These are organized under:

```text
Results/{English,Telugu}/variance_adaptor_plots/
├── duration/
├── pitch/
└── energy/
```

The complete interpretation of the loss curves, generated samples, variance-adaptor behavior, language comparison, limitations, and reproducibility is provided in [Saptarshi_Mukherjee_FastSpeech2_Technical_Report.pdf](Report/Saptarshi_Mukherjee_FastSpeech2_Technical_Report.pdf).

## Test Sentences

### English

| ID | Evaluation focus |
|---|---|
| E1 | Long technical word and long-sequence handling |
| E2a | Noun/verb ambiguity of `conduct` in a noun context |
| E2b | Noun/verb ambiguity of `conduct` in a verb context |
| E3 | Preserved input challenge containing the supplied form `rhyth m` |
| E4 | Time, date, mixed-case text, and the token `andaroo` |

### Telugu

| ID | Evaluation focus |
|---|---|
| T1 | Long Telugu technical sentence and grapheme coverage |
| T2a | First context for the written form `pattu` |
| T2b | Contrasting clothing-related context for `pattu` |
| T3 | Short controlled pronunciation case |
| T4 | Date, time, punctuation, and long-clause handling |

## Limitations and Future Improvements

- Grapheme tokenization is simple and transparent, but phoneme-aware modeling could improve pronunciation.
- Uniform duration initialization is a baseline; MFA or another forced aligner could provide stronger alignments.
- Griffin-Lim output is useful for evaluation and reproducibility but can sound robotic compared with a neural vocoder.
- The included report provides objective audio-file checks. Formal pronunciation, intelligibility, naturalness, and prosody scores require a human listening study or metrics such as MOS/WER.
- Longer training, increased model capacity, improved alignment, speaker conditioning, and HiFi-GAN/WaveGlow-style neural vocoders are promising next steps.

## Troubleshooting

### Out-of-memory during training

Reduce the batch size in `English/config.yaml` or `Telugu/config.yaml`:

```yaml
train:
  batch_size: 2
```

### Slow preprocessing

Set `num_workers: 0` for a lower-memory single-process run, or retry after ensuring sufficient disk space and RAM.

### Dataset download failure

Check Hugging Face access and network connectivity. Dataset downloads can require several gigabytes and may be resumed by rerunning preprocessing.

### Telugu characters not rendering in plots

Install a Telugu-capable font on the host operating system and rerun inference/plot generation.

## References

- Ren et al., [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [Reference implementation: ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- [Svarah dataset](https://huggingface.co/datasets/ai4bharat/Svarah)
- [IndicTTS Telugu dataset](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)

## Citation

```bibtex
@inproceedings{ren2020fastspeech2,
  title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
