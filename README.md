# FastSpeech 2-Based English and Telugu Text-to-Speech Synthesis

This repository implements separate FastSpeech 2 TTS models for English and Telugu, following the assignment requirements for the 07-11-2026 deadline.

## Overview

This project trains two independent FastSpeech 2 models:
- **English**: Using the [Svarah dataset](https://huggingface.co/datasets/ai4bharat/Svarah)
- **Telugu**: Using the [IndicTTS Telugu dataset](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)

Each model uses grapheme-level (character-level) tokenization. Telugu text is handled as Unicode graphemes without phonemization.

## Project Structure

The required submission layout is:

```
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
│   └── Technical_Report.pdf
├── README.md
└── requirements.txt
```

The repository also contains supporting implementation and verification files required by the entry points:

- `common/` — shared model, preprocessing, feature extraction, training, inference, plotting, and vocoder code.
- `English/data/` and `Telugu/data/` — language-specific manifests, features, and vocabularies generated during preprocessing.
- `tests/` — automated pipeline tests.
- `check_*.py` — optional dataset and training-status verification scripts.

## Environment Setup

### Requirements
- Python 3.10-3.12 recommended
- CUDA GPU recommended for training (CPU works but is much slower)
- ~20GB disk space for datasets and features

### Installation

```powershell
python -m pip install -r requirements.txt
```

### Verify Installation

```powershell
python -m pytest -q
```

All tests should pass (11 tests).

## Dataset Preparation

### English Dataset (Svarah)

#### 1. Inspect the dataset first (optional but recommended):

```powershell
python English\preprocess.py --config English\config.yaml --inspect 8
```

This shows 8 sample records with their text and audio information.

#### 2. Run preprocessing:

```powershell
python English\preprocess.py --config English\config.yaml
```

**What this does:**
- Downloads the Svarah dataset from HuggingFace
- Validates each record (text + audio)
- Measures actual audio duration
- Shuffles records with seed 1234
- Selects 18,000 seconds (5 hours) for training
- Selects 3,600 seconds (1 hour) for validation from remaining records
- Extracts mel spectrograms, pitch (F0), and energy features
- Computes uniform frame durations per character token
- Builds vocabulary from training texts
- Saves manifests to `English/data/train.jsonl` and `English/data/validation.jsonl`
- Saves features to `English/data/features/*.npz`
- Saves vocabulary to `English/data/vocab.json`

**Expected output:**
```
Loading dataset: ai4bharat/Svarah
Total records: XXXX
Processing records for English...
Valid records: YYYY/XXXX

Train: ZZZ records, 18000.0s (5.00h)
Validation: VVV records, 3600.0s (1.00h)

Saving features...
Building vocabulary...
Vocabulary size: NNN

Preprocessing complete!
```

### Telugu Dataset (IndicTTS Telugu)

#### 1. Inspect the dataset first:

```powershell
python Telugu\preprocess.py --config Telugu\config.yaml --inspect 8
```

#### 2. Run preprocessing:

```powershell
python Telugu\preprocess.py --config Telugu\config.yaml
```

Same process as English, but for Telugu text from `SPRINGLab/IndicTTS_Telugu`. The Telugu pipeline uses the configured `train` split and preserves Telugu Unicode graphemes during normalization and tokenization.

### Dataset selection and split guarantees

The two languages are prepared and trained independently. Each language has its own dataset configuration, manifests, feature directory, vocabulary, checkpoints, and results directory:

| Language | Dataset | Source split | Training target | Validation target |
|----------|---------|--------------|-----------------|-------------------|
| English | `ai4bharat/Svarah` | Automatically selected source split | 18,000 seconds (~5 hours) | 3,600 seconds (~1 hour) |
| Telugu | `SPRINGLab/IndicTTS_Telugu` | `train` | 18,000 seconds (~5 hours) | 3,600 seconds (~1 hour) |

For both datasets, preprocessing uses this deterministic subsampling strategy:

1. Load only the configured language dataset and source split.
2. Validate every candidate record by requiring non-empty transcript text and decodable audio of at least 0.3 seconds.
3. Measure the actual source-audio duration rather than estimating it from metadata.
4. Create a stable record ID from the dataset ID or audio path, sort records by ID, and shuffle with seed `1234`.
5. Accumulate complete utterances into the training split until the 18,000-second target is reached. Because utterances are not truncated, the final duration may be slightly above the target.
6. Build validation only from records not selected for training, accumulating complete utterances until 3,600 seconds. The implementation checks record IDs before adding validation records, so training and validation cannot overlap.
7. Extract features only for the selected records and build each language vocabulary from its training transcript text.

The current preprocessed manifests were verified as follows:

| Language | Train records | Train audio | Validation records | Validation audio | ID overlap |
|----------|---------------|-------------|--------------------|------------------|------------|
| English | 3,392 | 18,021.95 s (5.006 h) | 721 | 3,609.39 s (1.003 h) | 0 |
| Telugu | 891 | 18,013.36 s (5.004 h) | 162 | 3,614.00 s (1.004 h) | 0 |

These durations are calculated from the saved duration-alignment frame counts using the configured 22,050 Hz sample rate and 256-sample hop length. Both manifests have feature files for every record, and the zero-overlap check is performed separately for each language.

### Dataset preparation and preprocessing steps

For each language, the selected audio is:

- decoded from the dataset record and converted to mono;
- resampled to 22,050 Hz;
- transformed into 80-bin mel spectrograms using `n_fft=1024`, `hop_length=256`, `f_min=0`, and `f_max=8000`;
- analyzed for pitch (F0) and energy;
- tokenized at grapheme/character level, including Telugu Unicode graphemes;
- assigned uniform per-token frame durations as a reproducible baseline, with BOS/EOS tokens included;
- saved as compressed feature files under the language-specific `data/features/` directory;
- recorded in separate `train.jsonl` and `validation.jsonl` manifests; and
- used to build a language-specific `vocab.json` from training text only.

No English records are used by the Telugu model and no Telugu records are used by the English model. Training is launched separately with the corresponding entry point and configuration:

```powershell
python English\\train.py --config English\\config.yaml
python Telugu\\train.py --config Telugu\\config.yaml
```

## Training

### English Model

```powershell
python English\train.py --config English\config.yaml
```

**Training configuration:**
- Batch size: 4
- Epochs: 10
- Learning rate: 0.0001
- Optimizer: AdamW with weight decay 0.01
- Hidden size: 192
- Transformer layers: 4
- Attention heads: 2

**During training:**
- Prints loss per epoch: `{"epoch": 1, "train_loss": X.XX, "validation_loss": Y.YY}`
- Saves checkpoints to `English/checkpoints/`:
  - `last.pt` — most recent epoch
  - `best.pt` — best validation loss
- After training completes:
  - Saves `Results/English/history.json`
  - Generates `Results/English/training_loss.png` and `validation_loss.png`

### Telugu Model

```powershell
python Telugu\train.py --config Telugu\config.yaml
```

Same training process, independent from English model.

## Inference and Audio Generation

### English Inference

Generate speech for all test sentences:

```powershell
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt
```

**What this generates:**
- `Results/English/test_generated_audios/E1.wav` through `E4.wav`
- `Results/English/E1_mel.npy` through `E4_mel.npy` (mel spectrograms)
- `Results/English/E1_variance.csv` through `E4_variance.csv` (duration, pitch, energy values)
- `Results/English/variance_adaptor_plots/duration/E1_duration.png` through `E4_duration.png`
- `Results/English/variance_adaptor_plots/pitch/E1_pitch.png` through `E4_pitch.png`
- `Results/English/variance_adaptor_plots/energy/E1_energy.png` through `E4_energy.png`

Generate speech for a single test sentence:

```powershell
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt --id E1
```

Generate speech for custom text:

```powershell
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt --text "Your custom sentence here."
```

### Telugu Inference

```powershell
python Telugu\inference.py --config Telugu\config.yaml --checkpoint Telugu\checkpoints\best.pt
```

Generates T1.wav through T4.wav and corresponding plots.

## Test Sentences

### English Test Sentences

| ID | Sentence |
|----|----------|
| E1 | The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings. |
| E2a | The orchestra's conduct during the ceremony was praised by the audience. |
| E2b | The researchers will conduct an experiment to evaluate the proposed method. |
| E3 | The rhyth m of the orchestra changed unexpectedly. |
| E4 | The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time. |

### Telugu Test Sentences

| ID | Sentence |
|----|----------|
| T1 | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. |
| T2a | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. |
| T2b | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. |
| T3 | పులి అడవిలో నిశ్శబ్దంగా నడిచింది. |
| T4 | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. |

## Implementation Details

### Architecture

FastSpeech 2 architecture from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2):

1. **Phoneme/Grapheme Encoder**: 4-layer transformer encoder (192 hidden, 2 heads)
2. **Variance Adaptor**: Predicts duration, pitch (F0), and energy
3. **Length Regulator**: Expands encoder output based on predicted/ground-truth durations
4. **Mel Decoder**: 4-layer transformer decoder (192 hidden, 2 heads)
5. **Mel Linear**: Projects to 80-dimensional mel spectrogram

### Key Modifications from Reference Implementation

1. **Grapheme tokenization**: Character-level instead of phoneme-level
2. **Uniform duration initialization**: Simple uniform frame distribution instead of MFA alignment (baseline approach)
3. **Griffin-Lim vocoder**: Integrated waveform synthesis without external neural vocoder
4. **Automatic plotting**: Loss curves and variance adaptor plots generated automatically
5. **Independent language pipelines**: Separate configs, vocabs, and checkpoints

### Loss Function

Multi-task loss combining:
- Mel spectrogram reconstruction loss (MSE)
- Duration prediction loss (log-space MSE)
- Pitch prediction loss (MSE)
- Energy prediction loss (MSE)

All losses use masking to handle variable-length sequences.

### Vocoder

**Griffin-Lim algorithm** for mel-to-waveform conversion:
- Iterative phase reconstruction
- 32 iterations by default
- No external model required
- Output normalized to [-1, 1] range
- Saved as 16-bit PCM WAV at 22050 Hz

## Results

After running the complete pipeline:

1. **Training curves**: `Results/{Language}/training_loss.png` and `validation_loss.png`
2. **Generated audio**: `Results/{Language}/test_generated_audios/*.wav`
3. **Variance adaptor plots**: Duration, pitch, energy visualizations per test sentence
4. **History JSON**: Training metrics in `Results/{Language}/history.json`

## Dataset Selection Policy

Both languages use deterministic selection:

1. Load source dataset split
2. Validate all records (text + audio must be present and decodable)
3. Measure actual audio duration
4. Sort records by ID for reproducibility
5. Shuffle indices with seed 1234
6. Accumulate training set up to 18,000 seconds (5 hours)
7. Remove selected training records
8. Accumulate validation set from remainder up to 3,600 seconds (1 hour)
9. No overlap between training and validation

If the dataset has insufficient data, a warning is printed showing the shortfall.

## Duration Alignment

This implementation uses **uniform duration distribution** as a baseline:
- Total mel frames divided by number of tokens (including BOS/EOS)
- Remainder frames distributed to the first few tokens
- FastSpeech 2 learns to adjust these during training

For production systems, Montreal Forced Aligner (MFA) or other forced alignment tools would provide better initial durations.

## Troubleshooting

### Out of Memory during Training

Reduce batch size in config.yaml:
```yaml
train:
  batch_size: 2  # or 1
```

### Slow Preprocessing

Reduce `num_workers` or process on a machine with more RAM:
```yaml
train:
  num_workers: 0  # single-threaded
```

### Dataset Download Fails

Check internet connection and HuggingFace access. Datasets are large (~several GB).

### Audio Quality Issues

Griffin-Lim produces intelligible but robotic speech. For higher quality:
1. Train longer (more epochs)
2. Use a neural vocoder (HiFi-GAN, WaveGlow)
3. Increase model capacity

## References

- **FastSpeech 2 Paper**: [Ren et al., 2020](https://arxiv.org/abs/2006.04558)
- **Reference Implementation**: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- **English Dataset**: [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- **Telugu Dataset**: [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)

## Citation

If using this code, please cite the original FastSpeech 2 paper:

```bibtex
@inproceedings{ren2020fastspeech2,
  title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## License

This implementation is for educational purposes as part of an assignment.

## Author

Implemented for the FastSpeech 2-Based English and Telugu Text-to-Speech Synthesis assignment (Deadline: 07-11-2026).

---

For questions or issues, please refer to the assignment guidelines or contact the course instructor.
