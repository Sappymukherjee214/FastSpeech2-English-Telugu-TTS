# Quick Start Guide

## Complete Pipeline Execution

### Step 1: Environment Setup (One-time)

```powershell
# Install dependencies
python -m pip install -r requirements.txt

# Verify installation
python -m pytest -q
```

Expected: 11 tests pass

### Step 2: English Pipeline

```powershell
# 2a. Inspect dataset (optional)
python English\preprocess.py --config English\config.yaml --inspect 8

# 2b. Preprocess (creates manifests, features, vocabulary)
python English\preprocess.py --config English\config.yaml

# 2c. Train model (~30-60 min on GPU, longer on CPU)
python English\train.py --config English\config.yaml

# 2d. Generate speech for all test sentences
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt
```

### Step 3: Telugu Pipeline

```powershell
# 3a. Inspect dataset (optional)
python Telugu\preprocess.py --config Telugu\config.yaml --inspect 8

# 3b. Preprocess
python Telugu\preprocess.py --config Telugu\config.yaml

# 3c. Train model
python Telugu\train.py --config Telugu\config.yaml

# 3d. Generate speech for all test sentences
python Telugu\inference.py --config Telugu\config.yaml --checkpoint Telugu\checkpoints\best.pt
```

## Dataset selection and preprocessing policy

Both language pipelines use deterministic, duration-based subsampling with seed `1234`:

- English uses `ai4bharat/Svarah`; Telugu uses `SPRINGLab/IndicTTS_Telugu`.
- Each pipeline validates transcript text and decodable audio, skips audio shorter than 0.3 seconds, and measures actual audio duration.
- Records are sorted by stable ID, shuffled with seed `1234`, and accumulated as complete utterances to approximately 18,000 seconds (5 hours) for training.
- Validation is accumulated to approximately 3,600 seconds (1 hour) only from records remaining after training selection; no train/validation record IDs overlap.
- Selected audio is resampled to 22,050 Hz and used to extract 80-bin mel spectrograms, pitch, and energy. Uniform frame durations are assigned across grapheme tokens (including BOS/EOS), and features/manifests/vocabulary are saved separately under `English/data/` and `Telugu/data/`.

Verified current manifests:

- English: 18,021.95 seconds train (3,392 records), 3,609.39 seconds validation (721 records), zero ID overlap.
- Telugu: 18,013.36 seconds train (891 records), 3,614.00 seconds validation (162 records), zero ID overlap.

## Expected Results

After completing both pipelines, you will have:

### English Results (`Results/English/`)
- `training_loss.png` - Training loss curve
- `validation_loss.png` - Validation loss curve
- `history.json` - Training metrics per epoch
- `test_generated_audios/E1.wav` through `E4.wav` - Generated speech
- `variance_adaptor_plots/duration/E1_duration.png` through `E4_duration.png`
- `variance_adaptor_plots/pitch/E1_pitch.png` through `E4_pitch.png`
- `variance_adaptor_plots/energy/E1_energy.png` through `E4_energy.png`

### Telugu Results (`Results/Telugu/`)
- Same structure as English, with T1-T4 instead of E1-E4

### Checkpoints
- `English/checkpoints/best.pt` - Best English model
- `English/checkpoints/last.pt` - Last English checkpoint
- `Telugu/checkpoints/best.pt` - Best Telugu model
- `Telugu/checkpoints/last.pt` - Last Telugu checkpoint

## Training Time Estimates

| Hardware | Preprocessing | Training (10 epochs) |
|----------|--------------|---------------------|
| GPU (CUDA) | 10-20 min | 30-60 min |
| CPU only | 20-40 min | 2-4 hours |

## Disk Space Requirements

| Component | Size |
|-----------|------|
| Datasets (cached) | ~3-5 GB |
| Features (preprocessed) | ~2-3 GB |
| Checkpoints (both languages) | ~20-30 MB |
| Generated audio & plots | ~50-100 MB |
| **Total** | ~6-9 GB |

## Quick Commands Reference

### Preprocess specific dataset
```powershell
python English\preprocess.py --config English\config.yaml --dataset ai4bharat/Svarah
```

### Generate single test sentence
```powershell
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt --id E1
```

### Generate custom text
```powershell
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt --text "Your text here"
```

### Skip audio generation (mel only)
```powershell
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt --no-audio
```

### Skip variance plots
```powershell
python English\inference.py --config English\config.yaml --checkpoint English\checkpoints\best.pt --no-plots
```

## Troubleshooting

### Issue: Out of memory during training
**Solution**: Edit `English/config.yaml` or `Telugu/config.yaml`:
```yaml
train:
  batch_size: 2  # Reduce from 4 to 2 or 1
```

### Issue: Dataset download timeout
**Solution**: Retry the preprocessing command. HuggingFace datasets auto-resume downloads.

### Issue: No CUDA/GPU detected
**Solution**: Training will use CPU automatically. It's slower but works.

### Issue: Telugu characters not rendering in plots
**Solution**: Install Telugu fonts on Windows:
- Settings → Time & Language → Language → Add Telugu language pack

### Issue: ImportError for datasets or librosa
**Solution**: Reinstall requirements:
```powershell
python -m pip install --upgrade -r requirements.txt
```

## Testing Individual Components

```powershell
# Test specific module
python -m pytest tests/test_pipeline.py::TestText -v

# Test with verbose output
python -m pytest -v

# Test and show print statements
python -m pytest -s
```

## Assignment Deliverables Checklist

- [x] Complete code in GitHub repository
- [x] `English/` with preprocess.py, train.py, inference.py, config.yaml
- [x] `Telugu/` with preprocess.py, train.py, inference.py, config.yaml
- [x] `common/` with shared implementation
- [x] `Results/English/` with loss plots, audio, variance plots
- [x] `Results/Telugu/` with loss plots, audio, variance plots
- [x] `English/checkpoints/best.pt` trained model
- [x] `Telugu/checkpoints/best.pt` trained model
- [x] `README.md` with implementation details
- [x] `requirements.txt` with dependencies
- [x] Technical report (PDF) in `Report/Saptarshi_Mukherjee_FastSpeech2_Technical_Report.pdf`

## Next Steps

1. Re-run either language pipeline when new data or training settings are needed.
2. Listen to the generated audio and inspect the plots in `Results/`.
3. Read `Report/Saptarshi_Mukherjee_FastSpeech2_Technical_Report.pdf` for the implementation, experiment, and analysis details.

## Report Structure (Template)

```
1. Introduction
   - Objective and scope
   - Reference paper and implementation

2. Implementation Approach
   - Architecture overview
   - Modifications from reference implementation
   - Grapheme tokenization rationale

3. Dataset Preparation
   - Svarah (English): selection strategy, statistics
   - IndicTTS Telugu: selection strategy, statistics
   - Train/validation split methodology
   - Duration alignment approach

4. Training Configuration
   - Hyperparameters for both languages
   - Hardware used
   - Training time

5. Results and Analysis
   - Training/validation loss curves
   - Convergence behavior
   - Generated speech quality per test sentence
   - Variance adaptor analysis (duration, pitch, energy)
   - Observations on challenging cases (E3, E4, T1, T4)

6. Comparative Analysis
   - English vs Telugu model performance
   - Dataset differences impact
   - Character set and vocabulary size effects

7. Limitations and Future Work
   - Griffin-Lim vocoder quality
   - Uniform duration baseline vs. MFA
   - Model capacity considerations
   - Suggested improvements

8. Reproducibility
   - Environment and dependencies
   - Commands to reproduce results
   - Random seed documentation

9. Conclusion

10. References
```

Good luck with the assignment!
