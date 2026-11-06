import numpy as np
import pytest
import torch

from common.config import load_config
from common.model import FastSpeech2, VarianceAdaptor, length_mask, masked_mse
from common.plots import plot_loss_curves, plot_variance_adaptor
from common.text import (
    BOS,
    EOS,
    PAD,
    UNK,
    build_vocabulary,
    encode,
    normalize_text,
    tokenize,
)
from common.vocoder import griffin_lim_vocoder


class TestText:
    def test_normalize_text(self):
        assert normalize_text("  hello   world  ") == "hello world"
        assert normalize_text("తెలుగు") == "తెలుగు"

    def test_tokenize(self):
        tokens = tokenize("hello")
        assert tokens == ["h", "e", "l", "l", "o"]
        telugu_tokens = tokenize("నమస్తే")
        assert len(telugu_tokens) > 0

    def test_build_and_encode_vocabulary(self):
        texts = ["hello", "world"]
        vocab = build_vocabulary(texts)
        assert PAD in vocab
        assert UNK in vocab
        assert BOS in vocab
        assert EOS in vocab
        assert vocab[PAD] == 0

        encoded = encode("hello", vocab)
        assert encoded[0] == vocab[BOS]
        assert encoded[-1] == vocab[EOS]
        assert len(encoded) == len("hello") + 2


class TestModel:
    def test_length_mask(self):
        lengths = torch.tensor([3, 5])
        mask = length_mask(lengths, 6)
        assert mask.shape == (2, 6)
        assert mask[0, 2].item() is False
        assert mask[0, 3].item() is True
        assert mask[1, 4].item() is False
        assert mask[1, 5].item() is True

    def test_fastspeech2_shapes(self):
        vocab_size = 50
        n_mels = 80
        batch_size = 2
        seq_len = 10
        model = FastSpeech2(vocab_size=vocab_size, n_mels=n_mels, hidden=64, layers=2, heads=2)
        model.eval()

        text = torch.randint(1, vocab_size, (batch_size, seq_len))
        lengths = torch.tensor([8, 10])

        with torch.no_grad():
            output = model(text, lengths)

        assert "mel" in output
        assert "pitch" in output
        assert "energy" in output
        assert "duration" in output
        assert "log_duration" in output
        assert output["mel"].size(-1) == n_mels
        assert output["pitch"].shape == (batch_size, seq_len)
        assert output["energy"].shape == (batch_size, seq_len)
        assert output["duration"].shape == (batch_size, seq_len)

    def test_masked_mse(self):
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 10.0]])
        mask = torch.tensor([[False, False, True]])
        loss = masked_mse(pred, target, mask)
        assert loss.item() == 0.0


class TestVocoder:
    def test_griffin_lim_vocoder(self):
        config = {
            "audio": {
                "sample_rate": 22050,
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 80,
                "f_min": 0,
                "f_max": 8000,
            }
        }
        # Fake mel: [frames, n_mels]
        fake_mel = np.random.randn(50, 80).astype(np.float32)
        audio = griffin_lim_vocoder(fake_mel, config, n_iter=8)
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0


class TestConfig:
    def test_english_config(self):
        config = load_config("English/config.yaml")
        assert config["language"] == "English"
        assert config["audio"]["n_mels"] == 80
        assert config["targets"]["train_seconds"] == 18000

    def test_telugu_config(self):
        config = load_config("Telugu/config.yaml")
        assert config["language"] == "Telugu"
        assert config["audio"]["n_mels"] == 80
        assert config["targets"]["train_seconds"] == 18000


class TestPlots:
    def test_plot_variance_adaptor(self, tmp_path):
        config = {
            "audio": {
                "sample_rate": 22050,
                "hop_length": 256,
            }
        }
        seq_len = 6
        output = {
            "duration": torch.tensor([[2, 3, 4, 2, 3, 2]]),
            "pitch": torch.randn(1, seq_len),
            "energy": torch.randn(1, seq_len),
            "log_duration": torch.randn(1, seq_len),
        }
        plot_variance_adaptor("E1", "test", output, config, str(tmp_path))

        assert (tmp_path / "variance_adaptor_plots" / "duration" / "E1_duration.png").exists()
        assert (tmp_path / "variance_adaptor_plots" / "pitch" / "E1_pitch.png").exists()
        assert (tmp_path / "variance_adaptor_plots" / "energy" / "E1_energy.png").exists()

    def test_plot_loss_curves(self, tmp_path):
        import json
        history = [
            {"epoch": 1, "train_loss": 5.0, "validation_loss": 4.5},
            {"epoch": 2, "train_loss": 3.5, "validation_loss": 3.2},
            {"epoch": 3, "train_loss": 2.1, "validation_loss": 2.0},
        ]
        history_path = tmp_path / "history.json"
        history_path.write_text(json.dumps(history), encoding="utf-8")

        plot_loss_curves(history_path, tmp_path)
        assert (tmp_path / "training_loss.png").exists()
        assert (tmp_path / "validation_loss.png").exists()
