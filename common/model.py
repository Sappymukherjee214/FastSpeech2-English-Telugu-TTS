from typing import Optional

import torch
from torch import nn


def length_mask(lengths, maximum):
    return torch.arange(maximum, device=lengths.device)[None, :] >= lengths[:, None]


class VarianceAdaptor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.duration = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.pitch = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.energy = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, hidden, lengths, durations: Optional[torch.Tensor] = None):
        log_duration = self.duration(hidden).squeeze(-1)
        pitch = self.pitch(hidden).squeeze(-1)
        energy = self.energy(hidden).squeeze(-1)
        if durations is None:
            durations = (torch.exp(log_duration) - 1).round().clamp(1, 200).long()
        expanded, expanded_lengths = [], []
        for values, repeats, length in zip(hidden, durations, lengths):
            v_val = values[:length]
            v_rep = repeats[:length]
            expanded.append(torch.repeat_interleave(v_val, v_rep, dim=0))
            expanded_lengths.append(int(v_rep.sum()))
        maximum = max(expanded_lengths)
        output = hidden.new_zeros(len(expanded), maximum, hidden.size(-1))
        for index, values in enumerate(expanded):
            output[index, :len(values)] = values
        return output, pitch, energy, log_duration, durations, torch.tensor(expanded_lengths, device=hidden.device)


class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, n_mels, hidden=192, layers=4, heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.adaptor = VarianceAdaptor(hidden)
        decoder_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, layers)
        self.mel = nn.Linear(hidden, n_mels)

    def forward(self, text, lengths, durations=None):
        source_mask = length_mask(lengths, text.size(1))
        hidden = self.encoder(self.embedding(text), src_key_padding_mask=source_mask)
        expanded, pitch, energy, log_duration, duration, mel_lengths = self.adaptor(hidden, lengths, durations)
        mel_mask = length_mask(mel_lengths, expanded.size(1))
        return {
            "mel": self.mel(self.decoder(expanded, src_key_padding_mask=mel_mask)),
            "pitch": pitch, "energy": energy, "log_duration": log_duration,
            "duration": duration, "mel_lengths": mel_lengths,
        }


def masked_mse(prediction, target, mask):
    valid = ~mask
    return ((prediction[valid] - target[valid]) ** 2).mean() if valid.any() else prediction.new_zeros(())
