import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .config import load_config
from .features import load_features
from .manifest import read_jsonl
from .model import FastSpeech2, length_mask, masked_mse
from .text import encode, load_vocabulary


class SpeechDataset(Dataset):
    def __init__(self, records, vocabulary):
        self.records, self.vocabulary = records, vocabulary

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        features = load_features(record["feature_path"])
        boundaries = [0]
        for duration in record["duration"]:
            boundaries.append(boundaries[-1] + duration)
        def aggregate(name):
            values = features[name]
            return torch.tensor([float(values[start:min(end, len(values))].mean())
                                 if end > start and start < len(values) else 0.0
                                 for start, end in zip(boundaries[:-1], boundaries[1:])])
        return {"text": torch.tensor(encode(record["text"], self.vocabulary)),
                "duration": torch.tensor(record["duration"], dtype=torch.long),
                "mel": torch.from_numpy(features["mel"]),
                "pitch": aggregate("pitch"), "energy": aggregate("energy")}


def collate(items):
    text_lengths = torch.tensor([len(item["text"]) for item in items])
    mel_lengths = torch.tensor([len(item["mel"]) for item in items])
    max_text, max_mel = int(text_lengths.max()), int(mel_lengths.max())
    batch = {"text": torch.zeros(len(items), max_text, dtype=torch.long),
             "duration": torch.ones(len(items), max_text, dtype=torch.long),
             "mel": torch.zeros(len(items), max_mel, items[0]["mel"].size(1)),
             "pitch": torch.zeros(len(items), max_text), "energy": torch.zeros(len(items), max_text),
             "text_lengths": text_lengths, "mel_lengths": mel_lengths}
    for index, item in enumerate(items):
        text_len, mel_len = len(item["text"]), len(item["mel"])
        batch["text"][index, :text_len] = item["text"]
        batch["duration"][index, :len(item["duration"])] = item["duration"]
        batch["mel"][index, :mel_len] = item["mel"]
        batch["pitch"][index, :len(item["pitch"])] = item["pitch"]
        batch["energy"][index, :len(item["energy"])] = item["energy"]
    return batch


def epoch(model, loader, optimizer, device, training, micro_batch_size=1, epoch_num=None):
    model.train(training); total = 0.0
    step_size = micro_batch_size if device.type == "cpu" else None
    phase = "Train" if training else "Valid"
    for batch_idx, batch in enumerate(loader, 1):
        batch_size = len(batch["text"])
        cur_step = step_size if step_size is not None and batch_size > 1 else batch_size
        if training:
            optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for start in range(0, batch_size, cur_step):
            end = min(start + cur_step, batch_size)
            scale = (end - start) / batch_size

            text_lens = batch["text_lengths"][start:end].to(device)
            max_t = int(text_lens.max())
            text = batch["text"][start:end, :max_t].to(device)
            duration = batch["duration"][start:end, :max_t].to(device)
            pitch = batch["pitch"][start:end, :max_t].to(device)
            energy = batch["energy"][start:end, :max_t].to(device)

            mel_lens = batch["mel_lengths"][start:end].to(device)
            max_m = int(mel_lens.max())
            mel_tgt = batch["mel"][start:end, :max_m].to(device)

            output = model(text, text_lens, duration)
            text_mask = length_mask(text_lens, text.size(1))

            # Mel reconstruction loss (align lengths safely)
            mel_pred = output["mel"]
            min_mel_len = min(mel_pred.size(1), mel_tgt.size(1))
            mel_mask = length_mask(mel_lens, min_mel_len)
            loss = masked_mse(mel_pred[:, :min_mel_len], mel_tgt[:, :min_mel_len], mel_mask)

            # Variance adaptor losses
            dur_target = duration.float().clamp(min=1.0).log()
            min_text_len = min(output["log_duration"].size(1), dur_target.size(1), text.size(1))
            t_mask = text_mask[:, :min_text_len]
            loss = loss + masked_mse(output["log_duration"][:, :min_text_len], dur_target[:, :min_text_len], t_mask)
            loss = loss + masked_mse(output["pitch"][:, :min_text_len], pitch[:, :min_text_len], t_mask)
            loss = loss + masked_mse(output["energy"][:, :min_text_len], energy[:, :min_text_len], t_mask)

            if training:
                (loss * scale).backward()
            batch_loss += float(loss.detach()) * scale

        if training:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        log_interval = 25 if training else 50
        if batch_idx % log_interval == 0 or batch_idx == len(loader):
            ep_str = f"Epoch {epoch_num}" if epoch_num is not None else "Epoch"
            print(f"[{phase}] {ep_str} | Batch {batch_idx}/{len(loader)} ({batch_idx/len(loader)*100:.1f}%) | Loss: {batch_loss:.2f}", flush=True)
        total += batch_loss
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.config).resolve().parent
    results_dir = root.parent / config["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    vocabulary = load_vocabulary(root / "data" / "vocab.json")
    train_set = SpeechDataset(read_jsonl(root / "data" / "train.jsonl"), vocabulary)
    valid_set = SpeechDataset(read_jsonl(root / "data" / "validation.jsonl"), vocabulary)
    settings = {"batch_size": config["train"]["batch_size"], "collate_fn": collate, "num_workers": config["train"]["num_workers"]}
    train_loader = DataLoader(train_set, shuffle=True, **settings)
    valid_loader = DataLoader(valid_set, **settings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSpeech2(len(vocabulary), config["audio"]["n_mels"], **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=0.01)

    start_epoch = 1
    history = []
    best = float("inf")

    last_ckpt_path = checkpoint_dir / "last.pt"
    history_file = results_dir / "history.json"
    if args.resume and last_ckpt_path.exists():
        state = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0) + 1
        print(f"Resumed from checkpoint: last epoch was {start_epoch - 1}")
        if history_file.exists():
            try:
                history = json.loads(history_file.read_text(encoding="utf-8"))
                valid_losses = [h["validation_loss"] for h in history if "validation_loss" in h]
                if valid_losses:
                    best = min(valid_losses)
            except Exception:
                pass

    for current_epoch in range(start_epoch, config["train"]["epochs"] + 1):
        print(f"\n--- Starting Epoch {current_epoch}/{config['train']['epochs']} ---", flush=True)
        train_loss = epoch(model, train_loader, optimizer, device, True, epoch_num=current_epoch)
        with torch.no_grad():
            validation_loss = epoch(model, valid_loader, optimizer, device, False, epoch_num=current_epoch)
        row = {"epoch": current_epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": current_epoch}
        torch.save(state, checkpoint_dir / "last.pt")
        if validation_loss < best:
            best = validation_loss
            torch.save(state, checkpoint_dir / "best.pt")

        # Incrementally persist history and update loss plots each epoch
        history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        try:
            from .plots import plot_loss_curves
            plot_loss_curves(history_file, results_dir)
            print(f"Loss curves updated in {results_dir}", flush=True)
        except Exception as exc:
            print(f"Warning: could not generate loss plots: {exc}", flush=True)



if __name__ == "__main__":
    main()
