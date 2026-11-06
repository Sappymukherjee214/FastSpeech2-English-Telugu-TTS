import json
import unicodedata
from collections import Counter
from pathlib import Path

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(str(text).split()))


def tokenize(text: str) -> list[str]:
    return list(normalize_text(text))


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    symbols = [PAD, UNK, BOS, EOS] + sorted(counts)
    return {symbol: index for index, symbol in enumerate(symbols)}


def encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary[BOS], *(vocabulary.get(x, vocabulary[UNK]) for x in tokenize(text)), vocabulary[EOS]]


def save_vocabulary(value: dict[str, int], path: str | Path) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocabulary(path: str | Path) -> dict[str, int]:
    return {k: int(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}
