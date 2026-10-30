from __future__ import annotations

from pathlib import Path
from typing import Optional

MAX_HOTWORD_LENGTH = 80
MAX_HOTWORD_COUNT = 500
_RESERVED_PREFIXES = (":", "#", "@")


class HotwordValidationError(ValueError):
    """Raised when a hotword list cannot be stored or sent to the decoder."""


def parse_hotword_text(text: str) -> list[str]:
    """Parse editor text into a deduplicated hotword list.

    Blank lines are ignored. Internal whitespace, ``/``, and a leading
    ``:``, ``#``, or ``@`` are rejected because they break sherpa's hotword
    encoding. Duplicates keep the first occurrence and its case.

    Raises:
        HotwordValidationError: The text is not a safe hotword list.
    """
    seen: set[str] = set()
    kept: list[str] = []
    non_blank = 0
    for line_number, raw in enumerate(text.splitlines(), start=1):
        word = raw.strip()
        if not word:
            continue
        non_blank += 1
        if non_blank > MAX_HOTWORD_COUNT:
            raise HotwordValidationError(f"Use at most {MAX_HOTWORD_COUNT} hotwords.")
        if any(ch.isspace() for ch in word):
            raise HotwordValidationError(f"Line {line_number} contains a space. Put one hotword on each line.")
        if "/" in word:
            raise HotwordValidationError(f"Line {line_number} contains '/'. Hotwords cannot include that character.")
        if word[0] in _RESERVED_PREFIXES:
            raise HotwordValidationError(f"Line {line_number} cannot start with '{word[0]}'.")
        if len(word) > MAX_HOTWORD_LENGTH:
            raise HotwordValidationError(f"Line {line_number} is longer than {MAX_HOTWORD_LENGTH} characters.")
        if word in seen:
            continue
        seen.add(word)
        kept.append(word)
    return kept


def load_token_symbols(tokens_path: Path) -> set[str]:
    """Load the symbol column from a sherpa ``tokens.txt``."""
    symbols: set[str] = set()
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        symbol, separator, _token_id = line.rpartition(" ")
        if separator and symbol:
            symbols.add(symbol)
    return symbols


def export_bpe_vocab(model_path: Path, vocab_path: Path) -> None:
    """Write the two-column vocab sherpa-onnx reads from a SentencePiece model."""
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    lines: list[str] = []
    for index in range(processor.get_piece_size()):
        piece = processor.id_to_piece(index)
        if any(ch.isspace() for ch in piece):
            raise RuntimeError(f"BPE piece {piece!r} contains whitespace")
        lines.append(f"{piece} {processor.get_score(index)}")
    vocab_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def vocab_matches_tokens(vocab_path: Path, tokens_path: Path) -> Optional[str]:
    """Return an error when the exported vocab pieces differ from ``tokens.txt``."""
    pieces = {line.split(" ", 1)[0] for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    symbols = load_token_symbols(tokens_path)
    if pieces != symbols:
        return "Exported BPE vocab does not match tokens.txt"
    return None
