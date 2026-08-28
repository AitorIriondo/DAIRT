#!/usr/bin/env python3
"""Create exact SAM 3 C16 token-ID chunks without PyTorch.

This is a dependency-light port of SAM 3's VE/CLIP tokenizer.  It deliberately
keeps the two text-normalization dependencies (``ftfy`` and ``regex``) because
replacing either changes token IDs for valid Unicode prompts (pip install ftfy regex).
"""

from __future__ import annotations

import argparse
from array import array
import gzip
import html
import json
from pathlib import Path
import string
import sys
from typing import Iterable

try:
    import ftfy
    import regex as re
except ImportError as error:  # pragma: no cover - exercised on a bare target
    raise SystemExit(
        "tokenizer dependencies are missing; install ftfy and regex from "
        "third_party/wheelhouse before running this command"
    ) from error


CONTEXT_LENGTH = 32
BUCKET = 16


def bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(value) for value in codepoints)))


def pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


class Sam3Tokenizer:
    def __init__(self, bpe_path: Path) -> None:
        with gzip.open(bpe_path, "rt", encoding="utf-8") as handle:
            merges = handle.read().split("\n")[1 : 49152 - 256 - 2 + 1]
        merge_pairs = [tuple(item.split()) for item in merges if item]
        self.byte_encoder = bytes_to_unicode()
        vocabulary = list(self.byte_encoder.values())
        vocabulary += [item + "</w>" for item in vocabulary]
        vocabulary += ["".join(item) for item in merge_pairs]
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocabulary += special_tokens
        self.encoder = dict(zip(vocabulary, range(len(vocabulary))))
        self.ranks = dict(zip(merge_pairs, range(len(merge_pairs))))
        self.cache = {item: item for item in special_tokens}
        special = "|".join(special_tokens)
        self.pattern = re.compile(
            special + r"|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            re.IGNORECASE,
        )
        self.start_id = self.encoder[special_tokens[0]]
        self.end_id = self.encoder[special_tokens[1]]

    def _bpe(self, token: str) -> str:
        cached = self.cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        candidates = pairs(word)
        if not candidates:
            return token + "</w>"
        while True:
            first, second = min(
                candidates, key=lambda item: self.ranks.get(item, float("inf"))
            )
            if (first, second) not in self.ranks:
                break
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    match = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:match])
                index = match
                if index + 1 < len(word) and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            candidates = pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    @staticmethod
    def _clean(text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text)).strip()
        return re.sub(r"\s+", " ", text).strip().lower()

    def encode(self, text: str) -> list[int]:
        result: list[int] = [self.start_id]
        for token in re.findall(self.pattern, self._clean(text)):
            encoded = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            result.extend(self.encoder[item] for item in self._bpe(encoded).split(" "))
        result.append(self.end_id)
        if len(result) > CONTEXT_LENGTH:
            result = result[:CONTEXT_LENGTH]
            result[-1] = self.end_id
        return result


def chunked(values: list[str], size: int) -> Iterable[tuple[int, list[str]]]:
    for offset in range(0, len(values), size):
        yield offset, values[offset : offset + size]


def write_int32(path: Path, values: Iterable[int]) -> None:
    payload = array("i", values)
    if payload.itemsize != 4:
        raise RuntimeError("this Python build does not provide a 32-bit C int")
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def parse_classes(args: argparse.Namespace) -> list[str]:
    values = list(args.class_name or [])
    if args.classes_json is not None:
        loaded = json.loads(args.classes_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError("--classes-json must contain a JSON array of strings")
        values.extend(loaded)
    if not values or any(not item.strip() for item in values):
        raise ValueError("at least one non-empty class string is required")
    return values


def main() -> int:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Tokenize arbitrary SAM 3 class strings into C16 chunks")
    parser.add_argument("--class", dest="class_name", action="append", help="Class text; repeat as needed")
    parser.add_argument("--classes-json", type=Path, help="JSON array of class strings")
    parser.add_argument(
        "--bpe",
        type=Path,
        default=script_root / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    classes = parse_classes(args)
    tokenizer = Sam3Tokenizer(args.bpe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, object]] = []
    for chunk_index, (offset, texts) in enumerate(chunked(classes, BUCKET)):
        rows: list[int] = []
        for text in texts:
            tokens = tokenizer.encode(text)
            rows.extend(tokens + [0] * (CONTEXT_LENGTH - len(tokens)))
        rows.extend([0] * ((BUCKET - len(texts)) * CONTEXT_LENGTH))
        filename = f"token_ids_{chunk_index:04d}.i32le.bin"
        write_int32(args.output_dir / filename, rows)
        chunks.append(
            {
                "path": filename,
                "offset": offset,
                "valid": len(texts),
                "capacity": BUCKET,
                "shape": [BUCKET, CONTEXT_LENGTH],
                "dtype": "int32_little_endian",
                "classes": texts,
            }
        )
    plan = {
        "schema_version": "latticeq.orin.text_chunks.v1",
        "route": "text_c16",
        "class_count": len(classes),
        "chunks": chunks,
        "merge": "discard padded rows and preserve original class order",
    }
    (args.output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"complete": True, "classes": len(classes), "chunks": len(chunks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

