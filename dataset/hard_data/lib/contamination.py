"""SQLite-backed benchmark contamination index."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Iterable

from PIL import Image


class ContaminationIndex:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS texts (
                exact_hash TEXT NOT NULL,
                simhash INTEGER NOT NULL,
                normalized TEXT NOT NULL,
                source TEXT NOT NULL,
                location TEXT NOT NULL,
                UNIQUE(exact_hash, source, location)
            );
            CREATE INDEX IF NOT EXISTS idx_text_exact ON texts(exact_hash);
            CREATE INDEX IF NOT EXISTS idx_text_join
                ON texts(simhash, source, location);
            CREATE TABLE IF NOT EXISTS text_bands (
                band_index INTEGER NOT NULL,
                band_value INTEGER NOT NULL,
                simhash INTEGER NOT NULL,
                source TEXT NOT NULL,
                location TEXT NOT NULL,
                UNIQUE(band_index, band_value, simhash, source, location)
            );
            CREATE INDEX IF NOT EXISTS idx_text_band
                ON text_bands(band_index, band_value);
            CREATE TABLE IF NOT EXISTS text_ngrams (
                gram_hash INTEGER NOT NULL,
                simhash INTEGER NOT NULL,
                source TEXT NOT NULL,
                location TEXT NOT NULL,
                UNIQUE(gram_hash, simhash, source, location)
            );
            CREATE INDEX IF NOT EXISTS idx_text_ngram
                ON text_ngrams(gram_hash);
            CREATE TABLE IF NOT EXISTS images (
                sha256 TEXT NOT NULL,
                dhash INTEGER NOT NULL,
                source TEXT NOT NULL,
                location TEXT NOT NULL,
                UNIQUE(sha256, source, location)
            );
            CREATE INDEX IF NOT EXISTS idx_image_sha ON images(sha256);
            """
        )
        text_count = self.connection.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
        band_count = self.connection.execute("SELECT COUNT(*) FROM text_bands").fetchone()[0]
        if text_count and not band_count:
            for simhash, source, location in self.connection.execute(
                "SELECT simhash, source, location FROM texts"
            ):
                self._add_bands(simhash, source=source, location=location)
            self.connection.commit()
        ngram_count = self.connection.execute("SELECT COUNT(*) FROM text_ngrams").fetchone()[0]
        if text_count and not ngram_count:
            batch: list[tuple[int, int, str, str]] = []
            for normalized, simhash, source, location in self.connection.execute(
                "SELECT normalized, simhash, source, location FROM texts"
            ):
                batch.extend(
                    (gram_hash, simhash, source, location)
                    for gram_hash in token_ngram_hashes(normalized)
                )
                if len(batch) >= 10000:
                    self._insert_ngram_batch(batch)
                    batch.clear()
            if batch:
                self._insert_ngram_batch(batch)
            self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def add_text(self, text: str, *, source: str, location: str) -> None:
        normalized = normalize_text(text)
        if len(normalized) < 8:
            return
        exact = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        simhash = text_simhash(normalized)
        self.connection.execute(
            "INSERT OR IGNORE INTO texts(exact_hash, simhash, normalized, source, location) "
            "VALUES (?, ?, ?, ?, ?)",
            (exact, simhash, normalized, source, location),
        )
        self._add_bands(simhash, source=source, location=location)
        self._add_ngrams(normalized, simhash=simhash, source=source, location=location)

    def add_image(self, path: Path, *, source: str, location: str) -> None:
        sha = _file_sha256(path)
        dhash = image_dhash(path)
        self.connection.execute(
            "INSERT OR IGNORE INTO images(sha256, dhash, source, location) VALUES (?, ?, ?, ?)",
            (sha, dhash, source, location),
        )

    def text_matches(
        self,
        text: str,
        *,
        max_hamming: int = 14,
        min_jaccard: float = 0.5,
    ) -> list[dict[str, object]]:
        if not 0 <= min_jaccard <= 1:
            raise ValueError(f"expected min_jaccard in [0,1], got {min_jaccard!r}")
        normalized = normalize_text(text)
        exact = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        rows = self.connection.execute(
            "SELECT source, location, normalized FROM texts WHERE exact_hash=?", (exact,)
        ).fetchall()
        if rows:
            return [
                {"kind": "exact", "source": row[0], "location": row[1], "text": row[2]}
                for row in rows
            ]
        target = text_simhash(normalized)
        grams = token_ngram_hashes(normalized)
        if not grams:
            return []
        placeholders = ",".join("?" for _ in grams)
        candidates = self.connection.execute(
            "SELECT DISTINCT t.simhash, t.source, t.location, t.normalized "
            "FROM text_ngrams n JOIN texts t "
            "ON t.simhash=n.simhash AND t.source=n.source AND t.location=n.location "
            f"WHERE n.gram_hash IN ({placeholders})",
            grams,
        )
        matches: list[dict[str, object]] = []
        for simhash, source, location, stored in candidates:
            distance = (target ^ simhash).bit_count()
            jaccard = token_ngram_jaccard(normalized, stored, n=3)
            if distance <= max_hamming and jaccard >= min_jaccard:
                matches.append(
                    {
                        "kind": "simhash",
                        "distance": distance,
                        "jaccard": jaccard,
                        "source": source,
                        "location": location,
                        "text": stored,
                    }
                )
        return matches

    def _add_bands(self, simhash: int, *, source: str, location: str) -> None:
        for band_index, band_value in enumerate(simhash_bands(simhash)):
            self.connection.execute(
                "INSERT OR IGNORE INTO text_bands"
                "(band_index, band_value, simhash, source, location) VALUES (?, ?, ?, ?, ?)",
                (band_index, band_value, simhash, source, location),
            )

    def _add_ngrams(
        self,
        normalized: str,
        *,
        simhash: int,
        source: str,
        location: str,
    ) -> None:
        rows = [
            (gram_hash, simhash, source, location)
            for gram_hash in token_ngram_hashes(normalized)
        ]
        self._insert_ngram_batch(rows)

    def _insert_ngram_batch(self, rows: list[tuple[int, int, str, str]]) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO text_ngrams"
            "(gram_hash, simhash, source, location) VALUES (?, ?, ?, ?)",
            rows,
        )

    def image_matches(self, path: Path, *, max_hamming: int = 4) -> list[dict[str, object]]:
        sha = _file_sha256(path)
        rows = self.connection.execute(
            "SELECT source, location FROM images WHERE sha256=?", (sha,)
        ).fetchall()
        if rows:
            return [
                {"kind": "exact", "source": row[0], "location": row[1]} for row in rows
            ]
        target = image_dhash(path)
        matches: list[dict[str, object]] = []
        for dhash, source, location in self.connection.execute(
            "SELECT dhash, source, location FROM images"
        ):
            distance = (target ^ dhash).bit_count()
            if distance <= max_hamming:
                matches.append(
                    {
                        "kind": "dhash",
                        "distance": distance,
                        "source": source,
                        "location": location,
                    }
                )
        return matches

    def counts(self) -> dict[str, int]:
        texts = self.connection.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
        images = self.connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        return {"texts": int(texts), "images": int(images)}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def text_simhash(normalized: str) -> int:
    tokens = normalized.split()
    if not tokens:
        return 0
    grams = [" ".join(tokens[index : index + 3]) for index in range(max(1, len(tokens) - 2))]
    weights = [0] * 63
    for gram in grams:
        value = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(63):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def simhash_bands(value: int) -> tuple[int, ...]:
    """Split a 63-bit SimHash into 16 bands (15×4 bits + 1×3 bits)."""
    if value < 0 or value >= 1 << 63:
        raise ValueError(f"expected unsigned 63-bit SimHash, got {value!r}")
    bands = tuple((value >> (4 * index)) & 0xF for index in range(15))
    return bands + ((value >> 60) & 0x7,)


def token_ngram_jaccard(first: str, second: str, *, n: int = 3) -> float:
    if n < 1:
        raise ValueError(f"expected n >= 1, got {n!r}")

    def grams(value: str) -> set[tuple[str, ...]]:
        tokens = normalize_text(value).split()
        if not tokens:
            return set()
        width = min(n, len(tokens))
        return {
            tuple(tokens[index : index + width])
            for index in range(len(tokens) - width + 1)
        }

    left, right = grams(first), grams(second)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def token_ngram_hashes(value: str, *, n: int = 3) -> tuple[int, ...]:
    tokens = normalize_text(value).split()
    if not tokens:
        return ()
    width = min(n, len(tokens))
    grams = {
        " ".join(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    }
    return tuple(
        sorted(
            int.from_bytes(
                hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            & ((1 << 63) - 1)
            for gram in grams
        )
    )


def image_dhash(path: Path) -> int:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(grayscale.get_flattened_data())
    result = 0
    bit = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            if pixels[offset + column] > pixels[offset + column + 1]:
                result |= 1 << bit
            bit += 1
    # SQLite INTEGER is signed; clear the high bit without affecting useful Hamming distance.
    return result & ((1 << 63) - 1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_texts(
    index: ContaminationIndex,
    values: Iterable[tuple[str, str]],
    *,
    source: str,
) -> None:
    for location, text in values:
        index.add_text(text, source=source, location=location)
