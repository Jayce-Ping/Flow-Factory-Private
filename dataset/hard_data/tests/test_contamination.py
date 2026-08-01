from pathlib import Path

from PIL import Image

from lib.contamination import ContaminationIndex, image_dhash, normalize_text


def test_text_exact_and_near_duplicate(tmp_path: Path) -> None:
    index = ContaminationIndex(tmp_path / "index.sqlite")
    try:
        original = "A red ceramic teapot with the exact word ORBIT on the side."
        index.add_text(original, source="bench", location="row:1")
        exact = index.text_matches("A RED ceramic teapot—with the exact word orbit on the side!")
        assert exact and exact[0]["kind"] == "exact"
        near = index.text_matches(
            "A red ceramic teapot with the exact word ORBIT painted on its side.",
        )
        assert near
        unrelated = index.text_matches(
            "Is the knife present with a wolf logo etched near the pivot screw?",
        )
        assert unrelated == []
    finally:
        index.close()


def test_image_exact_and_dhash(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (32, 32), (120, 30, 30)).save(first)
    Image.new("RGB", (64, 64), (120, 30, 30)).save(second)
    assert image_dhash(first) == image_dhash(second)

    index = ContaminationIndex(tmp_path / "index.sqlite")
    try:
        index.add_image(first, source="bench", location="first.png")
        matches = index.image_matches(second)
        assert matches and matches[0]["kind"] == "dhash"
    finally:
        index.close()


def test_normalize_text_is_unicode_and_punctuation_stable() -> None:
    assert normalize_text("  Café—ORBIT! ") == "café orbit"
