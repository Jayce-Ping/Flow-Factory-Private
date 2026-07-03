# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/rewards/_ocr_worker.py
"""
Out-of-process PaddleOCR reward worker.

Runs in a DEDICATED conda env (default ``/opt/conda/envs/ocr``) that pins the
exact flow_grpo OCR stack (paddlepaddle==2.6.2 + paddleocr==2.9.1 + numpy<2),
which is incompatible with the training env (``ff``: numpy>=2 / torch 2.12).
The trainer-side client (``ocr.py``) spawns this script with the ``ocr`` env's
python and talks to it over stdin/stdout so the training env stays pristine.

This module is intentionally SELF-CONTAINED: it imports only the standard
library plus paddleocr / PIL / numpy / Levenshtein. It must NOT import anything
from ``flow_factory`` (the ``ocr`` env does not install the package).

Wire protocol (both directions): a 4-byte big-endian unsigned length header
followed by a ``pickle``-encoded payload.
  - client -> worker: {"cmd": "score", "images": [png_bytes, ...], "prompts": [str, ...]}
                      {"cmd": "shutdown"}
  - worker -> client: {"status": "ready"}                 (handshake, once)
                      {"rewards": [float, ...]}            (per score request)
                      {"error": "<message>"}              (fatal request error)
"""
import io
import struct
import sys
import traceback

import numpy as np
from PIL import Image

# IMPORT ORDER MATTERS (do not reorder): on this host (glibc 2.28 + a GCC-14 conda
# python), paddlepaddle 2.6.2 dlopens bundled libs that shadow the system zlib, after
# which pyclipper/scikit-image compiled extensions fail with
# "zlib.error: Error -2 ... inconsistent stream state". Importing them FIRST binds the
# correct zlib before paddle loads. See ocr.py setup notes.
import pyclipper  # noqa: F401  -- must precede paddleocr
import skimage  # noqa: F401  -- must precede paddleocr
from paddleocr import PaddleOCR
from Levenshtein import distance


def _recvall(stream, n: int) -> bytes:
    """Read exactly ``n`` bytes from a (possibly chunking) pipe, or return b'' at clean EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return b""
            raise EOFError(
                f"OCR worker: unexpected EOF (got {len(buf)} of {n} bytes)"
            )
        buf.extend(chunk)
    return bytes(buf)


def _read_msg(stream):
    header = _recvall(stream, 4)
    if header == b"":
        return None
    (length,) = struct.unpack(">I", header)
    import pickle

    return pickle.loads(_recvall(stream, length))


def _write_msg(stream, obj) -> None:
    import pickle

    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def _score_one(ocr: PaddleOCR, img_png: bytes, prompt: str) -> float:
    """Replicates the original OCRRewardModel scoring exactly (Levenshtein on quoted target)."""
    # Match the original: decode to an RGB ndarray before OCR.
    img = np.array(Image.open(io.BytesIO(img_png)).convert("RGB"))

    # Extract quoted target text (e.g. 'a sign saying "Hello World"' -> 'Hello World').
    parts = prompt.split('"')
    target_text = parts[1] if len(parts) >= 2 else prompt

    try:
        result = ocr.ocr(img, cls=False)
        recognized_text = (
            "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]])
            if result and result[0]
            else ""
        )
        recognized_text = recognized_text.replace(" ", "").lower()
        target_norm = target_text.replace(" ", "").lower()

        # Degenerate prompt with no target text: nothing to render -> no signal.
        if len(target_norm) == 0:
            return 0.0

        if target_norm in recognized_text:
            dist = 0
        else:
            dist = distance(recognized_text, target_norm)
        # Recognized many unrelated characters: cap at one-word penalty.
        if dist > len(target_norm):
            dist = len(target_norm)
    except Exception as e:  # noqa: BLE001 -- OCR inference can fail per-image; degrade to max penalty (as upstream did) rather than abort the whole eval.
        sys.stderr.write(f"[ocr_worker] OCR processing failed: {e}\n")
        sys.stderr.flush()
        target_norm = target_text.replace(" ", "").lower()
        if len(target_norm) == 0:
            return 0.0
        dist = len(target_norm)

    return 1.0 - dist / len(target_norm)


def main() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # Build the recognizer BEFORE the handshake so the client only sees "ready"
    # once OCR can actually serve requests. Any failure here propagates via a
    # non-zero exit + stderr, which the client surfaces.
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
    _write_msg(stdout, {"status": "ready"})

    while True:
        req = _read_msg(stdin)
        if req is None:
            break
        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd != "score":
            _write_msg(stdout, {"error": f"unknown cmd: {cmd!r}"})
            continue
        try:
            images = req["images"]
            prompts = req["prompts"]
            if len(images) != len(prompts):
                raise ValueError(
                    f"images/prompts length mismatch: {len(images)} vs {len(prompts)}"
                )
            rewards = [_score_one(ocr, im, p) for im, p in zip(images, prompts)]
            _write_msg(stdout, {"rewards": rewards})
        except Exception:  # noqa: BLE001 -- report a batch-fatal error back to the client, which will raise.
            _write_msg(stdout, {"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
