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

# src/flow_factory/rewards/ocr.py
"""
OCR Reward Model using PP-OCR (PaddleOCR), served OUT-OF-PROCESS.

The exact flow_grpo OCR stack (paddlepaddle==2.6.2 + paddleocr==2.9.1) hard-pins
``numpy<2`` and an older opencv, which conflicts with the training env (``ff``:
numpy>=2, torch 2.12, deepspeed). To keep ``ff`` pristine, PaddleOCR runs in a
DEDICATED conda env and this class talks to it over a stdin/stdout pipe (see
``_ocr_worker.py``). Nothing paddle/numpy<2 related is imported here.

Setup (per node), matching flow_grpo (https://github.com/yifan123/flow_grpo):
```bash
conda create -y -n ocr python=3.10
/opt/conda/envs/ocr/bin/pip install paddlepaddle==2.6.2 paddleocr==2.9.1 python-Levenshtein
# pre-download PP-OCR weights into ~/.paddleocr:
/opt/conda/envs/ocr/bin/python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)"
```
Override the worker interpreter with env var ``OCR_ENV_PYTHON`` or
``RewardArguments.ocr_env_python`` if the env lives elsewhere.
"""
import os
import io
import struct
import pickle
import threading
import subprocess
from typing import List, Optional

from accelerate import Accelerator
from PIL import Image
import numpy as np
import torch

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import *
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# Default interpreter for the isolated PaddleOCR env (see module docstring).
_DEFAULT_OCR_PYTHON = "/opt/conda/envs/ocr/bin/python"
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ocr_worker.py")
# Generous: a cold worker may (re)download PP-OCR weights on first ever run.
_HANDSHAKE_TIMEOUT_S = 600.0


def _write_msg(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def _recvall(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return b""
            raise EOFError(f"OCR worker: unexpected EOF (got {len(buf)} of {n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def _read_msg(stream):
    header = _recvall(stream, 4)
    if header == b"":
        return None
    (length,) = struct.unpack(">I", header)
    return pickle.loads(_recvall(stream, length))


def _encode_png(img) -> bytes:
    """PIL.Image or HxWxC ndarray -> lossless PNG bytes (RGB)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if not isinstance(img, Image.Image):
        raise TypeError(f"OCR reward expects PIL.Image or np.ndarray, got {type(img).__name__}")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


class OCRRewardModel(PointwiseRewardModel):
    required_fields = ("prompt", "image", "video")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        self._ocr_python = (
            os.environ.get("OCR_ENV_PYTHON")
            or getattr(config, "ocr_env_python", None)
            or _DEFAULT_OCR_PYTHON
        )
        if not os.path.exists(self._ocr_python):
            raise FileNotFoundError(
                f"OCR reward: python interpreter for the isolated PaddleOCR env not found at "
                f"{self._ocr_python!r}. Create it (see ocr.py docstring) or set OCR_ENV_PYTHON."
            )
        if not os.path.exists(_WORKER_SCRIPT):
            raise FileNotFoundError(f"OCR reward: worker script missing at {_WORKER_SCRIPT!r}")

        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        rank = getattr(accelerator, "local_process_index", 0)
        self._stderr_path = f"/tmp/ocr_worker_rank{rank}.log"
        self._start_worker()

    # ----------------------------- worker lifecycle -----------------------------
    def _start_worker(self) -> None:
        stderr_f = open(self._stderr_path, "wb")
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = env.get("OCR_OMP_NUM_THREADS", "4")  # bound CPU OCR threads per rank
        proc = subprocess.Popen(
            [self._ocr_python, "-u", _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_f,
            bufsize=0,
            env=env,
        )
        self._proc = proc

        # Wait for the "ready" handshake with a timeout (worker builds PaddleOCR first).
        result: dict = {}

        def _await_ready():
            try:
                result["msg"] = _read_msg(proc.stdout)
            except Exception as e:  # noqa: BLE001 -- surfaced below with worker stderr.
                result["exc"] = e

        t = threading.Thread(target=_await_ready, daemon=True)
        t.start()
        t.join(_HANDSHAKE_TIMEOUT_S)

        if t.is_alive() or result.get("msg") != {"status": "ready"}:
            self._kill()
            raise RuntimeError(
                f"OCR worker failed to become ready within {_HANDSHAKE_TIMEOUT_S:.0f}s "
                f"(interpreter={self._ocr_python}). Worker stderr tail:\n{self._stderr_tail()}"
            )
        logger.info(f"OCR reward worker ready (env python={self._ocr_python}, rank stderr={self._stderr_path})")

    def _stderr_tail(self, n_bytes: int = 4000) -> str:
        try:
            with open(self._stderr_path, "rb") as f:
                try:
                    f.seek(-n_bytes, os.SEEK_END)
                except OSError:
                    f.seek(0)
                return f.read().decode("utf-8", "replace")
        except FileNotFoundError:
            return "<no stderr captured>"

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        self._proc = None

    def _infer(self, images_png: List[bytes], prompts: List[str]) -> List[float]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError(
                    f"OCR worker is not running (exit code "
                    f"{None if self._proc is None else self._proc.returncode}). "
                    f"Worker stderr tail:\n{self._stderr_tail()}"
                )
            try:
                _write_msg(self._proc.stdin, {"cmd": "score", "images": images_png, "prompts": prompts})
                resp = _read_msg(self._proc.stdout)
            except (BrokenPipeError, EOFError) as e:
                self._kill()
                raise RuntimeError(
                    f"OCR worker died during inference ({type(e).__name__}: {e}). "
                    f"Worker stderr tail:\n{self._stderr_tail()}"
                ) from e

        if resp is None:
            raise RuntimeError(f"OCR worker closed the pipe. Worker stderr tail:\n{self._stderr_tail()}")
        if "error" in resp:
            raise RuntimeError(f"OCR worker reported an error:\n{resp['error']}")
        rewards = resp["rewards"]
        if len(rewards) != len(prompts):
            raise RuntimeError(
                f"OCR worker returned {len(rewards)} rewards for {len(prompts)} prompts"
            )
        return rewards

    # ------------------------------- scoring -------------------------------
    def _compute_scores_batch(
        self,
        prompt: List[str],
        image: List[Image.Image],
    ) -> torch.Tensor:
        """Compute OCR reward for a batch of image-prompt pairs (via the OCR worker)."""
        if len(prompt) != len(image):
            raise ValueError(
                f"OCR reward: prompt/image length mismatch: {len(prompt)} vs {len(image)}"
            )
        if len(prompt) == 0:
            return torch.zeros(0, dtype=torch.float32)
        images_png = [_encode_png(img) for img in image]
        rewards = self._infer(images_png, list(prompt))
        return torch.tensor(rewards, dtype=torch.float32)

    def _compute_video_scores(
        self,
        prompt: List[str],
        video: List[List[Image.Image]],
        batch_size: int,
    ) -> torch.Tensor:
        """Mean OCR reward across all frames for each video (flat-reconstruct)."""
        frame_counts = [len(clip) for clip in video]
        flat_images = [frame for clip in video for frame in clip]
        flat_prompts = [p for p, n in zip(prompt, frame_counts) for _ in range(n)]

        all_scores = []
        for i in range(0, len(flat_images), batch_size):
            all_scores.append(
                self._compute_scores_batch(
                    flat_prompts[i:i + batch_size],
                    flat_images[i:i + batch_size],
                )
            )
        flat_scores = torch.cat(all_scores, dim=0) if all_scores else torch.zeros(0, dtype=torch.float32)
        scores = flat_scores.split(frame_counts)
        scores = torch.stack([s.mean() for s in scores])
        return scores

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
    ) -> RewardModelOutput:
        if not isinstance(prompt, list):
            prompt = [prompt]
        if image is not None and video is not None:
            raise ValueError("Only one of image or video can be provided.")

        batch_size = getattr(self.config, "batch_size", len(prompt)) or len(prompt)

        if video is not None:
            scores = self._compute_video_scores(prompt, video, batch_size)
        else:
            if image is None:
                raise ValueError("OCR reward requires either `image` or `video`.")
            # Chunk large eval batches to bound per-message payload / memory.
            chunks = [
                self._compute_scores_batch(prompt[i:i + batch_size], image[i:i + batch_size])
                for i in range(0, len(prompt), batch_size)
            ]
            scores = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, dtype=torch.float32)

        return RewardModelOutput(rewards=scores, extra_info={})

    def __del__(self):
        try:
            if self._proc is not None and self._proc.poll() is None:
                with self._lock:
                    try:
                        _write_msg(self._proc.stdin, {"cmd": "shutdown"})
                    except Exception:  # noqa: BLE001
                        pass
                self._kill()
        except Exception:  # noqa: BLE001 -- never raise from a finalizer.
            pass


def download_model():
    """Trigger a PP-OCR weight download inside the isolated env (parity with __main__)."""
    py = os.environ.get("OCR_ENV_PYTHON", _DEFAULT_OCR_PYTHON)
    subprocess.run(
        [py, "-c",
         "from paddleocr import PaddleOCR; "
         "PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False); "
         "print('PaddleOCR initialized successfully')"],
        check=True,
    )


if __name__ == "__main__":
    download_model()
