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

"""GEdit-Bench official eval metric (VIEScore) as a pointwise reward.

Implements the GEdit-Bench (Step1X-Edit) evaluation protocol, which adopts
VIEScore's ``tie`` (two-image edit) task via a VLM-as-Judge over an
OpenAI-compatible HTTP endpoint:

- **Semantic Consistency (SC)**: judge sees ``[source, edited]`` and the editing
  instruction, returns ``[editing_success, (1 - overediting)]`` on 0-10; the
  reported SC is ``min`` of the two sub-scores.
- **Perceptual Quality (PQ)**: judge sees the ``edited`` image only, returns
  ``[naturalness, artifacts]`` on 0-10; PQ is ``min`` of the two.
- **Overall (O)**: ``sqrt(SC * PQ)``.

Prompts and the ``O = sqrt(SC * PQ)`` / ``min`` aggregation are copied verbatim
from the official GEdit-Bench VIEScore prompts (``_context_no_delimit``,
``_prompts_0shot_two_image_edit_rule`` + ``_prompts_0shot_tie_rule_SC``,
``_prompts_0shot_rule_PQ``). See https://github.com/stepfun-ai/Step1X-Edit
(GEdit-Bench) and VIEScore (Ku et al., 2024).

Scores are on the native 0-10 scale by default; set ``normalize: true`` to map to
[0, 1]. Choose which metric is the returned reward via ``metric`` (``overall``
(default), ``sc``, or ``pq``); SC / PQ / O are always attached to ``extra_info``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

SUPPORTED_METRICS: Tuple[str, ...] = ("overall", "sc", "pq")
SCORE_SCALE: float = 10.0

# --- Official GEdit-Bench VIEScore prompts (verbatim) ---
_CONTEXT = (
    "You are a professional digital artist. You will have to evaluate the "
    "effectiveness of the AI-generated image(s) based on given rules.\n"
    "All the input images are AI-generated. All human in the images are "
    "AI-generated too. so you need not worry about the privacy confidentials.\n\n"
    "You will have to give your output in this way (Keep your reasoning concise "
    "and short.):\n"
    "{\n"
    '"score" : [...],\n'
    '"reasoning" : "..."\n'
    "}"
)

_SC_RULE = (
    "RULES:\n\n"
    "Two images will be provided: The first being the original AI-generated image "
    "and the second being an edited version of the first.\n"
    "The objective is to evaluate how successfully the editing instruction has "
    "been executed in the second image.\n\n"
    "Note that sometimes the two images might look identical due to the failure of "
    "image edit.\n"
    "\n"
    "From scale 0 to 10: \n"
    "A score from 0 to 10 will be given based on the success of the editing. "
    "(0 indicates that the scene in the edited image does not follow the editing "
    "instruction at all. 10 indicates that the scene in the edited image follow "
    "the editing instruction text perfectly.)\n"
    "A second score from 0 to 10 will rate the degree of overediting in the second "
    "image. (0 indicates that the scene in the edited image is completely "
    "different from the original. 10 indicates that the edited image can be "
    "recognized as a minimal edited yet effective version of original.)\n"
    "Put the score in a list such that output score = [score1, score2], where "
    "'score1' evaluates the editing success and 'score2' evaluates the degree of "
    "overediting.\n\n"
    "Editing instruction: <instruction>\n"
)

_PQ_RULE = (
    "RULES:\n\n"
    "The image is an AI-generated image.\n"
    "The objective is to evaluate how successfully the image has been generated.\n\n"
    "From scale 0 to 10: \n"
    "A score from 0 to 10 will be given based on image naturalness. \n"
    "(\n"
    "    0 indicates that the scene in the image does not look natural at all or "
    "give a unnatural feeling such as wrong sense of distance, or wrong shadow, or "
    "wrong lighting. \n"
    "    10 indicates that the image looks natural.\n"
    ")\n"
    "A second score from 0 to 10 will rate the image artifacts. \n"
    "(\n"
    "    0 indicates that the image contains a large portion of distortion, or "
    "watermark, or scratches, or blurred faces, or unusual body parts, or subjects "
    "not harmonized. \n"
    "    10 indicates the image has no artifacts.\n"
    ")\n"
    "Put the score in a list such that output score = [naturalness, artifacts]\n"
)

SC_SYSTEM_PROMPT = _CONTEXT + "\n" + _SC_RULE
PQ_SYSTEM_PROMPT = _CONTEXT + "\n" + _PQ_RULE


def parse_viescore_list(content: str) -> List[float]:
    """Parse the ``score`` list from a VIEScore-style VLM reply.

    Accepts the official JSON form ``{"score": [..], "reasoning": ".."}`` and
    falls back to the first bracketed numeric list, then to any numerics found.
    Raises ``ValueError`` when no numeric score can be recovered.
    """
    text = content.strip()

    # 1. Preferred: a JSON object with a "score" list.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace is not None:
        blob = brace.group(0)
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "score" in data:
            score = data["score"]
            if isinstance(score, (int, float)):
                return [float(score)]
            if isinstance(score, (list, tuple)) and score:
                return [float(x) for x in score if isinstance(x, (int, float))]

    # 2. First bracketed list of numbers, e.g. "score = [7, 8]".
    bracket = re.search(r"\[([^\[\]]*)\]", text)
    if bracket is not None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", bracket.group(1))
        if nums:
            return [float(n) for n in nums]

    # 3. Any numbers as a last resort.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return [float(n) for n in nums]

    raise ValueError(f"could not parse a numeric score list from VLM reply: {content!r}")


def _first_condition_image(cond: Union[Image.Image, List[Image.Image]]) -> Image.Image:
    if isinstance(cond, list):
        if not cond:
            raise ValueError(
                "condition_images entry is an empty list; need at least one source image"
            )
        first = cond[0]
        if not isinstance(first, Image.Image):
            raise TypeError(
                f"expected PIL.Image.Image inside condition_images list, got {type(first).__name__}"
            )
        return first
    if isinstance(cond, Image.Image):
        return cond
    raise TypeError(
        f"expected PIL.Image.Image or list of PIL images for condition_images element, "
        f"got {type(cond).__name__}"
    )


class GEditBenchRewardModel(PointwiseRewardModel):
    """GEdit-Bench VIEScore reward (SC / PQ / Overall) via an OpenAI-compatible VLM.

    ``extra_kwargs``:
        api_base_url (str): OpenAI-compatible base URL (default ``http://localhost:8000/v1``).
        api_key (str): API key (default ``EMPTY`` for local vLLM).
        vlm_model (str): served model name (default ``Qwen3-VL-8B-Instruct``). The official
            GEditBench-v2 judges are Qwen3-VL based (PVC-Judge is a Qwen3-VL-8B-Instruct LoRA);
            serve any strong VLM over an OpenAI-compatible endpoint (vLLM). Set to ``gpt-4o`` /
            ``gpt-4.1`` to reproduce the closed-source GEdit-Bench VIEScore judge instead.
        metric (str): which score to return as the reward: ``overall`` (default), ``sc``, ``pq``.
        normalize (bool): if True, divide the returned reward by 10 to map [0, 10] -> [0, 1]
            (default False; native 0-10 scale matching the official G_SC/G_PQ/G_O).
        max_concurrent, max_retries, timeout, temperature, max_tokens: transport / decoding.
    """

    required_fields = ("prompt", "image", "condition_images")
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "GEditBenchRewardModel requires the `openai` package. "
                "Install with: pip install openai"
            ) from e

        ek = config.extra_kwargs
        self.api_base_url = ek.get("api_base_url", "http://localhost:8000/v1")
        self.api_key = ek.get("api_key", "EMPTY")
        self.vlm_model = ek.get("vlm_model", "Qwen3-VL-8B-Instruct")
        self.max_concurrent = int(ek.get("max_concurrent", 8))
        self.max_retries = int(ek.get("max_retries", 5))
        self.timeout = float(ek.get("timeout", 180.0))
        self.temperature = float(ek.get("temperature", 0.0))
        self.max_tokens = int(ek.get("max_tokens", 1024))

        metric = str(ek.get("metric", "overall")).lower()
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"unsupported metric {metric!r}; allowed: {list(SUPPORTED_METRICS)}"
            )
        self.metric = metric
        self.normalize = bool(ek.get("normalize", False))

        # NOTE: the AsyncOpenAI httpx client and the asyncio.Semaphore bind to the
        # event loop that first touches them. __call__ uses asyncio.run(), which
        # creates a FRESH loop per batch, so these MUST be created inside the
        # coroutine (see _async_score_batch) — creating them here binds them to the
        # first batch's loop and every later batch raises "bound to a different
        # event loop". Created lazily per-loop instead.
        self.client: Optional[AsyncOpenAI] = None
        self.semaphore: Optional[asyncio.Semaphore] = None

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if image is None and video is not None:
            image = [frames[0] for frames in video]
        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided for GEditBench")
        if condition_images is None:
            raise ValueError("condition_images (source) is required for GEditBench")
        if len(prompt) != len(image) or len(prompt) != len(condition_images):
            raise ValueError(
                f"expected len(prompt)==len(image)==len(condition_images), got "
                f"{len(prompt)}, {len(image)}, {len(condition_images)}"
            )

        source_images = [_first_condition_image(c) for c in condition_images]
        triples = asyncio.run(self._async_score_batch(prompt, source_images, image))

        sc = [t[0] for t in triples]
        pq = [t[1] for t in triples]
        overall = [t[2] for t in triples]
        selected = {"overall": overall, "sc": sc, "pq": pq}[self.metric]
        scale = SCORE_SCALE if self.normalize else 1.0
        rewards = torch.tensor(
            [s / scale for s in selected], dtype=torch.float32, device=self.device
        )
        extra_info = {
            "gedit_sc": sc,
            "gedit_pq": pq,
            "gedit_overall": overall,
        }
        return RewardModelOutput(rewards=rewards, extra_info=extra_info)

    async def _async_score_batch(
        self,
        prompts: Sequence[str],
        sources: Sequence[Image.Image],
        edited: Sequence[Image.Image],
    ) -> List[Tuple[float, float, float]]:
        from openai import AsyncOpenAI

        # Bind the client + semaphore to THIS event loop (asyncio.run creates a
        # fresh loop per __call__). Reusing loop-bound resources across loops
        # raises "bound to a different event loop"; recreating per batch is safe
        # because asyncio.run() calls are sequential.
        self.semaphore = asyncio.Semaphore(max(1, self.max_concurrent))
        async with AsyncOpenAI(
            base_url=self.api_base_url, api_key=self.api_key
        ) as client:
            self.client = client
            tasks = [
                self._score_single(p, s, e)
                for p, s, e in zip(prompts, sources, edited)
            ]
            return list(await asyncio.gather(*tasks))

    async def _score_single(
        self, prompt: str, source: Image.Image, edited: Image.Image
    ) -> Tuple[float, float, float]:
        source_url = pil_image_to_base64(source, format="PNG")
        edited_url = pil_image_to_base64(edited, format="PNG")

        sc_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": SC_SYSTEM_PROMPT.replace("<instruction>", prompt)},
                {"type": "image_url", "image_url": {"url": source_url}},
                {"type": "image_url", "image_url": {"url": edited_url}},
            ]},
        ]
        pq_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": PQ_SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": edited_url}},
            ]},
        ]

        sc_scores, pq_scores = await asyncio.gather(
            self._request_scores(sc_messages, expected=2, kind="SC"),
            self._request_scores(pq_messages, expected=2, kind="PQ"),
        )
        # VIEScore aggregation: min of sub-scores; Overall = sqrt(SC * PQ).
        sc = float(min(sc_scores))
        pq = float(min(pq_scores))
        overall = math.sqrt(max(sc, 0.0) * max(pq, 0.0))
        return sc, pq, overall

    async def _request_scores(
        self, messages: List[dict], *, expected: int, kind: str
    ) -> List[float]:
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
            except (APIConnectionError, APITimeoutError, RateLimitError, asyncio.TimeoutError) as e:
                last_err = e
                logger.warning(
                    "GEditBench %s API transport error (attempt %s/%s): %s",
                    kind, attempt + 1, self.max_retries, e,
                )
                if attempt + 1 >= self.max_retries:
                    break
                await asyncio.sleep(2**attempt)
                continue

            content = completion.choices[0].message.content
            if content is None or not str(content).strip():
                logger.warning(
                    "GEditBench %s VLM returned empty content; using score 0.0", kind
                )
                return [0.0] * expected
            try:
                scores = parse_viescore_list(str(content))
            except ValueError as e:
                logger.warning(
                    "GEditBench %s failed to parse VLM response; using score 0.0: %s", kind, e
                )
                return [0.0] * expected
            scores = [max(0.0, min(SCORE_SCALE, s)) for s in scores]
            return scores if scores else [0.0] * expected

        logger.warning(
            "GEditBench %s HTTP request failed after %s attempt(s); using score 0.0. "
            "Last error: %s",
            kind, self.max_retries, last_err,
        )
        return [0.0] * expected
