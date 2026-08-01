"""Resumable OpenAI-compatible client for Qwen3-VL hard-data construction."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


class QwenConstructorClient:
    def __init__(
        self,
        *,
        api_base: str = "http://28.7.185.156:8000/v1",
        model: str = "qwen3-vl-235b-a22b-instruct",
        cache_dir: Path = Path(".cache/qwen3vl-constructor"),
        timeout: float = 600.0,
        max_retries: int = 3,
        disable_proxy: bool = True,
    ) -> None:
        if not api_base.startswith(("http://", "https://")):
            raise ValueError(f"expected HTTP(S) api_base, got {api_base!r}")
        if not model.strip():
            raise ValueError(f"expected non-empty model name, got {model!r}")
        if timeout <= 0:
            raise ValueError(f"expected timeout > 0, got {timeout!r}")
        if max_retries < 1:
            raise ValueError(f"expected max_retries >= 1, got {max_retries!r}")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.cache_dir = cache_dir
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}) if disable_proxy else urllib.request.ProxyHandler()
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        images: Sequence[Path] = (),
        temperature: float = 0.0,
        max_tokens: int = 4096,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(images) > 10:
            raise ValueError(f"expected at most 10 images per request, got {len(images)}")
        if max_tokens < 1:
            raise ValueError(f"expected max_tokens >= 1, got {max_tokens!r}")
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for path in images:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if extra:
            payload.update(dict(extra))
        request_id = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{request_id}.json"
        if cache_path.exists():
            response = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            response = self._post(payload=payload, request_id=request_id)
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(response, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        text = _extract_assistant_text(response, request_id=request_id)
        return _parse_json_object(text, request_id=request_id)

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.api_base}/models", method="GET")
        try:
            with self.opener.open(request, timeout=min(self.timeout, 30.0)) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Qwen health endpoint returned HTTP {error.code}: {body[:500]}"
            ) from error
        except urllib.error.URLError as error:
            raise ConnectionError(
                f"could not reach Qwen health endpoint {self.api_base!r}: {error.reason}"
            ) from error
        return payload

    def _post(self, *, payload: Mapping[str, Any], request_id: str) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer EMPTY",
                "X-Request-ID": request_id,
            },
            method="POST",
        )
        last_error: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self.opener.open(request, timeout=self.timeout) as handle:
                    response = json.loads(handle.read().decode("utf-8"))
                if not isinstance(response, dict):
                    raise TypeError(
                        f"expected JSON object from Qwen request {request_id}, "
                        f"got {type(response).__name__}: {response!r}"
                    )
                return response
            except urllib.error.HTTPError as error:
                body_text = error.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Qwen request {request_id} returned HTTP {error.code}: {body_text[:1000]}"
                )
                if error.code < 500 and error.code != 429:
                    raise last_error from error
            except urllib.error.URLError as error:
                last_error = ConnectionError(
                    f"Qwen request {request_id} failed: {error.reason}"
                )
            if attempt < self.max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
        if last_error is None:
            raise RuntimeError(f"Qwen request {request_id} exhausted retries without an error")
        raise RuntimeError(
            f"Qwen request {request_id} failed after {self.max_retries} attempts"
        ) from last_error


def _image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"expected image file, got missing path {path}")
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        raise ValueError(f"expected image MIME type for {path}, got {mime!r}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _extract_assistant_text(response: Mapping[str, Any], *, request_id: str) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            f"Qwen response {request_id} has no non-empty choices list: {response!r}"
        )
    first = choices[0]
    if not isinstance(first, Mapping):
        raise TypeError(
            f"expected mapping for first Qwen choice {request_id}, "
            f"got {type(first).__name__}: {first!r}"
        )
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise TypeError(f"Qwen response {request_id} has invalid message: {message!r}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Qwen response {request_id} has empty text content: {content!r}")
    return content.strip()


def _parse_json_object(text: str, *, request_id: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError(
                f"Qwen response {request_id} has an unterminated markdown JSON fence"
            )
        stripped = "\n".join(lines[1:-1])
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:].lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Qwen response {request_id} is not valid JSON: {error.msg}; "
            f"content={text[:1000]!r}"
        ) from error
    remainder = stripped[end:].strip()
    if remainder:
        comment_lines = [line.strip() for line in remainder.splitlines() if line.strip()]
        if not comment_lines or not all(
            line.startswith(("//", "#")) for line in comment_lines
        ):
            raise ValueError(
                f"Qwen response {request_id} has non-comment content after its JSON object: "
                f"{remainder[:500]!r}"
            )
    if not isinstance(value, dict):
        raise TypeError(
            f"expected JSON object from Qwen response {request_id}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value
