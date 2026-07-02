#!/usr/bin/env python3
"""HTTP JSON wrapper for TimesFM."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np


MODEL_ID = os.environ.get("MODEL_ID", "google/timesfm-2.5-200m-pytorch")
os.environ.setdefault("HF_HOME", "/cache/huggingface")

_MODEL = None
_TIMESFM = None
_MODEL_LOCK = threading.Lock()
_COMPILED_KEY: tuple[Any, ...] | None = None
_COMPILED_CONFIG: dict[str, Any] | None = None


class ForecastError(ValueError):
  """Input validation error."""


def _jsonable(values: np.ndarray) -> list[float]:
  return np.asarray(values, dtype=np.float64).round(6).tolist()


def _clean_series(raw: Any) -> np.ndarray:
  if not isinstance(raw, list):
    raise ForecastError("series must be an array of numbers or an array of arrays.")

  values = []
  for item in raw:
    if item is None:
      values.append(float("nan"))
      continue
    try:
      value = float(item)
    except (TypeError, ValueError) as exc:
      raise ForecastError("series contains a non-numeric value.") from exc
    values.append(value if np.isfinite(value) else float("nan"))

  arr = np.asarray(values, dtype=np.float32)
  while len(arr) and np.isnan(arr[-1]):
    arr = arr[:-1]

  if int(np.isfinite(arr).sum()) < 8:
    raise ForecastError("series must contain at least 8 numeric values.")
  return arr


def _parse_inputs(payload: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
  raw_series = payload.get("series")
  labels = payload.get("labels")

  if (
    isinstance(raw_series, list)
    and raw_series
    and all(isinstance(item, list) for item in raw_series)
  ):
    inputs = [_clean_series(item) for item in raw_series]
  else:
    inputs = [_clean_series(raw_series)]

  if isinstance(labels, list) and len(labels) == len(inputs):
    clean_labels = [str(label)[:80] or f"Series {i + 1}" for i, label in enumerate(labels)]
  else:
    clean_labels = [f"Series {i + 1}" for i in range(len(inputs))]

  return inputs, clean_labels


def _bool(payload: dict[str, Any], key: str, default: bool) -> bool:
  value = payload.get(key, default)
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    return value.lower() in {"1", "true", "yes", "on"}
  return bool(value)


def _positive_by_default(inputs: list[np.ndarray]) -> bool:
  finite = np.concatenate([arr[np.isfinite(arr)] for arr in inputs])
  return bool(len(finite) and np.all(finite >= 0))


def _build_config(payload: dict[str, Any], inputs: list[np.ndarray]) -> tuple[int, dict[str, Any], tuple[Any, ...]]:
  try:
    horizon = int(payload.get("horizon", 12))
    max_context = int(payload.get("max_context", 256))
    per_core_batch_size = int(payload.get("per_core_batch_size", 1))
    requested_max_horizon = int(payload.get("max_horizon", horizon))
  except (TypeError, ValueError) as exc:
    raise ForecastError("horizon, max_context, max_horizon, and per_core_batch_size must be integers.") from exc

  if not 1 <= horizon <= 1024:
    raise ForecastError("horizon must be between 1 and 1024.")
  if not 32 <= max_context <= 16384:
    raise ForecastError("max_context must be between 32 and 16384.")
  if max_context + horizon > 16384:
    raise ForecastError("max_context + horizon must be at most 16384.")
  if not 1 <= per_core_batch_size <= 256:
    raise ForecastError("per_core_batch_size must be between 1 and 256.")

  max_horizon = max(horizon, requested_max_horizon)
  use_continuous_quantile_head = _bool(
    payload, "use_continuous_quantile_head", max_horizon <= 1024
  )
  if use_continuous_quantile_head and max_horizon > 1024:
    raise ForecastError("use_continuous_quantile_head requires max_horizon <= 1024.")

  infer_is_positive = payload.get("infer_is_positive")
  if infer_is_positive is None:
    infer_is_positive = _positive_by_default(inputs)

  config = {
    "max_context": max_context,
    "max_horizon": max_horizon,
    "normalize_inputs": _bool(payload, "normalize_inputs", True),
    "per_core_batch_size": per_core_batch_size,
    "use_continuous_quantile_head": use_continuous_quantile_head,
    "force_flip_invariance": _bool(payload, "force_flip_invariance", False),
    "infer_is_positive": bool(infer_is_positive),
    "fix_quantile_crossing": _bool(payload, "fix_quantile_crossing", True),
    "return_backcast": False,
  }
  return horizon, config, tuple(config.items())


def _load_model():
  global _MODEL, _TIMESFM

  if _MODEL is not None:
    return _MODEL, _TIMESFM

  import torch
  import timesfm

  torch.set_float32_matmul_precision("high")
  _TIMESFM = timesfm
  _MODEL = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    MODEL_ID,
    torch_compile=False,
  )
  return _MODEL, _TIMESFM


def run_forecast(payload: dict[str, Any]) -> dict[str, Any]:
  global _COMPILED_CONFIG, _COMPILED_KEY

  inputs, labels = _parse_inputs(payload)
  horizon, config, config_key = _build_config(payload, inputs)
  started = time.perf_counter()

  with _MODEL_LOCK:
    model, timesfm = _load_model()
    if _COMPILED_KEY != config_key:
      model.compile(timesfm.ForecastConfig(**config))
      _COMPILED_KEY = config_key
      _COMPILED_CONFIG = dataclasses.asdict(model.forecast_config)

    point, quantiles = model.forecast(horizon=horizon, inputs=inputs)
    device = str(getattr(model.model, "device", "unknown"))

  forecasts = []
  for index, label in enumerate(labels):
    forecasts.append(
      {
        "label": label,
        "point": _jsonable(point[index]),
        "mean": _jsonable(quantiles[index, :, 0]),
        "q10": _jsonable(quantiles[index, :, 1]),
        "q20": _jsonable(quantiles[index, :, 2]),
        "q50": _jsonable(quantiles[index, :, 5]),
        "q80": _jsonable(quantiles[index, :, 8]),
        "q90": _jsonable(quantiles[index, :, 9]),
      }
    )

  return {
    "model": MODEL_ID,
    "device": device,
    "horizon": horizon,
    "series_count": len(inputs),
    "history_lengths": [int(len(item)) for item in inputs],
    "config": _COMPILED_CONFIG,
    "elapsed_ms": int((time.perf_counter() - started) * 1000),
    "forecasts": forecasts,
  }


class Handler(BaseHTTPRequestHandler):
  server_version = "TimesFMDocker/1.0"

  def log_message(self, fmt: str, *args: Any) -> None:
    print(f"{self.address_string()} - {fmt % args}")

  def _send_json(self, body: dict[str, Any], status: int = 200) -> None:
    data = json.dumps(body).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(data)

  def _read_json(self) -> dict[str, Any]:
    length = int(self.headers.get("Content-Length", "0"))
    if length <= 0:
      raise ForecastError("request body is empty.")
    try:
      payload = json.loads(self.rfile.read(length).decode("utf-8"))
    except json.JSONDecodeError as exc:
      raise ForecastError("request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
      raise ForecastError("request body must be a JSON object.")
    return payload

  def do_GET(self) -> None:
    route = urlparse(self.path).path
    if route in {"/", "/healthz", "/readyz"}:
      self._send_json({"ok": True, "model_loaded": _MODEL is not None})
      return
    if route == "/api/status":
      hf_home = os.environ.get("HF_HOME", "")
      self._send_json(
        {
          "ok": True,
          "model": MODEL_ID,
          "model_loaded": _MODEL is not None,
          "compiled_config": _COMPILED_CONFIG,
          "hf_home": hf_home,
          "hf_cache_exists": Path(hf_home).exists(),
        }
      )
      return
    self.send_error(HTTPStatus.NOT_FOUND)

  def do_POST(self) -> None:
    if urlparse(self.path).path != "/api/forecast":
      self.send_error(HTTPStatus.NOT_FOUND)
      return

    try:
      result = run_forecast(self._read_json())
    except ForecastError as exc:
      self._send_json({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:  # pylint: disable=broad-except
      traceback.print_exc()
      self._send_json({"ok": False, "error": str(exc)}, status=500)
    else:
      self._send_json({"ok": True, **result})


def main() -> None:
  host = os.environ.get("HOST", "0.0.0.0")
  port = int(os.environ.get("PORT", "8765"))
  server = ThreadingHTTPServer((host, port), Handler)
  print(f"TimesFM API listening on http://{host}:{port}")
  print(f"MODEL_ID={MODEL_ID}")
  print(f"HF_HOME={os.environ.get('HF_HOME')}")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\nShutting down.")
  finally:
    server.server_close()


if __name__ == "__main__":
  main()
