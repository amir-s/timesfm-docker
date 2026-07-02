# timesfm-docker

Docker wrapper for Google's TimesFM 2.5 PyTorch model. The container exposes a
small JSON HTTP API and downloads model weights into a Docker volume on first use.

## Build

```bash
docker build -t timesfm-docker:latest .
```

## Run

With Docker Compose:

```bash
docker compose up -d
docker compose down
```

With Docker:

```bash
docker volume create timesfm-cache
docker run --rm -p 8765:8765 \
  -v timesfm-cache:/cache/huggingface \
  timesfm-docker:latest
```

Open `http://127.0.0.1:8765/healthz` to check the service.

## API

`GET /healthz`

```json
{"ok": true, "model_loaded": false}
```

`GET /api/status`

Returns model id, cache path, and whether the model is loaded.

`POST /api/forecast`

Single series:

```bash
curl -X POST http://127.0.0.1:8765/api/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "series": [10, 12, 13, 15, 18, 20, 21, 23],
    "labels": ["series"],
    "horizon": 4,
    "max_context": 64,
    "normalize_inputs": true
  }'
```

Multiple series:

```bash
curl -X POST http://127.0.0.1:8765/api/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "series": [
      [10, 12, 13, 15, 18, 20, 21, 23],
      [30, 31, 33, 35, 36, 38, 41, 43]
    ],
    "labels": ["series_a", "series_b"],
    "horizon": 4,
    "max_context": 64
  }'
```

Example response shape:

```json
{
  "ok": true,
  "horizon": 4,
  "series_count": 2,
  "forecasts": [
    {
      "label": "series_a",
      "point": [24.1, 25.0, 25.8, 26.5],
      "mean": [24.0, 24.9, 25.7, 26.4],
      "q10": [22.7, 23.4, 24.0, 24.5],
      "q50": [24.1, 25.0, 25.8, 26.5],
      "q90": [25.6, 26.7, 27.5, 28.3]
    },
    {
      "label": "series_b",
      "point": [44.7, 46.1, 47.4, 48.8],
      "mean": [44.6, 46.0, 47.3, 48.7],
      "q10": [42.9, 44.0, 45.0, 46.0],
      "q50": [44.7, 46.1, 47.4, 48.8],
      "q90": [46.3, 48.0, 49.8, 51.4]
    }
  ]
}
```

`point` is the main forecast. `q10`, `q50`, and `q90` are forecast quantiles;
lower-to-higher quantiles give a rough uncertainty band. `mean` is the model's
mean forecast. `horizon` is the number of future steps returned.
