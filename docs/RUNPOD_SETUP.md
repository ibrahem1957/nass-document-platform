# Qari OCR service and RunPod

> **Status at handoff:** the previously used RunPod service has been **stopped**. No endpoint in this
> repository is assumed to work. **No URL, pod ID, token or other credential is committed.** Nothing in
> the repository starts, recreates or calls a pod. Every request costs GPU time; start a pod only with
> the owner's explicit authorization.

## Why a GPU service

Nass uses one OCR model, **Qari**:
- base model `Qwen/Qwen3-VL-4B-Instruct` (a 4B-parameter vision-language model);
- PEFT adapter `NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct`.

The model needs a CUDA GPU to run at a usable speed. The machine running Studio, Nass and DocPrep
(Windows, CPU) does not run it. Nass sends images to an HTTP service on a GPU host. RunPod was used as
a development GPU host; a local GPU or an internal company server works the same way.

## What runs where

| Local machine (Studio + Nass + DocPrep) | GPU host (RunPod pod, local GPU, or internal server) |
|---|---|
| PDF analysis, page routing, text-layer extraction | `nass/runpod/qari_service.py` (FastAPI + uvicorn) |
| PP-DocLayout layout detection (PaddlePaddle, CPU or GPU) | Qwen3-VL-4B + Qari adapter (transformers, peft, torch/CUDA) |
| Crops, masking, job planning, assembly, validation | The fixed "proven" OCR recipe (prompt, decoding, limits, loop stopper, one retry) |
| All of DocPrep | — |

## Service API (`nass/runpod/qari_service.py`)

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Model, device and queue state. No document content and no secrets. |
| `POST /v1/ocr` | `Authorization: Bearer <token>` | JSON `{request_id, kind: "crop"\|"page", image_png_b64, image_sha256}`. Returns every attempt: text, tokens, seconds, `loop_stopped`, `hit_limit`, `timed_out`, params. |
| `GET /v1/results/{request_id}` | Bearer | Recovers a completed result held in memory (lost-response recovery). |

**Image handling:** the service uses the image exactly as received, with no resizing or re-encoding.

**Proven recipe** (`PROVEN`):
- prompt "Free OCR.", greedy decoding
- `max_new_tokens` 1536
- time limits of 30 s per crop and 60 s per page
- loop stopper: an n-gram (n = 1…40) repeated 5 or more times
- after a loop, timeout or token limit: one retry with `repetition_penalty` 1.1

Overrides are refused unless `QARI_ALLOW_PROVEN_OVERRIDES=true`; when allowed, they are reported in
every response.

### Service environment (on the GPU host)

| Variable | Meaning |
|---|---|
| `QARI_SERVICE_TOKEN` | **Required** bearer token. The service refuses to start without it. `QARI_SERVICE_ALLOW_NO_TOKEN=true` exists only for loopback tests. |
| `QARI_DEVICE_MAP`, `QARI_DTYPE` | transformers `device_map` (default `auto`); `float16`, `bfloat16` or `float32` |
| `QARI_SERVICE_CONCURRENCY` | Simultaneous generations (default 1) |
| `QARI_SERVICE_MAX_QUEUE` | Waiting requests before HTTP 429 (default 8) |
| `QARI_SERVICE_MAX_IMAGE_MB` | Request image limit (default 40) |
| `QARI_SERVICE_RESULT_CACHE` | Results kept for `/v1/results` (default 256) |
| `QARI_MODEL_CACHE_DIR` | Hugging Face cache directory. The weights are fetched by `from_pretrained` on first start unless already cached there. |
| `QARI_SERVICE_HOST`, `QARI_SERVICE_PORT` | Used when the file is run directly (`python qari_service.py`); defaults are `127.0.0.1:8000` |

**Install on the GPU host:**
1. Install `torch` for your CUDA version.
2. Run `pip install -r nass/runpod/requirements-qari-service.txt`. The file lists minimum versions
   only: transformers ≥ 4.57, peft, accelerate, pillow, fastapi, uvicorn.

**Start command** (from the module docstring):

```bash
cd nass/runpod
export QARI_SERVICE_TOKEN='<long random secret>'
uvicorn qari_service:app_from_environment --factory --host 0.0.0.0 --port 8000 --workers 1
```

## Client configuration (Studio machine, `nass/.env`)

Copy the root `.env.example` to `nass/.env`; real environment variables win over the file. The keys
read by `nass/lab/core/auto/config.py`:

| Variable | Meaning |
|---|---|
| `QARI_SERVICE_URL` | Base URL, e.g. `http://127.0.0.1:8000` or `https://<pod-id>-8000.proxy.runpod.net`. It must contain no credentials, query or fragment. |
| `QARI_DEPLOYMENT_MODE` | `local`: loopback host only. `internal`: private or internal host; public IPs are refused. `development_external`: https only, and the host must be listed in `QARI_TRUSTED_DEV_HOSTS`. |
| `QARI_TRUSTED_DEV_HOSTS` | Exact host names trusted in `development_external` mode, e.g. `<pod-id>-8000.proxy.runpod.net` |
| `QARI_SERVICE_TOKEN` | The same bearer token as the service. Read from the environment only, never from TOML; never written to manifests or logs. |
| `QARI_AUTOMATIC_ENABLED` | `true` lets Studio/Nass send jobs automatically |
| `QARI_REQUEST_TIMEOUT_S` | Per-request timeout (default 180 s) |
| `QARI_MAX_CONNECT_RETRIES` | Retries **only** when nothing was transmitted (default 2) |
| `QARI_CONCURRENCY` | Client-side parallel requests (default 1) |
| `AGENTICDOC_ENV` | `production` forbids `development_external` |
| `VERIFY_DIGITS_*` | Report-only second read of crops with Arabic-Indic digits (extra requests) |

These rules are enforced **before** anything is sent (`qari_client.endpoint_info`). An untrusted or
unconfigured endpoint raises `QariUntrusted` or `QariUnavailable`, and Studio refuses to start the
extraction.

## Configuring a new computer

1. Deploy or locate a Qari service: see "Handoff dependency" below.
2. Choose a long random token. Set it on the service (`QARI_SERVICE_TOKEN`) and in `nass/.env`.
3. In `nass/.env`, set `QARI_SERVICE_URL`, `QARI_DEPLOYMENT_MODE` and, for RunPod,
   `QARI_TRUSTED_DEV_HOSTS`.
4. Open Studio and check the readiness banner. Readiness inspects configuration only; it does **not**
   contact the service.

## Checking connectivity safely

`GET /health` sends no document content and needs no token:

```powershell
curl.exe -s https://<pod-id>-8000.proxy.runpod.net/health
```

Or use the Nass client, from `nass/`, with `nass/.env` configured:

```powershell
python -c "from lab.core.auto import config, qari_client; c=config.load_config(); print(qari_client.from_config(c).health())"
```

Do not send `/v1/ocr` requests to test connectivity: each one spends GPU time and transmits an image.

## Errors and recovery

| Error kind (`qari_client`) | Meaning | What happens |
|---|---|---|
| `untrusted` / `unavailable` before sending | URL missing, invalid or not trusted for the mode | Nothing is sent; Studio reports that Nass is not ready |
| `unavailable` (connection) | Service unreachable | Retried only if nothing was transmitted; the job stays pending |
| `unconfirmed` | The request may have reached the service, but no response arrived | **Never resent automatically**; it can be recovered through `GET /v1/results/{id}` |
| `service_error` | HTTP error from the service (e.g. 401 wrong token, 429 queue full, 5xx) | Recorded in `raw/<job>/attempt_NN.result.json` |

**In Studio:**
- If any job ends pending, unconfirmed or failed, the Nass stage fails ("Nass is incomplete"). The job
  folder and all attempt records are kept.
- **Retry** creates a **new** job folder and re-plans the document.
- Recovering an unconfirmed job inside the same run is available in the Nass lab (`python -m
  lab.launch`), not in Studio.

## When the service is unavailable

- **Studio still works for:**
  - launching;
  - opening previously processed documents (`existing_documents`);
  - viewing, downloading and inspecting saved results.
- **New extractions:**
  - Configuration must still be present: a Qari URL and `LAYOUT_MODEL_DIR`.
  - A PDF that produces **no OCR jobs** (a purely textual digital PDF) can finish without contacting
    Qari. This path is inferred from the code and **not** re-verified end to end in this handoff.
  - Any PDF with scanned pages, visuals or category-B Arabic text needs the service.
- **Offline alternative:** in the Nass lab, export a run's pending jobs
  (`pipeline.export_offline`), run `python nass/runpod/qari_offline_batch.py <unzipped export>` on any
  GPU host, then import the `results.jsonl` (`pipeline.import_offline`, hash-checked).
- **DocPrep:** never needs the service. It can be run again on any saved Nass run.

## Costs and timeouts

- **Pricing:** no prices are recorded in this repository; GPU pricing depends on the RunPod plan and
  GPU type.
- **What drives cost:** OCR requests are serialized by default (client and service concurrency 1). The
  service's own limits are 30 s per crop and 60 s per page, plus one possible retry. A scanned page
  typically produces one masked full-page request plus one request per selected visual. Costs come
  from pod uptime, not per request.
- **Client timeout:** `QARI_REQUEST_TIMEOUT_S` (180 s) must exceed the service's worst case for one
  request, including the retry.
- **Stopping the pod:** the stopped pod costs nothing for compute (storage may still be billed by the
  provider).

## Starting, stopping and managing the GPU service

The repository contains the **service code** (`qari_service.py`, `qari_offline_batch.py`,
`run_qari_v2.py`) and its requirement list. It does **not** contain a verified RunPod template,
container image, GPU type or volume layout.

On RunPod, the URL format used before was `https://<pod-id>-8000.proxy.runpod.net` (HTTP port 8000
exposed). Start and stop pods from the RunPod console or the owner's RunPod account. The service holds
the model in memory, so a restart reloads the weights.

### Handoff dependency (must be provided by the owner or the next developer)

- **A GPU host:** a RunPod account and pod, a local CUDA GPU, or an internal server.
- **The pod configuration:** GPU type, CUDA/torch versions, disk or volume for the model cache, the
  exposed port, and the start command. None of these were recorded in the source repositories.
- **Model access:** the model weights from Hugging Face (`Qwen/Qwen3-VL-4B-Instruct` and
  `NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct`), downloaded on the GPU host.
- **Licences:** review the licences of these model repositories before any non-development use.
- **Token:** a new `QARI_SERVICE_TOKEN`. Do not reuse old tokens.
