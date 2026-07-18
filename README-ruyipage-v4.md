# RuyiPage V4: persistent HTTP + local V11

V4 keeps V3 unchanged and adds a separate registration path:

1. A persistent `curl_cffi` session loads and submits every Battle.net form.
2. The registration country is always `GBR`.
3. The HTTP response from `set-battletag` supplies the Arkose blob.
4. RuyiPage Firefox starts only when Arkose solving is required.
5. The existing V3 image capture and click logic sends each `2000x400` image to the local V11 service.
6. RuyiPage returns the Arkose `onCompleted` token and exits.
7. The original HTTP session submits that token to `captcha-gate` and records the server result.

The fallback browser cookie import runs only when the HTTP response does not contain a usable blob.

## GitHub Actions

Run workflow `Battle.net auto register (HTTP + RuyiPage V4 + local V11)`.

Inputs:

- `count`: `1..256` matrix jobs.
- `max_parallel`: `1..20`; the workflow enforces the upper limit of 20.
- `proxy`: optional `ip:port:username:password`. Leave it blank to use the GitHub runner network directly.

The same normalized proxy is used by the persistent HTTP session and RuyiPage. Proxy credentials are masked in the job log and are not written to summaries or artifacts.

Also accepted by the script:

```text
host:port
host:port:username:password
http://username:password@host:port
socks5://username:password@host:port
```

## Low-traffic defaults

- No browser is used for form filling.
- No IP or geography probe runs.
- One lightweight GBR country probe runs before date-of-birth submission; this is required by the server-rendered flow.
- RuyiPage starts after the HTTP flow reaches `captcha-gate` and closes immediately after obtaining the token.
- Browser fonts, media, and common analytics hosts are blocked; Arkose assets and challenge images remain enabled.
- Per-wave screenshots are disabled unless `--debug-screenshots` is supplied.
- V11 is loaded once per matrix job and reused across all challenge waves and retries.
- V11 starts in parallel with the HTTP form flow; V4 waits for health only when Arkose inference is actually needed.
- The dependency cache keeps only `.venv` and the RuyiPage Firefox runtime. There is no wheelhouse job.

V4 does not launch CloakBrowser and does not use YesCaptcha or Mihomo.

## Local run

Install dependencies and the RuyiPage runtime:

```bash
python -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-ruyipage-v4.txt
.venv/bin/python -m ruyipage install
```

Start V11:

```bash
.venv/bin/python rank_v11/server.py \
  --host 127.0.0.1 --port 8765 \
  --device cpu --cpu-threads 2 --mode accurate
```

Run one registration on Linux:

```bash
xvfb-run -a --server-args="-screen 0 1920x1080x24" \
  .venv/bin/python register_ruyipage_v4.py \
    --proxy "" \
    --rank-v11-url http://127.0.0.1:8765 \
    --click-style balanced
```

Use `--proxy "HOST:PORT:USER:PASSWORD"` to enable a proxy.

Resume an interrupted protocol session with the same route:

```bash
.venv/bin/python register_ruyipage_v4.py \
  --resume ruyipage_http_v11_register/runs/run_TIMESTAMP \
  --proxy "HOST:PORT:USER:PASSWORD" \
  --rank-v11-url http://127.0.0.1:8765
```

When the saved state is already `token-ready`, V4 skips RuyiPage and V11 and submits the saved token directly through the restored HTTP cookies and CSRF form.

## Outputs

Each invocation writes to:

```text
ruyipage_http_v11_register/runs/run_*/
```

The workflow uploads one debug artifact and one account artifact per matrix job, then merges successful accounts into:

```text
all-ruyipage-v4-http-local-registered-accounts/all_accounts.txt
```
