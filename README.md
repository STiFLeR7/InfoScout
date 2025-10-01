# AI Web Agent (FastAPI + Playwright + Local Mistral .gguf)

**Tech README** — production-ish, developer-focused, and actionable.

This repository hosts a lightweight FastAPI wrapper around an autonomous browsing agent (`updated_main_agent.py`) that uses Playwright for web automation and optionally loads a local Mistral-style model stored as a `.gguf` file. The FastAPI server serves a static UI (`/static`) and provides a JSON API for starting/stopping the agent, executing instructions, loading a model at runtime, and checking health/status.

This README explains: architecture, endpoints, how `updated_main_agent.py` loads `.gguf` models, Windows-specific event-loop details (Proactor), environment variables, run commands, troubleshooting, and extension ideas.

---

## Table of contents

* Architecture overview
* File layout
* How the model loading works (what `updated_main_agent.py` expects)
* FastAPI endpoints (full list)
* Environment variables & configuration
* Running (dev & production)
* Troubleshooting & common fixes (Windows Proactor, Playwright transport warnings)
* Testing & examples (curl + JS snippets)
* Security & deployment notes
* Future improvements / research directions

---

## Architecture overview

* `fastapi_agent_server.py` — FastAPI app that:

  * mounts `static/` (UI) at `/`
  * manages an `AgentRunner` which runs `EnhancedWebAgent` in a dedicated worker thread with a dedicated asyncio event loop (important for Playwright subprocesses on Windows)
  * exposes REST endpoints for agent lifecycle, execution, model-loading and status
  * keeps an in-memory `recent_tasks` list for quick history in the UI

* `updated_main_agent.py` — the agent implementation (provided). It should expose at least the following surface:

  * `EnhancedWebAgent(headless: bool = True)` — constructor
  * `async def start(self)` — create Playwright browser / context / page and prepare any internal state
  * `async def stop(self)` — gracefully shut down Playwright/browser and any internal resources
  * `async def execute_instruction(self, instruction: str)` — perform the instruction (search, click, scrape, etc.) and return a JSON-serializable object or dataclass with fields like `success`, `data`, `error_message`, `steps_taken`.
  * `def load_local_mistral_model(self, model_path: str) -> bool` — (optional) synchronously load model artifacts / parser and set attributes used by `get_status()`
  * `self.mistral_parser` (optional attr) — object exposing `model_loaded`, `model_path`, `model_name`, `loader_repr` for status reporting.

The FastAPI server runs the agent in a **worker thread** with its **own event loop**. This isolates Playwright subprocess creation and prevents `NotImplementedError` / `asyncio` subprocess errors seen when mixing event loops on Windows.

---

## File layout (key files)

```
project/
├─ fastapi_agent_server.py   # FastAPI server + AgentRunner
├─ updated_main_agent.py     # Agent implementation (Playwright + LLM loader)
├─ static/
│  ├─ index.html
│  ├─ style.css
│  └─ app.js
├─ requirements.txt
└─ README.md
```

---

## How model loading works (updated_main_agent.py)

> This section describes the *convention* `fastapi_agent_server.py` expects from `updated_main_agent.py`. If your `updated_main_agent.py` follows these, the server will automatically load the `.gguf` file on agent start.

1. **Synchronous loader**: `EnhancedWebAgent` can optionally expose a synchronous method `load_local_mistral_model(model_path: str) -> bool`. The FastAPI runner calls this method in the worker thread **before** `agent.start()` so the parser/model metadata are available by the time the agent starts.

2. **Parser attributes**: If present, `agent.mistral_parser` should expose at least:

   * `model_loaded` (bool)
   * `model_path` (str)
   * `model_name` (str)
   * `loader_repr` (optional diagnostic string)

   The runner reads these for `/health` and `/model_status` responses.

3. **Start/Stop lifecycle**: `agent.start()` should create Playwright browser/context/page inside the same thread and loop. `agent.stop()` must close Playwright resources. If the loader also allocates GPU/torch resources, ensure `stop()` frees them.

4. **Return shape from `execute_instruction`**: Prefer returning a dataclass or simple namespace with attributes used by the server: `success`, `data`, `error_message`, `steps_taken`. If you return raw dicts, the server will pass them through.

Example minimal sketch inside `updated_main_agent.py` (illustrative only, not full code):

```py
class EnhancedWebAgent:
    def __init__(self, headless=True):
        self.headless = headless
        self.mistral_parser = SimpleNamespace(model_loaded=False, model_path=None)

    def load_local_mistral_model(self, model_path: str) -> bool:
        # load .gguf with your preferred backend (gguf loader, vllm shim, etc.)
        self.mistral_parser.model_path = model_path
        self.mistral_parser.model_loaded = True
        self.mistral_parser.model_name = 'mistral-7b-openorca'
        self.mistral_parser.loader_repr = 'local-gguf-loader-v1'
        return True

    async def start(self):
        # start Playwright browser, create page
        pass

    async def stop(self):
        # close browser + cleanup
        pass

    async def execute_instruction(self, instruction: str):
        # perform browsing / scraping / query orchestration
        return SimpleNamespace(success=True, data={"results": []}, steps_taken=["start"])
```

---

## FastAPI endpoints (complete)

All endpoints return JSON. Errors use HTTP 4xx/5xx and include exception `.detail` where appropriate.

* `GET /` — serves `static/index.html` (UI)

* `GET /health` and `GET /model_status` — return:

  ```json
  {
    "agent_running": true/false,
    "playwright_enabled": true/false,
    "startup_error": null | "traceback...",
    "model_info": {"model_loaded": bool, "model_path": str | null, "model_name": str | null}
  }
  ```

* `POST /start` and `POST /start_agent` — body: `{ "headless": true/false, "model_path": null | "/path/to.gguf" }`. Starts the agent in background thread. If you include `model_path`, the runner **attempts** to call `agent.load_local_mistral_model(model_path)` before `agent.start()`.

* `POST /stop` and `POST /stop_agent` — stops the agent; returns `{success:true, message:"Agent stopped."}`.

* `POST /execute` — body: `{ "instruction": "search for laptops" }` — schedules `agent.execute_instruction` on the worker event loop and returns normalized result. Example return shape:

  ```json
  { "success": true, "data": {"search_term":"laptops","results":[ ... ]}, "error_message": null }
  ```

* `POST /load_model` — body: `{ "model_path": "D:/.../model.gguf" }` — calls `agent.load_local_mistral_model(model_path)` (if implemented) and returns `{"success": true, "model_loaded": true, "model_path": ...}`.

* `GET /recent_tasks` — returns an in-memory slice of recent executed instructions and their results (useful for the UI). Not persisted.

---

## Environment variables & configuration

* `MODEL_PATH` — default path used for automatic model loading at agent start. You can set this to `D:\YASH\model\mistral-7b-openorca.gguf2.Q4_0.gguf` or another local path.
* `AUTO_START` — if `1`, the FastAPI app will attempt to start the agent automatically on FastAPI `startup` and load `MODEL_PATH`.

Set env in Windows (PowerShell):

```ps1
$env:MODEL_PATH = 'D:\YASH\model\mistral-7b-openorca.gguf2.Q4_0.gguf'
$env:AUTO_START = '1'
uvicorn fastapi_agent_server:app --reload --host 127.0.0.1 --port 8000
```

---

## Install & run (dev)

1. Create venv & install deps

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
# requirements.txt should include: fastapi, uvicorn[standard], playwright, pydantic
# plus your model loader libs (gguf, vllm, transformers, etc.)
```

2. Install Playwright browsers (if not already):

```bash
python -m playwright install
```

3. Run server

```bash
uvicorn fastapi_agent_server:app --reload --host 127.0.0.1 --port 8000
```

If you set `AUTO_START=1` and `MODEL_PATH` env var, the agent will auto-start and attempt to load the `.gguf` model.

---

## Troubleshooting (common Windows issues)

### `NotImplementedError` / `asyncio` subprocess errors

Cause: Playwright creates subprocesses and `asyncio` on Windows needs the **ProactorEventLoopPolicy** to support `create_subprocess_exec`. If Playwright is started from the uvicorn main loop (or a different loop), subprocess creation will fail.

Fix applied in this repo: the `AgentRunner` starts a **new worker thread**, sets `asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())` in that thread (on Windows), creates a new `asyncio` loop with `asyncio.new_event_loop()`, sets it, and runs `agent.start()` inside it. All Playwright interactions and model-related coroutines run inside that loop. When calling `execute` from FastAPI, the code uses `asyncio.run_coroutine_threadsafe(...)` into the worker loop.

If you still see `BaseSubprocessTransport` `ValueError: I/O operation on closed pipe`, it usually means some subprocess transport outlived its loop during shutdown — ensure `agent.stop()` closes browsers and that the worker loop is stopped via `loop.stop()` once `stop()` completes.

### `NoneType.send` while using Playwright

Symptom: UI shows error `AttributeError: 'NoneType' object has no attribute 'send'` during navigation.

Cause: Playwright browser/page objects were created in one loop and then used after that loop was closed or in a different loop/thread.

Fix: keep Playwright objects and their usage inside the same dedicated worker loop and thread. Use `asyncio.run_coroutine_threadsafe` to schedule tasks in that loop from the FastAPI thread.

---

## Example usage (curl)

Start agent (with optional model_path):

```bash
curl -X POST "http://127.0.0.1:8000/start_agent" -H "Content-Type: application/json" -d '{"headless": true, "model_path": "D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf"}'
```

Execute instruction:

```bash
curl -X POST "http://127.0.0.1:8000/execute" -H "Content-Type: application/json" -d '{"instruction":"search for laptops"}'
```

Check model status:

```bash
curl http://127.0.0.1:8000/model_status
```

Recent tasks:

```bash
curl http://127.0.0.1:8000/recent_tasks
```

---

## Deployment notes

* For Windows dev, keep the agent in-process (this repo uses a worker thread pattern). For production you might prefer separating agent into its own process (agent RPC server) to fully isolate Playwright and model memory.
* If you deploy in Linux, the Proactor policy step is not required, but the dedicated worker loop pattern is still a good design.
* Watch memory when loading large models — the `.gguf` file can be tens of GB. Ensure you run on a machine with enough RAM and GPU if your loader uses CUDA.

---

## Testing checklist

* [ ] Start uvicorn and confirm `GET /model_status` returns `model_loaded: true` when model path exists.
* [ ] Click `Start Agent` in the UI and confirm only one Playwright browser launches.
* [ ] Execute a query and confirm results in UI with clickable links.
* [ ] Download CSV/JSON and confirm formats match.
* [ ] Stop agent and confirm Playwright cleanly shut down with no `BaseSubprocessTransport` warnings.

---

## Future improvements

* Persist `recent_tasks` to a simple sqlite store with timestamps and optional user notes.
* Add websocket streaming for step-by-step task progress (stream `steps_taken` to UI as they happen).
* Add a lightweight supervisor to respawn agent if it crashes, with exponential backoff and failure telemetry.
* Add model quantization/adapter hooks and a capability negotiation endpoint to report RAM/VRAM requirements for loaded models.

---

If you want I will:

* add a `docker-compose` example that runs the FastAPI app and (optionally) a separate agent process; or
* open a small PR with instrumentation prints into `updated_main_agent.py` showing when the model loads and its loader repr.

Drop a note which of those you prefer and I’ll add it to the README or prepare the extra files.
