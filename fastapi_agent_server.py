import sys
import os
import asyncio
import threading
import traceback
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import your EnhancedWebAgent
from updated_main_agent import EnhancedWebAgent

# --- Config ---
MODEL_PATH = r"D:\YASH\model\mistral-7b-openorca.gguf2.Q4_0.gguf"

app = FastAPI(title="AI Web Agent (FastAPI)", version="auto-load-model-1.0")

# serve static UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class AgentRunner:
    def __init__(self):
        self._agent: Optional[EnhancedWebAgent] = None
        self._thread: Optional[threading.Thread] = None
        self._worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running_event = threading.Event()
        self._startup_exc: Optional[str] = None

    def _ensure_proactor(self):
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    def start_agent(self, headless: bool = True, timeout: float = 120.0) -> Dict[str, Any]:
        if self._thread and self._thread.is_alive() and self._running_event.is_set():
            return {"success": True, "message": "Agent already running."}

        self._startup_exc = None
        self._running_event.clear()
        self._worker_loop = None

        def worker():
            try:
                self._ensure_proactor()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._worker_loop = loop

                agent = EnhancedWebAgent(headless=headless)
                self._agent = agent

                # ✅ Auto-load mistral model here
                if hasattr(agent, "load_local_mistral_model"):
                    print(f"🔄 Loading model from {MODEL_PATH}")
                    agent.load_local_mistral_model(MODEL_PATH)

                async def start_and_wait():
                    await agent.start()
                loop.run_until_complete(start_and_wait())

                self._running_event.set()
                loop.run_forever()
            except Exception:
                self._startup_exc = traceback.format_exc()
                self._running_event.clear()
                self._agent = None

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

        # wait for startup
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._startup_exc:
                return {"success": False, "error": self._startup_exc}
            if self._running_event.is_set():
                return {"success": True, "message": "Agent + Model started."}
            time.sleep(0.1)
        return {"success": False, "error": "Agent start timed out."}

    def stop_agent(self):
        if not self._agent:
            return {"success": True, "message": "Agent not running."}
        loop = self._worker_loop

        async def _do_stop():
            await self._agent.stop()
            loop.stop()

        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(_do_stop(), loop)
        self._agent = None
        self._running_event.clear()
        return {"success": True, "message": "Agent stopped."}

    def execute_instruction_sync(self, instruction: str):
        if not self._agent:
            return {"success": False, "error": "Agent not running."}

        async def _call_exec():
            return await self._agent.execute_instruction(instruction)

        loop = self._worker_loop
        try:
            fut = asyncio.run_coroutine_threadsafe(_call_exec(), loop)
            res = fut.result(timeout=60)
        except Exception:
            return {"success": False, "error": traceback.format_exc()}

        return {
            "success": getattr(res, "success", True),
            "data": getattr(res, "data", res),
            "error_message": getattr(res, "error_message", None),
            "steps_taken": getattr(res, "steps_taken", None),
        }

    def get_status(self):
        return {
            "agent_running": bool(self._agent),
            "model_info": {
                "model_loaded": getattr(getattr(self._agent, "mistral_parser", None), "model_loaded", False),
                "model_path": getattr(getattr(self._agent, "mistral_parser", None), "model_path", None),
            },
        }


runner = AgentRunner()
recent_tasks: List[dict] = []


class ExecuteRequest(BaseModel):
    instruction: str


@app.get("/health")
@app.get("/model_status")
async def model_status():
    return runner.get_status()


@app.post("/start_agent")
async def start_agent():
    res = runner.start_agent(headless=True)
    if not res.get("success"):
        raise HTTPException(status_code=500, detail=res.get("error"))
    return res


@app.post("/stop_agent")
async def stop_agent():
    return runner.stop_agent()


@app.post("/execute")
async def execute(req: ExecuteRequest):
    res = runner.execute_instruction_sync(req.instruction)
    task = {"instruction": req.instruction, "timestamp": time.time(), "result": res}
    recent_tasks.append(task)
    return JSONResponse(content=res)


@app.get("/recent_tasks")
async def get_recent_tasks():
    return recent_tasks[-10:]


@app.get("/", include_in_schema=False)
async def root():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)


# ✅ Auto-start agent when FastAPI boots
@app.on_event("startup")
async def auto_startup():
    print("🚀 Auto-starting agent + loading model...")
    runner.start_agent(headless=True)
