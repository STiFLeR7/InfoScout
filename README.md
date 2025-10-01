Nice — you’ve got **InfoScout** structured in a clean, modular way. From the tree and code, here’s how the repo description and README could be shaped so anyone landing on your GitHub instantly “gets it” without having to dig through files:

---

# 📖 InfoScout

**InfoScout** is an **Agentic AI web scout** that autonomously searches Google (and falls back to DuckDuckGo when needed), extracts the **top 10 results**, and returns them as structured, clickable links. It uses **Playwright** for robust browser automation, with HTTP + BeautifulSoup + regex fallback to stay resilient against SERP changes.

### 🔹 Features

* **Autonomous Web Search:** Google-first, with DuckDuckGo fallback.
* **Top-10 Structured Results:** Titles, links, and metadata returned in JSON.
* **Agentic Memory:** SQLite-backed task memory stores past queries, results, and execution logs.
* **Multiple Extraction Modes:**

  * Playwright (Chrome/Chromium persistent profile)
  * BeautifulSoup HTML parsing
  * Regex-based emergency fallback
* **Local LLM Hook:** Mistral-7B (GGUF format) can be loaded via `llama_cpp` for agent reasoning.
* **FastAPI Wrapper:** Production-ready API (`fastapi_agent_server.py`).
* **CLI Debug Mode:** Run `python updated_main_agent.py` and interactively search from terminal.

---

### 📂 Repo Layout

* `model/` → Local Mistral-7B model (`mistral-7b-openorca.gguf2.Q4_0.gguf`).
* `static/` → FastAPI web frontend/static assets.
* `fastapi_agent_server.py` → FastAPI wrapper server for deploying InfoScout.
* `fast_production_agent.py` → Production-oriented runner for deployment.
* `updated_main_agent.py` → Main agent script (search execution, memory, Playwright integration).
* `llama_cpp_integration_and_patches.py` → Helper for local LLM model loading (Mistral).
* `requirements.txt` → Dependencies.
* `ModelFile` → Model reference config.
* `README.md` → Documentation (this file).

---

### 🚀 Quick Start

```bash
git clone https://github.com/STiFLeR7/InfoScout.git
cd InfoScout
pip install -r requirements.txt
```

Run in CLI debug mode:

```bash
python updated_main_agent.py
```

Start FastAPI server:

```bash
uvicorn fastapi_agent_server:app --reload --host 0.0.0.0 --port 8000
```

---

### 🌟 Example Usage

```json
{
  "search_term": "best AI research papers 2025",
  "results": [
    {
      "rank": 1,
      "title": "Top 2025 AI Research Papers - arXiv",
      "link": "https://arxiv.org/abs/...",
      "snippet": "",
      "source": "google"
    },
    ...
  ]
}
```

---

### 🛠️ Roadmap

* [ ] Add ranking & summarization of results via Mistral model
* [ ] Multi-query chaining for deeper research
* [ ] Web dashboard (FastAPI + JS frontend)
* [ ] Dockerized deployment

