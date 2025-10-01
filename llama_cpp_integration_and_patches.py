"""
llama_cpp_integration_and_patches.py

This file provides a drop-in replacement for the previous Ollama HTTP-based LLM calls
using llama-cpp-python (direct GGUF access).

What this contains:
- LlamaCppClient: thin wrapper around llama_cpp.Llama with synchronous interface and
  an async-friendly wrapper using asyncio.to_thread where needed.
- Updated parser classes that mirror the previous MistralLLMParser / FastLLMParser /
  AdvancedLLMParser / LocalLLM but call LlamaCppClient instead of Ollama HTTP.

How to use:
1. Install llama-cpp-python and its dependencies. On Windows you may need to follow
   the llama-cpp-python install instructions and have a compatible wheel or build toolchain.

   Suggested install (pip):
       pip install "llama-cpp-python>=0.1.58"

2. Place your GGUF at the path you gave me: D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf

3. Replace imports in your repo files:
   - from mistral_integration import MistralLLMParser -> you can keep the file but
     modify it to import and delegate to the LlamaCppClient in this document.
   - Alternatively, import LlamaCppClient and use it directly where needed.

4. This module uses blocking llama-cpp calls; calling code in async functions should
   call the async wrappers (e.g. .generate_async).

Notes:
- I intentionally avoid changing your repository files directly here; this document
  gives the exact replacements you can paste into each module or import from this module.
- If you want, I can now (in a follow-up) patch each file directly in the canvas with
  the minimal edits to import and use LlamaCppClient.

"""

from typing import Optional, Dict, Any, List
import time
import asyncio
import json
import os
import inspect

# Try to import llama_cpp; if not available, provide a helpful error when attempting to use.
try:
    from llama_cpp import Llama
    _LLAMA_AVAILABLE = True
except Exception as e:
    Llama = None
    _LLAMA_AVAILABLE = False


class LlamaCppClient:
    """Robust wrapper around llama_cpp.Llama that handles different API versions.

    - Detects whether the installed llama-cpp-python exposes `create()` (older),
      `generate()` (newer), or a chat-style `generate(messages=...)`.
    - Normalizes calls and responses to return a plain assistant string.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, verbose: bool = False, **kwargs):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.verbose = verbose
        self._client = None
        self._init_kwargs = kwargs
        self._api_type = None  # 'create', 'generate_prompt', 'generate_input', 'generate_messages'
        self._ensure_client()

    def _ensure_client(self):
        if self._client is not None:
            return

        # Lazy import already handled outside, but double-check
        if not _LLAMA_AVAILABLE:
            raise RuntimeError(
                "llama_cpp (llama-cpp-python) is not installed or failed to import."
            )

        # instantiate
        self._client = Llama(model_path=self.model_path, n_ctx=self.n_ctx, **self._init_kwargs)
        if self.verbose:
            print(f"[LlamaCppClient] Loaded model: {self.model_path}")

        # Detect API surface
        if hasattr(self._client, "create"):
            # older versions: client.create(prompt=...)
            self._api_type = "create"
            if self.verbose:
                print("[LlamaCppClient] Using 'create' API")
            return

        if hasattr(self._client, "generate"):
            sig = inspect.signature(self._client.generate)
            params = list(sig.parameters.keys())
            # try to infer whether generate accepts 'prompt', 'input', or 'messages'
            if "prompt" in params:
                self._api_type = "generate_prompt"
                if self.verbose:
                    print("[LlamaCppClient] Using 'generate(prompt=...)' API")
            elif "input" in params:
                self._api_type = "generate_input"
                if self.verbose:
                    print("[LlamaCppClient] Using 'generate(input=...)' API")
            elif "messages" in params:
                self._api_type = "generate_messages"
                if self.verbose:
                    print("[LlamaCppClient] Using 'generate(messages=...)' (chat) API")
            else:
                # fallback to generate with positional prompt
                self._api_type = "generate_fallback"
                if self.verbose:
                    print("[LlamaCppClient] Using 'generate(...)' fallback API")
            return

        # If nothing matched, raise
        raise RuntimeError("Unrecognized llama-cpp-python API: no create() or generate() found on Llama")

    def _extract_text_from_response(self, resp: Any) -> str:
        """Normalize various response shapes to assistant string."""
        if not resp:
            return ""

        # Common: resp is dict with choices -> text
        try:
            if isinstance(resp, dict) and "choices" in resp:
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    # choices entries sometimes have 'text' or 'message' keys
                    choice = choices[0]
                    if isinstance(choice, dict):
                        # chat-style might have 'message' -> {'role':..., 'content': ...}
                        if "text" in choice and choice["text"] is not None:
                            return str(choice["text"]).strip()
                        if "message" in choice and isinstance(choice["message"], dict):
                            # content might be list or string
                            content = choice["message"].get("content") or choice["message"].get("value")
                            if isinstance(content, list):
                                # list of dicts with 'content' keys -> join
                                texts = []
                                for c in content:
                                    if isinstance(c, dict):
                                        texts.append(str(c.get("content", "")).strip())
                                    else:
                                        texts.append(str(c))
                                return " ".join(texts).strip()
                            elif isinstance(content, str):
                                return content.strip()
                    # fallback stringify
                    return str(choice).strip()

            # Some versions return {'output':[{'id':..., 'content':[{'type':'text','text':"..."}]}]}
            if isinstance(resp, dict) and "output" in resp:
                out = resp["output"]
                if isinstance(out, list) and out:
                    first = out[0]
                    # search for content list
                    content = first.get("content") or first.get("contents")
                    if isinstance(content, list):
                        texts = []
                        for c in content:
                            if isinstance(c, dict) and ("text" in c or "content" in c):
                                texts.append(str(c.get("text") or c.get("content") or ""))
                        return " ".join(t for t in texts if t).strip()

            # If resp is plain string
            if isinstance(resp, str):
                return resp.strip()

            # If resp has __str__
            return str(resp).strip()
        except Exception as e:
            if self.verbose:
                print(f"[LlamaCppClient] Failed to normalize response: {e} -- raw resp: {resp}")
            try:
                return str(resp)
            except Exception:
                return ""

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Blocking generation call, robust across llama-cpp-python versions."""
        self._ensure_client()
        stop = stop or ["</s>", "###"]

        if self._api_type == "create":
            create_kwargs = dict(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.pop("top_p", 0.8),
                top_k=kwargs.pop("top_k", 40),
                stop=stop,
                **kwargs,
            )
            resp = self._client.create(**create_kwargs)
            return self._extract_text_from_response(resp)

        elif self._api_type == "generate_prompt":
            # newer generate(prompt=...)
            gen_kwargs = dict(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.pop("top_p", 0.8),
                top_k=kwargs.pop("top_k", 40),
                stop=stop,
                **kwargs,
            )
            resp = self._client.generate(**gen_kwargs)
            return self._extract_text_from_response(resp)

        elif self._api_type == "generate_input":
            # some builds expect 'input'
            gen_kwargs = dict(
                input=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.pop("top_p", 0.8),
                top_k=kwargs.pop("top_k", 40),
                stop=stop,
                **kwargs,
            )
            resp = self._client.generate(**gen_kwargs)
            return self._extract_text_from_response(resp)

        elif self._api_type == "generate_messages":
            # chat-style: convert the single prompt into a 'messages' list
            messages = [{"role": "user", "content": prompt}]
            gen_kwargs = dict(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs,
            )
            resp = self._client.generate(**gen_kwargs)
            return self._extract_text_from_response(resp)

        else:
            # fallback: try calling generate with prompt as positional arg
            try:
                resp = self._client.generate(prompt)
                return self._extract_text_from_response(resp)
            except Exception as e:
                raise RuntimeError(f"[LlamaCppClient] Failed to call model generate/create: {e}")

    async def generate_async(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Async wrapper that calls the blocking generate in a thread."""
        return await asyncio.to_thread(self.generate, prompt, max_tokens, temperature, stop, **kwargs)


# ------- Updated parser classes using LlamaCppClient -------

class MistralLlamaParser:
    """Replacement for prior MistralLLMParser but using llama-cpp-python directly."""

    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False, n_ctx: int = 2048):
        self.demo_mode = demo_mode
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.client: Optional[LlamaCppClient] = None

        if not demo_mode:
            self.client = LlamaCppClient(model_path=self.model_path, n_ctx=self.n_ctx, verbose=False)

    def setup_model(self) -> bool:
        """Validate model path and client creation. Returns True if model ready."""
        if self.demo_mode:
            return True
        try:
            self.client._ensure_client()
            return True
        except Exception as e:
            print(f"[MistralLlamaParser] Model setup failed: {e}")
            return False

    async def parse_complex_instruction(self, instruction: str, context: List[Dict] = None) -> List[Dict]:
        """Return a list of step-like dicts. Uses the Llama client.

        The original code expected TaskStep dataclasses; to keep changes minimal,
        callers can construct TaskStep from returned dicts. Here we return simple dicts.
        """
        if self.demo_mode:
            return self._rule_based_parse_instruction(instruction, context)

        if not self.setup_model():
            return self._rule_based_parse_instruction(instruction, context)

        prompt = self._create_mistral_prompt(instruction, context)
        # call the model
        response = await self.client.generate_async(prompt, max_tokens=800, temperature=0.2)

        steps = self._extract_steps_from_response(response)
        if steps:
            return steps
        return self._rule_based_parse_instruction(instruction, context)

    def _create_mistral_prompt(self, instruction: str, context: List[Dict] = None) -> str:
        context_str = f"\n\nPrevious task context:\n{json.dumps(context[-3:], indent=2)}" if context else ""
        prompt = (
            "You are an assistant that outputs ONLY a JSON array of steps. "
            "Each step is an object with: step_id, description, action, parameters, dependencies.\n\n"
            f"Instruction: {instruction}{context_str}\n\n"
            "Return only valid JSON array."
        )
        return prompt

    def _extract_steps_from_response(self, response: str) -> List[Dict]:
        # Try to find a JSON array in the response
        import re
        try:
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if not json_match:
                return []
            json_str = json_match.group(0)
            # sanitize common trailing commas
            json_str = json_str.replace(',}', '}').replace(',]', ']')
            data = json.loads(json_str)
            # ensure each step has required fields
            valid_steps = []
            for i, step in enumerate(data, 1):
                if 'step_id' not in step:
                    step['step_id'] = i
                valid_steps.append(step)
            return valid_steps
        except Exception as e:
            print(f"[MistralLlamaParser] Failed to parse steps: {e}")
            return []

    def _rule_based_parse_instruction(self, instruction: str, context: List[Dict] = None) -> List[Dict]:
        # Simple fallback similar to earlier rule-based implementations
        instruction_lower = instruction.lower()
        if 'compare' in instruction_lower:
            return [
                {"step_id": 1, "description": "Search for items", "action": "search", "parameters": {"search_term": instruction}, "dependencies": []},
                {"step_id": 2, "description": "Compare and rank", "action": "compare", "parameters": {"criteria": ["price", "rating"]}, "dependencies": [1]}
            ]
        return [{"step_id": 1, "description": f"Search for {instruction}", "action": "search", "parameters": {"search_term": instruction}, "dependencies": []}]


class FastLlamaParser:
    """Fast parser using llama-cpp-python for quick lightweight parsing."""

    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False, n_ctx: int = 1024):
        self.demo_mode = demo_mode
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.client: Optional[LlamaCppClient] = None
        if not self.demo_mode:
            self.client = LlamaCppClient(model_path=self.model_path, n_ctx=self.n_ctx, verbose=False)

    async def quick_parse(self, instruction: str) -> Dict[str, Any]:
        if self.demo_mode:
            # quick rule parse
            return {"action": "search_and_extract", "search_term": instruction, "limit": 5, "max_price": None, "site": "amazon.com"}

        prompt = f"Parse: \"{instruction}\"\nReturn JSON: { {'search_term': '...', 'limit': 5, 'max_price': None} }"
        resp = await self.client.generate_async(prompt, max_tokens=120, temperature=0.1)
        # try to extract JSON dict
        import re
        try:
            json_match = re.search(r"\{.*\}", resp, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
        except Exception:
            pass
        # fallback
        return {"action": "search_and_extract", "search_term": instruction, "limit": 5, "max_price": None, "site": "amazon.com"}


class AdvancedLlamaParser:
    """Advanced parser that can call the llama-cpp model for heavy parsing tasks."""

    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False, n_ctx: int = 2048):
        self.model_path = model_path
        self.demo_mode = demo_mode
        self.n_ctx = n_ctx
        self.client: Optional[LlamaCppClient] = None
        if not demo_mode:
            self.client = LlamaCppClient(model_path=self.model_path, n_ctx=self.n_ctx, verbose=False)

    async def parse_complex_instruction(self, instruction: str, context: List[Dict] = None) -> List[Dict]:
        if self.demo_mode:
            # fallback to simpler parser
            return FastLlamaParser(model_path=self.model_path, demo_mode=True).quick_parse(instruction)

        prompt = (
            "Parse this instruction into discrete executable steps for web automation:\n\n"
            f"Instruction: \"{instruction}\"\n\n"
            "Return a JSON array of steps, each with step_id, description, action, parameters, dependencies."
        )
        resp = await self.client.generate_async(prompt, max_tokens=400, temperature=0.15)
        # attempt to parse JSON array
        import re
        try:
            json_match = re.search(r"\[.*\]", resp, re.DOTALL)
            if json_match:
                steps = json.loads(json_match.group(0))
                return steps
        except Exception:
            pass
        # fallback
        return [{"step_id": 1, "description": f"Search for {instruction}", "action": "search", "parameters": {"search_term": instruction}, "dependencies": []}]


class LocalLlamaLLM:
    """Replacement for LocalLLM in ai_web_agent.py. Simple interface exposing parse_instruction.

    It uses FastLlamaParser or AdvancedLlamaParser depending on desired behavior.
    """

    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False):
        self.model_path = model_path
        self.demo_mode = demo_mode
        self.fast_parser = FastLlamaParser(model_path=model_path, demo_mode=demo_mode)
        self.advanced_parser = AdvancedLlamaParser(model_path=model_path, demo_mode=demo_mode)

    def parse_instruction(self, instruction: str, context: List[Dict] = None) -> Dict[str, Any]:
        """Synchronous wrapper that prefers fast parsing (rule-based), but will try Llama for complex queries."""
        instruct_lower = instruction.lower()
        complex_words = ['compare', 'filter', 'rank', 'best', 'analyze', 'export', 'save']
        if any(word in instruct_lower for word in complex_words) and not self.demo_mode:
            # call llama sync via to_thread
            parsed = asyncio.run(self.advanced_parser.parse_complex_instruction(instruction, context))
            # The advanced parser returns list of steps; we convert to a dict similar to old LocalLLM
            return {"action": "complex_workflow", "steps": parsed}

        # fallback to quick rule parsing
        return {"action": "search_and_extract", "search_term": instruction, "limit": 5, "filters": {}, "site": "amazon.com"}


# End of file
