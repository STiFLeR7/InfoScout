# fast_production_agent.py
"""
Production-Optimized AI Web Agent
Balances speed with real web scraping capabilities
(llama-cpp backend)
"""

import asyncio
import json
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from playwright.async_api import async_playwright, Page, Browser
from pathlib import Path

# Use the llama-cpp-based fast parser
from llama_cpp_integration_and_patches import FastLlamaParser as _FastLlamaParser

# Optimized Task Memory
class FastTaskMemory:
    def __init__(self, db_path: str = "fast_memory.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT,
                result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def save_task(self, instruction: str, result: Any):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tasks (instruction, result) VALUES (?, ?)",
            (instruction, json.dumps(result))
        )
        conn.commit()
        conn.close()

@dataclass
class FastTaskResult:
    success: bool
    data: Any
    error_message: Optional[str] = None
    steps_taken: List[str] = None
    execution_time: float = 0

class FastLLMParser:
    """
    Compatibility wrapper around FastLlamaParser (llama-cpp).
    Provides:
      - async quick_parse_async(instruction) -> Dict
      - sync quick_parse(instruction) -> Dict (uses asyncio.run when safe)
    """
    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False, n_ctx: int = 1024):
        self.model_path = model_path
        self.demo_mode = demo_mode
        try:
            if not demo_mode:
                self._impl = _FastLlamaParser(model_path=model_path, demo_mode=demo_mode, n_ctx=n_ctx)
            else:
                self._impl = None
        except Exception as e:
            print(f"[FastLLMParser] Failed to init FastLlamaParser: {e}")
            self._impl = None

    def is_ollama_available(self) -> bool:
        """Compatibility method name: returns True if local parser is initialized"""
        return self._impl is not None

    async def quick_parse_async(self, instruction: str) -> Dict[str, Any]:
        """Async parsing entrypoint used by the agent (await this inside async functions)."""
        # Fast rule-based fallback if _impl missing
        if not self._impl:
            return self._fast_rule_parse(instruction)

        try:
            return await self._impl.quick_parse(instruction)
        except Exception as e:
            print(f"[FastLLMParser] fast parser failed: {e}; falling back to rule-based.")
            return self._fast_rule_parse(instruction)

    def quick_parse(self, instruction: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for environments where async can't be awaited.
        WARNING: Do not call inside an already-running asyncio event loop (will raise).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are inside an event loop; prefer async entrypoint
                raise RuntimeError("Event loop is running; use quick_parse_async inside async functions.")
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.quick_parse_async(instruction))
        except Exception:
            # Fallback to run_until_complete on a new loop
            return asyncio.run(self.quick_parse_async(instruction))

    # ---------- original rule-based fallbacks (kept for speed) ----------
    def _fast_rule_parse(self, instruction: str) -> Dict[str, Any]:
        """Lightning-fast rule-based parsing (fallback)."""
        instruction_lower = instruction.lower()
        search_term = self._extract_search_term(instruction_lower)
        limit = self._extract_limit(instruction_lower)
        price_limit = self._extract_price(instruction_lower)
        
        return {
            "action": "search_and_extract",
            "search_term": search_term,
            "limit": limit,
            "max_price": price_limit,
            "site": "amazon.com"
        }

    def _is_complex_instruction(self, instruction: str) -> bool:
        complex_words = ['compare', 'filter', 'rank', 'best', 'analyze', 'export', 'save']
        return any(word in instruction.lower() for word in complex_words)

    def _extract_search_term(self, instruction: str) -> str:
        patterns = [
            (r'search for ([^,]+)', 1),
            (r'find ([^,]+)', 1), 
            (r'get ([^,]+)', 1)
        ]
        for pattern, group in patterns:
            match = re.search(pattern, instruction)
            if match:
                return match.group(group).strip().split(' and ')[0]
        words = instruction.split()
        return ' '.join([w for w in words if len(w) > 3][:2])

    def _extract_limit(self, instruction: str) -> int:
        match = re.search(r'(?:top|first)\s+(\d+)', instruction)
        return int(match.group(1)) if match else 5

    def _extract_price(self, instruction: str) -> Optional[int]:
        match = re.search(r'under\s+\$?(\d+(?:k|K)?)', instruction)
        if match:
            price_str = match.group(1)
            if price_str.endswith(('k', 'K')):
                return int(price_str[:-1]) * 1000
            return int(price_str)
        return None


class FastWebAgent:
    """Production-optimized web agent"""
    
    def __init__(self, model_path: str = r"D:/YASH/model/mistral-7b-openorca.gguf2.Q4_0.gguf", demo_mode: bool = False):
        self.parser = FastLLMParser(model_path=model_path, demo_mode=demo_mode)
        self.memory = FastTaskMemory()
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
        # Speed optimization flags
        self.headless = True
        self.disable_images = True
        self.disable_js = True
        
        print("⚡ Initializing Fast Production Web Agent...")
    
    async def start(self):
        """Initialize optimized browser"""
        playwright = await async_playwright().start()
        
        # Optimized browser launch
        self.browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-ipc-flooding-protection',
                '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows',
                '--disable-gpu',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-default-apps'
            ]
        )
        
        # Create optimized page context
        context = await self.browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        self.page = await context.new_page()
        
        # Block unnecessary resources for speed
        if self.disable_images:
            await self.page.route("**/*.{png,jpg,jpeg,gif,svg,webp}", lambda route: route.abort())
        
        if self.disable_js:
            await self.page.add_init_script("window.addEventListener = () => {};")
        
        print("✅ Fast browser initialized")
    
    async def stop(self):
        """Clean up"""
        if self.browser:
            await self.browser.close()
    
    async def execute_instruction(self, instruction: str) -> FastTaskResult:
        """Execute instruction with speed optimization"""
        start_time = asyncio.get_event_loop().time()
        steps = []
        
        try:
            # Ensure agent is started before execution
            if not self.page:
                await self.start()

            # Fast parsing
            steps.append("🧠 Parsing instruction (fast mode)")
            # Use async parser entrypoint
            parsed_task = await self.parser.quick_parse_async(instruction)
            
            # Execute search
            if parsed_task.get("action") == "search_and_extract":
                result = await self._fast_search(parsed_task, steps)
            else:
                return FastTaskResult(
                    success=False,
                    data=None,
                    error_message=f"Unknown action: {parsed_task.get('action')}"
                )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            # attach execution time to result
            if isinstance(result, FastTaskResult):
                result.execution_time = execution_time
            else:
                # wrap if needed
                result = FastTaskResult(success=result.success, data=result.data, error_message=getattr(result, 'error_message', None), steps_taken=getattr(result, 'steps_taken', None), execution_time=execution_time)
            
            # Save to memory
            if result.success:
                self.memory.save_task(instruction, result.data)
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return FastTaskResult(
                success=False,
                data=None,
                error_message=f"Execution error: {str(e)}",
                steps_taken=steps,
                execution_time=execution_time
            )
    
    async def _fast_search(self, task: Dict[str, Any], steps: List[str]) -> FastTaskResult:
        """Optimized search with minimal waiting"""
        
        try:
            search_term = task.get("search_term") or task.get("query") or ""
            limit = int(task.get("limit", 5))
            max_price = task.get("max_price")
            
            # Fast navigation
            search_url = f"https://www.amazon.com/s?k={search_term.replace(' ', '+')}&ref=sr_pg_1"
            steps.append(f"🔍 Fast search: {search_url}")
            
            # Optimized page load
            await self.page.goto(
                search_url,
                wait_until="domcontentloaded",  # Much faster than networkidle
                timeout=10000  # Short timeout
            )
            
            # Quick element wait
            steps.append("⚡ Extracting products (speed mode)")
            try:
                await self.page.wait_for_selector(
                    '[data-component-type="s-search-result"]',
                    timeout=3000  # Very short wait
                )
            except:
                # If specific selector fails, try generic
                try:
                    await self.page.wait_for_selector('div', timeout=1000)
                except:
                    pass
            
            # Fast extraction
            products = await self._extract_products_fast(limit, max_price)
            steps.append(f"✅ Extracted {len(products)} products")
            
            return FastTaskResult(
                success=True,
                data={
                    "search_term": search_term,
                    "products": products,
                    "total_found": len(products),
                    "speed_optimized": True
                },
                steps_taken=steps
            )
            
        except Exception as e:
            return FastTaskResult(
                success=False,
                data=None,
                error_message=f"Search failed: {str(e)}",
                steps_taken=steps
            )
    
    async def _extract_products_fast(self, limit: int, max_price: Optional[int]) -> List[Dict]:
        """Super fast product extraction"""
        
        products = []
        
        try:
            # Get all product containers quickly
            product_elements = await self.page.query_selector_all(
                '[data-component-type="s-search-result"]'
            )
            
            # If Amazon selector fails, try generic
            if not product_elements:
                product_elements = await self.page.query_selector_all('.s-result-item')
            
            # Fast extraction loop
            for i, element in enumerate(product_elements[:limit * 2]):  # Get extra for filtering
                try:
                    # Parallel extraction for speed
                    title_task = element.query_selector('h2 span, .s-size-mini span')
                    price_task = element.query_selector('.a-price-whole, .a-offscreen')
                    rating_task = element.query_selector('.a-icon-alt')
                    link_task = element.query_selector('h2 a')
                    
                    # Wait for all at once
                    title_elem, price_elem, rating_elem, link_elem = await asyncio.gather(
                        title_task, price_task, rating_task, link_task,
                        return_exceptions=True
                    )
                    
                    # Quick text extraction
                    title = await title_elem.text_content() if not isinstance(title_elem, Exception) and title_elem else f"Product {i+1}"
                    price_text = await price_elem.text_content() if not isinstance(price_elem, Exception) and price_elem else "N/A"
                    rating = await rating_elem.get_attribute('title') if not isinstance(rating_elem, Exception) and rating_elem else "N/A"
                    link = await link_elem.get_attribute('href') if not isinstance(link_elem, Exception) and link_elem else "N/A"
                    
                    # Quick price filtering
                    if max_price:
                        try:
                            price_num = float(re.search(r'(\d+(?:\.\d+)?)', price_text.replace(',', '')).group(1))
                            if price_num > float(max_price):
                                continue
                        except:
                            pass  # Include if price parsing fails
                        
                    product = {
                        "rank": len(products) + 1,
                        "title": title.strip()[:100] if title else f"Product {len(products) + 1}",
                        "price": price_text.strip() if price_text else "N/A",
                        "rating": rating.strip() if rating else "N/A",
                        "link": f"https://amazon.com{link}" if link and not link.startswith('http') else link,
                        "source": "Amazon",
                        "extraction_speed": "optimized"
                    }
                    
                    products.append(product)
                    
                    # Stop when we have enough
                    if len(products) >= limit:
                        break
                        
                except Exception:
                    continue  # Skip failed extractions
            
        except Exception as e:
            print(f"Extraction error: {e}")
            # Fallback: return minimal data
            return [{
                "rank": 1,
                "title": f"Search results for {limit} items",
                "price": "N/A",
                "rating": "N/A", 
                "link": "N/A",
                "source": "Amazon",
                "extraction_speed": "fallback"
            }]
        
        return products

# Fast CLI
async def main():
    """Fast CLI interface"""
    agent = FastWebAgent()
    await agent.start()
    
    print("\n" + "⚡" * 50)
    print("FAST PRODUCTION AI WEB AGENT")
    print("⚡" * 50)
    print("Optimized for speed while maintaining functionality")
    print("-" * 50)
    
    try:
        while True:
            instruction = input("\n💨 Quick instruction: ").strip()
            
            if instruction.lower() == 'quit':
                break
            elif not instruction:
                continue
            
            print("⚡ Processing at maximum speed...")
            result = await agent.execute_instruction(instruction)
            
            print(f"\n⏱️ Completed in {result.execution_time:.1f} seconds!")
            
            if result.success:
                print("✅ SUCCESS!")
                
                if result.steps_taken:
                    for step in result.steps_taken:
                        print(f"  {step}")
                
                data = result.data
                if isinstance(data, dict) and 'products' in data:
                    products = data['products']
                    print(f"\n🛍️ Found {len(products)} products:")
                    for product in products:
                        print(f"  • {product['title'][:60]}...")
                        print(f"    💰 {product['price']} | ⭐ {product['rating']}")
            else:
                print(f"❌ Error: {result.error_message}")
    
    finally:
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
