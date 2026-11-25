import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from grok_pool import GrokCookiePool
from telegram_bot import TelegramCookieAdminBot

COACH_API_KEY = os.getenv("COACH_API_KEY", "").strip()

bot_ref: Optional[TelegramCookieAdminBot] = None
main_loop: Optional[asyncio.AbstractEventLoop] = None


def _on_cookie_discarded(mask: str, reason: str, remaining: int) -> None:
    if bot_ref and bot_ref.enabled and main_loop and main_loop.is_running():
        msg = (
            f"⚠️ <b>Cookie Grok descartado (offline)</b>\n"
            f"• Cookie: <code>{mask}</code>\n"
            f"• Motivo: {reason[:120]}\n"
            f"• Cookies ativos restantes: <b>{remaining}</b>"
        )
        asyncio.run_coroutine_threadsafe(bot_ref.notify_admins(msg), main_loop)


pool = GrokCookiePool(on_cookie_discarded=_on_cookie_discarded)
bot = TelegramCookieAdminBot(pool)
bot_ref = bot


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global main_loop
    main_loop = asyncio.get_running_loop()

    # Optional seed cookies from Fly Secret INITIAL_GROK_SSOS if pool is empty
    initial_ssos = os.getenv("INITIAL_GROK_SSOS", "").strip()
    if initial_ssos and pool.active_count() == 0:
        asyncio.create_task(asyncio.to_thread(pool.add_ssos, initial_ssos))

    bot_task = asyncio.create_task(bot.run_forever())
    try:
        yield
    finally:
        bot.stop()
        bot_task.cancel()


app = FastAPI(title="pedroca-grok-coach", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "fast"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 700
    response_format: Optional[Dict[str, Any]] = None


@app.get("/")
@app.get("/health")
def health():
    st = pool.status()
    return {
        "status": "ok",
        "active_cookies": st["total_active"],
        "ready_now": st["ready_now"],
        "gemini_fallback": st["gemini_fallback"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
):
    if COACH_API_KEY:
        expected = f"Bearer {COACH_API_KEY}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    json_mode = (req.response_format or {}).get("type") == "json_object"
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        content, provider = await asyncio.to_thread(
            pool.complete, msgs, req.model or "fast", json_mode
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Nenhum cookie Grok disponível no momento.",
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": provider,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
