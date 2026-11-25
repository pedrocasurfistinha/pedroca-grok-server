import asyncio
import os
from typing import Optional, Set

import httpx

from grok_pool import GrokCookiePool


def _parse_admin_ids() -> Set[int]:
    raw = os.getenv("TELEGRAM_ADMIN_ID", "").strip()
    ids: Set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            ids.add(int(part))
    return ids


class TelegramCookieAdminBot:
    """
    Lightweight Telegram Bot (long polling via httpx) for managing Grok SSO cookies:
    - Send 1 or multiple raw `sso` tokens (or Netscape export) -> validates live, saves online, discards offline
    - /status -> active cookies, requests served, cooldowns
    - /check  -> re-tests all saved cookies live and auto-discards dead ones
    - /test   -> runs a live Grok test completion
    - /clear  -> removes all cookies
    """

    def __init__(self, pool: GrokCookiePool):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.admin_ids = _parse_admin_ids()
        self.pool = pool
        self._base_url = f"https://api.telegram.org/bot{self.token}"
        self._stop = asyncio.Event()
        self._offset = 0

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    async def send_message(self, chat_id: int, text: str) -> None:
        if not self.enabled:
            return
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.post(
                    f"{self._base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
        except Exception:
            pass

    async def notify_admins(self, text: str) -> None:
        for admin_id in self.admin_ids:
            await self.send_message(admin_id, text)

    def _is_authorized(self, user_id: Optional[int], chat_id: int) -> bool:
        if not self.admin_ids:
            return False
        return (user_id in self.admin_ids) or (chat_id in self.admin_ids)

    async def _handle_message(self, msg: dict) -> None:
        chat_id = msg.get("chat", {}).get("id")
        user_id = msg.get("from", {}).get("id")
        text = (msg.get("text") or "").strip()
        if not chat_id or not text:
            return

        if text.startswith("/myid"):
            await self.send_message(
                chat_id,
                f"Seu ID no Telegram: <code>{user_id}</code>\n"
                f"Configure esse valor no secret <code>TELEGRAM_ADMIN_ID</code> do Fly.io.",
            )
            return

        if not self._is_authorized(user_id, chat_id):
            await self.send_message(
                chat_id,
                "⛔ Acesso restrito. Use /myid para ver seu ID e configurar em TELEGRAM_ADMIN_ID.",
            )
            return

        cmd = text.split()[0].lower()

        if cmd in ("/start", "/help"):
            await self.send_message(
                chat_id,
                "<b>pedroca · Gerenciador de Cookies Grok</b>\n\n"
                "Envie um ou vários cookies <code>sso</code> direto aqui (um por linha) que eu testo na hora, salvo os online e descarto os offline!\n\n"
                "<b>Comandos:</b>\n"
                "• <code>/status</code> — ver cookies ativos e uso\n"
                "• <code>/check</code> — testar todos os cookies e descartar offline\n"
                "• <code>/test</code> — testar uma resposta real do Grok\n"
                "• <code>/clear</code> — limpar todos os cookies\n"
                "• <code>/add &lt;sso&gt;</code> — adicionar cookie(s) manualmente",
            )
            return

        if cmd == "/status":
            st = self.pool.status()
            lines = [
                f"<b>Status do Pool Grok</b>",
                f"• Cookies ativos: <b>{st['total_active']}</b> ({st['ready_now']} prontos agora)",
                f"• Backup Gemini: <b>{'Ativo' if st['gemini_fallback'] else 'Não configurado'}</b>",
            ]
            if st["cookies"]:
                lines.append("")
                for c in st["cookies"]:
                    cd_txt = (
                        f" ⏳ cooldown {c['cooldown_sec']}s"
                        if c["cooldown_sec"] > 0
                        else " 🟢 online"
                    )
                    lines.append(
                        f"• <code>{c['id']}</code> ({c['mask']}) — {c['requests']} reqs{cd_txt}"
                    )
            else:
                lines.append("\n⚠️ Nenhum cookie salvo. Cole seus cookies <code>sso</code> aqui!")
            await self.send_message(chat_id, "\n".join(lines))
            return

        if cmd == "/check":
            await self.send_message(chat_id, "🔍 Testando todos os cookies ao vivo...")
            report = await asyncio.to_thread(self.pool.check_all)
            lines = [
                f"<b>Verificação concluída</b>",
                f"• 🟢 Online mantidos: <b>{report['alive']}</b>",
                f"• 🗑️ Offline descartados: <b>{len(report['removed'])}</b>",
            ]
            for rem in report["removed"]:
                lines.append(f"  - <code>{rem['mask']}</code> ({rem['reason']})")
            await self.send_message(chat_id, "\n".join(lines))
            return

        if cmd == "/clear":
            removed = self.pool.clear()
            await self.send_message(
                chat_id, f"🗑️ Pool limpo ({removed} cookies removidos)."
            )
            return

        if cmd == "/test":
            await self.send_message(chat_id, "⚡ Testando resposta do Grok...")
            try:
                content, provider = await asyncio.to_thread(
                    self.pool.complete,
                    [{"role": "user", "content": "Responda apenas: Grok online no pedroca!"}],
                    "fast",
                    False,
                )
                await self.send_message(
                    chat_id,
                    f"✅ <b>Sucesso</b> (<code>{provider}</code>):\n{content[:300]}",
                )
            except Exception as e:
                await self.send_message(chat_id, f"❌ Falha no teste: {str(e)[:200]}")
            return

        # Default or /add -> extract and validate SSO cookie(s)
        payload = text[4:].strip() if cmd == "/add" else text
        await self.send_message(
            chat_id, "🔎 Validando cookie(s) <code>sso</code> ao vivo no Grok..."
        )
        res = await asyncio.to_thread(self.pool.add_ssos, payload)

        if res["total_found"] == 0:
            await self.send_message(
                chat_id,
                "⚠️ Não encontrei nenhum cookie <code>sso</code> (JWT começando com <code>eyJ...</code>) nessa mensagem.",
            )
            return

        lines = [
            "<b>Resultado da validação</b>",
            f"• ✅ Online adicionados: <b>{len(res['added'])}</b>",
            f"• ❌ Offline descartados: <b>{len(res['offline'])}</b>",
        ]
        if res["duplicates"]:
            lines.append(f"• ♻️ Já cadastrados: <b>{len(res['duplicates'])}</b>")
        for item in res["added"]:
            lines.append(f"  + <code>{item['id']}</code> ({item['mask']})")
        for item in res["offline"]:
            lines.append(f"  - <code>{item['mask']}</code> → {item['reason']}")
        lines.append(f"\nPool atual: <b>{res['active_count']}</b> cookie(s) ativos.")
        await self.send_message(chat_id, "\n".join(lines))

    async def run_forever(self) -> None:
        if not self.enabled:
            print("[TelegramBot] TELEGRAM_BOT_TOKEN não configurado; bot desativado.")
            return
        print("[TelegramBot] Iniciando long-polling...")
        async with httpx.AsyncClient(timeout=40.0) as client:
            while not self._stop.is_set():
                try:
                    r = await client.get(
                        f"{self._base_url}/getUpdates",
                        params={"offset": self._offset, "timeout": 25},
                    )
                    if r.status_code == 200:
                        data = r.json()
                        for upd in data.get("result", []):
                            self._offset = max(self._offset, upd["update_id"] + 1)
                            msg = upd.get("message")
                            if msg:
                                await self._handle_message(msg)
                    else:
                        await asyncio.sleep(5)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(5)

    def stop(self) -> None:
        self._stop.set()
