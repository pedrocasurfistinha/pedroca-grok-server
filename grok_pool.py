import json
import os
import random
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from curl_cffi import requests as cffi_requests

JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
UID_RE = re.compile(r"assets\.grok\.com/users/([0-9a-f-]{36})/")


def mask_sso(sso: str) -> str:
    if len(sso) <= 12:
        return "***"
    return f"{sso[:6]}…{sso[-8:]}"


def extract_ssos(raw_text: str) -> List[str]:
    """Extract unique Grok `sso` JWT tokens from raw text, multi-line paste, or Netscape cookie format."""
    found: List[str] = []
    seen = set()

    # 1. Check Netscape tab-delimited lines first so we don't grab non-sso tokens
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 7 and parts[5] == "sso":
                val = parts[6].strip()
                if val and val not in seen:
                    seen.add(val)
                    found.append(val)

    # 2. Fallback: extract any JWTs (eyJ...) from plain text / comma / newline lists
    if not found:
        for match in JWT_RE.findall(raw_text):
            if match not in seen:
                seen.add(match)
                found.append(match)

    return found


def _parse_proxies() -> List[str]:
    raw = os.getenv("PROXIES", "").strip()
    if not raw:
        return []
    items = re.split(r"[\r\n,]+", raw)
    return [p.strip() for p in items if p.strip() and not p.strip().startswith("#")]


class GrokCookiePool:
    """
    Thread-safe persistent pool of Grok `sso` cookies with:
    - Live validation (UID discovery + WebSocket session.create handshake)
    - Round-robin rotation
    - Automatic offline detection & permanent discard of dead cookies
    - Temporary cooldown on 429 rate limits
    - Optional Google Gemini silent fallback
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        on_cookie_discarded: Optional[Callable[[str, str, int], None]] = None,
    ):
        data_dir = os.getenv("DATA_DIR", "/data")
        if not os.path.isdir(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
            except OSError:
                data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
                os.makedirs(data_dir, exist_ok=True)

        self.storage_path = storage_path or os.path.join(data_dir, "grok_pool.json")
        self.on_cookie_discarded = on_cookie_discarded
        self.hf_dataset_repo = os.getenv("HF_DATASET_REPO", "").strip()
        self.hf_token = os.getenv("HF_TOKEN", "").strip()
        self._last_hf_sync = 0.0
        self._hf_sync_lock = threading.Lock()
        self._lock = threading.RLock()
        self._rr_index = 0
        self._entries: List[Dict[str, Any]] = []
        self._load()

    def _download_from_hf(self) -> None:
        if not (self.hf_dataset_repo and self.hf_token):
            return
        try:
            url = f"https://huggingface.co/datasets/{self.hf_dataset_repo}/resolve/main/grok_pool.json"
            r = httpx.get(
                url,
                headers={"Authorization": f"Bearer {self.hf_token}"},
                follow_redirects=True,
                timeout=30.0,
            )
            if r.status_code == 200 and r.text.strip().startswith("["):
                tmp = f"{self.storage_path}.dl"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(r.text)
                os.replace(tmp, self.storage_path)
                print(f"[HF] Loaded grok_pool.json from private dataset {self.hf_dataset_repo}", flush=True)
        except Exception as e:
            print(f"[HF] Failed to download pool: {e}", flush=True)

    def _upload_to_hf_bg(self) -> None:
        if not (self.hf_dataset_repo and self.hf_token):
            return
        now = time.time()
        if now - self._last_hf_sync < 45.0:
            return
        if not self._hf_sync_lock.acquire(blocking=False):
            return

        def _run():
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=self.hf_token)
                api.upload_file(
                    path_or_fileobj=self.storage_path,
                    path_in_repo="grok_pool.json",
                    repo_id=self.hf_dataset_repo,
                    repo_type="dataset",
                    commit_message="Sync grok_pool.json",
                )
                self._last_hf_sync = time.time()
            except Exception as e:
                print(f"[HF] Background sync error: {e}", flush=True)
            finally:
                self._hf_sync_lock.release()

        threading.Thread(target=_run, daemon=True).start()

    def _load(self) -> None:
        with self._lock:
            if not os.path.exists(self.storage_path):
                self._download_from_hf()
            if os.path.exists(self.storage_path):
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self._entries = data
                except Exception:
                    self._entries = []

    def _save(self) -> None:
        with self._lock:
            tmp = f"{self.storage_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.storage_path)
        self._upload_to_hf_bg()

    def _new_http_session(self, sso: str, use_proxy: bool = False) -> cffi_requests.Session:
        proxies = _parse_proxies() if use_proxy else []
        proxy = random.choice(proxies) if proxies else None
        s = (
            cffi_requests.Session(proxies={"http": proxy, "https": proxy})
            if proxy
            else cffi_requests.Session()
        )
        s.cookies.set("sso", sso, domain=".grok.com")
        s.cookies.set("sso-rw", sso, domain=".grok.com")
        return s

    def validate_sso(self, sso: str) -> Tuple[bool, Optional[str], str]:
        """
        Tests a single `sso` cookie live:
        1. Discovers `x-userid` via https://grok.com/
        2. Connects to wss://grok.com/ws/mgw/?uid=... and validates `session.created`
        """
        uid = None
        proxy_modes = [False, True] if _parse_proxies() else [False]
        last_err = "Cookie inválido ou expirado"

        for use_proxy in proxy_modes:
            try:
                s = self._new_http_session(sso, use_proxy=use_proxy)
                r = s.get("https://grok.com/", impersonate="chrome131", timeout=20)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code} ao abrir grok.com"
                    continue
                m = UID_RE.search(r.text)
                if not m:
                    last_err = "Sessão não autenticada (uid não encontrado)"
                    continue
                uid = m.group(1)

                # Validate WebSocket handshake (proves session is actually online)
                ws = s.ws_connect(
                    f"wss://grok.com/ws/mgw/?uid={uid}",
                    headers={
                        "origin": "https://grok.com",
                        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    },
                    impersonate="chrome131",
                    timeout=20,
                )
                try:
                    ws.send_str(
                        json.dumps(
                            {
                                "event": {
                                    "type": "session.create",
                                    "event_id": f"evt_val_{uuid.uuid4()}",
                                    "session": {
                                        "model": "fast",
                                        "x_grok": {
                                            "protocol_capabilities": [
                                                "conversation_attached",
                                                "custom_methods_v1",
                                            ],
                                            "use_chunk": True,
                                            "keep_context": False,
                                            "is_temporary": True,
                                            "enable_image_generation": False,
                                            "disable_artifact": True,
                                        },
                                    },
                                }
                            }
                        )
                    )
                    for _ in range(10):
                        frame = ws.recv_json()
                        etype = frame.get("event", {}).get("type")
                        if etype == "session.created":
                            return True, uid, "OK"
                        if etype == "error":
                            last_err = str(frame.get("event", {}).get("error") or "Erro WS")
                            break
                finally:
                    try:
                        ws.close()
                    except Exception:
                        pass
            except Exception as e:
                last_err = str(e)[:120]

        return False, None, last_err

    def add_ssos(self, raw_text: str) -> Dict[str, Any]:
        """Extract, validate live, save online cookies, and immediately discard offline ones."""
        candidates = extract_ssos(raw_text)
        added: List[Dict[str, str]] = []
        duplicates: List[str] = []
        offline: List[Dict[str, str]] = []

        with self._lock:
            existing_ssos = {e["sso"] for e in self._entries}

        for sso in candidates:
            if sso in existing_ssos:
                duplicates.append(mask_sso(sso))
                continue

            ok, uid, reason = self.validate_sso(sso)
            if ok and uid:
                entry = {
                    "id": str(uuid.uuid4())[:8],
                    "sso": sso,
                    "uid": uid,
                    "added_at": int(time.time()),
                    "last_ok": int(time.time()),
                    "requests": 0,
                    "cooldown_until": 0,
                }
                with self._lock:
                    # Re-check duplicate under lock
                    if not any(e["sso"] == sso for e in self._entries):
                        self._entries.append(entry)
                        self._save()
                        existing_ssos.add(sso)
                added.append({"id": entry["id"], "mask": mask_sso(sso), "uid": uid})
            else:
                offline.append({"mask": mask_sso(sso), "reason": reason})

        return {
            "total_found": len(candidates),
            "added": added,
            "duplicates": duplicates,
            "offline": offline,
            "active_count": self.active_count(),
        }

    def check_all(self) -> Dict[str, Any]:
        """Validate every stored cookie live and permanently discard any offline ones."""
        with self._lock:
            snapshot = list(self._entries)

        alive: List[Dict[str, Any]] = []
        removed: List[Dict[str, str]] = []

        for entry in snapshot:
            ok, uid, reason = self.validate_sso(entry["sso"])
            if ok and uid:
                entry["uid"] = uid
                entry["last_ok"] = int(time.time())
                entry["cooldown_until"] = 0
                alive.append(entry)
            else:
                removed.append(
                    {
                        "id": entry.get("id", "?"),
                        "mask": mask_sso(entry["sso"]),
                        "reason": reason,
                    }
                )

        with self._lock:
            self._entries = alive
            self._save()

        return {
            "alive": len(alive),
            "removed": removed,
        }

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries = []
            self._save()
            return count

    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def status(self) -> Dict[str, Any]:
        now = int(time.time())
        with self._lock:
            items = []
            for e in self._entries:
                cd = max(0, int(e.get("cooldown_until", 0)) - now)
                items.append(
                    {
                        "id": e.get("id", "?"),
                        "mask": mask_sso(e["sso"]),
                        "uid": e.get("uid", "?"),
                        "requests": e.get("requests", 0),
                        "cooldown_sec": cd,
                    }
                )
        return {
            "total_active": len(items),
            "ready_now": sum(1 for i in items if i["cooldown_sec"] == 0),
            "gemini_fallback": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "cookies": items,
        }

    def _discard_entry(self, entry_id: str, reason: str) -> None:
        mask = "???"
        remaining = 0
        removed = False
        with self._lock:
            for idx, e in enumerate(self._entries):
                if e.get("id") == entry_id:
                    mask = mask_sso(e["sso"])
                    self._entries.pop(idx)
                    self._save()
                    removed = True
                    break
            remaining = len(self._entries)

        if removed and self.on_cookie_discarded:
            try:
                self.on_cookie_discarded(mask, reason, remaining)
            except Exception:
                pass

    def _mark_cooldown(self, entry_id: str, seconds: int = 900) -> None:
        with self._lock:
            for e in self._entries:
                if e.get("id") == entry_id:
                    e["cooldown_until"] = int(time.time()) + seconds
                    self._save()
                    break

    def _mark_success(self, entry_id: str) -> None:
        with self._lock:
            for e in self._entries:
                if e.get("id") == entry_id:
                    e["requests"] = int(e.get("requests", 0)) + 1
                    e["last_ok"] = int(time.time())
                    e["cooldown_until"] = 0
                    self._save()
                    break

    def _next_candidates(self) -> List[Dict[str, Any]]:
        now = int(time.time())
        with self._lock:
            if not self._entries:
                return []
            n = len(self._entries)
            start = self._rr_index % n
            self._rr_index = (self._rr_index + 1) % n
            ordered = self._entries[start:] + self._entries[:start]
            ready = [e for e in ordered if int(e.get("cooldown_until", 0)) <= now]
            cooling = [e for e in ordered if int(e.get("cooldown_until", 0)) > now]
            return ready + cooling

    def _call_grok_single(
        self, entry: Dict[str, Any], prompt: str, model: str = "fast"
    ) -> str:
        sso = entry["sso"]
        uid = entry["uid"]
        proxy_modes = [False, True] if _parse_proxies() else [False]
        last_exc: Optional[Exception] = None

        for use_proxy in proxy_modes:
            ws = None
            try:
                s = self._new_http_session(sso, use_proxy=use_proxy)
                ws = s.ws_connect(
                    f"wss://grok.com/ws/mgw/?uid={uid}",
                    headers={
                        "origin": "https://grok.com",
                        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    },
                    impersonate="chrome131",
                    timeout=30,
                )

                ws.send_str(
                    json.dumps(
                        {
                            "event": {
                                "type": "session.create",
                                "event_id": f"evt_init_{uuid.uuid4()}",
                                "session": {
                                    "model": model,
                                    "x_grok": {
                                        "protocol_capabilities": [
                                            "conversation_attached",
                                            "custom_methods_v1",
                                        ],
                                        "use_chunk": True,
                                        "keep_context": False,
                                        "is_temporary": True,
                                        "enable_side_by_side": False,
                                        "force_side_by_side": False,
                                        "enable_image_generation": False,
                                        "disable_text_follow_ups": True,
                                        "disable_artifact": True,
                                        "force_concise": False,
                                    },
                                },
                            }
                        }
                    )
                )

                session_id = None
                for _ in range(10):
                    frame = ws.recv_json()
                    if frame.get("event", {}).get("type") == "session.created":
                        session_id = frame["session_id"]
                        break
                if not session_id:
                    raise RuntimeError("AUTH_FAILED: session.created nao recebido")

                ts = int(time.time() * 1000)
                ws.send_str(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "event": {
                                "type": "conversation.item.create",
                                "event_id": f"evt_msg_{ts}",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "x_grok": {
                                        "client_message_id": str(uuid.uuid4()),
                                        "input_chunks": [{"text": {"text": prompt}}],
                                    },
                                },
                            },
                        }
                    )
                )
                ws.send_str(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "event": {
                                "type": "response.create",
                                "event_id": f"evt_resp_{ts}",
                            },
                        }
                    )
                )

                full_text = ""
                while True:
                    try:
                        frame = ws.recv_json()
                    except Exception:
                        break

                    event = frame.get("event", {})
                    etype = event.get("type", "")

                    if etype == "response.chunk":
                        chunk = event.get("chunk", {})
                        text_obj = chunk.get("text")
                        if (
                            text_obj
                            and text_obj.get("channel") == "CHANNEL_ASSISTANT_RESPONSE"
                        ):
                            full_text += text_obj.get("text", "")
                    elif etype == "response.grok.output":
                        err = event.get("output", {}).get("stream_error")
                        if err:
                            raise RuntimeError(f"MODEL_ERROR: {err.get('message')}")
                    elif etype == "response.done":
                        break

                if not full_text.strip():
                    raise RuntimeError("EMPTY_RESPONSE")
                return full_text
            except Exception as e:
                last_exc = e
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

        assert last_exc is not None
        raise last_exc

    def _call_gemini_fallback(
        self, messages: List[Dict[str, str]], json_mode: bool = False
    ) -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        gemini_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
        system_parts = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "system" and m.get("content")
        ]
        user_parts = [
            m.get("content", "")
            for m in messages
            if m.get("role") != "system" and m.get("content")
        ]

        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": "\n\n".join(user_parts)}]}],
            "generationConfig": {"temperature": 0.2},
        }
        if system_parts:
            body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
        r = httpx.post(url, json=body, timeout=45.0)
        if r.status_code != 200:
            raise RuntimeError(f"Gemini fallback HTTP {r.status_code}")
        data = r.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return text.strip() or None

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "fast",
        json_mode: bool = False,
    ) -> Tuple[str, str]:
        """
        Executes a completion across the rotating Grok SSO pool.
        Auto-discards offline cookies and falls back to Gemini if configured.
        Returns (content, provider_used).
        """
        prompt_sections: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").replace("/no_think\n", "").strip()
            if not content:
                continue
            if role == "system":
                prompt_sections.append(content)
            else:
                prompt_sections.append(content)
        if json_mode:
            prompt_sections.append(
                "IMPORTANTE: Retorne EXCLUSIVAMENTE um JSON válido, sem markdown e sem texto fora do JSON."
            )
        prompt = "\n\n".join(prompt_sections)

        grok_model = model if model in ("fast", "auto", "expert") else "fast"
        candidates = self._next_candidates()
        last_error = "Nenhum cookie Grok ativo no pool."

        for entry in candidates:
            entry_id = entry["id"]
            try:
                text = self._call_grok_single(entry, prompt, model=grok_model)
                self._mark_success(entry_id)
                return text, f"grok:{entry_id}"
            except Exception as e:
                err_low = str(e).lower()
                last_error = str(e)[:160]
                if any(
                    k in err_low
                    for k in ("429", "rate limit", "too many", "limit reached", "quota")
                ):
                    self._mark_cooldown(entry_id, seconds=900)
                elif any(
                    k in err_low
                    for k in (
                        "401",
                        "403",
                        "unauthorized",
                        "auth_failed",
                        "session.created",
                        "forbidden",
                        "expired",
                    )
                ):
                    self._discard_entry(entry_id, last_error)
                else:
                    # Re-check live; if offline, discard immediately
                    ok, _, check_reason = self.validate_sso(entry["sso"])
                    if not ok:
                        self._discard_entry(entry_id, check_reason)
                    else:
                        self._mark_cooldown(entry_id, seconds=180)

        # Silent Google Gemini fallback if configured
        fallback_text = self._call_gemini_fallback(messages, json_mode=json_mode)
        if fallback_text:
            return fallback_text, "gemini-fallback"

        raise RuntimeError(f"Todos os cookies falharam ({last_error})")
