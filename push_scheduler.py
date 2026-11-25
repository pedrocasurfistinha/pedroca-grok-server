import asyncio
import datetime
import json
import os
import time
from typing import Any, Dict, List

import httpx
from pywebpush import WebPushException, webpush


class WebPushScheduler:
    """
    Background scheduler that reads user Web Push subscriptions from Supabase
    and dispatches real VAPID / APNs push notifications (works on iPhone PWA
    even when the app is closed and screen is locked).
    """

    def __init__(self) -> None:
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        self.service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        self.vapid_private_key = os.getenv("VAPID_PRIVATE_KEY", "").strip()
        self.vapid_subject = os.getenv(
            "VAPID_SUBJECT", "mailto:pedroca@pedroca-app.vercel.app"
        ).strip()
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return bool(
            self.supabase_url and self.service_role_key and self.vapid_private_key
        )

    def _sb_headers(self) -> Dict[str, str]:
        return {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }

    def stop(self) -> None:
        self._stop.set()

    def _send_push_sync(self, sub: Dict[str, Any], payload: Dict[str, Any]) -> str:
        """Returns 'ok', 'gone' (404/410), or 'error'."""
        subscription_info = {
            "endpoint": sub["endpoint"],
            "keys": {
                "p256dh": sub["p256dh"],
                "auth": sub["auth"],
            },
        }
        try:
            webpush(
                subscription_info=subscription_info,
                data=json.dumps(payload, ensure_ascii=False),
                vapid_private_key=self.vapid_private_key,
                vapid_claims={"sub": self.vapid_subject},
                ttl=3600,
            )
            return "ok"
        except WebPushException as ex:
            status = getattr(getattr(ex, "response", None), "status_code", None)
            if status in (404, 410):
                return "gone"
            print(f"[PUSH] WebPushException ({status}): {ex}", flush=True)
            return "error"
        except Exception as ex:
            print(f"[PUSH] Unexpected error: {ex}", flush=True)
            return "error"

    async def _fetch_subscriptions(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        r = await client.get(
            f"{self.supabase_url}/rest/v1/push_subscriptions?select=*",
            headers=self._sb_headers(),
            timeout=15.0,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return data
        return []

    async def _update_subscription(
        self, client: httpx.AsyncClient, sub_id: str, patch: Dict[str, Any]
    ) -> None:
        await client.patch(
            f"{self.supabase_url}/rest/v1/push_subscriptions?id=eq.{sub_id}",
            headers=self._sb_headers(),
            json=patch,
            timeout=15.0,
        )

    async def _delete_subscription(self, client: httpx.AsyncClient, sub_id: str) -> None:
        await client.delete(
            f"{self.supabase_url}/rest/v1/push_subscriptions?id=eq.{sub_id}",
            headers=self._sb_headers(),
            timeout=15.0,
        )

    async def tick(self) -> None:
        if not self.enabled:
            return

        now_ts = int(time.time())
        utc_now = datetime.datetime.now(datetime.timezone.utc)

        async with httpx.AsyncClient() as client:
            subs = await self._fetch_subscriptions(client)
            for sub in subs:
                sub_id = sub.get("id")
                if not sub_id:
                    continue

                tz_offset = int(sub.get("tz_offset_minutes") or 180)
                local_dt = utc_now - datetime.timedelta(minutes=tz_offset)
                local_hour = local_dt.hour
                local_min = local_dt.minute
                local_today = local_dt.strftime("%Y-%m-%d")

                # 1. Water reminder
                water_enabled = bool(sub.get("water_enabled", True))
                water_interval = max(15, int(sub.get("water_interval_minutes") or 30))
                water_start = int(sub.get("water_start_hour") or 8)
                water_end = int(sub.get("water_end_hour") or 22)
                last_water = int(sub.get("last_water_push_at") or 0)

                if water_enabled and water_start <= local_hour < water_end:
                    if last_water == 0:
                        # Set initial baseline so user doesn't get spammed immediately on first register
                        await self._update_subscription(
                            client, sub_id, {"last_water_push_at": now_ts}
                        )
                    elif (now_ts - last_water) >= (water_interval * 60 - 30):
                        res = await asyncio.to_thread(
                            self._send_push_sync,
                            sub,
                            {
                                "title": "pedroca · Hora de beber água 💧",
                                "body": "Beba um copo de água agora para manter sua hidratação em dia!",
                                "tag": "water-hydration-reminder",
                                "tab": "diet",
                            },
                        )
                        if res == "gone":
                            await self._delete_subscription(client, sub_id)
                            continue
                        if res == "ok":
                            await self._update_subscription(
                                client, sub_id, {"last_water_push_at": now_ts}
                            )

                # 2. Daily workout reminder
                workout_enabled = bool(sub.get("workout_enabled", True))
                workout_time = str(sub.get("workout_time") or "18:00")
                last_workout_date = str(sub.get("last_workout_push_date") or "")

                if workout_enabled and last_workout_date != local_today:
                    try:
                        t_h, t_m = [int(x) for x in workout_time.split(":")[:2]]
                        now_mins = local_hour * 60 + local_min
                        target_mins = t_h * 60 + t_m
                        if target_mins <= now_mins < target_mins + 15:
                            res = await asyncio.to_thread(
                                self._send_push_sync,
                                sub,
                                {
                                    "title": "pedroca · Hora de treinar! 💪",
                                    "body": "Bora fazer seu treino de hoje e manter seu streak firme!",
                                    "tag": "daily-workout-reminder",
                                    "tab": "workouts",
                                },
                            )
                            if res == "gone":
                                await self._delete_subscription(client, sub_id)
                                continue
                            if res == "ok":
                                await self._update_subscription(
                                    client,
                                    sub_id,
                                    {"last_workout_push_date": local_today},
                                )
                    except Exception:
                        pass

    async def run_forever(self) -> None:
        if not self.enabled:
            print("[PUSH] WebPushScheduler disabled (missing SUPABASE/VAPID env vars).", flush=True)
            return
        print("[PUSH] WebPushScheduler started (checking every 60s).", flush=True)
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception as ex:
                print(f"[PUSH] Tick error: {ex}", flush=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                pass
