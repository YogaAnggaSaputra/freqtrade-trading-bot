"""
macro_filter.py
================
Macro News & Economic Event Calendar Filter — "Rem Darurat" otomatis.

Memblokir semua trade baru selama ±N menit di sekitar event ekonomi
berdampak tinggi (High Impact): Fed CPI, FOMC, NFP, GDP, Interest Rate, dll.

Pada saat event makro terjadi, volatilitas pasar crypto bisa naik ratusan persen
dalam hitungan detik — kondisi ini mematikan bot karena:
  - Spread melebar ekstrem (slippage masif)
  - Stop-loss tersapu dalam sekejap
  - Model AI tidak bisa membaca momentum irasional

Source Data (free, no auth required):
  Primary: ForexFactory Calendar (via faireconomy.media mirror)
    URL This Week : https://nfs.faireconomy.media/ff_calendar_thisweek.json
    URL Next Week : https://nfs.faireconomy.media/ff_calendar_nextweek.json

Konfigurasi (env vars):
  MACRO_BLOCK_MINUTES_BEFORE=30    : Menit sebelum event → mulai blokir
  MACRO_BLOCK_MINUTES_AFTER=60     : Menit setelah event → masih blokir
  MACRO_CURRENCIES=USD,EUR,GBP     : Mata uang yang dipantau
  MACRO_REFRESH_HOURS=6            : Seberapa sering fetch calendar
  MACRO_ENABLED=true               : Matikan dengan "false" jika tidak diinginkan

Window untuk event CRITICAL (CPI, FOMC, Fed) diperluas 2x secara otomatis.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp

logger = logging.getLogger("risk_gateway.macro_filter")

# ─── Configuration ─────────────────────────────────────────────────────────────
MACRO_BLOCK_MINUTES_BEFORE = int(os.getenv("MACRO_BLOCK_MINUTES_BEFORE", "30"))
MACRO_BLOCK_MINUTES_AFTER = int(os.getenv("MACRO_BLOCK_MINUTES_AFTER", "60"))
MACRO_CURRENCIES = [c.strip().upper() for c in os.getenv("MACRO_CURRENCIES", "USD,EUR,GBP").split(",")]
MACRO_REFRESH_HOURS = int(os.getenv("MACRO_REFRESH_HOURS", "6"))
MACRO_ENABLED = os.getenv("MACRO_ENABLED", "true").lower() == "true"

# ForexFactory calendar URLs (free, no auth)
FF_THIS_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FF_NEXT_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

# Keyword yang menandai event paling kritis untuk crypto
CRITICAL_KEYWORDS = frozenset({
    "cpi", "fomc", "fed", "federal reserve", "interest rate", "inflation",
    "nonfarm", "nfp", "gdp", "unemployment", "jerome powell",
    "rate decision", "monetary policy", "consumer price", "producer price",
    "pce", "core inflation", "sec", "regulation",
})


class MacroEvent:
    """Satu event berita ekonomi high-impact."""

    __slots__ = ("title", "country", "impact", "dt", "is_critical")

    def __init__(self, title: str, country: str, impact: str, dt: datetime) -> None:
        self.title = title
        self.country = country
        self.impact = impact
        self.dt = dt
        self.is_critical = self._classify_critical(title)

    @staticmethod
    def _classify_critical(title: str) -> bool:
        title_lower = title.lower()
        return any(kw in title_lower for kw in CRITICAL_KEYWORDS)

    def get_block_window(self) -> tuple[datetime, datetime]:
        """Hitung window pemblokiran. Event critical mendapat window 2x lebih panjang."""
        multiplier = 2 if self.is_critical else 1
        window_before = timedelta(minutes=MACRO_BLOCK_MINUTES_BEFORE * multiplier)
        window_after = timedelta(minutes=MACRO_BLOCK_MINUTES_AFTER * multiplier)
        return (self.dt - window_before, self.dt + window_after)

    def check_blocking(self, now: datetime) -> tuple[bool, str]:
        """
        Cek apakah event ini sedang aktif memblokir trading.
        Returns: (is_blocking, reason_message)
        """
        window_start, window_end = self.get_block_window()

        if window_start <= now <= window_end:
            event_label = "⚠️ CRITICAL" if self.is_critical else "HIGH impact"
            if now < self.dt:
                mins = int((self.dt - now).total_seconds() / 60)
                reason = (
                    f"{event_label} event in {mins}min: "
                    f"{self.title} ({self.country}) — trading blocked"
                )
            else:
                mins = int((now - self.dt).total_seconds() / 60)
                reason = (
                    f"{event_label} event {mins}min ago: "
                    f"{self.title} ({self.country}) — cooldown active"
                )
            return True, reason

        return False, ""

    def to_dict(self) -> dict[str, Any]:
        window_start, window_end = self.get_block_window()
        return {
            "title": self.title,
            "country": self.country,
            "impact": self.impact,
            "event_time": self.dt.isoformat(),
            "is_critical": self.is_critical,
            "block_window_start": window_start.isoformat(),
            "block_window_end": window_end.isoformat(),
            "block_before_min": MACRO_BLOCK_MINUTES_BEFORE * (2 if self.is_critical else 1),
            "block_after_min": MACRO_BLOCK_MINUTES_AFTER * (2 if self.is_critical else 1),
        }


class MacroFilter:
    """
    Mengambil, menyimpan, dan memeriksa economic calendar untuk memblokir
    trading saat ada high-impact event makro.
    """

    def __init__(self) -> None:
        self._events: list[MacroEvent] = []
        self._last_refresh: datetime | None = None
        self._running = False

    async def start(self) -> None:
        """Mulai background refresh loop dan fetch calendar pertama kali."""
        if not MACRO_ENABLED:
            logger.info("MacroFilter disabled via MACRO_ENABLED=false")
            return

        self._running = True
        # Fetch segera saat startup
        await self._refresh_calendar()
        asyncio.create_task(self._refresh_loop())
        logger.info(
            "MacroFilter started — block window: -%dmin / +%dmin | currencies: %s",
            MACRO_BLOCK_MINUTES_BEFORE,
            MACRO_BLOCK_MINUTES_AFTER,
            ", ".join(MACRO_CURRENCIES),
        )

    async def stop(self) -> None:
        self._running = False

    async def _refresh_loop(self) -> None:
        """Refresh calendar setiap MACRO_REFRESH_HOURS jam."""
        while self._running:
            await asyncio.sleep(MACRO_REFRESH_HOURS * 3600)
            try:
                await self._refresh_calendar()
            except Exception as e:  # noqa: BLE001
                logger.error("MacroFilter refresh failed: %s", e)

    async def _refresh_calendar(self) -> None:
        """Ambil dan parse calendar dari ForexFactory."""
        events: list[MacroEvent] = []
        headers = {"User-Agent": "TradingBot/1.0 (Economic Calendar Monitor)"}

        try:
            timeout = aiohttp.ClientTimeout(total=12)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                for url in (FF_THIS_WEEK_URL, FF_NEXT_WEEK_URL):
                    try:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json(content_type=None)
                                parsed = self._parse_ff_events(data)
                                events.extend(parsed)
                                logger.debug("Fetched %d events from %s", len(parsed), url)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to fetch %s: %s", url, e)
        except Exception as e:  # noqa: BLE001
            logger.error("MacroFilter HTTP session error: %s", e)

        if events:
            self._events = events
            self._last_refresh = datetime.now(UTC)
            high = [e for e in events if e.impact.lower() == "high"]
            critical = [e for e in high if e.is_critical]
            logger.info(
                "MacroFilter calendar updated: %d total | %d high-impact | %d critical",
                len(events), len(high), len(critical),
            )
        else:
            logger.warning(
                "MacroFilter: calendar fetch returned no events — "
                "using %d cached events", len(self._events)
            )

    @staticmethod
    def _parse_ff_events(data: list[dict[str, Any]]) -> list[MacroEvent]:
        """
        Parse ForexFactory JSON format.
        Fields: title, country, date (YYYY-MM-DD), time (12:30am), impact (High/Medium/Low)
        """
        events: list[MacroEvent] = []

        for item in data:
            try:
                # Filter hanya High impact
                impact = str(item.get("impact", "")).strip().capitalize()
                if impact != "High":
                    continue

                # Filter hanya mata uang yang relevan
                country = str(item.get("country", "")).upper().strip()
                if not any(curr in country for curr in MACRO_CURRENCIES):
                    continue

                title = str(item.get("title", "")).strip()
                date_str = str(item.get("date", "")).strip()
                time_str = str(item.get("time", "")).strip()

                if not date_str or not title:
                    continue

                # Parse datetime — ForexFactory format: "2025-08-13", "8:30am"
                dt = _parse_ff_datetime(date_str, time_str)
                if dt is None:
                    continue

                events.append(MacroEvent(title=title, country=country, impact=impact, dt=dt))

            except Exception:  # noqa: BLE001
                continue

        return events

    # ─── Public API ────────────────────────────────────────────────────────────

    def check_blocking(self) -> tuple[bool, str, dict[str, Any] | None]:
        """
        Cek apakah saat ini ada event yang memblokir trading.

        Returns:
            (is_blocked, reason, event_details_dict)
        """
        if not MACRO_ENABLED:
            return False, "", None

        now = datetime.now(UTC)
        for event in self._events:
            is_blocking, reason = event.check_blocking(now)
            if is_blocking:
                return True, reason, event.to_dict()

        return False, "", None

    def get_upcoming_events(self, hours: int = 24) -> list[dict[str, Any]]:
        """Dapatkan semua high-impact event dalam N jam ke depan."""
        now = datetime.now(UTC)
        cutoff = now + timedelta(hours=hours)
        upcoming = []

        for event in self._events:
            if now <= event.dt <= cutoff:
                d = event.to_dict()
                d["minutes_away"] = int((event.dt - now).total_seconds() / 60)
                upcoming.append(d)

        return sorted(upcoming, key=lambda x: x["minutes_away"])

    def get_status(self) -> dict[str, Any]:
        """Status lengkap macro filter untuk health check / dashboard."""
        is_blocked, reason, blocking_event = self.check_blocking()
        return {
            "enabled": MACRO_ENABLED,
            "is_blocked": is_blocked,
            "block_reason": reason,
            "current_blocking_event": blocking_event,
            "total_events_cached": len(self._events),
            "high_impact_count": sum(1 for e in self._events if e.impact.lower() == "high"),
            "critical_count": sum(1 for e in self._events if e.is_critical),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "config": {
                "block_before_min": MACRO_BLOCK_MINUTES_BEFORE,
                "block_after_min": MACRO_BLOCK_MINUTES_AFTER,
                "monitored_currencies": MACRO_CURRENCIES,
                "refresh_hours": MACRO_REFRESH_HOURS,
            },
        }


# ─── Datetime Parsing Helper ───────────────────────────────────────────────────

def _parse_ff_datetime(date_str: str, time_str: str) -> datetime | None:
    """
    Parse ForexFactory date+time string ke UTC datetime.
    date_str: "2025-08-13"
    time_str: "8:30am" | "12:00pm" | "" | "All Day" | "Tentative"
    """
    try:
        if time_str.lower() in ("", "all day", "tentative", "tbd"):
            # Gunakan tengah hari sebagai placeholder
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
        else:
            # Parse "8:30am" / "12:00pm"
            time_str_clean = time_str.lower().replace(" ", "")
            try:
                dt_naive = datetime.strptime(f"{date_str} {time_str_clean}", "%Y-%m-%d %I:%M%p")
            except ValueError:
                # Try tanpa menit: "8am"
                dt_naive = datetime.strptime(f"{date_str} {time_str_clean}", "%Y-%m-%d %I%p")
            dt = dt_naive.replace(tzinfo=UTC)
        return dt
    except (ValueError, AttributeError):
        return None
