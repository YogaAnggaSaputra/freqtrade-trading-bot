"""
main.py � Discord Gateway Bot
==============================
Menggantikan Telegram Bot dengan Discord sebagai notification gateway.

Fitur:
  - Slash commands: /status /positions /kill /resume /equity /logs
  - Real-time alert forwarding dari Redis message bus ke Discord channel
  - Daily report otomatis via APScheduler
  - Authorization via DISCORD_AUTHORIZED_USER_ID
  - Discord Embeds (rich formatting dengan warna dan fields)
"""
import asyncio
import json
import os
from datetime import UTC, datetime

import discord
import redis.asyncio as aioredis
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from discord import app_commands
from sqlalchemy import case, desc, func, select

from shared.db.models import (
    Deployment,
    Incident,
    KillSwitchLog,
    Position,
    TradeDossier,
)
from shared.db.session import AsyncSessionLocal, init_db
from shared.messaging import Channels, MessageBus
from shared.schemas import KillSwitchLevelEnum
from shared.security import get_secret, load_secrets_into_env

load_secrets_into_env()
logger = structlog.get_logger("discord_bot")

DISCORD_TOKEN         = get_secret("discord_bot_token")
DISCORD_CHANNEL_ID    = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
DISCORD_ALERT_CHANNEL = int(os.getenv("DISCORD_ALERT_CHANNEL_ID", str(DISCORD_CHANNEL_ID)))
AUTHORIZED_USER_ID    = int(os.getenv("DISCORD_AUTHORIZED_USER_ID", "0"))

message_bus = MessageBus()
scheduler   = AsyncIOScheduler()
_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Lazy-init Redis untuk sinkronisasi kill_switch:level (dibaca risk-gateway)."""
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=get_secret("redis_password"),
            decode_responses=True,
        )
    return _redis


async def set_kill_switch_key(level: str) -> None:
    """Tulis kill switch level ke Redis — sumber kebenaran yang sama dengan risk-gateway.

    level 'green' menghapus key (normal operation). Gagal menulis hanya di-log,
    jangan sampai menghalangi feedback ke operator.
    """
    try:
        r = await get_redis()
        if not level or level.lower() in ("green", "none"):
            await r.delete("kill_switch:level")
        else:
            await r.set("kill_switch:level", level.lower())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to write kill_switch:level to Redis", error=str(e))

# Color palette
COLOR_GREEN  = 0x68D391
COLOR_RED    = 0xFC8181
COLOR_YELLOW = 0xF6E05E
COLOR_ORANGE = 0xF6AD55
COLOR_BLUE   = 0x63B3ED
COLOR_GRAY   = 0x4A5568


def authorized(interaction: discord.Interaction) -> bool:
    return AUTHORIZED_USER_ID == 0 or interaction.user.id == AUTHORIZED_USER_ID


class HermesBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()
        logger.info("Discord slash commands synced")

    async def on_ready(self):
        logger.info("Discord bot online", user=str(self.user))
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="Hermes Trading Bot"
            )
        )


bot = HermesBot()


async def send_embed(embed: discord.Embed, channel_id: int = DISCORD_ALERT_CHANNEL):
    channel = bot.get_channel(channel_id)
    if channel:
        await channel.send(embed=embed)


@bot.tree.command(name="status", description="Tampilkan status bot trading secara keseluruhan")
async def cmd_status(interaction: discord.Interaction):
    if not authorized(interaction):
        await interaction.response.send_message("Unauthorized", ephemeral=True)
        return
    await interaction.response.defer()
    async with AsyncSessionLocal() as db:
        dep_res = await db.execute(select(Deployment).where(Deployment.status == "active"))
        deployments = dep_res.scalars().all()
        pos_res = await db.execute(select(Position).where(Position.size > 0))
        positions = pos_res.scalars().all()
        ks_res = await db.execute(
            select(KillSwitchLog).order_by(desc(KillSwitchLog.activated_at)).limit(1)
        )
        ks = ks_res.scalars().first()
        eq_res = await db.execute(
            select(
                func.sum(TradeDossier.realized_pnl).label("total_pnl"),
                func.count(TradeDossier.trade_id).label("total_trades"),
            ).where(TradeDossier.closed_at.isnot(None))
        )
        equity = eq_res.first()

    ks_level = ks.level.value.upper() if ks else "GREEN"
    color_map = {
        "GREEN": COLOR_GREEN, "YELLOW": COLOR_YELLOW,
        "ORANGE": COLOR_ORANGE, "RED": COLOR_RED, "BLACK": 0x1A202C,
    }
    embed = discord.Embed(
        title="Status Report",
        color=color_map.get(ks_level, COLOR_GRAY),
        timestamp=datetime.now(UTC),
    )
    embed.add_field(
        name="Kill Switch",
        value=f"`{ks_level}` � {ks.reason if ks else 'Normal operation'}",
        inline=False,
    )
    deploy_text = "\n".join(
        f"- `{d.strategy_version}` v{d.config_version} [{d.environment}]"
        for d in deployments
    ) or "None"
    embed.add_field(name="Active Deployments", value=deploy_text, inline=False)
    pos_text = "\n".join(
        f"- `{p.pair}` {p.side.value.upper()} {p.size} @ {p.entry_price}"
        for p in positions
    ) or "No open positions"
    embed.add_field(name=f"Open Positions ({len(positions)})", value=pos_text, inline=False)
    if equity:
        embed.add_field(name="Total PnL", value=f"`{float(equity.total_pnl or 0):.4f}` USDT", inline=True)
        embed.add_field(name="Total Trades", value=str(equity.total_trades or 0), inline=True)
    embed.set_footer(text="Hermes Trading Bot")
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="positions", description="Tampilkan semua posisi yang sedang terbuka")
async def cmd_positions(interaction: discord.Interaction):
    if not authorized(interaction):
        await interaction.response.send_message("Unauthorized", ephemeral=True)
        return
    await interaction.response.defer()
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Position).where(Position.size > 0))
        positions = res.scalars().all()
    if not positions:
        embed = discord.Embed(title="Open Positions", description="No open positions", color=COLOR_GRAY)
        await interaction.followup.send(embed=embed)
        return
    embed = discord.Embed(title="Open Positions", color=COLOR_BLUE, timestamp=datetime.now(UTC))
    for p in positions:
        pnl = float(p.unrealized_pnl or 0)
        pnl_str = "+" if pnl >= 0 else ""
        embed.add_field(
            name=f"{'UP' if pnl >= 0 else 'DOWN'} {p.pair} {p.side.value.upper()}",
            value=(
                f"Size: `{p.size}` | Entry: `{p.entry_price}`\n"
                f"Mark: `{p.mark_price}` | PnL: `{pnl_str}{pnl:.4f} USDT`\n"
                f"SL: `{p.stop_loss}` | TP: `{p.take_profit}`\n"
                f"Lev: `{p.leverage}x` | Margin: `{p.margin_mode.value}`"
            ),
            inline=False,
        )
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="kill", description="Aktifkan Kill Switch darurat")
@app_commands.describe(
    level="Level kill switch: yellow | orange | red | black",
    reason="Alasan aktivasi kill switch",
)
@app_commands.choices(level=[
    app_commands.Choice(name="YELLOW - Pause entries, monitor", value="yellow"),
    app_commands.Choice(name="ORANGE - Halt all entries", value="orange"),
    app_commands.Choice(name="RED - Full trading halt", value="red"),
    app_commands.Choice(name="BLACK - Emergency nuclear stop", value="black"),
])
async def cmd_kill(interaction: discord.Interaction, level: str, reason: str = "Manual trigger via Discord"):
    if not authorized(interaction):
        await interaction.response.send_message("Unauthorized", ephemeral=True)
        return
    await interaction.response.defer()
    try:
        ks_level = KillSwitchLevelEnum(level)
    except ValueError:
        await interaction.followup.send("Invalid level.", ephemeral=True)
        return
    async with AsyncSessionLocal() as db:
        log = KillSwitchLog(
            level=ks_level,
            reason=reason,
            activated_by=f"discord:{interaction.user.name}",
            auto_recover=False,
        )
        db.add(log)
        await db.commit()
    await set_kill_switch_key(level)
    await message_bus.publish(Channels.KILL_SWITCH, {
        "level": level,
        "reason": reason,
        "activated_by": f"discord:{interaction.user.name}",
    })
    color_map = {"yellow": COLOR_YELLOW, "orange": COLOR_ORANGE, "red": COLOR_RED, "black": 0x1A202C}
    embed = discord.Embed(
        title=f"KILL SWITCH ACTIVATED: {level.upper()}",
        description=f"**Reason:** {reason}\n**By:** {interaction.user.mention}",
        color=color_map.get(level, COLOR_RED),
        timestamp=datetime.now(UTC),
    )
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="resume", description="Reset Kill Switch dan pulihkan trading normal")
async def cmd_resume(interaction: discord.Interaction):
    if not authorized(interaction):
        await interaction.response.send_message("Unauthorized", ephemeral=True)
        return
    await interaction.response.defer()
    async with AsyncSessionLocal() as db:
        log = KillSwitchLog(
            level=KillSwitchLevelEnum.YELLOW,
            reason="Manual resume by operator",
            activated_by=f"discord:{interaction.user.name}",
            auto_recover=True,
        )
        db.add(log)
        await db.commit()
    await set_kill_switch_key("green")
    await message_bus.publish(Channels.KILL_SWITCH, {
        "level": "green",
        "reason": "Manual resume",
        "activated_by": f"discord:{interaction.user.name}",
    })
    embed = discord.Embed(
        title="Kill Switch Reset",
        description="Trading has been **resumed**. All systems nominal.",
        color=COLOR_GREEN,
        timestamp=datetime.now(UTC),
    )
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="equity", description="Tampilkan ringkasan equity dan 10 trade terakhir")
async def cmd_equity(interaction: discord.Interaction):
    if not authorized(interaction):
        await interaction.response.send_message("Unauthorized", ephemeral=True)
        return
    await interaction.response.defer()
    async with AsyncSessionLocal() as db:
        eq_res = await db.execute(
            select(
                func.sum(TradeDossier.realized_pnl).label("total_pnl"),
                func.count(TradeDossier.trade_id).label("total_trades"),
                func.sum(case((TradeDossier.realized_pnl > 0, 1), else_=0)).label("wins"),
            ).where(TradeDossier.closed_at.isnot(None))
        )
        equity = eq_res.first()
        costs_res = await db.execute(
            select(TradeDossier.fees_funding_slippage).where(TradeDossier.closed_at.isnot(None))
        )
        total_costs = sum(
            (row[0] or {}).get("fees", 0) + (row[0] or {}).get("funding", 0)
            for row in costs_res.all()
        )
        recent_res = await db.execute(
            select(TradeDossier).where(TradeDossier.closed_at.isnot(None))
            .order_by(desc(TradeDossier.closed_at)).limit(10)
        )
        recent = recent_res.scalars().all()

    total_pnl    = float(equity.total_pnl or 0)
    total_trades = equity.total_trades or 0
    wins         = equity.wins or 0
    win_rate     = (wins / total_trades * 100) if total_trades > 0 else 0.0
    embed = discord.Embed(
        title="Equity Summary",
        color=COLOR_GREEN if total_pnl >= 0 else COLOR_RED,
        timestamp=datetime.now(UTC),
    )
    embed.add_field(name="Net PnL", value=f"`{total_pnl:.4f}` USDT", inline=True)
    embed.add_field(name="Total Trades", value=str(total_trades), inline=True)
    embed.add_field(name="Win Rate", value=f"`{win_rate:.1f}%`", inline=True)
    embed.add_field(name="Total Costs", value=f"`{total_costs:.4f}` USDT", inline=True)
    if recent:
        lines = []
        for t in recent:
            pnl_val = float(t.realized_pnl or 0)
            sign = "+" if pnl_val >= 0 else ""
            lines.append(f"`{t.pair}` {t.exit_reason} `{sign}{pnl_val:.4f}`")
        embed.add_field(name="Recent Trades", value="\n".join(lines), inline=False)
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="logs", description="Tampilkan N incident terbaru")
@app_commands.describe(count="Jumlah incident yang ditampilkan (default: 10)")
async def cmd_logs(interaction: discord.Interaction, count: int = 10):
    if not authorized(interaction):
        await interaction.response.send_message("Unauthorized", ephemeral=True)
        return
    await interaction.response.defer()
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(Incident).order_by(desc(Incident.created_at)).limit(count)
        )
        incidents = res.scalars().all()
    if not incidents:
        await interaction.followup.send("No recent incidents.")
        return
    embed = discord.Embed(
        title=f"Last {len(incidents)} Incidents",
        color=COLOR_RED,
        timestamp=datetime.now(UTC),
    )
    for i in incidents:
        embed.add_field(
            name=f"{i.severity.upper()} - {i.incident_type}",
            value=f"{i.title}\n`{i.created_at.strftime('%Y-%m-%d %H:%M:%S')}`",
            inline=False,
        )
    await interaction.followup.send(embed=embed)


async def alert_handler(msg: dict):
    """Forward alert dari Redis message bus ke Discord alert channel."""
    channel = bot.get_channel(DISCORD_ALERT_CHANNEL)
    if not channel:
        return
    alert_type = msg.get("type", "alert")
    if alert_type == "strategy_decay_alert":
        embed = discord.Embed(
            title="Strategy Decay Detected",
            color=COLOR_ORANGE,
            timestamp=datetime.now(UTC),
        )
        embed.add_field(name="Strategy", value=f"`{msg.get('strategy_version')}`", inline=True)
        embed.add_field(name="Decay Type", value=f"`{msg.get('decay_type')}`", inline=True)
        embed.add_field(name="Severity", value=f"`{msg.get('severity')}`", inline=True)
        embed.add_field(name="Recommendation", value=msg.get("recommendation", "-"), inline=False)
        await channel.send(embed=embed)
    elif alert_type == "hermes_proposal":
        embed = discord.Embed(
            title="New Hermes Proposal",
            description=f"**Problem Type:** `{msg.get('problem_type')}`",
            color=COLOR_BLUE,
            timestamp=datetime.now(UTC),
        )
        embed.add_field(name="Strategy", value=f"`{msg.get('strategy_version')}`", inline=True)
        embed.add_field(name="Proposal ID", value=f"`{msg.get('proposal_id')}`", inline=True)
        embed.set_footer(text="Experiment pipeline will start automatically if AUTO_ACCEPT_PROPOSALS=true")
        await channel.send(embed=embed)
    elif alert_type == "experiment_auto_started":
        embed = discord.Embed(
            title="Experiment Pipeline Started",
            description="Hermes proposal diterima dan pipeline pengujian dimulai otomatis.",
            color=COLOR_BLUE,
            timestamp=datetime.now(UTC),
        )
        embed.add_field(name="Experiment ID", value=f"`{msg.get('experiment_id')}`", inline=False)
        embed.add_field(name="Proposal ID", value=f"`{msg.get('proposal_id')}`", inline=True)
        embed.add_field(name="Problem Type", value=f"`{msg.get('problem_type')}`", inline=True)
        embed.add_field(name="Strategy", value=f"`{msg.get('strategy_version')}`", inline=True)
        embed.add_field(
            name="Pipeline Stages",
            value="Backtest -> Walk-forward -> Stress Test -> Monte Carlo\nPromote ke live tetap membutuhkan approval manual",
            inline=False,
        )
        await channel.send(embed=embed)
    elif alert_type == "loss_pattern_detected":
        severity = msg.get("severity", "medium")
        color_map = {"high": COLOR_RED, "medium": COLOR_ORANGE, "low": COLOR_YELLOW}
        embed = discord.Embed(
            title="Loss Pattern Detected",
            description=f"**Pattern:** `{msg.get('pattern')}`",
            color=color_map.get(severity, COLOR_YELLOW),
            timestamp=datetime.now(UTC),
        )
        embed.add_field(name="Severity", value=f"`{severity.upper()}`", inline=True)
        embed.add_field(name="Incident ID", value=f"`{msg.get('incident_id', 'N/A')}`", inline=True)
        embed.set_footer(text="Hermes agent will analyze and generate proposal")
        await channel.send(embed=embed)
    else:
        text = msg.get("text", json.dumps(msg, indent=2))
        embed = discord.Embed(
            title="Alert",
            description=f"```{text[:2000]}```",
            color=COLOR_YELLOW,
            timestamp=datetime.now(UTC),
        )
        await channel.send(embed=embed)


async def handle_order_update(msg: dict):
    """Notifikasi order terisi (entry/exit fill) ke Discord."""
    channel = bot.get_channel(DISCORD_ALERT_CHANNEL)
    if not channel:
        return
    order = msg.get("order", {})
    side = str(order.get("side", "")).upper()
    status = str(order.get("status", "")).upper()
    pair = order.get("pair", "?")
    filled = order.get("filled", "0")
    avg = order.get("avg_price") or order.get("price", "0")
    # Hanya notif kalau order FILLED (bukan NEW/partial baru)
    if status not in ("FILLED", "PARTIALLY_FILLED", "CLOSED"):
        return
    is_entry = side in ("BUY", "OPEN", "LONG")
    color = COLOR_GREEN if is_entry else COLOR_RED
    title = "🟢 Entry Filled" if is_entry else "🔴 Exit Filled"
    embed = discord.Embed(
        title=title,
        description=f"`{pair}`",
        color=color,
        timestamp=datetime.now(UTC),
    )
    embed.add_field(name="Side", value=f"`{side}`", inline=True)
    embed.add_field(name="Filled", value=f"`{filled}`", inline=True)
    embed.add_field(name="Avg Price", value=f"`{avg}`", inline=True)
    embed.add_field(name="Leverage", value=f"`{order.get('leverage', '1')}x`", inline=True)
    await channel.send(embed=embed)


# Track posisi terakhir (pair -> size) buat deteksi transisi open/close
_last_position_sizes: dict[str, float] = {}


async def handle_position_update(msg: dict):
    """Notifikasi pas posisi dibuka/ditutup (transisi size 0 <-> >0)."""
    channel = bot.get_channel(DISCORD_ALERT_CHANNEL)
    if not channel:
        return
    pos = msg.get("position", {})
    pair = pos.get("pair", "?")
    side = str(pos.get("side", "")).upper()
    size = float(pos.get("size", 0) or 0)
    prev = _last_position_sizes.get(pair, 0.0)
    _last_position_sizes[pair] = size

    if size > 0 and prev == 0:
        # Transisi: posisi BARU dibuka
        embed = discord.Embed(
            title="📈 Position Opened",
            description=f"`{pair}`",
            color=COLOR_GREEN,
            timestamp=datetime.now(UTC),
        )
        embed.add_field(name="Side", value=f"`{side}`", inline=True)
        embed.add_field(name="Size", value=f"`{size}`", inline=True)
        embed.add_field(name="Entry", value=f"`{pos.get('entry_price', '?')}`", inline=True)
        embed.add_field(name="Leverage", value=f"`{pos.get('leverage', '1')}x`", inline=True)
        embed.add_field(name="Liq. Price", value=f"`{pos.get('liquidation_price', 'N/A')}`", inline=True)
        await channel.send(embed=embed)
    elif size == 0 and prev > 0:
        # Transisi: posisi DITUTUP
        embed = discord.Embed(
            title="📉 Position Closed",
            description=f"`{pair}`",
            color=COLOR_RED,
            timestamp=datetime.now(UTC),
        )
        embed.add_field(name="Side", value=f"`{side}`", inline=True)
        embed.add_field(name="Realized PnL", value=f"`{pos.get('realized_pnl', '?')}`", inline=True)
        await channel.send(embed=embed)


async def handle_trade_closed(msg: dict):
    """Notifikasi trade closed + PnL dari event TRADE_CLOSED."""
    channel = bot.get_channel(DISCORD_ALERT_CHANNEL)
    if not channel:
        return
    outcome = msg.get("outcome", msg)
    pair = outcome.get("pair", outcome.get("symbol", "?"))
    pnl = outcome.get("realized_pnl", outcome.get("pnl"))
    try:
        pnl_f = float(pnl) if pnl is not None else None
        color = COLOR_GREEN if (pnl_f or 0) >= 0 else COLOR_RED
        pnl_str = f"{pnl_f:+.2f} USDT" if pnl_f is not None else "?"
    except (TypeError, ValueError):
        color, pnl_str = COLOR_GRAY, str(pnl)
    embed = discord.Embed(
        title="💰 Trade Closed",
        description=f"`{pair}`",
        color=color,
        timestamp=datetime.now(UTC),
    )
    embed.add_field(name="Realized PnL", value=f"`{pnl_str}`", inline=True)
    if outcome.get("exit_reason"):
        embed.add_field(name="Exit Reason", value=f"`{outcome.get('exit_reason')}`", inline=True)
    await channel.send(embed=embed)


async def setup_alerts():
    await message_bus.connect()
    await message_bus.subscribe(Channels.ALERT, alert_handler)
    await message_bus.subscribe(Channels.ORDER_UPDATE, handle_order_update)
    await message_bus.subscribe(Channels.POSITION_UPDATE, handle_position_update)
    await message_bus.subscribe(Channels.TRADE_CLOSED, handle_trade_closed)
    asyncio.create_task(message_bus.start_listening())


async def daily_report():
    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(
                func.count(TradeDossier.trade_id).label("trades"),
                func.sum(TradeDossier.realized_pnl).label("pnl"),
                func.sum(case((TradeDossier.realized_pnl > 0, 1), else_=0)).label("wins"),
                func.sum(case((TradeDossier.realized_pnl <= 0, 1), else_=0)).label("losses"),
            ).where(TradeDossier.closed_at.isnot(None))
        )
        stats = res.first()
    if not stats or not stats.trades:
        return
    wins     = stats.wins or 0
    losses   = stats.losses or 0
    trades   = stats.trades or 0
    pnl      = float(stats.pnl or 0)
    win_rate = wins / trades * 100 if trades > 0 else 0
    embed = discord.Embed(
        title=f"Daily Report - {datetime.now().strftime('%Y-%m-%d')}",
        color=COLOR_GREEN if pnl >= 0 else COLOR_RED,
        timestamp=datetime.now(UTC),
    )
    embed.add_field(name="Total Trades", value=str(trades), inline=True)
    embed.add_field(name="Wins", value=str(wins), inline=True)
    embed.add_field(name="Losses", value=str(losses), inline=True)
    embed.add_field(name="Win Rate", value=f"`{win_rate:.1f}%`", inline=True)
    embed.add_field(name="Net PnL", value=f"`{pnl:.4f}` USDT", inline=True)
    channel = bot.get_channel(DISCORD_ALERT_CHANNEL)
    if channel:
        await channel.send(embed=embed)


async def main():
    await init_db()
    await setup_alerts()
    scheduler.add_job(daily_report, CronTrigger(hour=0, minute=5))
    scheduler.start()
    logger.info("Discord bot starting...")
    async with bot:
        await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
